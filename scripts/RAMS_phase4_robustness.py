#!/usr/bin/env python3
"""RAMS Phase 4 robustness: 
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import RAMS
import RAMS_phase4 as P4


def paired_test_ci(y, p, frozen_t, final_t, nboot, alpha, seed):
    """Within-experiment paired participant bootstrap; NOT survey-design adjusted.
    The three differences use the SAME resampled participant indices.
    """
    y = np.asarray(y, dtype=int)
    f = np.asarray(p >= frozen_t, dtype=bool)
    r = np.asarray(p >= final_t, dtype=bool)
    positive = y == 1
    npos = int(positive.sum())
    if npos == 0:
        raise ValueError('No positive test outcomes')
    d_ref = r.astype(float) - f.astype(float)
    d_miss = ((positive & ~r).astype(float) - (positive & ~f).astype(float))
    point = {
        'delta_referral_fraction': float(d_ref.mean()),
        'delta_missed_per_1000': float(d_miss.mean() * 1000),
        'delta_sensitivity': float(-d_miss.sum() / npos),
    }
    rng = np.random.default_rng(seed)
    boot = {key: np.empty(nboot) for key in point}
    for start in range(0, nboot, 100):
        stop = min(nboot, start + 100)
        ix = rng.integers(0, len(y), size=(stop - start, len(y)))
        positives = positive[ix].sum(axis=1)
        boot['delta_referral_fraction'][start:stop] = d_ref[ix].mean(axis=1)
        boot['delta_missed_per_1000'][start:stop] = d_miss[ix].mean(axis=1) * 1000
        boot['delta_sensitivity'][start:stop] = np.divide(
            -d_miss[ix].sum(axis=1), positives,
            out=np.full(len(positives), np.nan), where=positives > 0)
    result = dict(point)
    for key, vals in boot.items():
        valid = vals[np.isfinite(vals)]
        result[key + '_ci_low'] = float(np.quantile(valid, alpha / 2)) if len(valid) else np.nan
        result[key + '_ci_high'] = float(np.quantile(valid, 1 - alpha / 2)) if len(valid) else np.nan
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root')
    ap.add_argument('--archive')
    ap.add_argument('--out')
    ap.add_argument('--penalties', nargs='+', type=float, default=[2, 5, 10, 20, 40])
    ap.add_argument('--uncertainty-penalty', type=float, default=1.)
    ap.add_argument('--alpha', type=float, default=.05)
    ap.add_argument('--selection-fraction', type=float, default=.60)
    ap.add_argument('--split-seed', type=int, default=2026)
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if not args.penalties or any(not np.isfinite(x) or x <= 0 for x in args.penalties):
        ap.error('Penalties must be finite and positive')
    if len(set(args.penalties)) != len(args.penalties):
        ap.error('Duplicate penalties are not permitted')
    if not np.isfinite(args.uncertainty_penalty) or args.uncertainty_penalty < 0:
        ap.error('Uncertainty penalty must be nonnegative')
    if not (0 < args.alpha < .5 and 0 < args.selection_fraction < 1):
        ap.error('alpha must be in (0,.5), selection fraction in (0,1)')
    if args.bootstrap < 100:
        ap.error('Bootstrap iterations must be >= 100')

    root = Path(args.root).expanduser().resolve() if args.root else Path(__file__).resolve().parent.parent
    archive_path = Path(args.archive).expanduser() if args.archive else root/'processed'/'08_model_results.zip'
    if not archive_path.is_absolute():
        archive_path = root/archive_path
    out = Path(args.out).expanduser() if args.out else root/'processed'/'RAMS'/'phase4_robustness'
    if not out.is_absolute():
        out = root/out
    out.mkdir(parents=True, exist_ok=True)

    arc = RAMS.Archive(archive_path)
    try:
        manifest = arc.table('07_model_design/05_experiment_manifest.csv')
        if len(manifest) != 336:
            raise ValueError(f'Expected 336 experiments; got {len(manifest)}')
        chosen = manifest.loc[manifest.apply(
            lambda r: (r.feature_tier, r.model) == RAMS.MODELS[r.study], axis=1)].copy()
        if len(chosen) != 28:
            raise ValueError(f'Expected 28 primary experiments; got {len(chosen)}')
        if args.quick:
            chosen = chosen.head(2)
        frozen_lookup = RAMS.archived_thresholds(arc)
        audits, policies, paired = [], [], []
        for ei, r in enumerate(chosen.itertuples(index=False)):
            exp = str(r.experiment_id)
            ids, ytr, ptr = RAMS.seed_average(arc, exp, 'oof', 'sigmoid')
            tids, yte, pte = RAMS.seed_average(arc, exp, 'test', 'sigmoid')
            if len(np.unique(ids)) != len(ids) or len(np.unique(tids)) != len(tids):
                raise ValueError(f'{exp}: duplicate SEQN')
            if np.intersect1d(ids, tids).size:
                raise ValueError(f'{exp}: training/test SEQN overlap')
            if not np.isfinite(ptr).all() or not np.isfinite(pte).all():
                raise ValueError(f'{exp}: nonfinite scores')
            if not np.isin(ytr, [0, 1]).all() or not np.isin(yte, [0, 1]).all():
                raise ValueError(f'{exp}: nonbinary outcomes')
            target = RAMS.TARGETS[r.study]
            frozen_t, source = RAMS.historical_threshold(
                frozen_lookup, exp, target, ytr, ptr)
            s, v = train_test_split(np.arange(len(ytr)),
                train_size=args.selection_fraction, stratify=ytr,
                random_state=args.split_seed)
            if (ytr[s] == 1).sum() == 0 or (ytr[v] == 1).sum() == 0:
                raise ValueError(f'{exp}: no positives in a split')
            base = RAMS.metrics(yte, pte, pte >= frozen_t)
            for pi, penalty in enumerate(args.penalties):
                proposed = P4.select_threshold(ytr[s], ptr[s], penalty,
                    args.uncertainty_penalty, args.alpha)
                delta, upper, accepted = P4.paired_verification(
                    ytr[v], ptr[v], proposed['threshold'], frozen_t,
                    penalty, args.alpha, args.bootstrap,
                    args.split_seed + 100003*ei + pi)
                final_t = proposed['threshold'] if accepted else frozen_t
                new = RAMS.metrics(yte, pte, pte >= final_t)
                common = dict(experiment_id=exp, study=r.study,
                    design=r.temporal_design, test_cycle=r.test_cycle,
                    miss_penalty=float(penalty),
                    uncertainty_penalty=args.uncertainty_penalty,
                    nominal_target=target)
                audits.append(dict(**common, frozen_threshold=float(frozen_t),
                    frozen_source=source, selection_n=len(s), verification_n=len(v),
                    selection_positives=int(ytr[s].sum()),
                    verification_positives=int(ytr[v].sum()),
                    proposed_threshold=float(proposed['threshold']),
                    selection_objective=float(proposed['objective']),
                    selection_missed_upper=float(proposed['missed_upper']),
                    verification_delta_loss=delta, verification_delta_upper=upper,
                    accepted=bool(accepted),
                    decision='adopt_candidate' if accepted else 'retain_frozen',
                    final_threshold=float(final_t)))
                for name, threshold, m in [
                    ('frozen_target', frozen_t, base), ('RAMS_RCO', final_t, new)]:
                    policies.append(dict(**common, policy=name, threshold=float(threshold),
                        target_held=bool(m['sensitivity'] >= target), **m))
                ci = paired_test_ci(yte, pte, frozen_t, final_t,
                    args.bootstrap, args.alpha, args.split_seed + 200003*ei + pi)
                paired.append(dict(**common, accepted=bool(accepted),
                    frozen_target_held=bool(base['sensitivity'] >= target),
                    rco_target_held=bool(new['sensitivity'] >= target), **ci))
                print(f'{exp} | lambda={penalty:g} | '
                      f'{"adopt" if accepted else "retain"} | '
                      f'referrals={new["referral_fraction"]:.3f} | '
                      f'sensitivity={new["sensitivity"]:.3f}', flush=True)

        audits_df, policy_df, paired_df = map(pd.DataFrame, (audits, policies, paired))
        audits_df.to_csv(out/'RAMS_phase4_robustness_threshold_audit.csv', index=False)
        policy_df.to_csv(out/'RAMS_phase4_robustness_policy_results.csv', index=False)
        paired_df.to_csv(out/'RAMS_phase4_robustness_paired_differences.csv', index=False)
        summary = policy_df.groupby(['study', 'miss_penalty', 'policy'], as_index=False).agg(
            experiments=('experiment_id', 'size'),
            mean_sensitivity=('sensitivity', 'mean'),
            minimum_sensitivity=('sensitivity', 'min'),
            mean_referral_fraction=('referral_fraction', 'mean'),
            mean_missed_per_1000=('missed_per_1000', 'mean'),
            fraction_targets_held=('target_held', 'mean'))
        adoption = audits_df.groupby(['study', 'miss_penalty'], as_index=False).agg(
            candidates_adopted=('accepted', 'sum'),
            experiments=('experiment_id', 'size'))
        summary = summary.merge(adoption, on=['study', 'miss_penalty'], how='left')
        summary.to_csv(out/'RAMS_phase4_robustness_summary.csv', index=False)
        config = dict(method='RAMS-RCO penalty robustness',
            n_experiments=int(len(chosen)), penalties=args.penalties,
            uncertainty_penalty=args.uncertainty_penalty, alpha=args.alpha,
            selection_fraction=args.selection_fraction,
            split_seed=args.split_seed, bootstrap=args.bootstrap,
            n_policy_rows=int(len(policy_df)), n_paired_rows=int(len(paired_df)),
            cautions=[
                'Penalty grid is a sensitivity analysis; no penalty is chosen using test results',
                'Per-experiment paired bootstrap CIs are exploratory and not NHANES survey adjusted',
                'Intervals do not correct for testing multiple penalties',
                'Temporal experiments may share participants; do not pool their intervals as independent',
                'Miss penalty is not a clinically validated cost ratio',
                'Verification loss uses referral plus penalty times missed fraction, without gamma',
                'No sensitivity guarantee and no prospective clinical validation'])
        (out/'RAMS_phase4_robustness_manifest.json').write_text(
            json.dumps(config, indent=2), encoding='utf-8')
        print(f'COMPLETED: {len(chosen)} experiments x {len(args.penalties)} penalties; outputs: {out}')
    finally:
        arc.close()


if __name__ == '__main__':
    main()
