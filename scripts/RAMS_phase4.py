#!/usr/bin/env python3
"""RAMS Phase 4: 
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import RAMS


def wilson_upper(k, n, alpha):
    if n == 0:
        raise ValueError('Empty sample')
    p = k / n
    z = norm.ppf(1 - alpha)
    den = 1 + z*z/n
    return float((p + z*z/(2*n) + z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / den)


def assess(y, p, threshold, penalty, gamma, alpha):
    refer = p >= threshold
    missed = (y == 1) & ~refer
    n = len(y)
    mr = missed.mean()
    upper = wilson_upper(int(missed.sum()), n, alpha)
    return dict(threshold=float(threshold), referral_fraction=float(refer.mean()),
                missed_fraction=float(mr), missed_upper=float(upper),
                uncertainty_allowance=float(upper-mr),
                objective=float(refer.mean()+penalty*mr+gamma*(upper-mr)),
                sensitivity=float(1-missed.sum()/sum(y==1)) if (y==1).any() else np.nan,
                missed_per_1000=float(1000*mr))


def candidate_thresholds(p):
    # All distinct score cutoffs plus refer-all and refer-none endpoints.
    return np.unique(np.r_[0., np.unique(p), np.nextafter(1., 2.)])


def select_threshold(y, p, penalty, gamma, alpha):
    # Evaluate exactly, rather than using test data or a heuristic threshold grid.
    best = None
    for threshold in candidate_thresholds(p):
        result = assess(y, p, threshold, penalty, gamma, alpha)
        # Tie-break toward fewer referrals, then lower threshold for stability.
        key = (result['objective'], result['referral_fraction'], threshold)
        if best is None or key < best[0]:
            best = (key, result)
    return best[1]


def paired_verification(y, p, candidate, frozen, penalty, alpha, nboot, seed):
    candidate_ref = p >= candidate
    frozen_ref = p >= frozen
    # Per-participant loss, bounded; no test outcomes enter verification.
    c = candidate_ref.astype(float) + penalty*((y == 1) & ~candidate_ref)
    f = frozen_ref.astype(float) + penalty*((y == 1) & ~frozen_ref)
    delta = c-f
    rng = np.random.default_rng(seed)
    # Chunk to avoid large bootstrap arrays.
    means = np.empty(nboot)
    for start in range(0, nboot, 100):
        end = min(start+100, nboot)
        ix = rng.integers(0, len(delta), size=(end-start, len(delta)))
        means[start:end] = delta[ix].mean(axis=1)
    upper = float(np.quantile(means, 1-alpha))
    # Require a strictly negative upper bound to adopt.
    return float(delta.mean()), upper, bool(upper < 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root')
    ap.add_argument('--archive')
    ap.add_argument('--out')
    ap.add_argument('--miss-penalty', type=float, default=20.)
    ap.add_argument('--uncertainty-penalty', type=float, default=1.)
    ap.add_argument('--alpha', type=float, default=.05)
    ap.add_argument('--selection-fraction', type=float, default=.60)
    ap.add_argument('--split-seed', type=int, default=2026)
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.miss_penalty <= 0 or args.uncertainty_penalty < 0:
        ap.error('Penalties must be positive (uncertainty penalty may be zero)')
    if not 0 < args.alpha < .5 or not 0 < args.selection_fraction < 1:
        ap.error('alpha must be in (0,.5); selection fraction in (0,1)')
    if args.bootstrap < 100:
        ap.error('--bootstrap must be at least 100')

    root = Path(args.root).expanduser().resolve() if args.root else Path(__file__).resolve().parent.parent
    archive_path = Path(args.archive).expanduser() if args.archive else root/'processed'/'08_model_results.zip'
    out = Path(args.out).expanduser() if args.out else root/'processed'/'RAMS'/'phase4'
    if not archive_path.is_absolute(): archive_path = root/archive_path
    if not out.is_absolute(): out = root/out
    out.mkdir(parents=True, exist_ok=True)

    arc = RAMS.Archive(archive_path)
    try:
        manifest = arc.table('07_model_design/05_experiment_manifest.csv')
        if len(manifest) != 336:
            raise ValueError(f'Expected 336 manifest experiments; got {len(manifest)}')
        selected = manifest.loc[manifest.apply(
            lambda r: (r.feature_tier, r.model) == RAMS.MODELS[r.study], axis=1)].copy()
        if len(selected) != 28:
            raise ValueError(f'Expected 28 primary experiments; got {len(selected)}')
        if args.quick: selected = selected.head(2)
        frozen = RAMS.archived_thresholds(arc)
        audits, policies, paired = [], [], []
        for r in selected.itertuples(index=False):
            exp = str(r.experiment_id)
            train_ids, ytr, ptr = RAMS.seed_average(arc, exp, 'oof', 'sigmoid')
            test_ids, yte, pte = RAMS.seed_average(arc, exp, 'test', 'sigmoid')
            if len(np.unique(train_ids)) != len(train_ids) or len(np.unique(test_ids)) != len(test_ids):
                raise ValueError(f'{exp}: duplicate participant identifiers')
            if np.intersect1d(train_ids, test_ids).size:
                raise ValueError(f'{exp}: OOF/test SEQN overlap')
            if not (np.isfinite(ptr).all() and np.isfinite(pte).all()):
                raise ValueError(f'{exp}: nonfinite predictions')
            target = RAMS.TARGETS[r.study]
            frozen_t, frozen_source = RAMS.historical_threshold(frozen, exp, target, ytr, ptr)
            ix = np.arange(len(ytr))
            s, v = train_test_split(ix, train_size=args.selection_fraction,
                                    stratify=ytr, random_state=args.split_seed)
            if (ytr[s] == 1).sum() == 0 or (ytr[v] == 1).sum() == 0:
                raise ValueError(f'{exp}: no positives in selection/verification')
            proposed = select_threshold(ytr[s], ptr[s], args.miss_penalty,
                                        args.uncertainty_penalty, args.alpha)
            delta, upper, accepted = paired_verification(
                ytr[v], ptr[v], proposed['threshold'], frozen_t,
                args.miss_penalty, args.alpha, args.bootstrap, args.split_seed+int(exp.split('_')[-1]))
            final_t = proposed['threshold'] if accepted else frozen_t
            decision = 'adopt_candidate' if accepted else 'retain_frozen'
            audits.append(dict(experiment_id=exp, study=r.study, design=r.temporal_design,
                               test_cycle=r.test_cycle, selection_n=len(s), verification_n=len(v),
                               selection_positives=int(ytr[s].sum()), verification_positives=int(ytr[v].sum()),
                               frozen_threshold=frozen_t, proposed_threshold=proposed['threshold'],
                               selection_objective=proposed['objective'],
                               selection_missed_upper=proposed['missed_upper'],
                               verification_delta_loss=delta, verification_delta_upper=upper,
                               accepted=accepted, decision=decision, final_threshold=final_t))
            for name, threshold in [('frozen_target', frozen_t), ('RAMS_RCO', final_t)]:
                m = RAMS.metrics(yte, pte, pte >= threshold)
                policies.append(dict(experiment_id=exp, study=r.study, design=r.temporal_design,
                                     test_cycle=r.test_cycle, policy=name, threshold=threshold,
                                     nominal_target=target, target_held=bool(m['sensitivity'] >= target), **m))
            b, u = policies[-2], policies[-1]
            paired.append(dict(experiment_id=exp, study=r.study, design=r.temporal_design,
                               test_cycle=r.test_cycle, accepted=accepted,
                               delta_sensitivity=u['sensitivity']-b['sensitivity'],
                               delta_referral_fraction=u['referral_fraction']-b['referral_fraction'],
                               delta_missed_per_1000=u['missed_per_1000']-b['missed_per_1000'],
                               frozen_target_held=b['target_held'], rco_target_held=u['target_held']))
            print(f'{exp} {r.study}: {decision}; verification upper={upper:.4f}; '
                  f'test referral={u["referral_fraction"]:.3f}; sensitivity={u["sensitivity"]:.3f}')
        df = pd.DataFrame(policies)
        pd.DataFrame(audits).to_csv(out/'RAMS_phase4_threshold_audit.csv', index=False)
        df.to_csv(out/'RAMS_phase4_policy_results.csv', index=False)
        pd.DataFrame(paired).to_csv(out/'RAMS_phase4_paired_comparisons.csv', index=False)
        df.groupby(['study','policy'], as_index=False).agg(
            n_experiments=('experiment_id','size'), mean_sensitivity=('sensitivity','mean'),
            min_sensitivity=('sensitivity','min'), mean_referral=('referral_fraction','mean'),
            mean_missed_per_1000=('missed_per_1000','mean'),
            fraction_targets_held=('target_held','mean')).to_csv(
                out/'RAMS_phase4_summary.csv', index=False)
        (out/'RAMS_phase4_run_manifest.json').write_text(json.dumps(dict(
            method='RAMS-RCO', n_experiments=len(selected),
            n_adopted=sum(x['accepted'] for x in audits),
            miss_penalty=args.miss_penalty, uncertainty_penalty=args.uncertainty_penalty,
            alpha=args.alpha, selection_fraction=args.selection_fraction,
            split_seed=args.split_seed, bootstrap=args.bootstrap,
            limitations=['Penalties are illustrative, not clinically elicited',
                         'Bootstrap verification is not NHANES survey-design adjusted',
                         'Test cohorts and temporal experiments are not independent',
                         'No guaranteed sensitivity floor or prospective validation']), indent=2))
        print(f'Completed {len(selected)} experiments; {sum(x["accepted"] for x in audits)} candidates adopted. Outputs: {out}')
    finally:
        arc.close()


if __name__ == '__main__':
    main()
