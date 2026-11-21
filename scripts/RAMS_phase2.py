
#!/usr/bin/env python3
"""RAMS Phase 2:.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import RAMS


def record(r, policy, capacity, threshold, source, target, y, p, decision):
    return dict(experiment_id=r.experiment_id, study=r.study,
                design=r.temporal_design, train_cycles=r.train_cycles,
                test_cycle=r.test_cycle, policy=policy, capacity=capacity,
                threshold=threshold, threshold_source=source,
                nominal_target=target, target_held=bool(
                    np.sum(decision & (y == 1)) / np.sum(y == 1) >= target),
                **RAMS.metrics(y, p, decision))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root')
    ap.add_argument('--archive')
    ap.add_argument('--out')
    ap.add_argument('--capacities', nargs='+', type=float,
                    default=[.40, .50, .60, .70, .80, .90, 1.0])
    ap.add_argument('--local-reference', help='Independent experiment_id,seqn,score,y CSV')
    ap.add_argument('--min-local-class', type=int, default=20)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if any(not 0 < c <= 1 for c in args.capacities):
        ap.error('Capacities must be in (0,1]')
    root = Path(args.root).expanduser().resolve() if args.root else Path(__file__).resolve().parent.parent
    archive = Path(args.archive) if args.archive else root/'processed'/'08_model_results.zip'
    out = Path(args.out) if args.out else root/'processed'/'RAMS'/'phase2'
    if not archive.is_absolute(): archive = root/archive
    if not out.is_absolute(): out = root/out
    out.mkdir(parents=True, exist_ok=True)
    local = None
    if args.local_reference:
        locpath = Path(args.local_reference)
        if not locpath.is_absolute(): locpath = root/locpath
        local = pd.read_csv(locpath)
        if not {'experiment_id','seqn','score','y'}.issubset(local.columns):
            raise ValueError('Local CSV requires experiment_id,seqn,score,y')
        local['seqn'] = pd.to_numeric(local.seqn, errors='raise').astype('int64')
        local['score'] = pd.to_numeric(local.score, errors='raise')
        local['y'] = pd.to_numeric(local.y, errors='raise')
        if local.duplicated(['experiment_id','seqn']).any():
            raise ValueError('Duplicate experiment_id/seqn in local reference')
    arc = RAMS.Archive(archive)
    try:
        manifest = arc.table('07_model_design/05_experiment_manifest.csv')
        if len(manifest) != 336: raise ValueError('Expected 336 frozen experiments')
        selected = manifest[manifest.apply(
            lambda r: (r.feature_tier, r.model) == RAMS.MODELS[r.study], axis=1)]
        if len(selected) != 28: raise ValueError('Expected 28 primary experiments')
        if args.quick: selected = selected.head(2)
        lookup = RAMS.archived_thresholds(arc)
        rows = []
        for r in selected.itertuples(index=False):
            train_id, train_y, train_p = RAMS.seed_average(arc,r.experiment_id,'oof','sigmoid')
            test_id, test_y, test_p = RAMS.seed_average(arc,r.experiment_id,'test','sigmoid')
            target = RAMS.TARGETS[r.study]
            threshold, source = RAMS.historical_threshold(
                lookup,r.experiment_id,target,train_y,train_p)
            rows.append(record(r,'frozen_target',np.nan,threshold,source,target,
                               test_y,test_p,test_p >= threshold))
            for cap in sorted(set(args.capacities)):
                choose = RAMS.choose_capacity(test_p,test_id,cap)
                rows.append(record(r,'capacity_top_k',cap,np.nan,
                                   'evaluation_batch_rank',target,test_y,test_p,choose))
            if local is not None:
                ref = local[local.experiment_id.astype(str) == str(r.experiment_id)].copy()
                if not ref.empty:
                    if ref.seqn.isna().any() or ref.seqn.duplicated().any():
                        raise ValueError(f'{r.experiment_id}: invalid local SEQN')
                    if set(ref.seqn).intersection(set(train_id)) or set(ref.seqn).intersection(set(test_id)):
                        raise ValueError(f'{r.experiment_id}: local/train/test SEQN overlap')
                    if not np.isfinite(ref.score).all() or not ref.score.between(0,1).all() or not ref.y.isin([0,1]).all():
                        raise ValueError(f'{r.experiment_id}: invalid local scores/outcomes')
                    if min(int(ref.y.sum()),int((ref.y==0).sum())) < args.min_local_class:
                        raise ValueError(f'{r.experiment_id}: insufficient local class counts')
                    # Fit using independent reference ONLY; held-out outcomes never used.
                    model, local_threshold = RAMS.local_calibration(
                        ref.assign(study=r.study),r.study,'target',target)
                    adapted = RAMS.apply_local(model,test_p)
                    rows.append(record(r,'independent_local_target',np.nan,
                                       local_threshold,'independent_local_reference',
                                       target,test_y,adapted,adapted >= local_threshold))
            print(r.experiment_id, r.study, r.temporal_design, r.test_cycle, 'done')
        results = pd.DataFrame(rows)
        results.to_csv(out/'RAMS_phase2_policy_results.csv',index=False)
        base = results[results.policy=='frozen_target'].set_index('experiment_id')
        comparisons = []
        for row in results[results.policy!='frozen_target'].itertuples(index=False):
            b = base.loc[row.experiment_id]
            comparisons.append(dict(experiment_id=row.experiment_id,study=row.study,
                design=row.design,test_cycle=row.test_cycle,policy=row.policy,
                capacity=row.capacity,nominal_target=row.nominal_target,
                baseline_sensitivity=b.sensitivity,policy_sensitivity=row.sensitivity,
                delta_sensitivity=row.sensitivity-b.sensitivity,
                baseline_referral_fraction=b.referral_fraction,
                policy_referral_fraction=row.referral_fraction,
                delta_referral_fraction=row.referral_fraction-b.referral_fraction,
                baseline_missed_per_1000=b.missed_per_1000,
                policy_missed_per_1000=row.missed_per_1000,
                delta_missed_per_1000=row.missed_per_1000-b.missed_per_1000,
                baseline_target_held=bool(b.target_held),
                policy_target_held=bool(row.target_held)))
        pd.DataFrame(comparisons).to_csv(out/'RAMS_phase2_paired_comparisons.csv',index=False)
        summary = results.groupby(['study','policy','capacity'],dropna=False,as_index=False).agg(
            n_experiments=('experiment_id','size'),
            mean_sensitivity=('sensitivity','mean'),
            minimum_sensitivity=('sensitivity','min'),
            mean_referral_fraction=('referral_fraction','mean'),
            mean_missed_per_1000=('missed_per_1000','mean'),
            fraction_targets_held=('target_held','mean'))
        summary.to_csv(out/'RAMS_phase2_summary.csv',index=False)
        manifest_info = dict(n_experiments=len(selected),capacities=sorted(set(args.capacities)),
            local_reference_supplied=local is not None,
            local_experiments=int((results.policy=='independent_local_target').sum()),
            limitations=['No refitting of frozen models',
                         'Capacity is retrospective batch top-k, not a fixed prospective rule',
                         'Local calibration requires independent labeled scores from same model',
                         'Temporal experiments reuse participants and are not independent',
                         'Not validated for clinical deployment'])
        (out/'RAMS_phase2_run_manifest.json').write_text(json.dumps(manifest_info,indent=2))
        print(f'Completed {len(selected)} experiments; {len(results)} policy rows; output: {out}')
        if local is None: print('Local adaptation skipped: independent labeled reference not supplied.')
    finally:
        arc.close()

if __name__=='__main__': main()
