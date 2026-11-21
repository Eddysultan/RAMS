
#!/usr/bin/env python3
"""RAMS — Resource-Adaptive Micronutrient Screening.

"""

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression


TARGETS = {
    'iron_observed': .90,
    'iron_matched': .90,
    'vitamin_d': .95
}

MODELS = {
    'iron_observed': ('T2', 'elastic_net_logistic'),
    'iron_matched': ('T2', 'elastic_net_logistic'),
    'vitamin_d': ('T1', 'catboost')
}

TARGET_GRID = [.80, .85, .90, .95, .975]


class Archive:
    def __init__(self, path):
        self.path = Path(path)
        self.z = zipfile.ZipFile(self.path) if self.path.is_file() else None
        self.names = set(self.z.namelist()) if self.z else set()

    def read(self, name):
        if self.z:
            if name not in self.names:
                raise FileNotFoundError(name)
            return self.z.read(name)
        return (self.path / name).read_bytes()

    def table(self, name):
        return pd.read_csv(io.BytesIO(self.read(name)))

    def npz(self, name):
        with np.load(io.BytesIO(self.read(name))) as f:
            return {k: f[k].copy() for k in f.files}

    def list(self, prefix):
        if self.z:
            return sorted(
                x for x in self.names
                if x.startswith(prefix) and x.endswith('.npz')
            )
        return sorted(
            str(p.relative_to(self.path)).replace('\\', '/')
            for p in (self.path / prefix).glob('*.npz')
        )

    def close(self):
        if self.z:
            self.z.close()


def seed_average(arc, exp, kind, calibration='sigmoid'):
    prefix = (
        f'08_model_results/checkpoints/'
        f'seed_{kind}_predictions/{exp}__seed_'
    )
    paths = arc.list(prefix)

    if len(paths) != 10:
        raise ValueError(
            f'{exp} {kind}: expected 10 frozen seeds, got {len(paths)}'
        )

    arrays = [arc.npz(p) for p in paths]
    base = arrays[0]

    for a in arrays[1:]:
        if not (
            np.array_equal(a['seqn'], base['seqn'])
            and np.array_equal(a['y'], base['y'])
        ):
            raise ValueError(
                f'Participant alignment failed: {exp}, {kind}'
            )

    return (
        base['seqn'].astype('int64'),
        base['y'].astype(int),
        np.mean([a[calibration] for a in arrays], axis=0)
    )


def cutoff(y, p, target):
    """Maximum score cutoff with training TPR >= target, score >= cutoff."""
    positive = np.sort(
        np.asarray(p)[np.asarray(y) == 1]
    )

    if not len(positive):
        raise ValueError('No positives in reference sample')

    # At least ceil(target*n) positives must be >= cutoff.
    index = max(
        0,
        len(positive) - int(np.ceil(target * len(positive)))
    )
    return float(positive[index])


def metrics(y, p, selected):
    y = np.asarray(y).astype(int)
    selected = np.asarray(selected).astype(bool)

    tp = int(np.sum(selected & (y == 1)))
    fn = int(np.sum(~selected & (y == 1)))

    n = len(y)
    positives = int(y.sum())

    return dict(
        n=n,
        positives=positives,
        sensitivity=tp / positives if positives else np.nan,
        referrals=int(selected.sum()),
        referral_fraction=float(selected.mean()),
        tests_avoided_fraction=float(1 - selected.mean()),
        tp=tp,
        fn=fn,
        missed_per_1000=1000 * fn / n,
        auroc=(
            float(roc_auc_score(y, p))
            if len(np.unique(y)) == 2 else np.nan
        ),
        brier=float(brier_score_loss(y, p))
    )


def local_calibration(reference, study, mode, target):
    d = reference[reference.study == study].copy()

    if len(d) == 0:
        raise ValueError(f'No local reference rows for {study}')

    if not {'score', 'y'}.issubset(d.columns):
        raise ValueError(
            'Local reference requires study, score, y'
        )

    if (
        not set(d.y.unique()).issubset({0, 1})
        or d.y.nunique() < 2
    ):
        raise ValueError(
            'Local reference needs both classes'
        )

    scores = np.clip(
        d.score.to_numpy(float), 1e-6, 1 - 1e-6
    )
    logits = np.log(scores / (1 - scores)).reshape(-1, 1)

    model = LogisticRegression(
        C=1e6,
        max_iter=2000
    ).fit(logits, d.y.to_numpy(int))

    calibrated = model.predict_proba(logits)[:, 1]

    return model, cutoff(
        d.y.to_numpy(int), calibrated, target
    )


def apply_local(model, scores):
    x = np.clip(
        np.asarray(scores), 1e-6, 1 - 1e-6
    )
    return model.predict_proba(
        np.log(x / (1 - x)).reshape(-1, 1)
    )[:, 1]


def choose_capacity(scores, seqn, fraction):
    """Deterministic batch top-k; tie-break by SEQN. No outcome used."""
    n = len(scores)
    k = int(np.floor(n * fraction + 1e-10))

    selected = np.zeros(n, bool)

    if k:
        selected[
            np.lexsort((seqn, -scores))[:k]
        ] = True

    return selected


def archived_thresholds(arc):
    """Authoritative Stage 08 cutoffs inherited from Stage 06 testing burden."""
    audit = arc.table(
        '10_clinical_utility/08_operating_threshold_audit.csv'
    )

    required = {
        'experiment_id',
        'target_training_sensitivity',
        'threshold',
        'threshold_source'
    }

    if not required.issubset(audit.columns):
        raise ValueError(
            'Stage 08 threshold audit has missing columns'
        )

    if audit.duplicated([
        'experiment_id',
        'target_training_sensitivity'
    ]).any():
        raise ValueError(
            'Duplicate archived experiment/target thresholds'
        )

    if not (
        audit.threshold_source == 'stage06_testing_burden'
    ).all():
        raise ValueError(
            'Unexpected archived threshold source'
        )

    return {
        (
            str(r.experiment_id),
            round(float(r.target_training_sensitivity), 6)
        ): float(r.threshold)
        for r in audit.itertuples(index=False)
    }


def historical_threshold(
    lookup, experiment_id, target,
    y_train=None, p_train=None
):
    """Use frozen Stage 06/08 threshold where available; otherwise OOF rule.

    Stage 08 archives targets 90%, 95%, 97.5%. Exploratory 80%, 85%
    are calculated from frozen training OOF scores, NOT archived points.
    """
    key = (
        str(experiment_id),
        round(float(target), 6)
    )

    if key in lookup:
        return lookup[key], 'stage06_testing_burden'

    if y_train is None or p_train is None:
        raise KeyError(
            f'No archived threshold for {key}'
        )

    return (
        cutoff(y_train, p_train, target),
        'derived_from_training_oof'
    )


def validate_stage08(
    arc, manifest, calibration, lookup, limit=None
):
    reference = arc.table(
        '10_clinical_utility/'
        '08_operating_point_experiment_summary.csv'
    )

    reference = reference[
        reference.calibration == calibration
    ]

    rows = []

    for i, r in enumerate(
        manifest.itertuples(index=False)
    ):
        if limit is not None and i >= limit:
            break

        _, yte, pte = seed_average(
            arc, r.experiment_id, 'test', calibration
        )

        for q in reference[
            reference.experiment_id == r.experiment_id
        ].itertuples(index=False):

            t, source = historical_threshold(
                lookup,
                r.experiment_id,
                q.target_training_sensitivity
            )

            m = metrics(yte, pte, pte >= t)

            threshold_error = abs(
                t - float(q.threshold)
            )

            sensitivity_error = abs(
                m['sensitivity'] - float(q.Sensitivity)
            )

            referral_error = abs(
                m['referral_fraction'] - float(q.ReferralRate)
            )

            count_match = (
                m['tp'] == int(q.TP)
                and m['fn'] == int(q.FN)
                and m['referrals'] == int(q.TP + q.FP)
            )

            rows.append(dict(
                experiment_id=r.experiment_id,
                target=float(q.target_training_sensitivity),
                threshold=t,
                threshold_source=source,
                reference_threshold=float(q.threshold),
                threshold_error=threshold_error,
                sensitivity_error=sensitivity_error,
                referral_error=referral_error,
                counts_match=count_match,
                pass_reconciliation=(
                    threshold_error < 1e-10
                    and sensitivity_error < 1e-10
                    and referral_error < 1e-10
                    and count_match
                )
            ))

    return pd.DataFrame(rows)


def resolve_paths(args):
    """Locate project root from Scripts/RAMS.py, regardless of terminal directory.

    Default archive:
      <project>/processed/08_model_results.zip

    Default output:
      <project>/processed/RAMS

    Explicit relative archive/output paths are interpreted
    relative to project root.
    """

    script = Path(__file__).resolve()

    root = (
        Path(args.root).expanduser().resolve()
        if args.root
        else (
            script.parent.parent
            if script.parent.name.lower() == 'scripts'
            else script.parent
        )
    )

    processed = root / 'processed'

    if args.archive:
        archive = Path(args.archive).expanduser()

        if not archive.is_absolute():
            archive = root / archive
    else:
        choices = [
            processed / '08_model_results.zip',
            processed / '08_model_results',
            root / '08_model_results.zip',
            root / '08_model_results'
        ]

        archive = next(
            (x for x in choices if x.exists()),
            choices[0]
        )

    if not archive.exists():
        raise FileNotFoundError(
            f'Conference prediction archive not found: '
            f'{archive}\n'
            f'Expected processed/08_model_results.zip '
            f'or processed/08_model_results'
        )

    output = (
        Path(args.out).expanduser()
        if args.out
        else processed / 'RAMS'
    )

    if not output.is_absolute():
        output = root / output

    if args.local_reference:
        reference = Path(
            args.local_reference
        ).expanduser()

        if not reference.is_absolute():
            reference = root / reference

        args.local_reference = str(reference)

    args.root = str(root)
    args.archive = str(archive)
    args.out = str(output)

    return archive, output


def main():
    ap = argparse.ArgumentParser(
        description='RAMS on frozen conference predictions'
    )

    ap.add_argument(
        '--root',
        help='Project root; default is parent of the Scripts folder'
    )

    ap.add_argument(
        '--archive',
        help='Optional archive override; defaults to processed/08_model_results.zip'
    )

    ap.add_argument(
        '--out',
        help='Optional output override; defaults to processed/RAMS'
    )

    ap.add_argument(
        '--mode',
        choices=['target', 'capacity'],
        default='target'
    )

    ap.add_argument(
        '--capacity',
        type=float,
        default=.70,
        help='Fraction of participants referred in capacity mode'
    )

    ap.add_argument(
        '--calibration',
        choices=['sigmoid', 'raw', 'isotonic'],
        default='sigmoid'
    )

    ap.add_argument(
        '--local-reference',
        help='Independent labeled reference CSV with study,score,y; no evaluation participants'
    )

    ap.add_argument(
        '--validate-all',
        action='store_true',
        help='Reconcile every archived Stage 08 operating point (slower)'
    )

    ap.add_argument(
        '--quick',
        action='store_true',
        help='First 2 selected experiments only, for testing'
    )

    args = ap.parse_args()

    if not 0 <= args.capacity <= 1:
        ap.error('--capacity must be between 0 and 1')

    archive_path, out = resolve_paths(args)
    out.mkdir(parents=True, exist_ok=True)

    print(f'Project root: {args.root}')
    print(f'Conference archive: {archive_path}')
    print(f'RAMS output directory: {out}')

    arc = Archive(archive_path)

    try:
        manifest = arc.table(
            '07_model_design/05_experiment_manifest.csv'
        )

        if len(manifest) != 336:
            raise ValueError(
                f'Expected 336 conference experiments; '
                f'found {len(manifest)}'
            )

        frozen_cutoffs = archived_thresholds(arc)

        if args.validate_all:
            audit = validate_stage08(
                arc, manifest, args.calibration,
                frozen_cutoffs
            )

            audit.to_csv(
                out / 'RAMS_stage08_reconciliation.csv',
                index=False
            )

            print(
                f'Stage 08 reconciliation: '
                f'{audit.pass_reconciliation.sum()}/'
                f'{len(audit)} pass'
            )

            if not audit.pass_reconciliation.all():
                raise RuntimeError(
                    'Stage 08 reconciliation FAILED; inspect report'
                )

        selected = manifest[
            manifest.apply(
                lambda r: (
                    r.feature_tier, r.model
                ) == MODELS[r.study],
                axis=1
            )
        ].copy()

        if len(selected) != 28:
            raise ValueError(
                f'Expected 28 primary-model temporal experiments; '
                f'found {len(selected)}'
            )

        if args.quick:
            selected = selected.head(2)

        local = (
            pd.read_csv(args.local_reference)
            if args.local_reference
            else None
        )

        rows = []
        actions = []
        validation = []

        for r in selected.itertuples(index=False):
            ids_train, y_train, p_train = seed_average(
                arc, r.experiment_id, 'oof',
                args.calibration
            )

            ids_test, y_test, p_test = seed_average(
                arc, r.experiment_id, 'test',
                args.calibration
            )

            target = TARGETS[r.study]

            threshold, threshold_source = historical_threshold(
                frozen_cutoffs,
                r.experiment_id,
                target,
                y_train,
                p_train
            )

            model = None

            if local is not None:
                model, threshold = local_calibration(
                    local, r.study, args.mode, target
                )

                p_test = apply_local(model, p_test)
                threshold_source = (
                    'independent_local_recalibration'
                )

            if args.mode == 'target':
                refer = p_test >= threshold
            else:
                refer = choose_capacity(
                    p_test, ids_test, args.capacity
                )

            m = metrics(y_test, p_test, refer)

            rows.append(dict(
                experiment_id=r.experiment_id,
                study=r.study,
                design=r.temporal_design,
                train_cycles=r.train_cycles,
                test_cycle=r.test_cycle,
                model=r.model,
                tier=r.feature_tier,
                mode=args.mode,
                nominal_target=target,
                threshold=(
                    threshold
                    if args.mode == 'target'
                    else np.nan
                ),
                threshold_source=(
                    threshold_source
                    if args.mode == 'target'
                    else 'batch_top_k_capacity'
                ),
                capacity=(
                    args.capacity
                    if args.mode == 'capacity'
                    else np.nan
                ),
                target_held=bool(
                    m['sensitivity'] >= target
                ),
                **m
            ))

            for seqn, score, action, y in zip(
                ids_test, p_test, refer, y_test
            ):
                actions.append(dict(
                    experiment_id=r.experiment_id,
                    study=r.study,
                    seqn=int(seqn),
                    score=float(score),
                    screen_action=(
                        (
                            'REFER_FOR_FERRITIN'
                            if r.study.startswith('iron')
                            else 'REFER_FOR_25OHD'
                        )
                        if action
                        else 'SCREEN_NEGATIVE'
                    ),
                    reference_outcome=int(y)
                ))

            for t in TARGET_GRID:
                if model is None:
                    ct, ct_source = historical_threshold(
                        frozen_cutoffs,
                        r.experiment_id,
                        t,
                        y_train,
                        p_train
                    )
                else:
                    ct = local_calibration(
                        local, r.study, args.mode, t
                    )[1]

                    ct_source = (
                        'independent_local_recalibration'
                    )

                mt = metrics(
                    y_test, p_test, p_test >= ct
                )

                validation.append(dict(
                    experiment_id=r.experiment_id,
                    study=r.study,
                    design=r.temporal_design,
                    test_cycle=r.test_cycle,
                    training_target=t,
                    threshold=ct,
                    threshold_source=ct_source,
                    achieved_sensitivity=mt['sensitivity'],
                    referral_fraction=mt['referral_fraction'],
                    target_held=bool(
                        mt['sensitivity'] >= t
                    ),
                    missed_per_1000=mt['missed_per_1000']
                ))

            print(
                f"{r.experiment_id} "
                f"{r.study} "
                f"{r.temporal_design} "
                f"{r.test_cycle}: "
                f"sensitivity={m['sensitivity']:.3f}, "
                f"referrals={m['referral_fraction']:.3f}"
            )

        results = pd.DataFrame(rows)

        results.to_csv(
            out / 'RAMS_primary_temporal_results.csv',
            index=False
        )

        pd.DataFrame(actions).to_csv(
            out / 'RAMS_retrospective_screening_actions.csv',
            index=False
        )

        pd.DataFrame(validation).to_csv(
            out / 'RAMS_all_target_validation.csv',
            index=False
        )

        summary = results.groupby(
            ['study', 'design'],
            as_index=False
        ).agg(
            n_pairs=('experiment_id', 'size'),
            mean_sensitivity=('sensitivity', 'mean'),
            min_sensitivity=('sensitivity', 'min'),
            mean_referral=('referral_fraction', 'mean'),
            mean_missed_per_1000=(
                'missed_per_1000', 'mean'
            ),
            fraction_targets_held=('target_held', 'mean')
        )

        summary.to_csv(
            out / 'RAMS_summary.csv',
            index=False
        )

        (
            out / 'RAMS_run_manifest.json'
        ).write_text(
            json.dumps(
                dict(
                    vars(args),
                    n_selected_experiments=len(selected),
                    model_routing=MODELS,
                    targets=TARGETS,
                    warning=(
                        'Retrospective screening simulation; '
                        'no new-patient model fitting or inference'
                    )
                ),
                indent=2
            )
        )

        print(
            f'Completed {len(selected)} '
            f'primary-model experiments; outputs in {out}'
        )

    finally:
        arc.close()


if __name__ == '__main__':
    main()
