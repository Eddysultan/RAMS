
#!/usr/bin/env python3
"""
RAMS PHASE 3
Uncertainty-Aware Capacity Selection (RAMS-UAC)

"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import norm
from sklearn.model_selection import train_test_split

import RAMS


DEFAULT_ALPHA = 0.05
DEFAULT_SELECTION_FRACTION = 0.60
DEFAULT_SPLIT_SEED = 2026


def wilson_lower_bound(tp, positives, alpha=0.05):
    """
    One-sided Wilson lower confidence bound for sensitivity.

    Sensitivity = TP / (TP + FN).

    This treats positive-case outcomes as binomial observations.
    It does not account for survey design, spatial clustering,
    or dependence between repeated temporal experiments.
    """
    n = int(positives)

    if n <= 0:
        return np.nan

    phat = float(tp) / n
    z = float(norm.ppf(1.0 - alpha))

    denominator = 1.0 + z * z / n

    center = phat + z * z / (2.0 * n)

    adjustment = z * np.sqrt(
        phat * (1.0 - phat) / n
        + z * z / (4.0 * n * n)
    )

    return float(
        (center - adjustment) / denominator
    )


def evaluate_threshold(y, p, threshold, alpha):
    """Evaluate sensitivity, uncertainty, and referral burden."""

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    referred = p >= threshold

    positives = int(np.sum(y == 1))
    tp = int(np.sum(referred & (y == 1)))
    fn = positives - tp

    sensitivity = (
        tp / positives if positives > 0 else np.nan
    )

    lcb = wilson_lower_bound(
        tp, positives, alpha
    )

    return {
        "n": len(y),
        "positives": positives,
        "tp": tp,
        "fn": fn,
        "sensitivity": sensitivity,
        "sensitivity_lcb": lcb,
        "referral_fraction": float(np.mean(referred)),
        "tests_avoided_fraction": float(
            1.0 - np.mean(referred)
        ),
        "missed_per_1000": float(
            1000.0 * fn / len(y)
        )
    }


def choose_uac_threshold(y, p, target, alpha):
    """
    Choose the largest feasible threshold on the selection set.

    Larger thresholds generally produce fewer referrals.

    Candidate thresholds are observed positive-case scores,
    plus zero (refer all for probability scores in [0,1]).

    The selection-set confidence bound is used to screen
    candidate thresholds. This bound is NOT an independent
    post-selection confidence guarantee.
    """

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    if len(y) == 0:
        raise ValueError("Empty selection set")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Outcomes must be binary")

    if not np.isfinite(p).all():
        raise ValueError("Nonfinite prediction scores")

    if np.any((p < 0) | (p > 1)):
        raise ValueError(
            "Expected probability scores between 0 and 1"
        )

    positive_scores = np.unique(p[y == 1])

    if len(positive_scores) == 0:
        raise ValueError(
            "No positive cases in selection set"
        )

    candidates = np.unique(
        np.concatenate(([0.0], positive_scores))
    )

    # Descending thresholds = fewest referrals first.
    candidates = candidates[::-1]

    for threshold in candidates:
        result = evaluate_threshold(
            y, p, threshold, alpha
        )

        if result["sensitivity_lcb"] >= target:
            return float(threshold), result, True

    # Feasibility may fail for very small positive samples.
    fallback = evaluate_threshold(
        y, p, 0.0, alpha
    )

    return 0.0, fallback, False


def split_oof(seqn, y, fraction, seed):
    """
    Stratified, reproducible split of frozen training OOF records.

    No temporal test outcomes enter this operation.
    """

    seqn = np.asarray(seqn)
    y = np.asarray(y, dtype=int)

    if len(np.unique(seqn)) != len(seqn):
        raise ValueError("Duplicate SEQN in OOF records")

    if len(np.unique(y)) != 2:
        raise ValueError(
            "Both outcome classes are required"
        )

    counts = np.bincount(y, minlength=2)

    if counts.min() < 2:
        raise ValueError(
            "At least two examples of each class required"
        )

    indices = np.arange(len(y))

    select_idx, verify_idx = train_test_split(
        indices,
        train_size=fraction,
        random_state=seed,
        stratify=y
    )

    return select_idx, verify_idx


def make_result_row(
    experiment,
    study,
    design,
    test_cycle,
    policy,
    threshold,
    source,
    target,
    alpha,
    y,
    p
):
    """Standardize policy evaluation results."""

    result = evaluate_threshold(
        y, p, threshold, alpha
    )

    return {
        "experiment_id": experiment,
        "study": study,
        "design": design,
        "test_cycle": test_cycle,
        "policy": policy,
        "threshold": threshold,
        "threshold_source": source,
        "nominal_target": target,
        "confidence_level": 1.0 - alpha,
        "target_held_point_estimate": bool(
            result["sensitivity"] >= target
        ),
        "target_held_lcb": bool(
            result["sensitivity_lcb"] >= target
        ),
        **result
    }


def main():

    parser = argparse.ArgumentParser(
        description="RAMS Phase 3: uncertainty-aware capacity selection"
    )

    parser.add_argument("--root")
    parser.add_argument("--archive")
    parser.add_argument("--out")

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="One-sided confidence alpha; default .05"
    )

    parser.add_argument(
        "--selection-fraction",
        type=float,
        default=DEFAULT_SELECTION_FRACTION
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED
    )

    parser.add_argument(
        "--quick",
        action="store_true"
    )

    args = parser.parse_args()

    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")

    if not 0 < args.selection_fraction < 1:
        parser.error(
            "--selection-fraction must be between 0 and 1"
        )

    root = (
        Path(args.root).expanduser().resolve()
        if args.root
        else Path(__file__).resolve().parent.parent
    )

    archive_path = (
        Path(args.archive).expanduser()
        if args.archive
        else root / "processed" / "08_model_results.zip"
    )

    if not archive_path.is_absolute():
        archive_path = root / archive_path

    output = (
        Path(args.out).expanduser()
        if args.out
        else root / "processed" / "RAMS" / "phase3"
    )

    if not output.is_absolute():
        output = root / output

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nRAMS PHASE 3")
    print("Uncertainty-Aware Capacity Selection")
    print("-------------------------------------")
    print("Archive:", archive_path)
    print("Output:", output)
    print("Confidence level:", 1 - args.alpha)

    archive = RAMS.Archive(archive_path)

    try:

        manifest = archive.table(
            "07_model_design/05_experiment_manifest.csv"
        )

        if len(manifest) != 336:
            raise ValueError(
                f"Expected 336 experiments, got {len(manifest)}"
            )

        frozen_thresholds = RAMS.archived_thresholds(
            archive
        )

        selected = manifest[
            manifest.apply(
                lambda r: (
                    r.feature_tier,
                    r.model
                ) == RAMS.MODELS[r.study],
                axis=1
            )
        ].copy()

        if len(selected) != 28:
            raise ValueError(
                f"Expected 28 primary experiments, got {len(selected)}"
            )

        if args.quick:
            selected = selected.head(2)

        policy_rows = []
        selection_rows = []
        comparison_rows = []

        for r in selected.itertuples(index=False):

            experiment = str(r.experiment_id)
            study = str(r.study)
            target = RAMS.TARGETS[study]

            train_id, train_y, train_p = (
                RAMS.seed_average(
                    archive,
                    experiment,
                    "oof",
                    "sigmoid"
                )
            )

            test_id, test_y, test_p = (
                RAMS.seed_average(
                    archive,
                    experiment,
                    "test",
                    "sigmoid"
                )
            )

            if len(set(train_id).intersection(set(test_id))):
                raise ValueError(
                    f"{experiment}: train/test participant overlap"
                )

            # ----------------------------------
            # Original frozen RAMS threshold
            # ----------------------------------

            frozen_threshold, frozen_source = (
                RAMS.historical_threshold(
                    frozen_thresholds,
                    experiment,
                    target,
                    train_y,
                    train_p
                )
            )

            # ----------------------------------
            # Selection / verification split
            # ----------------------------------

            select_idx, verify_idx = split_oof(
                train_id,
                train_y,
                args.selection_fraction,
                args.split_seed
            )

            y_select = train_y[select_idx]
            p_select = train_p[select_idx]

            y_verify = train_y[verify_idx]
            p_verify = train_p[verify_idx]

            # ----------------------------------
            # UAC threshold selection
            # ----------------------------------

            proposed_threshold, select_metrics, feasible = (
                choose_uac_threshold(
                    y_select,
                    p_select,
                    target,
                    args.alpha
                )
            )

            verify_metrics = evaluate_threshold(
                y_verify,
                p_verify,
                proposed_threshold,
                args.alpha
            )

            verification_pass = bool(
                feasible
                and verify_metrics["sensitivity_lcb"] >= target
            )

            if verification_pass:
                final_threshold = proposed_threshold
                final_source = (
                    "selection_lcb_verified"
                )
            else:
                # Conservative policy: refer everyone.
                # No test outcomes are used in fallback selection.
                final_threshold = 0.0
                final_source = (
                    "refer_all_verification_fallback"
                )

            # ----------------------------------
            # Held-out temporal evaluation
            # ----------------------------------

            frozen_result = make_result_row(
                experiment,
                study,
                r.temporal_design,
                r.test_cycle,
                "frozen_target",
                frozen_threshold,
                frozen_source,
                target,
                args.alpha,
                test_y,
                test_p
            )

            uac_result = make_result_row(
                experiment,
                study,
                r.temporal_design,
                r.test_cycle,
                "RAMS_UAC",
                final_threshold,
                final_source,
                target,
                args.alpha,
                test_y,
                test_p
            )

            policy_rows.extend([
                frozen_result,
                uac_result
            ])

            selection_rows.append({
                "experiment_id": experiment,
                "study": study,
                "design": r.temporal_design,
                "test_cycle": r.test_cycle,
                "target": target,
                "selection_n": len(select_idx),
                "selection_positives": int(
                    y_select.sum()
                ),
                "verification_n": len(verify_idx),
                "verification_positives": int(
                    y_verify.sum()
                ),
                "proposed_threshold": proposed_threshold,
                "selection_sensitivity": (
                    select_metrics["sensitivity"]
                ),
                "selection_lcb": (
                    select_metrics["sensitivity_lcb"]
                ),
                "selection_referral": (
                    select_metrics["referral_fraction"]
                ),
                "candidate_feasible": bool(feasible),
                "verification_sensitivity": (
                    verify_metrics["sensitivity"]
                ),
                "verification_lcb": (
                    verify_metrics["sensitivity_lcb"]
                ),
                "verification_pass": verification_pass,
                "final_threshold": final_threshold,
                "final_source": final_source
            })

            comparison_rows.append({
                "experiment_id": experiment,
                "study": study,
                "design": r.temporal_design,
                "test_cycle": r.test_cycle,
                "nominal_target": target,
                "frozen_sensitivity": (
                    frozen_result["sensitivity"]
                ),
                "uac_sensitivity": (
                    uac_result["sensitivity"]
                ),
                "delta_sensitivity": (
                    uac_result["sensitivity"]
                    - frozen_result["sensitivity"]
                ),
                "frozen_referral": (
                    frozen_result["referral_fraction"]
                ),
                "uac_referral": (
                    uac_result["referral_fraction"]
                ),
                "delta_referral": (
                    uac_result["referral_fraction"]
                    - frozen_result["referral_fraction"]
                ),
                "frozen_missed_per_1000": (
                    frozen_result["missed_per_1000"]
                ),
                "uac_missed_per_1000": (
                    uac_result["missed_per_1000"]
                ),
                "delta_missed_per_1000": (
                    uac_result["missed_per_1000"]
                    - frozen_result["missed_per_1000"]
                ),
                "frozen_target_held": (
                    frozen_result["target_held_point_estimate"]
                ),
                "uac_target_held": (
                    uac_result["target_held_point_estimate"]
                ),
                "verification_pass": verification_pass,
                "uac_fallback": not verification_pass
            })

            print(
                f"{experiment} | {study} | "
                f"{r.temporal_design} | "
                f"Frozen sensitivity: "
                f"{frozen_result['sensitivity']:.3f} | "
                f"UAC sensitivity: "
                f"{uac_result['sensitivity']:.3f} | "
                f"UAC referrals: "
                f"{uac_result['referral_fraction']:.3f} | "
                f"Verified: {verification_pass}"
            )

        # ----------------------------------
        # Save results
        # ----------------------------------

        policies = pd.DataFrame(policy_rows)
        selections = pd.DataFrame(selection_rows)
        comparisons = pd.DataFrame(comparison_rows)

        policies.to_csv(
            output / "RAMS_phase3_policy_results.csv",
            index=False
        )

        selections.to_csv(
            output / "RAMS_phase3_threshold_selection.csv",
            index=False
        )

        comparisons.to_csv(
            output / "RAMS_phase3_paired_comparisons.csv",
            index=False
        )

        summary = policies.groupby(
            ["study", "policy"],
            as_index=False
        ).agg(
            n_experiments=("experiment_id", "size"),
            mean_sensitivity=("sensitivity", "mean"),
            minimum_sensitivity=("sensitivity", "min"),
            mean_referral_fraction=(
                "referral_fraction", "mean"
            ),
            mean_missed_per_1000=(
                "missed_per_1000", "mean"
            ),
            fraction_point_targets_held=(
                "target_held_point_estimate", "mean"
            ),
            fraction_lcb_targets_held=(
                "target_held_lcb", "mean"
            )
        )

        summary.to_csv(
            output / "RAMS_phase3_summary.csv",
            index=False
        )

        manifest_info = {
            "phase": 3,
            "method": "RAMS-UAC",
            "description": (
                "Uncertainty-aware referral threshold selection"
            ),
            "alpha": args.alpha,
            "confidence_level": 1.0 - args.alpha,
            "selection_fraction": args.selection_fraction,
            "split_seed": args.split_seed,
            "n_experiments": len(selected),
            "n_policy_evaluations": len(policies),
            "n_verification_passed": int(
                selections["verification_pass"].sum()
            ),
            "n_fallbacks": int(
                (~selections["verification_pass"]).sum()
            ),
            "notes": [
                "Frozen conference predictions; no refitting",
                "Thresholds selected using training OOF records",
                "Separate verification subset used for acceptance",
                "Held-out temporal outcomes used only for evaluation",
                "Wilson bounds assume binomial sampling",
                "Selection-set bounds are not post-selection guarantees",
                "Verification is a one-time acceptance check",
                "No adjustment for NHANES survey design",
                "Repeated temporal experiments are not independent",
                "Refer-all fallback may eliminate testing savings",
                "Not prospectively or clinically validated"
            ]
        }

        (
            output / "RAMS_phase3_run_manifest.json"
        ).write_text(
            json.dumps(manifest_info, indent=2),
            encoding="utf-8"
        )

        print("\n-------------------------------------")
        print("RAMS PHASE 3 COMPLETED")
        print("-------------------------------------")

        print(
            "Experiments:", len(selected)
        )

        print(
            "Policy evaluations:", len(policies)
        )

        print(
            "Verified thresholds:",
            int(selections["verification_pass"].sum())
        )

        print(
            "Refer-all fallbacks:",
            int((~selections["verification_pass"]).sum())
        )

        print("Output directory:", output)

    finally:
        archive.close()


if __name__ == "__main__":
    main()
