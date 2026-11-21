
"""
RAMS PHASE 5: STATISTICAL AUDIT
================================

"""

from pathlib import Path
from datetime import datetime, timezone
import json
import math
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

# Resolve project root from this script's location.
# Works whether terminal is in Scripts or project root.

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PROCESSED = PROJECT_ROOT / "processed"
RAMS_DIR = PROCESSED / "RAMS"
COHORT_DIR = PROCESSED / "05_analytic_cohorts"

OUTPUT_DIR = RAMS_DIR / "phase5_statistical_audit"

PRIMARY_FILE = "RAMS_primary_temporal_results.csv"

ACTIONS_FILE = "RAMS_retrospective_screening_actions.csv"

SUBGROUP_COLUMNS = [
    "race_ethnicity_5cat",
    "sex",
    "age_group",
    "pir_group"
]

ALPHA = 0.05

# Allow only very small floating-point differences.
TOLERANCE = 1e-8


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def log(message):
    print(f"[RAMS Phase 5] {message}", flush=True)


def locate_file(folder, filename):
    """
    Locate an exact filename recursively.

    Excludes Phase 5 outputs to avoid reading previous audits.
    """
    if not folder.exists():
        raise FileNotFoundError(
            f"Required folder not found:\n{folder}"
        )

    matches = [
        p for p in folder.rglob(filename)
        if "phase5_statistical_audit" not in p.parts
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"\nCould not locate: {filename}\n"
            f"Searched under: {folder}\n"
            "Check that the original RAMS files are extracted."
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple copies of {filename} found:\n"
            + "\n".join(str(p) for p in matches)
        )

    return matches[0]


def wilson_interval(successes, total, alpha=ALPHA):
    """
    Wilson score interval for a binomial proportion.

    These intervals describe individual analytic test samples.
    They are NOT NHANES complex-survey confidence intervals.
    """
    if total <= 0:
        return np.nan, np.nan

    z = norm.ppf(1 - alpha / 2)
    p = successes / total

    denominator = 1 + z**2 / total

    center = (
        p + z**2 / (2 * total)
    ) / denominator

    half_width = (
        z * math.sqrt(
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        )
    ) / denominator

    return (
        max(0.0, center - half_width),
        min(1.0, center + half_width)
    )


def calculate_metrics(df):
    """
    Calculate screening performance from actual ground truth
    and archived referral decisions.
    """
    y = df["reference_outcome"].to_numpy(dtype=int)
    a = df["referred"].to_numpy(dtype=int)

    tp = int(((y == 1) & (a == 1)).sum())
    fn = int(((y == 1) & (a == 0)).sum())
    fp = int(((y == 0) & (a == 1)).sum())
    tn = int(((y == 0) & (a == 0)).sum())

    n = len(df)
    positives = tp + fn
    negatives = tn + fp
    referrals = tp + fp

    sensitivity = tp / positives if positives else np.nan
    specificity = tn / negatives if negatives else np.nan
    referral_fraction = referrals / n if n else np.nan

    se_low, se_high = wilson_interval(tp, positives)
    sp_low, sp_high = wilson_interval(tn, negatives)
    rf_low, rf_high = wilson_interval(referrals, n)

    return {
        "n": n,
        "unique_participants": int(df["seqn"].nunique()),
        "positives": positives,
        "negatives": negatives,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "referrals": referrals,
        "sensitivity": sensitivity,
        "sensitivity_ci_low": se_low,
        "sensitivity_ci_high": se_high,
        "specificity": specificity,
        "specificity_ci_low": sp_low,
        "specificity_ci_high": sp_high,
        "referral_fraction": referral_fraction,
        "referral_ci_low": rf_low,
        "referral_ci_high": rf_high,
        "tests_avoided_fraction": (
            1 - referral_fraction if n else np.nan
        ),
        "missed_per_1000": (
            1000 * fn / n if n else np.nan
        )
    }


def save_csv(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    log(f"Saved: {filename} ({len(df)} rows)")


# ============================================================
# 3. LOAD ORIGINAL RAMS ARCHIVES
# ============================================================

def load_rams_data():

    log(f"Project root: {PROJECT_ROOT}")

    primary_path = locate_file(
        RAMS_DIR, PRIMARY_FILE
    )

    actions_path = locate_file(
        RAMS_DIR, ACTIONS_FILE
    )

    log(f"Primary file: {primary_path}")
    log(f"Actions file: {actions_path}")

    primary = pd.read_csv(primary_path)
    actions = pd.read_csv(actions_path)

    primary_required = {
        "experiment_id", "study", "n",
        "positives", "sensitivity",
        "referral_fraction", "tp", "fn",
        "threshold"
    }

    actions_required = {
        "experiment_id", "study", "seqn",
        "score", "screen_action",
        "reference_outcome"
    }

    missing_primary = primary_required - set(primary.columns)
    missing_actions = actions_required - set(actions.columns)

    if missing_primary:
        raise ValueError(
            f"Missing primary columns: {missing_primary}"
        )

    if missing_actions:
        raise ValueError(
            f"Missing actions columns: {missing_actions}"
        )

    # One row per primary temporal evaluation.
    keys = ["experiment_id", "study"]

    if primary.duplicated(keys).any():
        raise ValueError(
            "Duplicate primary evaluation identifiers found."
        )

    # Map original screening actions to binary referral.
    actions["screen_action"] = (
        actions["screen_action"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    valid_actions = {
        "SCREEN_NEGATIVE",
        "REFER_FOR_25OHD",
        "REFER_FOR_FERRITIN"
    }

    unexpected = set(actions["screen_action"]) - valid_actions

    if unexpected:
        raise ValueError(
            f"Unexpected screening actions: {unexpected}"
        )

    actions["referred"] = (
        actions["screen_action"] != "SCREEN_NEGATIVE"
    ).astype(int)

    actions["seqn"] = pd.to_numeric(
        actions["seqn"], errors="raise"
    ).astype("int64")

    actions["reference_outcome"] = pd.to_numeric(
        actions["reference_outcome"], errors="raise"
    ).astype(int)

    actions["score"] = pd.to_numeric(
        actions["score"], errors="raise"
    )

    if not actions["reference_outcome"].isin([0, 1]).all():
        raise ValueError("Ground-truth labels must be 0 or 1.")

    if actions["score"].isna().any():
        raise ValueError("Missing prediction scores found.")

    if not actions["score"].between(0, 1).all():
        raise ValueError("Scores outside [0, 1] found.")

    if actions.duplicated(
        ["experiment_id", "study", "seqn"]
    ).any():
        raise ValueError(
            "Duplicate participant within an evaluation."
        )

    # Verify that all actions belong to registered primary
    # evaluations. Do not silently drop unmatched records.
    merged = actions.merge(
        primary[keys].drop_duplicates(),
        on=keys,
        how="left",
        indicator=True,
        validate="many_to_one"
    )

    if (merged["_merge"] != "both").any():
        raise ValueError(
            "Some actions have no matching primary evaluation."
        )

    actions = merged.drop(columns="_merge")

    log(f"Primary evaluations: {len(primary)}")
    log(f"Participant-evaluation rows: {len(actions)}")

    return primary, actions, primary_path, actions_path


# ============================================================
# 4. PRIMARY PER-EVALUATION CONFIDENCE INTERVALS
# ============================================================

def audit_primary(primary, actions):

    log("Calculating primary screening confidence intervals...")

    rows = []

    for (exp_id, study), group in actions.groupby(
        ["experiment_id", "study"], sort=False
    ):

        metadata = primary.loc[
            (primary["experiment_id"] == exp_id)
            & (primary["study"] == study)
        ].iloc[0]

        metrics = calculate_metrics(group)

        nominal_target = float(metadata["nominal_target"])

        rows.append({
            "experiment_id": exp_id,
            "study": study,
            "design": metadata["design"],
            "train_cycles": metadata["train_cycles"],
            "test_cycle": metadata["test_cycle"],
            "model": metadata["model"],
            "tier": metadata["tier"],
            "nominal_target": nominal_target,
            "threshold": metadata["threshold"],
            **metrics,
            "target_met_point_estimate": (
                metrics["sensitivity"] >= nominal_target
            ),
            "ci_lower_bound_meets_target": (
                metrics["sensitivity_ci_low"] >= nominal_target
            )
        })

    result = pd.DataFrame(rows)

    save_csv(
        result,
        "phase5_primary_per_cycle_ci.csv"
    )

    return result


# ============================================================
# 5. VERIFY ORIGINAL MANUSCRIPT RESULTS
# ============================================================

def reconcile_primary(primary, audit):

    log("Reconciling primary results against archives...")

    merged = primary.merge(
        audit,
        on=["experiment_id", "study"],
        suffixes=("_archive", "_audit"),
        validate="one_to_one"
    )

    if len(merged) != len(primary):
        raise ValueError(
            "Not all primary evaluations were reconstructed."
        )

    fields = [
        "n",
        "positives",
        "tp",
        "fn",
        "referrals",
        "sensitivity",
        "referral_fraction",
        "missed_per_1000"
    ]

    records = []

    for field in fields:

        archived = pd.to_numeric(
            merged[f"{field}_archive"],
            errors="coerce"
        )

        calculated = pd.to_numeric(
            merged[f"{field}_audit"],
            errors="coerce"
        )

        delta = (archived - calculated).abs()

        records.append({
            "metric": field,
            "max_absolute_difference": float(delta.max()),
            "all_match": bool(
                delta.notna().all()
                and (delta <= TOLERANCE).all()
            )
        })

    result = pd.DataFrame(records)

    save_csv(
        result,
        "phase5_primary_reconciliation.csv"
    )

    if not result["all_match"].all():
        raise ValueError(
            "Primary reconciliation failed. "
            "Review the reconciliation CSV."
        )

    log("Primary screening results reconciled successfully.")

    return result


# ============================================================
# 6. LOAD ANALYTIC COHORTS
# ============================================================

def load_cohorts():

    log("Loading original analytic cohorts...")

    paths = {
        "vitamin_d":
            COHORT_DIR / "vitamin_d"
            / "NHANES_vitaminD_primary_multicycle.csv",

        "iron_observed":
            COHORT_DIR / "iron" / "observed"
            / "NHANES_iron_observed_multicycle.csv",

        "iron_matched":
            COHORT_DIR / "iron" / "matched"
            / "NHANES_iron_matched_multicycle.csv"
    }

    cohorts = {}

    for study, path in paths.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing cohort file:\n{path}"
            )

        df = pd.read_csv(path)

        if "SEQN" not in df.columns:
            raise ValueError(
                f"SEQN missing from {path.name}"
            )

        df = df.rename(columns={"SEQN": "seqn"})

        df["seqn"] = pd.to_numeric(
            df["seqn"], errors="raise"
        ).astype("int64")

        if df["seqn"].duplicated().any():
            raise ValueError(
                f"Duplicate SEQN in {study} cohort."
            )

        cohorts[study] = df

        log(f"Loaded {study}: {len(df)} participants")

    return cohorts


# ============================================================
# 7. ADD SUBGROUP METADATA
# ============================================================

def enrich_actions(actions, cohorts):

    log("Linking predictions to participant characteristics...")

    combined = []

    for study, group in actions.groupby("study"):

        if study not in cohorts:
            raise ValueError(
                f"No analytic cohort found for study: {study}"
            )

        cohort = cohorts[study].copy()

        keep = [
            "seqn",
            "survey_cycle",
            "age_years",
            "sex",
            "race_ethnicity_5cat",
            "poverty_income_ratio",
            "cbc_hemoglobin"
        ]

        keep = [c for c in keep if c in cohort.columns]

        cohort = cohort[keep]

        joined = group.merge(
            cohort,
            on="seqn",
            how="left",
            validate="many_to_one",
            indicator=True
        )

        if (joined["_merge"] != "both").any():
            raise ValueError(
                f"Unmatched participant IDs for {study}."
            )

        joined = joined.drop(columns="_merge")

        # Age and poverty categories for subgroup analysis.
        joined["age_group"] = pd.cut(
            joined["age_years"],
            bins=[18, 30, 40, 50, 60],
            right=False,
            labels=["18-29", "30-39", "40-49", "50-59"]
        )

        joined["pir_group"] = pd.cut(
            joined["poverty_income_ratio"],
            bins=[-np.inf, 1, 2, 4, np.inf],
            right=False,
            labels=["<1", "1-<2", "2-<4", ">=4"]
        )

        combined.append(joined)

    enriched = pd.concat(
        combined,
        ignore_index=True
    )

    return enriched


# ============================================================
# 8. SUBGROUP SCREENING PERFORMANCE
# ============================================================

def subgroup_audit(enriched, primary):

    log("Calculating per-cycle subgroup diagnostics...")

    rows = []
    score_rows = []

    for (exp_id, study), group in enriched.groupby(
        ["experiment_id", "study"], sort=False
    ):

        metadata = primary.loc[
            (primary["experiment_id"] == exp_id)
            & (primary["study"] == study)
        ].iloc[0]

        for variable in SUBGROUP_COLUMNS:

            if variable not in group.columns:
                continue

            for subgroup_value, sub in group.groupby(
                variable,
                dropna=False,
                observed=True
            ):

                label = (
                    "Missing"
                    if pd.isna(subgroup_value)
                    else str(subgroup_value)
                )

                metrics = calculate_metrics(sub)

                base = {
                    "experiment_id": exp_id,
                    "study": study,
                    "design": metadata["design"],
                    "test_cycle": metadata["test_cycle"],
                    "subgroup_variable": variable,
                    "subgroup_value": label
                }

                rows.append({
                    **base,
                    **metrics
                })

                scores = sub["score"]

                score_rows.append({
                    **base,
                    "n": len(sub),
                    "positives": int(
                        sub["reference_outcome"].sum()
                    ),
                    "mean_predicted_risk": float(
                        scores.mean()
                    ),
                    "median_predicted_risk": float(
                        scores.median()
                    ),
                    "score_q10": float(
                        scores.quantile(0.10)
                    ),
                    "score_q90": float(
                        scores.quantile(0.90)
                    ),
                    "observed_outcome_fraction": float(
                        sub["reference_outcome"].mean()
                    ),
                    "mean_score_positive": (
                        float(
                            sub.loc[
                                sub["reference_outcome"] == 1,
                                "score"
                            ].mean()
                        )
                    ),
                    "mean_score_negative": (
                        float(
                            sub.loc[
                                sub["reference_outcome"] == 0,
                                "score"
                            ].mean()
                        )
                    )
                })

    subgroup_df = pd.DataFrame(rows)
    score_df = pd.DataFrame(score_rows)

    save_csv(
        subgroup_df,
        "phase5_subgroup_per_cycle_ci.csv"
    )

    save_csv(
        score_df,
        "phase5_subgroup_score_calibration_descriptive.csv"
    )

    return subgroup_df, score_df


# ============================================================
# 9. IRON COHORT COMPARABILITY
# ============================================================

def iron_cohort_comparison(cohorts):

    log("Comparing 2017-2018 iron cohorts...")

    rows = []

    for study in ["iron_observed", "iron_matched"]:

        df = cohorts[study].copy()

        df = df[
            df["survey_cycle"]
            .astype(str)
            .str.contains("2017")
        ]

        if df.empty:
            raise ValueError(
                f"No 2017-2018 records for {study}"
            )

        row = {
            "cohort": study,
            "n": len(df),
            "unique_participants": df["seqn"].nunique()
        }

        for col in [
            "age_years",
            "hemoglobin_g_dL",
            "cbc_hemoglobin",
            "ferritin_ug_L",
            "poverty_income_ratio"
        ]:

            if col in df.columns:
                values = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

                row[f"{col}_mean"] = values.mean()
                row[f"{col}_median"] = values.median()
                row[f"{col}_missing"] = values.isna().sum()

        if "primary_outcome" in df.columns:
            outcome = pd.to_numeric(
                df["primary_outcome"],
                errors="coerce"
            )

            row["positive_cases"] = int(outcome.sum())
            row["positive_fraction"] = outcome.mean()

        if "sex" in df.columns:
            for category, count in df["sex"].value_counts(
                dropna=False
            ).items():
                row[f"sex_{category}_n"] = int(count)

        if "race_ethnicity_5cat" in df.columns:
            for category, count in df[
                "race_ethnicity_5cat"
            ].value_counts(dropna=False).items():

                row[f"race_{category}_n"] = int(count)

        rows.append(row)

    result = pd.DataFrame(rows)

    save_csv(
        result,
        "phase5_iron_2017_2018_cohort_comparison.csv"
    )

    return result


# ============================================================
# 10. RUN MANIFEST
# ============================================================

def write_manifest(
    primary_path,
    actions_path,
    primary_audit,
    subgroup_df,
    reconciliation
):

    manifest = {
        "audit_name": "RAMS Phase 5 Statistical Audit",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "primary_source": str(primary_path),
        "actions_source": str(actions_path),
        "primary_evaluations": len(primary_audit),
        "subgroup_evaluation_rows": len(subgroup_df),
        "reconciliation_passed": bool(
            reconciliation["all_match"].all()
        ),
        "models_refitted": False,
        "thresholds_changed": False,
        "method": "Wilson score confidence intervals",
        "confidence_level": 0.95,
        "limitations": [
            "Individual temporal evaluation intervals are "
            "unweighted binomial intervals.",
            "They do not incorporate the NHANES complex "
            "survey design.",
            "Temporal evaluations overlap and are not "
            "independent replications.",
            "Subgroup score summaries are descriptive.",
            "This audit does not independently validate "
            "the development-selected RAMS configuration."
        ]
    }

    path = OUTPUT_DIR / "phase5_manifest.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log("Saved: phase5_manifest.json")


# ============================================================
# 11. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 65)
    print("RAMS PHASE 5: STATISTICAL AUDIT")
    print("=" * 65)

    # Check inputs before creating outputs.
    if not RAMS_DIR.exists():
        raise FileNotFoundError(
            f"RAMS folder not found:\n{RAMS_DIR}"
        )

    if not COHORT_DIR.exists():
        raise FileNotFoundError(
            f"Analytic cohort folder not found:\n{COHORT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    primary, actions, primary_path, actions_path = (
        load_rams_data()
    )

    # Primary statistical results
    primary_audit = audit_primary(
        primary, actions
    )

    # Numerical reconciliation
    reconciliation = reconcile_primary(
        primary, primary_audit
    )

    # Cohort metadata
    cohorts = load_cohorts()

    # Link metadata
    enriched = enrich_actions(
        actions, cohorts
    )

    # Subgroup diagnostics
    subgroup_df, score_df = subgroup_audit(
        enriched, primary
    )

    # Iron cohort comparability
    iron_comparison = iron_cohort_comparison(
        cohorts
    )

    # Manifest
    write_manifest(
        primary_path,
        actions_path,
        primary_audit,
        subgroup_df,
        reconciliation
    )

    print()
    print("=" * 65)
    print("RAMS PHASE 5 COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print(
        "Primary evaluations:",
        len(primary_audit)
    )

    print(
        "Subgroup evaluation rows:",
        len(subgroup_df)
    )

    print(
        "Reconciliation:",
        "PASS" if reconciliation["all_match"].all()
        else "FAIL"
    )

    print()
    print("Output directory:")
    print(OUTPUT_DIR)
    print("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("RAMS PHASE 5 FAILED:")
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
