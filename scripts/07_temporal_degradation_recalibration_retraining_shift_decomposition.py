"""
===============================================================================
STAGE 07 — CORRECTED CANONICAL VERSION
Temporal Degradation, Recalibration vs Retraining,
Tier Contribution, Eligibility Shift, and Operational Burden
===============================================================================

===============================================================================
"""

from __future__ import annotations

import shutil
import warnings

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.stats import spearmanr


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_ROOT = PROJECT_ROOT / "processed"

STAGE06_DIR = (
    PROCESSED_ROOT
    / "08_model_results"
)

OUTPUT_DIR = (
    PROCESSED_ROOT
    / "09_temporal_interpretation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. STAGE 06 INPUTS
# =============================================================================

EXPERIMENT_METRICS_PATH = (
    STAGE06_DIR
    / "06_experiment_metrics.csv"
)

BOOTSTRAP_PATH = (
    STAGE06_DIR
    / "06_bootstrap_ci.csv"
)

TESTING_BURDEN_PATH = (
    STAGE06_DIR
    / "06_testing_burden.csv"
)

TIER_COMPARISON_PATH = (
    STAGE06_DIR
    / "06_tier1_vs_tier2_comparison.csv"
)

IRON_COMPARISON_PATH = (
    STAGE06_DIR
    / "06_iron_observed_vs_matched_performance.csv"
)

EXECUTION_STATUS_PATH = (
    STAGE06_DIR
    / "06_execution_status.csv"
)


# =============================================================================
# 3. STAGE 04 FILENAMES
# =============================================================================

STAGE04_FILENAMES = {
    "continuous":
        "04_continuous_shift_metrics.csv",

    "categorical":
        "04_categorical_shift_metrics.csv",

    "iron_matched_shift":
        "04_iron_observed_vs_matched_shift.csv",

    "shift_source":
        "04_shift_source_summary.csv",
}


# =============================================================================
# 4. STAGE 07 OUTPUTS
# =============================================================================

TEMPORAL_PROFILE_PATH = (
    OUTPUT_DIR
    / "07_temporal_transport_profile.csv"
)

TEMPORAL_ASSOCIATION_PATH = (
    OUTPUT_DIR
    / "07_temporal_distance_association.csv"
)

RECALIBRATION_EFFECT_PATH = (
    OUTPUT_DIR
    / "07_recalibration_effect.csv"
)

RECALIBRATION_SUMMARY_PATH = (
    OUTPUT_DIR
    / "07_recalibration_summary.csv"
)

RETRAINING_EFFECT_PATH = (
    OUTPUT_DIR
    / "07_retraining_effect.csv"
)

RETRAINING_SUMMARY_PATH = (
    OUTPUT_DIR
    / "07_retraining_summary.csv"
)

TIER_MODEL_PATH = (
    OUTPUT_DIR
    / "07_tier_contribution_model_level.csv"
)

TIER_SUMMARY_PATH = (
    OUTPUT_DIR
    / "07_tier_contribution_summary.csv"
)

IRON_MODEL_PATH = (
    OUTPUT_DIR
    / "07_iron_natural_experiment_model_level.csv"
)

IRON_SUMMARY_PATH = (
    OUTPUT_DIR
    / "07_iron_natural_experiment_summary.csv"
)

SHIFT_PAIR_PATH = (
    OUTPUT_DIR
    / "07_shift_pair_summary.csv"
)

SHIFT_LINKAGE_PATH = (
    OUTPUT_DIR
    / "07_shift_performance_linkage.csv"
)

SHIFT_DECOMPOSITION_PATH = (
    OUTPUT_DIR
    / "07_shift_decomposition_descriptive.csv"
)

BURDEN_MODEL_PATH = (
    OUTPUT_DIR
    / "07_testing_burden_model_level.csv"
)

BURDEN_SUMMARY_PATH = (
    OUTPUT_DIR
    / "07_testing_burden_summary.csv"
)

MODEL_STABILITY_PATH = (
    OUTPUT_DIR
    / "07_model_stability_summary.csv"
)

KEY_FINDINGS_PATH = (
    OUTPUT_DIR
    / "07_key_findings.csv"
)

INTEGRITY_PATH = (
    OUTPUT_DIR
    / "07_integrity_check.csv"
)

DECISION_LEDGER_PATH = (
    OUTPUT_DIR
    / "decision_ledger_stage07.csv"
)

WORKBOOK_PATH = (
    OUTPUT_DIR
    / "07_stage07_summary.xlsx"
)


# =============================================================================
# 5. DESIGN CONSTANTS
# =============================================================================

PRIMARY_CALIBRATION = "sigmoid"

PRIMARY_METRICS = [
    "AUROC",
    "AUPRC",
    "Brier",
    "ECE",
]

HIGHER_IS_BETTER = {
    "AUROC": True,
    "AUPRC": True,
    "Brier": False,
    "ECE": False,
}


# =============================================================================
# 6. ACTUAL NHANES CYCLE DEFINITIONS
# =============================================================================

CYCLE_ORDER = {
    "2009_2010": 1,
    "2011_2012": 2,
    "2013_2014": 3,
    "2015_2016": 4,
    "2017_2018": 5,
    "2017_2020_PrePandemic": 6,
    "2021_2023": 7,
}

CYCLE_MIDPOINT = {
    "2009_2010": 2009.5,
    "2011_2012": 2011.5,
    "2013_2014": 2013.5,
    "2015_2016": 2015.5,
    "2017_2018": 2017.5,
    "2017_2020_PrePandemic": 2018.5,
    "2021_2023": 2022.0,
}

CYCLE_SHORT = {
    "2009_2010": "F",
    "2011_2012": "G",
    "2013_2014": "H",
    "2015_2016": "I",
    "2017_2018": "J",
    "2017_2020_PrePandemic": "P",
    "2021_2023": "L",
}


# =============================================================================
# 7. BACKUP PREVIOUS FAILED STAGE 07
# =============================================================================

def backup_previous_stage07():

    existing_files = list(
        OUTPUT_DIR.glob(
            "07_*"
        )
    )

    if not existing_files:
        return None

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_dir = (
        OUTPUT_DIR
        / "previous_stage07_attempts"
        / f"attempt_backup_{timestamp}"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for source in existing_files:

        if source.is_file():

            shutil.copy2(
                source,
                backup_dir / source.name,
            )

    print(
        f"Previous Stage-07 outputs backed up to:\n"
        f"  {backup_dir}"
    )

    return backup_dir


# =============================================================================
# 8. GENERIC HELPERS
# =============================================================================

def require_file(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{path}"
        )


def read_csv_required(
    path: Path,
) -> pd.DataFrame:

    require_file(
        path
    )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def find_processed_file(
    filename: str,
) -> Path:

    candidates = list(
        PROCESSED_ROOT.rglob(
            filename
        )
    )

    if not candidates:

        # Useful when running against a local test directory
        candidates = list(
            PROJECT_ROOT.rglob(
                filename
            )
        )

    if not candidates:

        raise FileNotFoundError(
            f"Could not locate {filename}"
        )

    # Prefer files underneath processed/
    processed_matches = [
        p
        for p in candidates
        if "processed" in p.parts
    ]

    if processed_matches:

        return processed_matches[0]

    return candidates[0]


def split_cycles(
    text,
) -> List[str]:

    if pd.isna(text):
        return []

    text = str(
        text
    ).strip()

    if not text:
        return []

    return [
        x.strip()
        for x in text.split("|")
        if x.strip()
    ]


def earliest_train_cycle(
    train_cycles,
) -> Optional[str]:

    cycles = [
        c
        for c in split_cycles(
            train_cycles
        )
        if c in CYCLE_ORDER
    ]

    if not cycles:
        return None

    return min(
        cycles,
        key=lambda c:
            CYCLE_ORDER[c],
    )


def most_recent_train_cycle(
    train_cycles,
) -> Optional[str]:

    cycles = [
        c
        for c in split_cycles(
            train_cycles
        )
        if c in CYCLE_ORDER
    ]

    if not cycles:
        return None

    return max(
        cycles,
        key=lambda c:
            CYCLE_ORDER[c],
    )


def temporal_distance(
    reference_cycle: str,
    target_cycle: str,
) -> Tuple[float, float]:

    if (
        reference_cycle
        not in CYCLE_MIDPOINT
        or target_cycle
        not in CYCLE_MIDPOINT
    ):

        return (
            np.nan,
            np.nan,
        )

    year_difference = (
        CYCLE_MIDPOINT[
            target_cycle
        ]
        -
        CYCLE_MIDPOINT[
            reference_cycle
        ]
    )

    cycle_steps = (
        CYCLE_ORDER[
            target_cycle
        ]
        -
        CYCLE_ORDER[
            reference_cycle
        ]
    )

    return (
        float(
            year_difference
        ),
        float(
            cycle_steps
        ),
    )


def safe_numeric_mean(
    series: pd.Series,
) -> float:

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    return float(
        values.mean()
    )


def safe_numeric_median(
    series: pd.Series,
) -> float:

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    return float(
        values.median()
    )


def improvement(
    new_value,
    old_value,
    metric: str,
):

    if (
        pd.isna(new_value)
        or pd.isna(old_value)
    ):

        return np.nan

    if HIGHER_IS_BETTER[
        metric
    ]:

        return (
            float(new_value)
            -
            float(old_value)
        )

    return (
        float(old_value)
        -
        float(new_value)
    )


# =============================================================================
# 9. VALIDATE STAGE 06
# =============================================================================

def validate_stage06():

    status = read_csv_required(
        EXECUTION_STATUS_PATH
    )

    latest = (
        status
        .drop_duplicates(
            subset=[
                "experiment_id"
            ],
            keep="last",
        )
        .copy()
    )

    failures = (
        latest[
            latest[
                "status"
            ]
            != "SUCCESS"
        ]
    )

    if not failures.empty:

        raise RuntimeError(
            f"Stage 06 has "
            f"{len(failures)} "
            f"unfinished experiments."
        )

    if len(latest) != 336:

        raise RuntimeError(
            f"Expected 336 completed experiments; "
            f"found {len(latest)}."
        )

    print(
        "Stage 06 validation: "
        "336/336 experiments SUCCESS."
    )


# =============================================================================
# 10. LOAD INPUT DATA
# =============================================================================

def load_inputs():

    validate_stage06()

    metrics = read_csv_required(
        EXPERIMENT_METRICS_PATH
    )

    bootstrap = read_csv_required(
        BOOTSTRAP_PATH
    )

    burden = read_csv_required(
        TESTING_BURDEN_PATH
    )

    tier = read_csv_required(
        TIER_COMPARISON_PATH
    )

    iron = read_csv_required(
        IRON_COMPARISON_PATH
    )

    stage04 = {}

    for key, filename in (
        STAGE04_FILENAMES.items()
    ):

        path = find_processed_file(
            filename
        )

        print(
            f"Stage 04 {key}: "
            f"{path}"
        )

        stage04[
            key
        ] = pd.read_csv(
            path,
            low_memory=False,
        )

    return (
        metrics,
        bootstrap,
        burden,
        tier,
        iron,
        stage04,
    )


# =============================================================================
# 11. TEMPORAL TRANSPORT PROFILE
# =============================================================================

def build_temporal_profile(
    metrics: pd.DataFrame,
):

    primary = (
        metrics[
            metrics[
                "calibration"
            ]
            == PRIMARY_CALIBRATION
        ]
        .copy()
    )

    primary[
        "earliest_train_cycle"
    ] = primary[
        "train_cycles"
    ].apply(
        earliest_train_cycle
    )

    primary[
        "most_recent_train_cycle"
    ] = primary[
        "train_cycles"
    ].apply(
        most_recent_train_cycle
    )

    primary[
        "train_cycle_count"
    ] = primary[
        "train_cycles"
    ].apply(
        lambda x:
            len(
                split_cycles(
                    x
                )
            )
    )

    signed_years = []
    signed_steps = []

    for _, row in primary.iterrows():

        years, steps = temporal_distance(
            row[
                "most_recent_train_cycle"
            ],
            row[
                "test_cycle"
            ],
        )

        signed_years.append(
            years
        )

        signed_steps.append(
            steps
        )

    primary[
        "signed_temporal_distance_years"
    ] = signed_years

    primary[
        "absolute_temporal_distance_years"
    ] = np.abs(
        primary[
            "signed_temporal_distance_years"
        ]
    )

    primary[
        "signed_temporal_distance_steps"
    ] = signed_steps

    primary[
        "absolute_temporal_distance_steps"
    ] = np.abs(
        primary[
            "signed_temporal_distance_steps"
        ]
    )

    primary[
        "transport_direction"
    ] = np.select(
        [
            primary[
                "signed_temporal_distance_years"
            ] > 0,

            primary[
                "signed_temporal_distance_years"
            ] < 0,
        ],
        [
            "forward",
            "backward",
        ],
        default="same_period",
    )

    primary[
        "earliest_train_cycle_short"
    ] = primary[
        "earliest_train_cycle"
    ].map(
        CYCLE_SHORT
    )

    primary[
        "most_recent_train_cycle_short"
    ] = primary[
        "most_recent_train_cycle"
    ].map(
        CYCLE_SHORT
    )

    primary[
        "test_cycle_short"
    ] = primary[
        "test_cycle"
    ].map(
        CYCLE_SHORT
    )

    primary.to_csv(
        TEMPORAL_PROFILE_PATH,
        index=False,
    )

    return primary


# =============================================================================
# 12. TEMPORAL DISTANCE ASSOCIATION
# =============================================================================

def build_temporal_distance_association(
    temporal: pd.DataFrame,
):

    rows = []

    # Fixed-origin is the cleanest temporal-distance experiment because
    # the training origin is unchanged while temporal distance grows.
    fixed = (
        temporal[
            temporal[
                "temporal_design"
            ]
            == "fixed_origin"
        ]
        .copy()
    )

    for (
        study,
        feature_tier,
        model,
    ), group in fixed.groupby(
        [
            "study",
            "feature_tier",
            "model",
        ]
    ):

        x = pd.to_numeric(
            group[
                "absolute_temporal_distance_years"
            ],
            errors="coerce",
        )

        for metric in PRIMARY_METRICS:

            y = pd.to_numeric(
                group[
                    metric
                ],
                errors="coerce",
            )

            valid = (
                x.notna()
                & y.notna()
            )

            n = int(
                valid.sum()
            )

            if n < 3:
                continue

            rho, p = spearmanr(
                x[
                    valid
                ],
                y[
                    valid
                ],
            )

            rows.append(
                {
                    "analysis_scope":
                        "fixed_origin",

                    "study":
                        study,

                    "feature_tier":
                        feature_tier,

                    "model":
                        model,

                    "metric":
                        metric,

                    "n_transport_points":
                        n,

                    "spearman_rho":
                        rho,

                    "spearman_p_descriptive":
                        p,

                    "interpretation":
                        (
                            "Descriptive association "
                            "between temporal distance "
                            "and predictive performance."
                        ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        TEMPORAL_ASSOCIATION_PATH,
        index=False,
    )

    return result


# =============================================================================
# 13. RECALIBRATION
# =============================================================================

def build_recalibration_analysis(
    metrics: pd.DataFrame,
):

    keys = [
        "experiment_id",
        "study",
        "temporal_design",
        "split_role",
        "train_cycles",
        "test_cycle",
        "feature_tier",
        "model",
    ]

    raw = (
        metrics[
            metrics[
                "calibration"
            ]
            == "raw"
        ][
            keys
            + PRIMARY_METRICS
        ]
        .copy()
    )

    sigmoid = (
        metrics[
            metrics[
                "calibration"
            ]
            == "sigmoid"
        ][
            keys
            + PRIMARY_METRICS
        ]
        .copy()
    )

    isotonic = (
        metrics[
            metrics[
                "calibration"
            ]
            == "isotonic"
        ][
            keys
            + PRIMARY_METRICS
        ]
        .copy()
    )

    raw = raw.rename(
        columns={
            metric:
                f"{metric}_raw"
            for metric
            in PRIMARY_METRICS
        }
    )

    sigmoid = sigmoid.rename(
        columns={
            metric:
                f"{metric}_sigmoid"
            for metric
            in PRIMARY_METRICS
        }
    )

    isotonic = isotonic.rename(
        columns={
            metric:
                f"{metric}_isotonic"
            for metric
            in PRIMARY_METRICS
        }
    )

    merged = (
        raw
        .merge(
            sigmoid,
            on=keys,
            how="inner",
        )
        .merge(
            isotonic,
            on=keys,
            how="inner",
        )
    )

    for method in [
        "sigmoid",
        "isotonic",
    ]:

        for metric in PRIMARY_METRICS:

            merged[
                f"{method}_improvement_{metric}"
            ] = merged.apply(
                lambda row:
                    improvement(
                        row[
                            f"{metric}_{method}"
                        ],
                        row[
                            f"{metric}_raw"
                        ],
                        metric,
                    ),
                axis=1,
            )

    merged.to_csv(
        RECALIBRATION_EFFECT_PATH,
        index=False,
    )

    rows = []

    for (
        study,
        tier,
    ), group in merged.groupby(
        [
            "study",
            "feature_tier",
        ]
    ):

        row = {
            "study":
                study,

            "feature_tier":
                tier,

            "n_experiments":
                len(group),
        }

        for method in [
            "sigmoid",
            "isotonic",
        ]:

            for metric in PRIMARY_METRICS:

                column = (
                    f"{method}_improvement_"
                    f"{metric}"
                )

                row[
                    f"mean_{column}"
                ] = safe_numeric_mean(
                    group[
                        column
                    ]
                )

                row[
                    f"median_{column}"
                ] = safe_numeric_median(
                    group[
                        column
                    ]
                )

                row[
                    f"proportion_improved_"
                    f"{method}_{metric}"
                ] = float(
                    (
                        group[
                            column
                        ]
                        > 0
                    )
                    .mean()
                )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        RECALIBRATION_SUMMARY_PATH,
        index=False,
    )

    return (
        merged,
        summary,
    )


# =============================================================================
# 14. FIXED-ORIGIN VS ROLLING-EXPANDING RETRAINING
# =============================================================================

def build_retraining_analysis(
    temporal: pd.DataFrame,
):

    fixed = (
        temporal[
            temporal[
                "temporal_design"
            ]
            == "fixed_origin"
        ]
        .copy()
    )

    rolling = (
        temporal[
            temporal[
                "temporal_design"
            ]
            == "rolling_expanding"
        ]
        .copy()
    )

    keys = [
        "study",
        "test_cycle",
        "feature_tier",
        "model",
    ]

    fixed_columns = (
        keys
        + [
            "experiment_id",
            "train_cycles",
        ]
        + PRIMARY_METRICS
    )

    rolling_columns = (
        keys
        + [
            "experiment_id",
            "train_cycles",
        ]
        + PRIMARY_METRICS
    )

    fixed2 = fixed[
        fixed_columns
    ].rename(
        columns={
            "experiment_id":
                "fixed_experiment_id",

            "train_cycles":
                "fixed_train_cycles",

            **{
                metric:
                    f"{metric}_fixed"
                for metric
                in PRIMARY_METRICS
            },
        }
    )

    rolling2 = rolling[
        rolling_columns
    ].rename(
        columns={
            "experiment_id":
                "rolling_experiment_id",

            "train_cycles":
                "rolling_train_cycles",

            **{
                metric:
                    f"{metric}_rolling"
                for metric
                in PRIMARY_METRICS
            },
        }
    )

    comparison = fixed2.merge(
        rolling2,
        on=keys,
        how="inner",
    )

    for metric in PRIMARY_METRICS:

        comparison[
            f"retraining_improvement_{metric}"
        ] = comparison.apply(
            lambda row:
                improvement(
                    row[
                        f"{metric}_rolling"
                    ],
                    row[
                        f"{metric}_fixed"
                    ],
                    metric,
                ),
            axis=1,
        )

    comparison.to_csv(
        RETRAINING_EFFECT_PATH,
        index=False,
    )

    rows = []

    for (
        study,
        tier,
    ), group in comparison.groupby(
        [
            "study",
            "feature_tier",
        ]
    ):

        row = {
            "study":
                study,

            "feature_tier":
                tier,

            "n_comparisons":
                len(group),
        }

        for metric in PRIMARY_METRICS:

            column = (
                f"retraining_improvement_{metric}"
            )

            row[
                f"mean_{column}"
            ] = safe_numeric_mean(
                group[
                    column
                ]
            )

            row[
                f"median_{column}"
            ] = safe_numeric_median(
                group[
                    column
                ]
            )

            row[
                f"proportion_improved_{metric}"
            ] = float(
                (
                    group[
                        column
                    ]
                    > 0
                )
                .mean()
            )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        RETRAINING_SUMMARY_PATH,
        index=False,
    )

    return (
        comparison,
        summary,
    )


# =============================================================================
# 15. TIER CONTRIBUTION
# =============================================================================

def build_tier_analysis(
    tier: pd.DataFrame,
):

    tier = tier.copy()

    tier.to_csv(
        TIER_MODEL_PATH,
        index=False,
    )

    delta_columns = [
        column
        for column in tier.columns
        if column.startswith(
            "delta_"
        )
    ]

    rows = []

    for study, group in tier.groupby(
        "study"
    ):

        row = {
            "study":
                study,

            "n_comparisons":
                len(group),
        }

        for column in delta_columns:

            row[
                f"mean_{column}"
            ] = safe_numeric_mean(
                group[
                    column
                ]
            )

            row[
                f"median_{column}"
            ] = safe_numeric_median(
                group[
                    column
                ]
            )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        TIER_SUMMARY_PATH,
        index=False,
    )

    return (
        tier,
        summary,
    )


# =============================================================================
# 16. IRON NATURAL EXPERIMENT
# =============================================================================

def build_iron_natural_experiment(
    iron: pd.DataFrame,
):

    iron = iron.copy()

    iron.to_csv(
        IRON_MODEL_PATH,
        index=False,
    )

    difference_columns = [
        column
        for column in iron.columns
        if column.endswith(
            "_matched_minus_observed"
        )
    ]

    rows = []

    grouping = [
        "temporal_design",
        "test_cycle",
        "feature_tier",
    ]

    for keys, group in iron.groupby(
        grouping
    ):

        row = {
            "temporal_design":
                keys[0],

            "test_cycle":
                keys[1],

            "feature_tier":
                keys[2],

            "n_models":
                len(group),
        }

        for column in difference_columns:

            row[
                f"mean_{column}"
            ] = safe_numeric_mean(
                group[
                    column
                ]
            )

            row[
                f"median_{column}"
            ] = safe_numeric_median(
                group[
                    column
                ]
            )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        IRON_SUMMARY_PATH,
        index=False,
    )

    return (
        iron,
        summary,
    )


# =============================================================================
# 17. BUILD EXACT STAGE-04 SHIFT PAIR SUMMARY
# =============================================================================

def build_shift_pair_summary(
    stage04: Dict[str, pd.DataFrame],
):

    continuous = stage04[
        "continuous"
    ].copy()

    categorical = stage04[
        "categorical"
    ].copy()

    source = stage04[
        "shift_source"
    ].copy()

    # -------------------------------------------------------------------------
    # Continuous shift aggregation
    # -------------------------------------------------------------------------

    continuous_summary = (
        continuous.groupby(
            [
                "study",
                "comparison_design",
                "reference_cycle",
                "comparison_cycle",
                "temporal_distance_years",
            ],
            dropna=False,
        )
        .agg(
            mean_abs_smd=(
                "abs_smd",
                "mean",
            ),

            max_abs_smd=(
                "abs_smd",
                "max",
            ),

            mean_psi=(
                "psi",
                "mean",
            ),

            max_psi=(
                "psi",
                "max",
            ),

            mean_ks=(
                "ks_statistic",
                "mean",
            ),

            max_ks=(
                "ks_statistic",
                "max",
            ),

            n_continuous_features=(
                "feature",
                "nunique",
            ),
        )
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # Categorical shift aggregation
    # -------------------------------------------------------------------------

    categorical_summary = (
        categorical.groupby(
            [
                "study",
                "comparison_design",
                "reference_cycle",
                "comparison_cycle",
                "temporal_distance_years",
            ],
            dropna=False,
        )
        .agg(
            mean_js_divergence=(
                "js_divergence",
                "mean",
            ),

            max_js_divergence=(
                "js_divergence",
                "max",
            ),

            mean_max_category_difference=(
                "max_abs_category_difference",
                "mean",
            ),

            max_category_difference=(
                "max_abs_category_difference",
                "max",
            ),

            n_categorical_features=(
                "feature",
                "nunique",
            ),
        )
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    pair = continuous_summary.merge(
        categorical_summary,
        on=[
            "study",
            "comparison_design",
            "reference_cycle",
            "comparison_cycle",
            "temporal_distance_years",
        ],
        how="outer",
    )

    # -------------------------------------------------------------------------
    # Add weighted prevalence at reference and comparison cycles
    # -------------------------------------------------------------------------

    prevalence = source[
        [
            "study",
            "cycle",
            "weighted_prevalence_pct",
            "known_protocol_or_eligibility_shift",
        ]
    ].copy()

    reference_prevalence = (
        prevalence.rename(
            columns={
                "cycle":
                    "reference_cycle",

                "weighted_prevalence_pct":
                    "reference_weighted_prevalence_pct",

                "known_protocol_or_eligibility_shift":
                    "reference_known_protocol_or_eligibility_shift",
            }
        )
    )

    comparison_prevalence = (
        prevalence.rename(
            columns={
                "cycle":
                    "comparison_cycle",

                "weighted_prevalence_pct":
                    "comparison_weighted_prevalence_pct",

                "known_protocol_or_eligibility_shift":
                    "comparison_known_protocol_or_eligibility_shift",
            }
        )
    )

    pair = pair.merge(
        reference_prevalence,
        on=[
            "study",
            "reference_cycle",
        ],
        how="left",
    )

    pair = pair.merge(
        comparison_prevalence,
        on=[
            "study",
            "comparison_cycle",
        ],
        how="left",
    )

    pair[
        "prevalence_shift_pp"
    ] = (
        pair[
            "comparison_weighted_prevalence_pct"
        ]
        -
        pair[
            "reference_weighted_prevalence_pct"
        ]
    )

    pair[
        "absolute_prevalence_shift_pp"
    ] = pair[
        "prevalence_shift_pp"
    ].abs()

    # -------------------------------------------------------------------------
    # Create reverse-direction copies.
    #
    # |SMD|, PSI, KS and JS are used as magnitude measures here.
    # This allows J -> I bridge experiments to connect to the measured I -> J
    # population-shift magnitude without pretending direction is identical.
    #
    # Prevalence change IS directional and is reversed explicitly.
    # -------------------------------------------------------------------------

    reverse = pair.copy()

    reverse[
        [
            "reference_cycle",
            "comparison_cycle",
        ]
    ] = reverse[
        [
            "comparison_cycle",
            "reference_cycle",
        ]
    ].values

    reverse[
        "temporal_distance_years"
    ] = (
        -reverse[
            "temporal_distance_years"
        ]
    )

    reverse[
        [
            "reference_weighted_prevalence_pct",
            "comparison_weighted_prevalence_pct",
        ]
    ] = reverse[
        [
            "comparison_weighted_prevalence_pct",
            "reference_weighted_prevalence_pct",
        ]
    ].values

    reverse[
        [
            "reference_known_protocol_or_eligibility_shift",
            "comparison_known_protocol_or_eligibility_shift",
        ]
    ] = reverse[
        [
            "comparison_known_protocol_or_eligibility_shift",
            "reference_known_protocol_or_eligibility_shift",
        ]
    ].values

    reverse[
        "prevalence_shift_pp"
    ] = (
        -reverse[
            "prevalence_shift_pp"
        ]
    )

    reverse[
        "absolute_prevalence_shift_pp"
    ] = reverse[
        "prevalence_shift_pp"
    ].abs()

    reverse[
        "shift_direction_source"
    ] = "reverse_of_stage04_pair"

    pair[
        "shift_direction_source"
    ] = "direct_stage04_pair"

    pair_both = pd.concat(
        [
            pair,
            reverse,
        ],
        ignore_index=True,
    )

    pair_both = (
        pair_both
        .drop_duplicates(
            subset=[
                "study",
                "comparison_design",
                "reference_cycle",
                "comparison_cycle",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    pair_both.to_csv(
        SHIFT_PAIR_PATH,
        index=False,
    )

    return pair_both


# =============================================================================
# 18. MAP STAGE-06 EXPERIMENTS TO STAGE-04 SHIFT PAIRS
# =============================================================================

def map_shift_design(
    temporal_design: str,
) -> str:

    if temporal_design == "fixed_origin":
        return "fixed_origin"

    if temporal_design in {
        "rolling_expanding",
        "bridge_J_to_I",
        "bridge_J_to_L",
    }:
        return "rolling"

    return "rolling"


def build_shift_performance_linkage(
    temporal: pd.DataFrame,
    shift_pairs: pd.DataFrame,
):

    performance = temporal.copy()

    performance[
        "shift_comparison_design"
    ] = performance[
        "temporal_design"
    ].apply(
        map_shift_design
    )

    performance[
        "shift_reference_cycle"
    ] = np.where(
        performance[
            "temporal_design"
        ]
        == "fixed_origin",

        performance[
            "earliest_train_cycle"
        ],

        performance[
            "most_recent_train_cycle"
        ],
    )

    performance[
        "shift_comparison_cycle"
    ] = performance[
        "test_cycle"
    ]

    shift = shift_pairs.rename(
        columns={
            "comparison_design":
                "shift_comparison_design",

            "reference_cycle":
                "shift_reference_cycle",

            "comparison_cycle":
                "shift_comparison_cycle",
        }
    )

    linkage = performance.merge(
        shift,
        on=[
            "study",
            "shift_comparison_design",
            "shift_reference_cycle",
            "shift_comparison_cycle",
        ],
        how="left",
        suffixes=(
            "",
            "_shift",
        ),
    )

    linkage[
        "shift_linked"
    ] = linkage[
        "mean_abs_smd"
    ].notna()

    linkage.to_csv(
        SHIFT_LINKAGE_PATH,
        index=False,
    )

    return linkage


# =============================================================================
# 19. DESCRIPTIVE SHIFT-PERFORMANCE DECOMPOSITION
# =============================================================================

def build_shift_decomposition(
    linkage: pd.DataFrame,
):

    shift_metrics = [
        "mean_abs_smd",
        "max_abs_smd",
        "mean_psi",
        "max_psi",
        "mean_ks",
        "max_ks",
        "mean_js_divergence",
        "max_js_divergence",
        "mean_max_category_difference",
        "max_category_difference",
        "absolute_prevalence_shift_pp",
    ]

    rows = []

    # Aggregate across model families first to avoid pretending that six models
    # applied to the same temporal transition are six independent shifts.
    aggregation = (
        linkage[
            linkage[
                "shift_linked"
            ]
        ]
        .groupby(
            [
                "study",
                "temporal_design",
                "feature_tier",
                "shift_reference_cycle",
                "shift_comparison_cycle",
            ],
            dropna=False,
        )
        .agg(
            AUROC=("AUROC", "mean"),
            AUPRC=("AUPRC", "mean"),
            Brier=("Brier", "mean"),
            ECE=("ECE", "mean"),

            **{
                metric:
                    (
                        metric,
                        "first",
                    )
                for metric
                in shift_metrics
            },
        )
        .reset_index()
    )

    for (
        study,
        tier,
    ), group in aggregation.groupby(
        [
            "study",
            "feature_tier",
        ]
    ):

        for shift_metric in shift_metrics:

            x = pd.to_numeric(
                group[
                    shift_metric
                ],
                errors="coerce",
            )

            for performance_metric in (
                PRIMARY_METRICS
            ):

                y = pd.to_numeric(
                    group[
                        performance_metric
                    ],
                    errors="coerce",
                )

                valid = (
                    x.notna()
                    & y.notna()
                )

                n = int(
                    valid.sum()
                )

                if n < 4:
                    continue

                rho, p = spearmanr(
                    x[
                        valid
                    ],
                    y[
                        valid
                    ],
                )

                rows.append(
                    {
                        "study":
                            study,

                        "feature_tier":
                            tier,

                        "shift_metric":
                            shift_metric,

                        "performance_metric":
                            performance_metric,

                        "n_transition_points":
                            n,

                        "spearman_rho":
                            rho,

                        "spearman_p_descriptive":
                            p,

                        "interpretation_scope":
                            (
                                "Descriptive association only; "
                                "not causal decomposition."
                            ),
                    }
                )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        SHIFT_DECOMPOSITION_PATH,
        index=False,
    )

    return result


# =============================================================================
# 20. TESTING BURDEN
# =============================================================================

def build_testing_burden(
    burden: pd.DataFrame,
):

    burden = (
        burden[
            burden[
                "calibration"
            ]
            == PRIMARY_CALIBRATION
        ]
        .copy()
    )

    burden.to_csv(
        BURDEN_MODEL_PATH,
        index=False,
    )

    measures = [
        "ReferralRate",
        "TestsAvoided",
        "CasesDetected",
        "CasesMissed",
        "TestsPerDetectedCase",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
    ]

    rows = []

    for (
        study,
        tier,
        target,
    ), group in burden.groupby(
        [
            "study",
            "feature_tier",
            "target_training_sensitivity",
        ]
    ):

        row = {
            "study":
                study,

            "feature_tier":
                tier,

            "target_training_sensitivity":
                target,

            "n_experiments":
                len(group),
        }

        for measure in measures:

            if measure not in group.columns:
                continue

            row[
                f"mean_{measure}"
            ] = safe_numeric_mean(
                group[
                    measure
                ]
            )

            row[
                f"median_{measure}"
            ] = safe_numeric_median(
                group[
                    measure
                ]
            )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        BURDEN_SUMMARY_PATH,
        index=False,
    )

    return (
        burden,
        summary,
    )


# =============================================================================
# 21. MODEL STABILITY
# =============================================================================

def build_model_stability(
    temporal: pd.DataFrame,
):

    rows = []

    for (
        study,
        tier,
        model,
    ), group in temporal.groupby(
        [
            "study",
            "feature_tier",
            "model",
        ]
    ):

        row = {
            "study":
                study,

            "feature_tier":
                tier,

            "model":
                model,

            "n_temporal_tests":
                len(group),
        }

        for metric in PRIMARY_METRICS:

            values = pd.to_numeric(
                group[
                    metric
                ],
                errors="coerce",
            )

            row[
                f"mean_{metric}"
            ] = values.mean()

            row[
                f"sd_{metric}"
            ] = values.std(
                ddof=1
            )

            row[
                f"min_{metric}"
            ] = values.min()

            row[
                f"max_{metric}"
            ] = values.max()

            row[
                f"range_{metric}"
            ] = (
                values.max()
                -
                values.min()
            )

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        MODEL_STABILITY_PATH,
        index=False,
    )

    return result


# =============================================================================
# 22. KEY FINDINGS
# =============================================================================

def build_key_findings(
    temporal: pd.DataFrame,
    tier_summary: pd.DataFrame,
    recalibration_summary: pd.DataFrame,
    retraining_summary: pd.DataFrame,
    burden_summary: pd.DataFrame,
):

    rows = []

    primary_summary = (
        temporal.groupby(
            [
                "study",
                "feature_tier",
            ]
        )[
            PRIMARY_METRICS
        ]
        .mean()
        .reset_index()
    )

    for _, row in primary_summary.iterrows():

        rows.append(
            {
                "finding_domain":
                    "Primary performance",

                "study":
                    row[
                        "study"
                    ],

                "feature_tier":
                    row[
                        "feature_tier"
                    ],

                "measure":
                    "mean_AUROC",

                "value":
                    row[
                        "AUROC"
                    ],
            }
        )

        rows.append(
            {
                "finding_domain":
                    "Primary performance",

                "study":
                    row[
                        "study"
                    ],

                "feature_tier":
                    row[
                        "feature_tier"
                    ],

                "measure":
                    "mean_AUPRC",

                "value":
                    row[
                        "AUPRC"
                    ],
            }
        )

    if (
        not tier_summary.empty
        and
        "mean_delta_AUROC_T2_minus_T1"
        in tier_summary.columns
    ):

        for _, row in tier_summary.iterrows():

            rows.append(
                {
                    "finding_domain":
                        "Tier contribution",

                    "study":
                        row[
                            "study"
                        ],

                    "feature_tier":
                        "T2_minus_T1",

                    "measure":
                        "mean_delta_AUROC",

                    "value":
                        row[
                            "mean_delta_AUROC_T2_minus_T1"
                        ],
                }
            )

    findings = pd.DataFrame(
        rows
    )

    findings.to_csv(
        KEY_FINDINGS_PATH,
        index=False,
    )

    return findings


# =============================================================================
# 23. INTEGRITY CHECKS
# =============================================================================

def build_integrity_check(
    temporal: pd.DataFrame,
    temporal_association: pd.DataFrame,
    shift_pairs: pd.DataFrame,
    linkage: pd.DataFrame,
    shift_decomposition: pd.DataFrame,
    recalibration_effect: pd.DataFrame,
    retraining_effect: pd.DataFrame,
):

    checks = []

    # -------------------------------------------------------------------------
    # Primary metric rows
    # -------------------------------------------------------------------------

    checks.append(
        {
            "check":
                "Primary sigmoid experiment count",

            "expected":
                336,

            "observed":
                len(
                    temporal
                ),

            "pass":
                len(
                    temporal
                )
                == 336,
        }
    )

    # -------------------------------------------------------------------------
    # Temporal distance
    # -------------------------------------------------------------------------

    missing_distance = int(
        temporal[
            "absolute_temporal_distance_years"
        ]
        .isna()
        .sum()
    )

    checks.append(
        {
            "check":
                "Missing temporal distances",

            "expected":
                0,

            "observed":
                missing_distance,

            "pass":
                missing_distance
                == 0,
        }
    )

    checks.append(
        {
            "check":
                "Temporal-distance association non-empty",

            "expected":
                ">0",

            "observed":
                len(
                    temporal_association
                ),

            "pass":
                len(
                    temporal_association
                )
                > 0,
        }
    )

    # -------------------------------------------------------------------------
    # Shift linkage
    # -------------------------------------------------------------------------

    linked_count = int(
        linkage[
            "shift_linked"
        ]
        .sum()
    )

    checks.append(
        {
            "check":
                "Stage04-Stage06 shift linkage",

            "expected":
                336,

            "observed":
                linked_count,

            "pass":
                linked_count
                == 336,
        }
    )

    checks.append(
        {
            "check":
                "Shift-pair summary non-empty",

            "expected":
                ">0",

            "observed":
                len(
                    shift_pairs
                ),

            "pass":
                len(
                    shift_pairs
                )
                > 0,
        }
    )

    checks.append(
        {
            "check":
                "Shift decomposition non-empty",

            "expected":
                ">0",

            "observed":
                len(
                    shift_decomposition
                ),

            "pass":
                len(
                    shift_decomposition
                )
                > 0,
        }
    )

    # -------------------------------------------------------------------------
    # Recalibration
    # -------------------------------------------------------------------------

    checks.append(
        {
            "check":
                "Recalibration experiment count",

            "expected":
                336,

            "observed":
                len(
                    recalibration_effect
                ),

            "pass":
                len(
                    recalibration_effect
                )
                == 336,
        }
    )

    # -------------------------------------------------------------------------
    # Retraining comparisons
    # -------------------------------------------------------------------------

    checks.append(
        {
            "check":
                "Retraining comparison non-empty",

            "expected":
                ">0",

            "observed":
                len(
                    retraining_effect
                ),

            "pass":
                len(
                    retraining_effect
                )
                > 0,
        }
    )

    integrity = pd.DataFrame(
        checks
    )

    integrity.to_csv(
        INTEGRITY_PATH,
        index=False,
    )

    return integrity


# =============================================================================
# 24. DECISION LEDGER
# =============================================================================

def build_decision_ledger():

    rows = [
        {
            "decision_id":
                "D071",

            "stage":
                "07",

            "topic":
                "Canonical replacement",

            "decision":
                (
                    "The unsuccessful first Stage-07 run is "
                    "superseded by this corrected canonical Stage 07."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },

        {
            "decision_id":
                "D072",

            "stage":
                "07",

            "topic":
                "Temporal distance",

            "decision":
                (
                    "Temporal distance is calculated from actual "
                    "NHANES cycle names and the most recent training cycle."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },

        {
            "decision_id":
                "D073",

            "stage":
                "07",

            "topic":
                "Fixed-origin degradation",

            "decision":
                (
                    "Fixed-origin experiments are the primary analysis "
                    "for performance versus temporal distance."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },

        {
            "decision_id":
                "D074",

            "stage":
                "07",

            "topic":
                "Recalibration",

            "decision":
                (
                    "Sigmoid calibration remains primary; raw and isotonic "
                    "outputs are comparative sensitivity analyses."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },

        {
            "decision_id":
                "D075",

            "stage":
                "07",

            "topic":
                "Retraining",

            "decision":
                (
                    "Rolling-expanding models are compared with fixed-origin "
                    "models only for the same study, target cycle, tier, "
                    "and model family."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },

        {
            "decision_id":
                "D076",

            "stage":
                "07",

            "topic":
                "Shift-performance linkage",

            "decision":
                (
                    "Stage-04 distributional shift is linked using exact "
                    "study, reference-cycle, target-cycle and design keys."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },

        {
            "decision_id":
                "D077",

            "stage":
                "07",

            "topic":
                "Backward bridge",

            "decision":
                (
                    "For J-to-I bridge analyses, symmetric shift-magnitude "
                    "measures are taken from the corresponding I-to-J pair; "
                    "prevalence shift direction is reversed explicitly."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },

        {
            "decision_id":
                "D078",

            "stage":
                "07",

            "topic":
                "Causal interpretation",

            "decision":
                (
                    "Shift-performance associations are descriptive and "
                    "will not be interpreted causally."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },
    ]

    ledger = pd.DataFrame(
        rows
    )

    ledger.to_csv(
        DECISION_LEDGER_PATH,
        index=False,
    )

    return ledger


# =============================================================================
# 25. WRITE WORKBOOK
# =============================================================================

def write_workbook(
    key_findings,
    temporal,
    temporal_association,
    recalibration_effect,
    recalibration_summary,
    retraining_effect,
    retraining_summary,
    tier_model,
    tier_summary,
    iron_model,
    iron_summary,
    shift_pairs,
    linkage,
    shift_decomposition,
    burden_model,
    burden_summary,
    model_stability,
    integrity,
    ledger,
):

    with pd.ExcelWriter(
        WORKBOOK_PATH,
        engine="openpyxl",
    ) as writer:

        key_findings.to_excel(
            writer,
            sheet_name="Key_Findings",
            index=False,
        )

        integrity.to_excel(
            writer,
            sheet_name="Integrity_Check",
            index=False,
        )

        temporal.to_excel(
            writer,
            sheet_name="Temporal_Profile",
            index=False,
        )

        temporal_association.to_excel(
            writer,
            sheet_name="Temporal_Distance",
            index=False,
        )

        recalibration_effect.to_excel(
            writer,
            sheet_name="Recalibration_Model",
            index=False,
        )

        recalibration_summary.to_excel(
            writer,
            sheet_name="Recalibration_Summary",
            index=False,
        )

        retraining_effect.to_excel(
            writer,
            sheet_name="Retraining_Model",
            index=False,
        )

        retraining_summary.to_excel(
            writer,
            sheet_name="Retraining_Summary",
            index=False,
        )

        tier_model.to_excel(
            writer,
            sheet_name="Tier_Model",
            index=False,
        )

        tier_summary.to_excel(
            writer,
            sheet_name="Tier_Summary",
            index=False,
        )

        iron_model.to_excel(
            writer,
            sheet_name="Iron_NaturalExp_Model",
            index=False,
        )

        iron_summary.to_excel(
            writer,
            sheet_name="Iron_NaturalExp_Summary",
            index=False,
        )

        shift_pairs.to_excel(
            writer,
            sheet_name="Shift_Pairs",
            index=False,
        )

        linkage.to_excel(
            writer,
            sheet_name="Shift_Performance_Link",
            index=False,
        )

        shift_decomposition.to_excel(
            writer,
            sheet_name="Shift_Decomposition",
            index=False,
        )

        burden_model.to_excel(
            writer,
            sheet_name="Testing_Burden_Model",
            index=False,
        )

        burden_summary.to_excel(
            writer,
            sheet_name="Testing_Burden_Summary",
            index=False,
        )

        model_stability.to_excel(
            writer,
            sheet_name="Model_Stability",
            index=False,
        )

        ledger.to_excel(
            writer,
            sheet_name="Decision_Ledger",
            index=False,
        )


# =============================================================================
# 26. MAIN
# =============================================================================

def main():

    warnings.filterwarnings(
        "ignore"
    )

    print("\n")
    print(
        "=" * 105
    )

    print(
        "STAGE 07 — CORRECTED CANONICAL RUN"
    )

    print(
        "Temporal degradation | recalibration | retraining | "
        "tier contribution | shift decomposition"
    )

    print(
        "=" * 105
    )

    # -------------------------------------------------------------------------
    # Backup unsuccessful previous Stage 07
    # -------------------------------------------------------------------------

    backup_previous_stage07()

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    (
        metrics,
        bootstrap,
        burden,
        tier,
        iron,
        stage04,
    ) = load_inputs()

    # -------------------------------------------------------------------------
    # Temporal profile
    # -------------------------------------------------------------------------

    temporal = (
        build_temporal_profile(
            metrics
        )
    )

    temporal_association = (
        build_temporal_distance_association(
            temporal
        )
    )

    # -------------------------------------------------------------------------
    # Recalibration
    # -------------------------------------------------------------------------

    (
        recalibration_effect,
        recalibration_summary,
    ) = build_recalibration_analysis(
        metrics
    )

    # -------------------------------------------------------------------------
    # Retraining
    # -------------------------------------------------------------------------

    (
        retraining_effect,
        retraining_summary,
    ) = build_retraining_analysis(
        temporal
    )

    # -------------------------------------------------------------------------
    # Tier
    # -------------------------------------------------------------------------

    (
        tier_model,
        tier_summary,
    ) = build_tier_analysis(
        tier
    )

    # -------------------------------------------------------------------------
    # Iron natural experiment
    # -------------------------------------------------------------------------

    (
        iron_model,
        iron_summary,
    ) = build_iron_natural_experiment(
        iron
    )

    # -------------------------------------------------------------------------
    # Stage 04 pair summary
    # -------------------------------------------------------------------------

    shift_pairs = (
        build_shift_pair_summary(
            stage04
        )
    )

    # -------------------------------------------------------------------------
    # Stage 04 ↔ Stage 06 linkage
    # -------------------------------------------------------------------------

    linkage = (
        build_shift_performance_linkage(
            temporal=
                temporal,
            shift_pairs=
                shift_pairs,
        )
    )

    # -------------------------------------------------------------------------
    # Descriptive shift decomposition
    # -------------------------------------------------------------------------

    shift_decomposition = (
        build_shift_decomposition(
            linkage
        )
    )

    # -------------------------------------------------------------------------
    # Testing burden
    # -------------------------------------------------------------------------

    (
        burden_model,
        burden_summary,
    ) = build_testing_burden(
        burden
    )

    # -------------------------------------------------------------------------
    # Stability
    # -------------------------------------------------------------------------

    model_stability = (
        build_model_stability(
            temporal
        )
    )

    # -------------------------------------------------------------------------
    # Key findings
    # -------------------------------------------------------------------------

    key_findings = (
        build_key_findings(
            temporal=
                temporal,
            tier_summary=
                tier_summary,
            recalibration_summary=
                recalibration_summary,
            retraining_summary=
                retraining_summary,
            burden_summary=
                burden_summary,
        )
    )

    # -------------------------------------------------------------------------
    # Integrity
    # -------------------------------------------------------------------------

    integrity = (
        build_integrity_check(
            temporal=
                temporal,
            temporal_association=
                temporal_association,
            shift_pairs=
                shift_pairs,
            linkage=
                linkage,
            shift_decomposition=
                shift_decomposition,
            recalibration_effect=
                recalibration_effect,
            retraining_effect=
                retraining_effect,
        )
    )

    # -------------------------------------------------------------------------
    # Decision ledger
    # -------------------------------------------------------------------------

    ledger = (
        build_decision_ledger()
    )

    # -------------------------------------------------------------------------
    # Workbook
    # -------------------------------------------------------------------------

    write_workbook(
        key_findings=
            key_findings,
        temporal=
            temporal,
        temporal_association=
            temporal_association,
        recalibration_effect=
            recalibration_effect,
        recalibration_summary=
            recalibration_summary,
        retraining_effect=
            retraining_effect,
        retraining_summary=
            retraining_summary,
        tier_model=
            tier_model,
        tier_summary=
            tier_summary,
        iron_model=
            iron_model,
        iron_summary=
            iron_summary,
        shift_pairs=
            shift_pairs,
        linkage=
            linkage,
        shift_decomposition=
            shift_decomposition,
        burden_model=
            burden_model,
        burden_summary=
            burden_summary,
        model_stability=
            model_stability,
        integrity=
            integrity,
        ledger=
            ledger,
    )

    # =========================================================================
    # CONSOLE REPORT
    # =========================================================================

    print("\n")
    print(
        "-" * 105
    )

    print(
        "PRIMARY SIGMOID PERFORMANCE"
    )

    print(
        "-" * 105
    )

    primary_summary = (
        temporal.groupby(
            [
                "study",
                "feature_tier",
            ]
        )[
            PRIMARY_METRICS
        ]
        .mean()
        .round(
            4
        )
    )

    print(
        primary_summary.to_string()
    )

    print("\n")
    print(
        "-" * 105
    )

    print(
        "TEMPORAL DISTANCE"
    )

    print(
        "-" * 105
    )

    print(
        f"Temporal profile rows: "
        f"{len(temporal):,}"
    )

    print(
        f"Missing temporal distances: "
        f"{temporal['absolute_temporal_distance_years'].isna().sum():,}"
    )

    print(
        f"Temporal association rows: "
        f"{len(temporal_association):,}"
    )

    print("\n")
    print(
        "-" * 105
    )

    print(
        "SHIFT LINKAGE"
    )

    print(
        "-" * 105
    )

    print(
        f"Shift pair rows: "
        f"{len(shift_pairs):,}"
    )

    print(
        f"Performance rows linked to Stage 04 shift: "
        f"{int(linkage['shift_linked'].sum()):,}"
        f" / "
        f"{len(linkage):,}"
    )

    print(
        f"Shift decomposition rows: "
        f"{len(shift_decomposition):,}"
    )

    print("\n")
    print(
        "-" * 105
    )

    print(
        "TIER CONTRIBUTION"
    )

    print(
        "-" * 105
    )

    print(
        tier_summary
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print("\n")
    print(
        "-" * 105
    )

    print(
        "RECALIBRATION"
    )

    print(
        "-" * 105
    )

    print(
        recalibration_summary
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print("\n")
    print(
        "-" * 105
    )

    print(
        "RETRAINING"
    )

    print(
        "-" * 105
    )

    print(
        retraining_summary
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print("\n")
    print(
        "-" * 105
    )

    print(
        "INTEGRITY CHECK"
    )

    print(
        "-" * 105
    )

    print(
        integrity.to_string(
            index=False
        )
    )

    all_pass = bool(
        integrity[
            "pass"
        ]
        .all()
    )

    print("\n")
    print(
        "=" * 105
    )

    if all_pass:

        print(
            "STAGE 07 EXECUTION: PASS"
        )

        print(
            "All required analyses were generated."
        )

        print(
            "Stage 07 is ready for scientific interpretation "
            "before final freezing."
        )

    else:

        print(
            "STAGE 07 EXECUTION: FAIL"
        )

        print(
            "Do not freeze Stage 07."
        )

        print(
            "Review the failed integrity checks."
        )

    print(
        "=" * 105
    )

    print(
        f"\nWorkbook:\n"
        f"  {WORKBOOK_PATH}"
    )


# =============================================================================
# 27. ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()