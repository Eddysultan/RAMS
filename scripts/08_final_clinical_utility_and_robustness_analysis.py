"""
===============================================================================
STAGE 08 — FINAL CANONICAL VERSION
Clinical Utility, Survey-Weighted Robustness, Subgroup Transportability,
Structural Sensitivity, and Final Study Robustness
===============================================================================

===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import warnings

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import norm

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "processed"
)

STAGE03_DIR = (
    PROCESSED_ROOT
    / "05_analytic_cohorts"
)

STAGE06_DIR = (
    PROCESSED_ROOT
    / "08_model_results"
)

STAGE07_DIR = (
    PROCESSED_ROOT
    / "09_temporal_interpretation"
)

OUTPUT_DIR = (
    PROCESSED_ROOT
    / "10_clinical_utility"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. STAGE 03 COHORT PATHS
# =============================================================================

COHORT_PATHS = {

    "vitamin_d":
        STAGE03_DIR
        / "vitamin_d"
        / "NHANES_vitaminD_primary_multicycle.csv",

    "iron_observed":
        STAGE03_DIR
        / "iron"
        / "observed"
        / "NHANES_iron_observed_multicycle.csv",

    "iron_matched":
        STAGE03_DIR
        / "iron"
        / "matched"
        / "NHANES_iron_matched_multicycle.csv",
}


OUTCOME_COLUMNS = {

    "vitamin_d":
        "outcome_vitd_lt50",

    "iron_observed":
        "outcome_iron_lt15",

    "iron_matched":
        "outcome_iron_lt15",
}


# =============================================================================
# 3. STAGE 06 INPUTS
# =============================================================================

EXPERIMENT_METRICS_PATH = (
    STAGE06_DIR
    / "06_experiment_metrics.csv"
)

TESTING_BURDEN_PATH = (
    STAGE06_DIR
    / "06_testing_burden.csv"
)

DECISION_CURVE_PATH = (
    STAGE06_DIR
    / "06_decision_curve.csv"
)

TIER_COMPARISON_PATH = (
    STAGE06_DIR
    / "06_tier1_vs_tier2_comparison.csv"
)

EXECUTION_STATUS_PATH = (
    STAGE06_DIR
    / "06_execution_status.csv"
)

SEED_TEST_DIR = (
    STAGE06_DIR
    / "checkpoints"
    / "seed_test_predictions"
)

SEED_OOF_DIR = (
    STAGE06_DIR
    / "checkpoints"
    / "seed_oof_predictions"
)


# =============================================================================
# 4. STAGE 07 INPUTS
# =============================================================================

TEMPORAL_PROFILE_PATH = (
    STAGE07_DIR
    / "07_temporal_transport_profile.csv"
)

STAGE07_INTEGRITY_PATH = (
    STAGE07_DIR
    / "07_integrity_check.csv"
)


# =============================================================================
# 5. STAGE 08 OUTPUTS
# =============================================================================

DCA_EXPERIMENT_PATH = (
    OUTPUT_DIR
    / "08_dca_experiment_summary.csv"
)

DCA_STRATEGY_PATH = (
    OUTPUT_DIR
    / "08_dca_strategy_summary.csv"
)

DCA_THRESHOLD_PATH = (
    OUTPUT_DIR
    / "08_dca_threshold_advantage.csv"
)

OPERATING_EXPERIMENT_PATH = (
    OUTPUT_DIR
    / "08_operating_point_experiment_summary.csv"
)

OPERATING_SUMMARY_PATH = (
    OUTPUT_DIR
    / "08_operating_point_summary.csv"
)

OPERATING_TARGET_PATH = (
    OUTPUT_DIR
    / "08_operating_target_robustness.csv"
)

MODEL_ROBUSTNESS_PATH = (
    OUTPUT_DIR
    / "08_model_family_tier_robustness.csv"
)

MODEL_CONSENSUS_PATH = (
    OUTPUT_DIR
    / "08_model_family_consensus.csv"
)

LINKAGE_AUDIT_PATH = (
    OUTPUT_DIR
    / "08_prediction_linkage_audit.csv"
)

THRESHOLD_AUDIT_PATH = (
    OUTPUT_DIR
    / "08_operating_threshold_audit.csv"
)

WEIGHTED_PERFORMANCE_PATH = (
    OUTPUT_DIR
    / "08_survey_weighted_performance.csv"
)

WEIGHTED_SUMMARY_PATH = (
    OUTPUT_DIR
    / "08_weighted_vs_unweighted_summary.csv"
)

SUBGROUP_PERFORMANCE_PATH = (
    OUTPUT_DIR
    / "08_subgroup_performance.csv"
)

SUBGROUP_DISPARITY_PATH = (
    OUTPUT_DIR
    / "08_subgroup_disparity_summary.csv"
)

AGE_SENSITIVITY_PATH = (
    OUTPUT_DIR
    / "08_age20_59_sensitivity.csv"
)

AGE_SENSITIVITY_SUMMARY_PATH = (
    OUTPUT_DIR
    / "08_age20_59_sensitivity_summary.csv"
)

SURVEY_DESCRIPTIVE_PATH = (
    OUTPUT_DIR
    / "08_survey_design_descriptive.csv"
)

TEMPORAL_UTILITY_PATH = (
    OUTPUT_DIR
    / "08_temporal_design_utility_summary.csv"
)

FINAL_ROBUSTNESS_PATH = (
    OUTPUT_DIR
    / "08_final_robustness_audit.csv"
)

KEY_FINDINGS_PATH = (
    OUTPUT_DIR
    / "08_key_clinical_findings.csv"
)

INTEGRITY_PATH = (
    OUTPUT_DIR
    / "08_integrity_check.csv"
)

DECISION_LEDGER_PATH = (
    OUTPUT_DIR
    / "decision_ledger_stage08.csv"
)

WORKBOOK_PATH = (
    OUTPUT_DIR
    / "08_stage08_summary.xlsx"
)


# =============================================================================
# 6. FROZEN DESIGN CONSTANTS
# =============================================================================

PRIMARY_CALIBRATION = "sigmoid"

EXPECTED_EXPERIMENTS = 336

LOCKED_SEEDS = [
    1107,
    2203,
    3301,
    4409,
    5501,
    6607,
    7703,
    8807,
    9901,
    10103,
]

SENSITIVITY_TARGETS = [
    0.90,
    0.95,
    0.975,
]

PRIMARY_SUBGROUP_TARGET = 0.95

MIN_SUBGROUP_N = 100
MIN_SUBGROUP_CASES = 20
MIN_SUBGROUP_NONCASES = 20

ECE_BINS = 10

PRIMARY_METRICS = [
    "AUROC",
    "AUPRC",
    "Brier",
    "ECE",
]


# =============================================================================
# 7. BACK UP PREVIOUS STAGE 08 ATTEMPT
# =============================================================================

def backup_previous_stage08():

    files = []

    for pattern in [
        "08_*",
        "decision_ledger_stage08.csv",
    ]:

        files.extend(
            OUTPUT_DIR.glob(
                pattern
            )
        )

    files = [
        p
        for p in files
        if p.is_file()
    ]

    if not files:
        return None

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_dir = (
        OUTPUT_DIR
        / "previous_stage08_attempts"
        / f"attempt_backup_{timestamp}"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for source in files:

        shutil.copy2(
            source,
            backup_dir
            / source.name,
        )

    print(
        "\nPrevious Stage-08 outputs backed up to:"
    )

    print(
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
            f"Required file not found:\n{path}"
        )


def require_directory(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Required directory not found:\n{path}"
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


def find_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:

    lookup = {
        str(column)
        .strip()
        .lower():
            column
        for column in df.columns
    }

    for candidate in candidates:

        key = (
            candidate
            .strip()
            .lower()
        )

        if key in lookup:

            return lookup[
                key
            ]

    return None


def safe_mean(
    series: pd.Series,
):

    return float(
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .mean()
    )


def safe_median(
    series: pd.Series,
):

    return float(
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .median()
    )


def safe_sd(
    series: pd.Series,
):

    return float(
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .std(
            ddof=1
        )
    )


def sha256_file(
    path: Path,
) -> str:

    hasher = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        for chunk in iter(
            lambda:
                file.read(
                    1024 * 1024
                ),
            b"",
        ):

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


# =============================================================================
# 9. VERIFY FROZEN PREVIOUS STAGES
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

    success = (
        latest[
            "status"
        ]
        == "SUCCESS"
    )

    if len(
        latest
    ) != EXPECTED_EXPERIMENTS:

        raise RuntimeError(
            f"Expected {EXPECTED_EXPERIMENTS} Stage-06 experiments; "
            f"found {len(latest)}."
        )

    if int(
        success.sum()
    ) != EXPECTED_EXPERIMENTS:

        raise RuntimeError(
            "Stage 06 does not have 336/336 successful experiments."
        )


def validate_stage07():

    integrity = read_csv_required(
        STAGE07_INTEGRITY_PATH
    )

    pass_col = find_column(
        integrity,
        [
            "pass",
        ]
    )

    if pass_col is None:

        raise RuntimeError(
            "Stage-07 integrity file has no pass column."
        )

    passed = (
        integrity[
            pass_col
        ]
        .astype(
            str
        )
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )

    if not passed.all():

        raise RuntimeError(
            "Stage 07 has failed integrity checks."
        )


# =============================================================================
# 10. STAGE 06 IMMUTABILITY SNAPSHOT
# =============================================================================

def stage06_hash_snapshot():

    files = [
        EXPERIMENT_METRICS_PATH,
        TESTING_BURDEN_PATH,
        DECISION_CURVE_PATH,
        TIER_COMPARISON_PATH,
        EXECUTION_STATUS_PATH,
    ]

    result = {}

    for path in files:

        require_file(
            path
        )

        result[
            str(path)
        ] = sha256_file(
            path
        )

    return result


# =============================================================================
# 11. LOAD INPUT TABLES
# =============================================================================

def load_inputs():

    validate_stage06()
    validate_stage07()

    require_directory(
        SEED_TEST_DIR
    )

    require_directory(
        SEED_OOF_DIR
    )

    for path in COHORT_PATHS.values():

        require_file(
            path
        )

    metrics = read_csv_required(
        EXPERIMENT_METRICS_PATH
    )

    burden = read_csv_required(
        TESTING_BURDEN_PATH
    )

    dca = read_csv_required(
        DECISION_CURVE_PATH
    )

    tier = read_csv_required(
        TIER_COMPARISON_PATH
    )

    temporal = read_csv_required(
        TEMPORAL_PROFILE_PATH
    )

    cohorts = {
        study:
            read_csv_required(
                path
            )
        for study, path
        in COHORT_PATHS.items()
    }

    return {
        "metrics":
            metrics,

        "burden":
            burden,

        "dca":
            dca,

        "tier":
            tier,

        "temporal":
            temporal,

        "cohorts":
            cohorts,
    }


# =============================================================================
# 12. METRIC HELPERS
# =============================================================================

def weighted_mean(
    values,
    weights,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    valid = (
        np.isfinite(
            values
        )
        &
        np.isfinite(
            weights
        )
        &
        (
            weights
            > 0
        )
    )

    if not np.any(
        valid
    ):

        return np.nan

    return np.average(
        values[
            valid
        ],
        weights=weights[
            valid
        ],
    )


def ece_score(
    y,
    p,
    weights=None,
    n_bins=ECE_BINS,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    if weights is None:

        weights = np.ones(
            len(
                y
            ),
            dtype=float,
        )

    else:

        weights = np.asarray(
            weights,
            dtype=float,
        )

    valid = (
        np.isfinite(
            y
        )
        &
        np.isfinite(
            p
        )
        &
        np.isfinite(
            weights
        )
        &
        (
            weights
            > 0
        )
    )

    y = y[
        valid
    ]

    p = p[
        valid
    ]

    weights = weights[
        valid
    ]

    if len(
        y
    ) == 0:

        return np.nan

    bins = np.linspace(
        0,
        1,
        n_bins + 1,
    )

    index = np.digitize(
        p,
        bins[
            1:-1
        ],
        right=False,
    )

    total_weight = weights.sum()

    ece = 0.0

    for bin_id in range(
        n_bins
    ):

        mask = (
            index
            == bin_id
        )

        if not np.any(
            mask
        ):

            continue

        bin_weight = (
            weights[
                mask
            ]
            .sum()
        )

        observed = np.average(
            y[
                mask
            ],
            weights=weights[
                mask
            ],
        )

        predicted = np.average(
            p[
                mask
            ],
            weights=weights[
                mask
            ],
        )

        ece += (
            bin_weight
            /
            total_weight
        ) * abs(
            observed
            -
            predicted
        )

    return float(
        ece
    )


def calibration_intercept_slope(
    y,
    p,
    weights=None,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    if weights is None:

        weights = np.ones(
            len(
                y
            ),
            dtype=float,
        )

    else:

        weights = np.asarray(
            weights,
            dtype=float,
        )

    valid = (
        np.isfinite(
            y
        )
        &
        np.isfinite(
            p
        )
        &
        np.isfinite(
            weights
        )
        &
        (
            weights
            > 0
        )
    )

    y = y[
        valid
    ]

    p = p[
        valid
    ]

    weights = weights[
        valid
    ]

    if (
        len(
            np.unique(
                y
            )
        )
        < 2
    ):

        return (
            np.nan,
            np.nan,
        )

    eps = 1e-6

    p = np.clip(
        p,
        eps,
        1 - eps,
    )

    logit_p = np.log(
        p
        /
        (
            1 - p
        )
    )

    weights = (
        weights
        /
        np.mean(
            weights
        )
    )

    def objective(
        params,
    ):

        intercept = params[
            0
        ]

        slope = params[
            1
        ]

        linear = (
            intercept
            +
            slope
            * logit_p
        )

        probability = (
            1
            /
            (
                1
                +
                np.exp(
                    -np.clip(
                        linear,
                        -35,
                        35,
                    )
                )
            )
        )

        probability = np.clip(
            probability,
            eps,
            1 - eps,
        )

        loglik = (
            y
            * np.log(
                probability
            )
            +
            (
                1 - y
            )
            * np.log(
                1 - probability
            )
        )

        return -np.sum(
            weights
            * loglik
        )

    result = minimize(
        objective,
        x0=np.array(
            [
                0.0,
                1.0,
            ]
        ),
        method="BFGS",
    )

    if not result.success:

        return (
            np.nan,
            np.nan,
        )

    return (
        float(
            result.x[
                0
            ]
        ),
        float(
            result.x[
                1
            ]
        ),
    )


def evaluate_binary_predictions(
    y,
    p,
    weights=None,
    threshold=None,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    if weights is None:

        weights = np.ones(
            len(
                y
            ),
            dtype=float,
        )

    else:

        weights = np.asarray(
            weights,
            dtype=float,
        )

    valid = (
        np.isfinite(
            p
        )
        &
        np.isfinite(
            weights
        )
        &
        (
            weights
            > 0
        )
    )

    y = y[
        valid
    ]

    p = p[
        valid
    ]

    weights = weights[
        valid
    ]

    result = {
        "n":
            len(
                y
            ),

        "cases":
            int(
                np.sum(
                    y == 1
                )
            ),

        "noncases":
            int(
                np.sum(
                    y == 0
                )
            ),
    }

    if len(
        np.unique(
            y
        )
    ) >= 2:

        result[
            "AUROC"
        ] = roc_auc_score(
            y,
            p,
            sample_weight=weights,
        )

        result[
            "AUPRC"
        ] = average_precision_score(
            y,
            p,
            sample_weight=weights,
        )

    else:

        result[
            "AUROC"
        ] = np.nan

        result[
            "AUPRC"
        ] = np.nan

    result[
        "Brier"
    ] = np.average(
        (
            y - p
        ) ** 2,
        weights=weights,
    )

    result[
        "ECE"
    ] = ece_score(
        y,
        p,
        weights=weights,
    )

    intercept, slope = (
        calibration_intercept_slope(
            y,
            p,
            weights=weights,
        )
    )

    result[
        "CalibrationIntercept"
    ] = intercept

    result[
        "CalibrationSlope"
    ] = slope

    result[
        "WeightedPrevalence"
    ] = np.average(
        y,
        weights=weights,
    )

    if threshold is not None:

        predicted = (
            p
            >= threshold
        ).astype(
            int
        )

        tp = weights[
            (
                y == 1
            )
            &
            (
                predicted == 1
            )
        ].sum()

        fn = weights[
            (
                y == 1
            )
            &
            (
                predicted == 0
            )
        ].sum()

        tn = weights[
            (
                y == 0
            )
            &
            (
                predicted == 0
            )
        ].sum()

        fp = weights[
            (
                y == 0
            )
            &
            (
                predicted == 1
            )
        ].sum()

        result[
            "Sensitivity"
        ] = (
            tp
            /
            (
                tp + fn
            )
            if (
                tp + fn
            )
            > 0
            else np.nan
        )

        result[
            "FNR"
        ] = (
            fn
            /
            (
                tp + fn
            )
            if (
                tp + fn
            )
            > 0
            else np.nan
        )

        result[
            "Specificity"
        ] = (
            tn
            /
            (
                tn + fp
            )
            if (
                tn + fp
            )
            > 0
            else np.nan
        )

        result[
            "PPV"
        ] = (
            tp
            /
            (
                tp + fp
            )
            if (
                tp + fp
            )
            > 0
            else np.nan
        )

        result[
            "NPV"
        ] = (
            tn
            /
            (
                tn + fn
            )
            if (
                tn + fn
            )
            > 0
            else np.nan
        )

        result[
            "ReferralRate"
        ] = np.average(
            predicted,
            weights=weights,
        )

        result[
            "TestsAvoided"
        ] = (
            1
            -
            result[
                "ReferralRate"
            ]
        )

    return result


# =============================================================================
# 13. LOAD AND AVERAGE STAGE-06 TEST PREDICTIONS
# =============================================================================

def load_seed_prediction_mean(
    experiment_id: str,
    directory: Path,
):

    reference_seqn = None
    reference_y = None

    probabilities = []

    files_found = []

    for seed in LOCKED_SEEDS:

        path = (
            directory
            / (
                f"{experiment_id}"
                f"__seed_{seed}.npz"
            )
        )

        require_file(
            path
        )

        data = np.load(
            path,
            allow_pickle=False,
        )

        required_keys = {
            "y",
            "sigmoid",
        }

        missing = (
            required_keys
            -
            set(
                data.files
            )
        )

        if missing:

            raise RuntimeError(
                f"{path.name} is missing NPZ keys: "
                f"{sorted(missing)}"
            )

        seqn = (
            data[
                "seqn"
            ]
            if "seqn"
            in data.files
            else None
        )

        y = np.asarray(
            data[
                "y"
            ],
            dtype=int,
        )

        p = np.asarray(
            data[
                "sigmoid"
            ],
            dtype=float,
        )

        if len(
            y
        ) != len(
            p
        ):

            raise RuntimeError(
                f"Length mismatch in {path}"
            )

        if reference_y is None:

            reference_y = y.copy()

            if seqn is not None:

                reference_seqn = (
                    np.asarray(
                        seqn
                    )
                    .copy()
                )

        else:

            if not np.array_equal(
                reference_y,
                y,
            ):

                raise RuntimeError(
                    f"Outcome ordering differs across seeds "
                    f"for {experiment_id}."
                )

            if (
                reference_seqn
                is not None
                and seqn
                is not None
                and not np.array_equal(
                    reference_seqn,
                    seqn,
                )
            ):

                raise RuntimeError(
                    f"SEQN ordering differs across seeds "
                    f"for {experiment_id}."
                )

        probabilities.append(
            p
        )

        files_found.append(
            path.name
        )

    p_mean = np.mean(
        np.vstack(
            probabilities
        ),
        axis=0,
    )

    return {
        "seqn":
            reference_seqn,

        "y":
            reference_y,

        "p_sigmoid":
            p_mean,

        "n_seeds":
            len(
                probabilities
            ),

        "files":
            files_found,
    }


# =============================================================================
# 14. COHORT SCHEMA DETECTION
# =============================================================================

def detect_cohort_schema(
    df: pd.DataFrame,
    study: str,
):

    seqn = find_column(
        df,
        [
            "SEQN",
            "seqn",
        ]
    )

    cycle = find_column(
        df,
        [
            "cycle",
            "nhanes_cycle",
            "survey_cycle",
        ]
    )

    weight = find_column(
        df,
        [
            "WTMEC2YR",
            "wtmec2yr",
            "mec_weight",
            "survey_weight",
        ]
    )

    strata = find_column(
        df,
        [
            "SDMVSTRA",
            "sdmvstra",
        ]
    )

    psu = find_column(
        df,
        [
            "SDMVPSU",
            "sdmvpsu",
        ]
    )

    age = find_column(
        df,
        [
            "age_years",
            "RIDAGEYR",
            "ridageyr",
        ]
    )

    sex = find_column(
        df,
        [
            "sex",
            "RIAGENDR",
            "riagendr",
        ]
    )

    race = find_column(
        df,
        [
            "race_ethnicity_5cat",
            "race_ethnicity",
            "RIDRETH3",
            "ridreth3",
        ]
    )

    poverty = find_column(
        df,
        [
            "poverty_income_ratio",
            "INDFMPIR",
            "indfmpir",
        ]
    )

    education = find_column(
        df,
        [
            "education_adult_5cat",
            "DMDEDUC2",
            "dmdeduc2",
        ]
    )

    outcome = find_column(
        df,
        [
            OUTCOME_COLUMNS[
                study
            ],
        ]
    )

    required = {
        "SEQN":
            seqn,

        "cycle":
            cycle,

        "weight":
            weight,

        "age":
            age,

        "outcome":
            outcome,
    }

    missing = [
        name
        for name, column
        in required.items()
        if column is None
    ]

    if missing:

        raise RuntimeError(
            f"{study} cohort is missing required columns: "
            f"{missing}"
        )

    return {
        "seqn":
            seqn,

        "cycle":
            cycle,

        "weight":
            weight,

        "strata":
            strata,

        "psu":
            psu,

        "age":
            age,

        "sex":
            sex,

        "race":
            race,

        "poverty":
            poverty,

        "education":
            education,

        "outcome":
            outcome,
    }


# =============================================================================
# 15. LINK ONE EXPERIMENT TO STAGE-03 PARTICIPANTS
# =============================================================================

def link_experiment_predictions(
    experiment_row: pd.Series,
    cohorts: Dict[str, pd.DataFrame],
    schemas: Dict[str, Dict],
):

    experiment_id = (
        experiment_row[
            "experiment_id"
        ]
    )

    study = (
        experiment_row[
            "study"
        ]
    )

    test_cycle = (
        experiment_row[
            "test_cycle"
        ]
    )

    prediction = (
        load_seed_prediction_mean(
            experiment_id=
                experiment_id,
            directory=
                SEED_TEST_DIR,
        )
    )

    seqn = prediction[
        "seqn"
    ]

    if seqn is None:

        raise RuntimeError(
            f"{experiment_id}: test checkpoint "
            f"does not contain SEQN."
        )

    if len(
        np.unique(
            seqn
        )
    ) != len(
        seqn
    ):

        raise RuntimeError(
            f"{experiment_id}: duplicate SEQN in predictions."
        )

    cohort = cohorts[
        study
    ]

    schema = schemas[
        study
    ]

    target = (
        cohort[
            cohort[
                schema[
                    "cycle"
                ]
            ].astype(
                str
            )
            == str(
                test_cycle
            )
        ]
        .copy()
    )

    if target.empty:

        raise RuntimeError(
            f"{experiment_id}: no Stage-03 cohort rows found "
            f"for {study}, {test_cycle}."
        )

    if target[
        schema[
            "seqn"
        ]
    ].duplicated().any():

        raise RuntimeError(
            f"{experiment_id}: duplicate SEQN in "
            f"Stage-03 test cohort."
        )

    prediction_df = pd.DataFrame(
        {
            "SEQN":
                seqn,

            "prediction_y":
                prediction[
                    "y"
                ],

            "p_sigmoid":
                prediction[
                    "p_sigmoid"
                ],
        }
    )

    cohort_columns = [
        schema[
            "seqn"
        ],
        schema[
            "outcome"
        ],
        schema[
            "weight"
        ],
        schema[
            "age"
        ],
    ]

    for key in [
        "strata",
        "psu",
        "sex",
        "race",
        "poverty",
        "education",
    ]:

        column = schema[
            key
        ]

        if (
            column is not None
            and column not in cohort_columns
        ):

            cohort_columns.append(
                column
            )

    target = target[
        cohort_columns
    ].copy()

    target = target.rename(
        columns={
            schema[
                "seqn"
            ]:
                "SEQN",

            schema[
                "outcome"
            ]:
                "cohort_y",

            schema[
                "weight"
            ]:
                "survey_weight",

            schema[
                "age"
            ]:
                "age_years",
        }
    )

    optional_rename = {}

    if schema[
        "strata"
    ] is not None:

        optional_rename[
            schema[
                "strata"
            ]
        ] = "SDMVSTRA"

    if schema[
        "psu"
    ] is not None:

        optional_rename[
            schema[
                "psu"
            ]
        ] = "SDMVPSU"

    if schema[
        "sex"
    ] is not None:

        optional_rename[
            schema[
                "sex"
            ]
        ] = "sex"

    if schema[
        "race"
    ] is not None:

        optional_rename[
            schema[
                "race"
            ]
        ] = "race_ethnicity"

    if schema[
        "poverty"
    ] is not None:

        optional_rename[
            schema[
                "poverty"
            ]
        ] = "poverty_income_ratio"

    if schema[
        "education"
    ] is not None:

        optional_rename[
            schema[
                "education"
            ]
        ] = "education"

    target = target.rename(
        columns=
            optional_rename
    )

    linked = prediction_df.merge(
        target,
        on="SEQN",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int(
        (
            linked[
                "_merge"
            ]
            != "both"
        )
        .sum()
    )

    if unmatched > 0:

        raise RuntimeError(
            f"{experiment_id}: {unmatched} prediction participants "
            f"did not link to the frozen Stage-03 cohort."
        )

    linked = linked.drop(
        columns=[
            "_merge"
        ]
    )

    cohort_y = pd.to_numeric(
        linked[
            "cohort_y"
        ],
        errors="coerce",
    )

    if cohort_y.isna().any():

        raise RuntimeError(
            f"{experiment_id}: linked Stage-03 outcome contains missing values."
        )

    outcome_agreement = np.mean(
        linked[
            "prediction_y"
        ].astype(
            int
        )
        .to_numpy()
        ==
        cohort_y.astype(
            int
        )
        .to_numpy()
    )

    if outcome_agreement != 1.0:

        raise RuntimeError(
            f"{experiment_id}: Stage-06 prediction outcome and "
            f"Stage-03 cohort outcome do not agree 100%."
        )

    linked[
        "y"
    ] = (
        linked[
            "prediction_y"
        ]
        .astype(
            int
        )
    )

    linked[
        "study"
    ] = study

    linked[
        "test_cycle"
    ] = test_cycle

    linked[
        "experiment_id"
    ] = experiment_id

    linked[
        "feature_tier"
    ] = experiment_row[
        "feature_tier"
    ]

    linked[
        "model"
    ] = experiment_row[
        "model"
    ]

    linked[
        "temporal_design"
    ] = experiment_row[
        "temporal_design"
    ]

    audit = {
        "experiment_id":
            experiment_id,

        "study":
            study,

        "test_cycle":
            test_cycle,

        "feature_tier":
            experiment_row[
                "feature_tier"
            ],

        "model":
            experiment_row[
                "model"
            ],

        "n_prediction_participants":
            len(
                prediction_df
            ),

        "n_stage03_test_participants":
            len(
                target
            ),

        "n_linked":
            len(
                linked
            ),

        "n_unmatched":
            unmatched,

        "outcome_agreement":
            outcome_agreement,

        "n_seeds":
            prediction[
                "n_seeds"
            ],

        "n_unique_seqn":
            linked[
                "SEQN"
            ].nunique(),

        "n_positive_survey_weights":
            int(
                (
                    pd.to_numeric(
                        linked[
                            "survey_weight"
                        ],
                        errors="coerce",
                    )
                    > 0
                )
                .sum()
            ),
    }

    return (
        linked,
        audit,
    )


# =============================================================================
# 16. PRIMARY EXPERIMENT TABLE
# =============================================================================

def get_primary_experiments(
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

    if len(
        primary
    ) != EXPECTED_EXPERIMENTS:

        raise RuntimeError(
            f"Expected {EXPECTED_EXPERIMENTS} sigmoid experiments; "
            f"found {len(primary)}."
        )

    if primary[
        "experiment_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate experiment IDs in primary Stage-06 metrics."
        )

    return primary


# =============================================================================
# 17. OPERATING THRESHOLD EXTRACTION
# =============================================================================

def choose_threshold_at_sensitivity(
    y,
    p,
    target,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    fpr, tpr, thresholds = roc_curve(
        y,
        p,
    )

    valid = (
        np.isfinite(
            thresholds
        )
        &
        (
            tpr
            >= target
        )
    )

    if not np.any(
        valid
    ):

        return float(
            np.min(
                p
            )
        )

    return float(
        np.max(
            thresholds[
                valid
            ]
        )
    )


def build_threshold_table(
    burden: pd.DataFrame,
    experiments: pd.DataFrame,
):

    burden_primary = burden.copy()

    if (
        "calibration"
        in burden_primary.columns
    ):

        burden_primary = (
            burden_primary[
                burden_primary[
                    "calibration"
                ]
                == PRIMARY_CALIBRATION
            ]
            .copy()
        )

    target_column = find_column(
        burden_primary,
        [
            "target_training_sensitivity",
            "target_sensitivity",
            "sensitivity_target",
        ]
    )

    threshold_column = find_column(
        burden_primary,
        [
            "threshold",
            "Threshold",
            "decision_threshold",
            "probability_threshold",
            "training_threshold",
            "oof_threshold",
            "threshold_from_oof",
        ]
    )

    rows = []

    for _, experiment in experiments.iterrows():

        experiment_id = (
            experiment[
                "experiment_id"
            ]
        )

        burden_rows = (
            burden_primary[
                burden_primary[
                    "experiment_id"
                ]
                == experiment_id
            ]
            .copy()
        )

        for target in SENSITIVITY_TARGETS:

            threshold = np.nan
            source = None

            if (
                target_column is not None
                and threshold_column is not None
                and not burden_rows.empty
            ):

                temp = (
                    burden_rows[
                        np.isclose(
                            pd.to_numeric(
                                burden_rows[
                                    target_column
                                ],
                                errors="coerce",
                            ),
                            target,
                            atol=1e-8,
                        )
                    ]
                )

                if not temp.empty:

                    candidate = (
                        pd.to_numeric(
                            temp[
                                threshold_column
                            ],
                            errors="coerce",
                        )
                        .dropna()
                    )

                    if not candidate.empty:

                        threshold = float(
                            candidate.iloc[
                                0
                            ]
                        )

                        source = (
                            "stage06_testing_burden"
                        )

            if not np.isfinite(
                threshold
            ):

                oof = (
                    load_seed_prediction_mean(
                        experiment_id=
                            experiment_id,
                        directory=
                            SEED_OOF_DIR,
                    )
                )

                threshold = (
                    choose_threshold_at_sensitivity(
                        y=
                            oof[
                                "y"
                            ],
                        p=
                            oof[
                                "p_sigmoid"
                            ],
                        target=
                            target,
                    )
                )

                source = (
                    "recomputed_from_mean_OOF"
                )

            rows.append(
                {
                    "experiment_id":
                        experiment_id,

                    "target_training_sensitivity":
                        target,

                    "threshold":
                        threshold,

                    "threshold_source":
                        source,
                }
            )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        THRESHOLD_AUDIT_PATH,
        index=False,
    )

    return result


# =============================================================================
# 18. DECISION-CURVE SYNTHESIS
# =============================================================================

def build_dca_analysis(
    dca: pd.DataFrame,
):

    df = dca.copy()

    if (
        "calibration"
        in df.columns
    ):

        df = (
            df[
                df[
                    "calibration"
                ]
                == PRIMARY_CALIBRATION
            ]
            .copy()
        )

    threshold_col = find_column(
        df,
        [
            "threshold",
            "threshold_probability",
            "probability_threshold",
            "pt",
        ]
    )

    model_nb_col = find_column(
        df,
        [
            "NetBenefit",
            "net_benefit",
            "model_net_benefit",
            "nb_model",
        ]
    )

    all_nb_col = find_column(
        df,
        [
            "NetBenefit_TestAll",
            "net_benefit_test_all",
            "test_all_net_benefit",
            "nb_all",
        ]
    )

    none_nb_col = find_column(
        df,
        [
            "NetBenefit_TestNone",
            "net_benefit_test_none",
            "test_none_net_benefit",
            "nb_none",
        ]
    )

    if (
        threshold_col is None
        or model_nb_col is None
        or all_nb_col is None
    ):

        raise RuntimeError(
            "Required decision-curve columns could not be identified."
        )

    if none_nb_col is None:

        df[
            "_test_none"
        ] = 0.0

        none_nb_col = (
            "_test_none"
        )

    df[
        "threshold_probability"
    ] = pd.to_numeric(
        df[
            threshold_col
        ],
        errors="coerce",
    )

    df[
        "model_net_benefit"
    ] = pd.to_numeric(
        df[
            model_nb_col
        ],
        errors="coerce",
    )

    df[
        "test_all_net_benefit"
    ] = pd.to_numeric(
        df[
            all_nb_col
        ],
        errors="coerce",
    )

    df[
        "test_none_net_benefit"
    ] = pd.to_numeric(
        df[
            none_nb_col
        ],
        errors="coerce",
    )

    df[
        "best_default_net_benefit"
    ] = df[
        [
            "test_all_net_benefit",
            "test_none_net_benefit",
        ]
    ].max(
        axis=1
    )

    df[
        "net_benefit_advantage"
    ] = (
        df[
            "model_net_benefit"
        ]
        -
        df[
            "best_default_net_benefit"
        ]
    )

    df[
        "model_best_strategy"
    ] = (
        df[
            "net_benefit_advantage"
        ]
        > 0
    )

    grouping = [
        "experiment_id",
        "study",
        "temporal_design",
        "train_cycles",
        "test_cycle",
        "feature_tier",
        "model",
    ]

    experiment_rows = []

    for keys, group in df.groupby(
        grouping,
        dropna=False,
    ):

        experiment_rows.append(
            {
                "experiment_id":
                    keys[0],

                "study":
                    keys[1],

                "temporal_design":
                    keys[2],

                "train_cycles":
                    keys[3],

                "test_cycle":
                    keys[4],

                "feature_tier":
                    keys[5],

                "model":
                    keys[6],

                "n_thresholds":
                    len(
                        group
                    ),

                "proportion_thresholds_model_best":
                    float(
                        group[
                            "model_best_strategy"
                        ]
                        .mean()
                    ),

                "mean_net_benefit_advantage":
                    safe_mean(
                        group[
                            "net_benefit_advantage"
                        ]
                    ),

                "max_net_benefit_advantage":
                    float(
                        group[
                            "net_benefit_advantage"
                        ]
                        .max()
                    ),
            }
        )

    experiment_summary = pd.DataFrame(
        experiment_rows
    )

    experiment_summary.to_csv(
        DCA_EXPERIMENT_PATH,
        index=False,
    )

    strategy = (
        experiment_summary.groupby(
            [
                "study",
                "feature_tier",
            ]
        )
        .agg(
            n_experiments=(
                "experiment_id",
                "nunique",
            ),

            mean_proportion_thresholds_model_best=(
                "proportion_thresholds_model_best",
                "mean",
            ),

            median_proportion_thresholds_model_best=(
                "proportion_thresholds_model_best",
                "median",
            ),

            mean_net_benefit_advantage=(
                "mean_net_benefit_advantage",
                "mean",
            ),

            median_max_net_benefit_advantage=(
                "max_net_benefit_advantage",
                "median",
            ),
        )
        .reset_index()
    )

    strategy.to_csv(
        DCA_STRATEGY_PATH,
        index=False,
    )

    threshold_summary = (
        df.groupby(
            [
                "study",
                "feature_tier",
                "threshold_probability",
            ]
        )
        .agg(
            n_experiments=(
                "experiment_id",
                "nunique",
            ),

            mean_model_net_benefit=(
                "model_net_benefit",
                "mean",
            ),

            mean_test_all_net_benefit=(
                "test_all_net_benefit",
                "mean",
            ),

            mean_test_none_net_benefit=(
                "test_none_net_benefit",
                "mean",
            ),

            mean_net_benefit_advantage=(
                "net_benefit_advantage",
                "mean",
            ),

            proportion_model_best=(
                "model_best_strategy",
                "mean",
            ),
        )
        .reset_index()
    )

    threshold_summary.to_csv(
        DCA_THRESHOLD_PATH,
        index=False,
    )

    return (
        experiment_summary,
        strategy,
        threshold_summary,
    )


# =============================================================================
# 19. OPERATING-POINT SYNTHESIS
# =============================================================================

def build_operating_analysis(
    burden: pd.DataFrame,
):

    df = burden.copy()

    if "calibration" in df.columns:

        df = (
            df[
                df[
                    "calibration"
                ]
                == PRIMARY_CALIBRATION
            ]
            .copy()
        )

    target_col = find_column(
        df,
        [
            "target_training_sensitivity",
            "target_sensitivity",
            "sensitivity_target",
        ]
    )

    if target_col is None:

        raise RuntimeError(
            "Operating target column could not be found."
        )

    df[
        "target_training_sensitivity"
    ] = pd.to_numeric(
        df[
            target_col
        ],
        errors="coerce",
    )

    df.to_csv(
        OPERATING_EXPERIMENT_PATH,
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
    ), group in df.groupby(
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
                len(
                    group
                ),
        }

        for measure in measures:

            if measure not in group.columns:
                continue

            row[
                f"mean_{measure}"
            ] = safe_mean(
                group[
                    measure
                ]
            )

            row[
                f"median_{measure}"
            ] = safe_median(
                group[
                    measure
                ]
            )

            row[
                f"sd_{measure}"
            ] = safe_sd(
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
        OPERATING_SUMMARY_PATH,
        index=False,
    )

    target_rows = []

    for (
        study,
        tier,
    ), group in summary.groupby(
        [
            "study",
            "feature_tier",
        ]
    ):

        group = group.sort_values(
            "target_training_sensitivity"
        )

        row = {
            "study":
                study,

            "feature_tier":
                tier,

            "n_targets":
                group[
                    "target_training_sensitivity"
                ]
                .nunique(),
        }

        if (
            "mean_TestsAvoided"
            in group.columns
        ):

            row[
                "tests_avoided_at_90"
            ] = group.loc[
                np.isclose(
                    group[
                        "target_training_sensitivity"
                    ],
                    0.90,
                ),
                "mean_TestsAvoided",
            ].mean()

            row[
                "tests_avoided_at_95"
            ] = group.loc[
                np.isclose(
                    group[
                        "target_training_sensitivity"
                    ],
                    0.95,
                ),
                "mean_TestsAvoided",
            ].mean()

            row[
                "tests_avoided_at_975"
            ] = group.loc[
                np.isclose(
                    group[
                        "target_training_sensitivity"
                    ],
                    0.975,
                ),
                "mean_TestsAvoided",
            ].mean()

        target_rows.append(
            row
        )

    target_summary = pd.DataFrame(
        target_rows
    )

    target_summary.to_csv(
        OPERATING_TARGET_PATH,
        index=False,
    )

    return (
        df,
        summary,
        target_summary,
    )


# =============================================================================
# 20. MODEL-FAMILY ROBUSTNESS
# =============================================================================

def build_model_family_robustness(
    tier: pd.DataFrame,
):

    delta_auroc = find_column(
        tier,
        [
            "delta_AUROC_T2_minus_T1",
        ]
    )

    delta_auprc = find_column(
        tier,
        [
            "delta_AUPRC_T2_minus_T1",
        ]
    )

    delta_brier = find_column(
        tier,
        [
            "delta_Brier_T2_minus_T1",
        ]
    )

    delta_ece = find_column(
        tier,
        [
            "delta_ECE_T2_minus_T1",
        ]
    )

    if delta_auroc is None:

        raise RuntimeError(
            "Tier comparison AUROC delta was not found."
        )

    rows = []

    for (
        study,
        model,
    ), group in tier.groupby(
        [
            "study",
            "model",
        ]
    ):

        row = {
            "study":
                study,

            "model":
                model,

            "n_temporal_comparisons":
                len(
                    group
                ),

            "mean_delta_AUROC":
                safe_mean(
                    group[
                        delta_auroc
                    ]
                ),

            "median_delta_AUROC":
                safe_median(
                    group[
                        delta_auroc
                    ]
                ),

            "proportion_T2_AUROC_better":
                float(
                    (
                        pd.to_numeric(
                            group[
                                delta_auroc
                            ],
                            errors="coerce",
                        )
                        > 0
                    )
                    .mean()
                ),
        }

        if delta_auprc:

            row[
                "mean_delta_AUPRC"
            ] = safe_mean(
                group[
                    delta_auprc
                ]
            )

            row[
                "proportion_T2_AUPRC_better"
            ] = float(
                (
                    pd.to_numeric(
                        group[
                            delta_auprc
                        ],
                        errors="coerce",
                    )
                    > 0
                )
                .mean()
            )

        if delta_brier:

            row[
                "mean_delta_Brier_T2_minus_T1"
            ] = safe_mean(
                group[
                    delta_brier
                ]
            )

        if delta_ece:

            row[
                "mean_delta_ECE_T2_minus_T1"
            ] = safe_mean(
                group[
                    delta_ece
                ]
            )

        rows.append(
            row
        )

    model_summary = pd.DataFrame(
        rows
    )

    model_summary.to_csv(
        MODEL_ROBUSTNESS_PATH,
        index=False,
    )

    consensus_rows = []

    for study, group in model_summary.groupby(
        "study"
    ):

        support = (
            group[
                "mean_delta_AUROC"
            ]
            > 0
        )

        consensus_rows.append(
            {
                "study":
                    study,

                "n_model_families":
                    len(
                        group
                    ),

                "n_models_with_positive_mean_delta_AUROC":
                    int(
                        support.sum()
                    ),

                "proportion_models_supporting_T2":
                    float(
                        support.mean()
                    ),

                "minimum_model_mean_delta_AUROC":
                    float(
                        group[
                            "mean_delta_AUROC"
                        ]
                        .min()
                    ),

                "maximum_model_mean_delta_AUROC":
                    float(
                        group[
                            "mean_delta_AUROC"
                        ]
                        .max()
                    ),

                "mean_across_models_delta_AUROC":
                    safe_mean(
                        group[
                            "mean_delta_AUROC"
                        ]
                    ),
            }
        )

    consensus = pd.DataFrame(
        consensus_rows
    )

    consensus.to_csv(
        MODEL_CONSENSUS_PATH,
        index=False,
    )

    return (
        model_summary,
        consensus,
    )


# =============================================================================
# 21. PARTICIPANT-LEVEL LINKAGE + WEIGHTED PERFORMANCE
# =============================================================================

def build_participant_robustness(
    experiments: pd.DataFrame,
    cohorts: Dict[str, pd.DataFrame],
    thresholds: pd.DataFrame,
):

    schemas = {
        study:
            detect_cohort_schema(
                df,
                study,
            )
        for study, df
        in cohorts.items()
    }

    linkage_rows = []
    weighted_rows = []
    subgroup_rows = []
    age_rows = []

    stage06_metric_lookup = (
        experiments.set_index(
            "experiment_id"
        )
    )

    for counter, (
        _,
        experiment,
    ) in enumerate(
        experiments.iterrows(),
        start=1,
    ):

        experiment_id = (
            experiment[
                "experiment_id"
            ]
        )

        if (
            counter == 1
            or counter % 25 == 0
            or counter == len(
                experiments
            )
        ):

            print(
                f"Participant analysis "
                f"{counter}/{len(experiments)}: "
                f"{experiment_id}"
            )

        linked, audit = (
            link_experiment_predictions(
                experiment,
                cohorts=
                    cohorts,
                schemas=
                    schemas,
            )
        )

        linkage_rows.append(
            audit
        )

        threshold_row = (
            thresholds[
                (
                    thresholds[
                        "experiment_id"
                    ]
                    == experiment_id
                )
                &
                (
                    np.isclose(
                        thresholds[
                            "target_training_sensitivity"
                        ],
                        PRIMARY_SUBGROUP_TARGET,
                    )
                )
            ]
        )

        if threshold_row.empty:

            raise RuntimeError(
                f"No 95% training-sensitivity threshold "
                f"for {experiment_id}."
            )

        threshold = float(
            threshold_row[
                "threshold"
            ].iloc[
                0
            ]
        )

        y = linked[
            "y"
        ].to_numpy()

        p = linked[
            "p_sigmoid"
        ].to_numpy()

        weights = pd.to_numeric(
            linked[
                "survey_weight"
            ],
            errors="coerce",
        ).to_numpy()

        # ---------------------------------------------------------------------
        # Recomputed unweighted performance
        # ---------------------------------------------------------------------

        unweighted = (
            evaluate_binary_predictions(
                y=y,
                p=p,
                weights=np.ones(
                    len(
                        y
                    )
                ),
                threshold=threshold,
            )
        )

        # ---------------------------------------------------------------------
        # Survey-weighted sensitivity evaluation
        # ---------------------------------------------------------------------

        weighted = (
            evaluate_binary_predictions(
                y=y,
                p=p,
                weights=weights,
                threshold=threshold,
            )
        )

        row = {
            "experiment_id":
                experiment_id,

            "study":
                experiment[
                    "study"
                ],

            "temporal_design":
                experiment[
                    "temporal_design"
                ],

            "train_cycles":
                experiment[
                    "train_cycles"
                ],

            "test_cycle":
                experiment[
                    "test_cycle"
                ],

            "feature_tier":
                experiment[
                    "feature_tier"
                ],

            "model":
                experiment[
                    "model"
                ],

            "threshold_95":
                threshold,
        }

        for metric in [
            "AUROC",
            "AUPRC",
            "Brier",
            "ECE",
            "CalibrationIntercept",
            "CalibrationSlope",
            "Sensitivity",
            "FNR",
            "Specificity",
            "PPV",
            "NPV",
            "ReferralRate",
            "TestsAvoided",
            "WeightedPrevalence",
        ]:

            row[
                f"unweighted_{metric}"
            ] = unweighted.get(
                metric,
                np.nan,
            )

            row[
                f"weighted_{metric}"
            ] = weighted.get(
                metric,
                np.nan,
            )

            if (
                np.isfinite(
                    row[
                        f"weighted_{metric}"
                    ]
                )
                and np.isfinite(
                    row[
                        f"unweighted_{metric}"
                    ]
                )
            ):

                row[
                    f"weighted_minus_unweighted_{metric}"
                ] = (
                    row[
                        f"weighted_{metric}"
                    ]
                    -
                    row[
                        f"unweighted_{metric}"
                    ]
                )

            else:

                row[
                    f"weighted_minus_unweighted_{metric}"
                ] = np.nan

        # Validate prediction reconstruction against frozen Stage06 metrics

        for metric in PRIMARY_METRICS:

            frozen_value = (
                experiment[
                    metric
                ]
            )

            recomputed = (
                unweighted[
                    metric
                ]
            )

            row[
                f"stage06_{metric}"
            ] = frozen_value

            row[
                f"recomputed_minus_stage06_{metric}"
            ] = (
                recomputed
                -
                frozen_value
            )

        weighted_rows.append(
            row
        )

        # ---------------------------------------------------------------------
        # Age 20–59 sensitivity
        # ---------------------------------------------------------------------

        age_numeric = pd.to_numeric(
            linked[
                "age_years"
            ],
            errors="coerce",
        )

        age_mask = (
            age_numeric
            .between(
                20,
                59,
                inclusive="both",
            )
            .to_numpy()
        )

        if age_mask.sum() > 0:

            age_unweighted = (
                evaluate_binary_predictions(
                    y=y[
                        age_mask
                    ],
                    p=p[
                        age_mask
                    ],
                    weights=np.ones(
                        int(
                            age_mask.sum()
                        )
                    ),
                    threshold=threshold,
                )
            )

            age_weighted = (
                evaluate_binary_predictions(
                    y=y[
                        age_mask
                    ],
                    p=p[
                        age_mask
                    ],
                    weights=weights[
                        age_mask
                    ],
                    threshold=threshold,
                )
            )

            age_row = {
                "experiment_id":
                    experiment_id,

                "study":
                    experiment[
                        "study"
                    ],

                "temporal_design":
                    experiment[
                        "temporal_design"
                    ],

                "test_cycle":
                    experiment[
                        "test_cycle"
                    ],

                "feature_tier":
                    experiment[
                        "feature_tier"
                    ],

                "model":
                    experiment[
                        "model"
                    ],

                "n_full_18_59":
                    len(
                        linked
                    ),

                "n_age20_59":
                    int(
                        age_mask.sum()
                    ),

                "threshold_95":
                    threshold,
            }

            for metric in [
                "AUROC",
                "AUPRC",
                "Brier",
                "ECE",
                "Sensitivity",
                "FNR",
                "Specificity",
                "PPV",
                "NPV",
            ]:

                age_row[
                    f"full_unweighted_{metric}"
                ] = unweighted.get(
                    metric,
                    np.nan,
                )

                age_row[
                    f"age20_59_unweighted_{metric}"
                ] = age_unweighted.get(
                    metric,
                    np.nan,
                )

                age_row[
                    f"delta_age20_59_minus_full_unweighted_{metric}"
                ] = (
                    age_unweighted.get(
                        metric,
                        np.nan,
                    )
                    -
                    unweighted.get(
                        metric,
                        np.nan,
                    )
                )

                age_row[
                    f"full_weighted_{metric}"
                ] = weighted.get(
                    metric,
                    np.nan,
                )

                age_row[
                    f"age20_59_weighted_{metric}"
                ] = age_weighted.get(
                    metric,
                    np.nan,
                )

                age_row[
                    f"delta_age20_59_minus_full_weighted_{metric}"
                ] = (
                    age_weighted.get(
                        metric,
                        np.nan,
                    )
                    -
                    weighted.get(
                        metric,
                        np.nan,
                    )
                )

            age_rows.append(
                age_row
            )

        # ---------------------------------------------------------------------
        # Construct subgroup variables
        # ---------------------------------------------------------------------

        subgroup_definitions = {}

        if "sex" in linked.columns:

            subgroup_definitions[
                "sex"
            ] = (
                linked[
                    "sex"
                ]
                .astype(
                    str
                )
            )

        if (
            "race_ethnicity"
            in linked.columns
        ):

            subgroup_definitions[
                "race_ethnicity"
            ] = (
                linked[
                    "race_ethnicity"
                ]
                .astype(
                    str
                )
            )

        age_group = pd.cut(
            age_numeric,
            bins=[
                18,
                30,
                45,
                60,
            ],
            right=False,
            include_lowest=True,
            labels=[
                "18-29",
                "30-44",
                "45-59",
            ],
        )

        subgroup_definitions[
            "age_group"
        ] = (
            age_group.astype(
                str
            )
        )

        if (
            "poverty_income_ratio"
            in linked.columns
        ):

            pir = pd.to_numeric(
                linked[
                    "poverty_income_ratio"
                ],
                errors="coerce",
            )

            poverty_group = pd.cut(
                pir,
                bins=[
                    -np.inf,
                    1.30,
                    3.50,
                    np.inf,
                ],
                right=False,
                labels=[
                    "<1.30",
                    "1.30-3.49",
                    ">=3.50",
                ],
            )

            subgroup_definitions[
                "poverty_group"
            ] = (
                poverty_group.astype(
                    str
                )
            )

        # ---------------------------------------------------------------------
        # Evaluate subgroups
        # ---------------------------------------------------------------------

        for dimension, values in subgroup_definitions.items():

            values = values.replace(
                {
                    "nan":
                        np.nan,

                    "None":
                        np.nan,
                }
            )

            for category in (
                pd.Series(
                    values
                )
                .dropna()
                .unique()
            ):

                mask = (
                    values
                    == category
                ).to_numpy()

                n = int(
                    mask.sum()
                )

                cases = int(
                    y[
                        mask
                    ].sum()
                )

                noncases = (
                    n
                    -
                    cases
                )

                sufficient = (
                    n
                    >= MIN_SUBGROUP_N
                    and cases
                    >= MIN_SUBGROUP_CASES
                    and noncases
                    >= MIN_SUBGROUP_NONCASES
                )

                subgroup_row = {
                    "experiment_id":
                        experiment_id,

                    "study":
                        experiment[
                            "study"
                        ],

                    "temporal_design":
                        experiment[
                            "temporal_design"
                        ],

                    "test_cycle":
                        experiment[
                            "test_cycle"
                        ],

                    "feature_tier":
                        experiment[
                            "feature_tier"
                        ],

                    "model":
                        experiment[
                            "model"
                        ],

                    "subgroup_dimension":
                        dimension,

                    "subgroup":
                        category,

                    "n":
                        n,

                    "cases":
                        cases,

                    "noncases":
                        noncases,

                    "meets_minimum_support":
                        sufficient,

                    "threshold_95":
                        threshold,
                }

                if sufficient:

                    sg_unweighted = (
                        evaluate_binary_predictions(
                            y=y[
                                mask
                            ],
                            p=p[
                                mask
                            ],
                            weights=np.ones(
                                n
                            ),
                            threshold=threshold,
                        )
                    )

                    sg_weighted = (
                        evaluate_binary_predictions(
                            y=y[
                                mask
                            ],
                            p=p[
                                mask
                            ],
                            weights=weights[
                                mask
                            ],
                            threshold=threshold,
                        )
                    )

                    for metric in [
                        "AUROC",
                        "AUPRC",
                        "Brier",
                        "ECE",
                        "CalibrationIntercept",
                        "CalibrationSlope",
                        "Sensitivity",
                        "FNR",
                        "Specificity",
                        "PPV",
                        "NPV",
                    ]:

                        subgroup_row[
                            f"unweighted_{metric}"
                        ] = sg_unweighted.get(
                            metric,
                            np.nan,
                        )

                        subgroup_row[
                            f"weighted_{metric}"
                        ] = sg_weighted.get(
                            metric,
                            np.nan,
                        )

                subgroup_rows.append(
                    subgroup_row
                )

    linkage_audit = pd.DataFrame(
        linkage_rows
    )

    weighted_performance = pd.DataFrame(
        weighted_rows
    )

    subgroup_performance = pd.DataFrame(
        subgroup_rows
    )

    age_sensitivity = pd.DataFrame(
        age_rows
    )

    linkage_audit.to_csv(
        LINKAGE_AUDIT_PATH,
        index=False,
    )

    weighted_performance.to_csv(
        WEIGHTED_PERFORMANCE_PATH,
        index=False,
    )

    subgroup_performance.to_csv(
        SUBGROUP_PERFORMANCE_PATH,
        index=False,
    )

    age_sensitivity.to_csv(
        AGE_SENSITIVITY_PATH,
        index=False,
    )

    return (
        linkage_audit,
        weighted_performance,
        subgroup_performance,
        age_sensitivity,
    )


# =============================================================================
# 22. WEIGHTED VS UNWEIGHTED SUMMARY
# =============================================================================

def build_weighted_summary(
    weighted: pd.DataFrame,
):

    metrics = [
        "AUROC",
        "AUPRC",
        "Brier",
        "ECE",
        "Sensitivity",
        "FNR",
        "Specificity",
        "PPV",
        "NPV",
        "TestsAvoided",
    ]

    rows = []

    for (
        study,
        tier,
    ), group in weighted.groupby(
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
                len(
                    group
                ),
        }

        for metric in metrics:

            unweighted_col = (
                f"unweighted_{metric}"
            )

            weighted_col = (
                f"weighted_{metric}"
            )

            if (
                unweighted_col
                not in group.columns
                or weighted_col
                not in group.columns
            ):

                continue

            row[
                f"mean_unweighted_{metric}"
            ] = safe_mean(
                group[
                    unweighted_col
                ]
            )

            row[
                f"mean_weighted_{metric}"
            ] = safe_mean(
                group[
                    weighted_col
                ]
            )

            row[
                f"mean_weighted_minus_unweighted_{metric}"
            ] = safe_mean(
                group[
                    weighted_col
                ]
                -
                group[
                    unweighted_col
                ]
            )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        WEIGHTED_SUMMARY_PATH,
        index=False,
    )

    return summary


# =============================================================================
# 23. SUBGROUP DISPARITY SUMMARY
# =============================================================================

def build_subgroup_disparity(
    subgroup: pd.DataFrame,
):

    valid = (
        subgroup[
            subgroup[
                "meets_minimum_support"
            ]
            == True
        ]
        .copy()
    )

    metrics = [
        "unweighted_AUROC",
        "unweighted_Sensitivity",
        "unweighted_FNR",
        "unweighted_ECE",
        "weighted_AUROC",
        "weighted_Sensitivity",
        "weighted_FNR",
        "weighted_ECE",
    ]

    rows = []

    grouping = [
        "experiment_id",
        "study",
        "temporal_design",
        "test_cycle",
        "feature_tier",
        "model",
        "subgroup_dimension",
    ]

    for keys, group in valid.groupby(
        grouping,
        dropna=False,
    ):

        if group[
            "subgroup"
        ].nunique() < 2:

            continue

        row = {
            "experiment_id":
                keys[0],

            "study":
                keys[1],

            "temporal_design":
                keys[2],

            "test_cycle":
                keys[3],

            "feature_tier":
                keys[4],

            "model":
                keys[5],

            "subgroup_dimension":
                keys[6],

            "n_supported_groups":
                group[
                    "subgroup"
                ]
                .nunique(),
        }

        for metric in metrics:

            if metric not in group.columns:
                continue

            values = pd.to_numeric(
                group[
                    metric
                ],
                errors="coerce",
            )

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
        SUBGROUP_DISPARITY_PATH,
        index=False,
    )

    return result


# =============================================================================
# 24. AGE 20–59 SUMMARY
# =============================================================================

def build_age_sensitivity_summary(
    age: pd.DataFrame,
):

    delta_columns = [
        column
        for column in age.columns
        if column.startswith(
            "delta_age20_59_minus_full_"
        )
    ]

    rows = []

    for (
        study,
        tier,
    ), group in age.groupby(
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
                len(
                    group
                ),

            "mean_fraction_retained":
                np.mean(
                    group[
                        "n_age20_59"
                    ]
                    /
                    group[
                        "n_full_18_59"
                    ]
                ),
        }

        for column in delta_columns:

            row[
                f"mean_{column}"
            ] = safe_mean(
                group[
                    column
                ]
            )

            row[
                f"median_{column}"
            ] = safe_median(
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
        AGE_SENSITIVITY_SUMMARY_PATH,
        index=False,
    )

    return summary


# =============================================================================
# 25. SURVEY-DESIGN-AWARE DESCRIPTIVE PREVALENCE
# =============================================================================

def survey_weighted_binary_mean(
    y,
    weight,
    strata,
    psu,
):

    df = pd.DataFrame(
        {
            "y":
                pd.to_numeric(
                    y,
                    errors="coerce",
                ),

            "w":
                pd.to_numeric(
                    weight,
                    errors="coerce",
                ),

            "strata":
                strata,

            "psu":
                psu,
        }
    )

    df = df[
        df[
            "y"
        ].notna()
        &
        df[
            "w"
        ].notna()
        &
        (
            df[
                "w"
            ]
            > 0
        )
        &
        df[
            "strata"
        ].notna()
        &
        df[
            "psu"
        ].notna()
    ].copy()

    if df.empty:

        return {
            "estimate":
                np.nan,

            "se":
                np.nan,

            "ci_low":
                np.nan,

            "ci_high":
                np.nan,

            "n":
                0,

            "n_strata":
                0,

            "n_psu":
                0,

            "singleton_strata":
                np.nan,
        }

    total_weight = df[
        "w"
    ].sum()

    estimate = (
        (
            df[
                "w"
            ]
            *
            df[
                "y"
            ]
        )
        .sum()
        /
        total_weight
    )

    df[
        "linearized"
    ] = (
        df[
            "w"
        ]
        *
        (
            df[
                "y"
            ]
            -
            estimate
        )
        /
        total_weight
    )

    psu_totals = (
        df.groupby(
            [
                "strata",
                "psu",
            ]
        )[
            "linearized"
        ]
        .sum()
        .reset_index()
    )

    variance = 0.0
    singleton_strata = 0

    for _, group in psu_totals.groupby(
        "strata"
    ):

        m = len(
            group
        )

        if m < 2:

            singleton_strata += 1
            continue

        values = (
            group[
                "linearized"
            ]
            .to_numpy()
        )

        center = (
            values.mean()
        )

        variance += (
            m
            /
            (
                m - 1
            )
        ) * np.sum(
            (
                values
                -
                center
            ) ** 2
        )

    se = math.sqrt(
        max(
            variance,
            0.0,
        )
    )

    ci_low = max(
        0.0,
        estimate
        -
        1.96
        * se,
    )

    ci_high = min(
        1.0,
        estimate
        +
        1.96
        * se,
    )

    return {
        "estimate":
            estimate,

        "se":
            se,

        "ci_low":
            ci_low,

        "ci_high":
            ci_high,

        "n":
            len(
                df
            ),

        "n_strata":
            df[
                "strata"
            ]
            .nunique(),

        "n_psu":
            df[
                [
                    "strata",
                    "psu",
                ]
            ]
            .drop_duplicates()
            .shape[
                0
            ],

        "singleton_strata":
            singleton_strata,
    }


def build_survey_descriptive(
    cohorts: Dict[str, pd.DataFrame],
):

    rows = []

    for study, cohort in cohorts.items():

        schema = (
            detect_cohort_schema(
                cohort,
                study,
            )
        )

        if (
            schema[
                "strata"
            ]
            is None
            or schema[
                "psu"
            ]
            is None
        ):

            continue

        for cycle, group in cohort.groupby(
            schema[
                "cycle"
            ]
        ):

            result = (
                survey_weighted_binary_mean(
                    y=
                        group[
                            schema[
                                "outcome"
                            ]
                        ],
                    weight=
                        group[
                            schema[
                                "weight"
                            ]
                        ],
                    strata=
                        group[
                            schema[
                                "strata"
                            ]
                        ],
                    psu=
                        group[
                            schema[
                                "psu"
                            ]
                        ],
                )
            )

            rows.append(
                {
                    "study":
                        study,

                    "cycle":
                        cycle,

                    "weighted_prevalence":
                        result[
                            "estimate"
                        ],

                    "weighted_prevalence_pct":
                        (
                            result[
                                "estimate"
                            ]
                            * 100
                        ),

                    "survey_linearized_se":
                        result[
                            "se"
                        ],

                    "ci95_low":
                        result[
                            "ci_low"
                        ],

                    "ci95_high":
                        result[
                            "ci_high"
                        ],

                    "n":
                        result[
                            "n"
                        ],

                    "n_strata":
                        result[
                            "n_strata"
                        ],

                    "n_psu":
                        result[
                            "n_psu"
                        ],

                    "singleton_strata":
                        result[
                            "singleton_strata"
                        ],

                    "scope_note":
                        (
                            "Taylor-linearized descriptive estimate "
                            "within the frozen analytic cohort. "
                            "Use full survey-domain software for "
                            "publication-final population inference "
                            "if required."
                        ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        SURVEY_DESCRIPTIVE_PATH,
        index=False,
    )

    return result


# =============================================================================
# 26. TEMPORAL-DESIGN UTILITY
# =============================================================================

def build_temporal_utility(
    operating: pd.DataFrame,
):

    measures = [
        "ReferralRate",
        "TestsAvoided",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "TestsPerDetectedCase",
    ]

    aggregation = {}

    for measure in measures:

        if measure in operating.columns:

            aggregation[
                f"mean_{measure}"
            ] = (
                measure,
                "mean",
            )

    result = (
        operating.groupby(
            [
                "study",
                "temporal_design",
                "feature_tier",
                "target_training_sensitivity",
            ],
            dropna=False,
        )
        .agg(
            n_experiments=(
                "experiment_id",
                "nunique",
            ),

            **aggregation,
        )
        .reset_index()
    )

    result.to_csv(
        TEMPORAL_UTILITY_PATH,
        index=False,
    )

    return result


# =============================================================================
# 27. FINAL ROBUSTNESS AUDIT
# =============================================================================

def build_final_robustness_audit():

    rows = [
        {
            "analysis":
                "Temporal external validation",

            "status":
                "COMPLETE",

            "evidence_source":
                "Stages 06–07",

            "note":
                (
                    "Fixed-origin, rolling-expanding, "
                    "and backward/forward bridge analyses completed."
                ),
        },

        {
            "analysis":
                "Probability calibration",

            "status":
                "COMPLETE",

            "evidence_source":
                "Stages 06–07",

            "note":
                (
                    "Leakage-safe OOF sigmoid calibration primary; "
                    "raw and isotonic outputs retained."
                ),
        },

        {
            "analysis":
                "Decision-curve analysis",

            "status":
                "COMPLETE",

            "evidence_source":
                "Stages 06 and 08",

            "note":
                (
                    "Model-guided testing compared with "
                    "test-all and test-none."
                ),
        },

        {
            "analysis":
                "Testing-burden analysis",

            "status":
                "COMPLETE",

            "evidence_source":
                "Stages 06–08",

            "note":
                (
                    "90%, 95%, and 97.5% prespecified "
                    "training-sensitivity targets evaluated."
                ),
        },

        {
            "analysis":
                "Model-family robustness",

            "status":
                "COMPLETE",

            "evidence_source":
                "Stage 08",

            "note":
                (
                    "Tier conclusions evaluated across all "
                    "six locked model families."
                ),
        },

        {
            "analysis":
                "Participant-level linkage",

            "status":
                "COMPLETE",

            "evidence_source":
                "Stage 06 checkpoints + Stage 03 cohorts",

            "note":
                (
                    "SEQN linkage and 100% outcome agreement "
                    "required before robustness metrics are calculated."
                ),
        },

        {
            "analysis":
                "Survey-weighted predictive sensitivity",

            "status":
                "COMPLETE",

            "evidence_source":
                "Stage 08",

            "note":
                (
                    "Individual MEC weights incorporated into "
                    "predictive metric sensitivity analyses. "
                    "These are not labeled as full complex-survey "
                    "estimators of every ML metric."
                ),
        },

        {
            "analysis":
                "Subgroup temporal transportability",

            "status":
                "COMPLETE_WITH_SUPPORT_RULE",

            "evidence_source":
                "Stage 08",

            "note":
                (
                    "Sex, race/ethnicity, age, and poverty groups "
                    "evaluated only when minimum N/case/noncase "
                    "requirements are met."
                ),
        },

        {
            "analysis":
                "Age 20–59 structural sensitivity",

            "status":
                "COMPLETE",

            "evidence_source":
                "Stage 08",

            "note":
                (
                    "Frozen models evaluated after excluding "
                    "ages 18–19; no model retraining performed."
                ),
        },

        {
            "analysis":
                "Survey-design descriptive inference",

            "status":
                "COMPLETE_WITH_SCOPE_LIMITATION",

            "evidence_source":
                "Stage 08",

            "note":
                (
                    "Taylor-linearized analytic-cohort prevalence "
                    "estimates generated using weights, strata and PSU. "
                    "Full survey-domain software remains preferable "
                    "for publication-final population inference."
                ),
        },
    ]

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        FINAL_ROBUSTNESS_PATH,
        index=False,
    )

    return result


# =============================================================================
# 28. KEY FINDINGS
# =============================================================================

def build_key_findings(
    model_consensus: pd.DataFrame,
    operating_summary: pd.DataFrame,
    dca_strategy: pd.DataFrame,
    weighted_summary: pd.DataFrame,
    age_summary: pd.DataFrame,
):

    rows = []

    for _, row in model_consensus.iterrows():

        rows.append(
            {
                "domain":
                    "Model-family robustness",

                "study":
                    row[
                        "study"
                    ],

                "feature_tier":
                    "T2_minus_T1",

                "measure":
                    "proportion_models_supporting_T2",

                "value":
                    row[
                        "proportion_models_supporting_T2"
                    ],
            }
        )

        rows.append(
            {
                "domain":
                    "Model-family robustness",

                "study":
                    row[
                        "study"
                    ],

                "feature_tier":
                    "T2_minus_T1",

                "measure":
                    "mean_across_models_delta_AUROC",

                "value":
                    row[
                        "mean_across_models_delta_AUROC"
                    ],
            }
        )

    if (
        "mean_TestsAvoided"
        in operating_summary.columns
    ):

        op90 = (
            operating_summary[
                np.isclose(
                    operating_summary[
                        "target_training_sensitivity"
                    ],
                    0.90,
                )
            ]
        )

        for _, row in op90.iterrows():

            rows.append(
                {
                    "domain":
                        "Testing burden",

                    "study":
                        row[
                            "study"
                        ],

                    "feature_tier":
                        row[
                            "feature_tier"
                        ],

                    "measure":
                        "mean_tests_avoided_at_90pct_target",

                    "value":
                        row[
                            "mean_TestsAvoided"
                        ],
                }
            )

    for _, row in dca_strategy.iterrows():

        rows.append(
            {
                "domain":
                    "Decision curve",

                "study":
                    row[
                        "study"
                    ],

                "feature_tier":
                    row[
                        "feature_tier"
                    ],

                "measure":
                    "mean_fraction_thresholds_model_best",

                "value":
                    row[
                        "mean_proportion_thresholds_model_best"
                    ],
            }
        )

    for _, row in weighted_summary.iterrows():

        if (
            "mean_weighted_minus_unweighted_AUROC"
            in row.index
        ):

            rows.append(
                {
                    "domain":
                        "Survey-weight sensitivity",

                    "study":
                        row[
                            "study"
                        ],

                    "feature_tier":
                        row[
                            "feature_tier"
                        ],

                    "measure":
                        "mean_weighted_minus_unweighted_AUROC",

                    "value":
                        row[
                            "mean_weighted_minus_unweighted_AUROC"
                        ],
                }
            )

    for _, row in age_summary.iterrows():

        column = (
            "mean_delta_age20_59_minus_full_"
            "unweighted_AUROC"
        )

        if column in row.index:

            rows.append(
                {
                    "domain":
                        "Age sensitivity",

                    "study":
                        row[
                            "study"
                        ],

                    "feature_tier":
                        row[
                            "feature_tier"
                        ],

                    "measure":
                        "mean_delta_AUROC_age20_59_minus_full",

                    "value":
                        row[
                            column
                        ],
                }
            )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        KEY_FINDINGS_PATH,
        index=False,
    )

    return result


# =============================================================================
# 29. FINAL INTEGRITY CHECK
# =============================================================================

def build_integrity_check(
    experiments,
    dca_experiment,
    operating,
    model_robustness,
    linkage,
    weighted,
    subgroup,
    age,
    survey,
    stage06_hash_before,
    stage06_hash_after,
):

    checks = []

    def add(
        check,
        expected,
        observed,
        passed,
    ):

        checks.append(
            {
                "check":
                    check,

                "expected":
                    expected,

                "observed":
                    observed,

                "pass":
                    bool(
                        passed
                    ),
            }
        )

    add(
        "Primary sigmoid experiment count",
        EXPECTED_EXPERIMENTS,
        len(
            experiments
        ),
        len(
            experiments
        )
        == EXPECTED_EXPERIMENTS,
    )

    add(
        "DCA experiment coverage",
        EXPECTED_EXPERIMENTS,
        dca_experiment[
            "experiment_id"
        ].nunique(),
        dca_experiment[
            "experiment_id"
        ].nunique()
        == EXPECTED_EXPERIMENTS,
    )

    add(
        "Testing-burden experiment coverage",
        EXPECTED_EXPERIMENTS,
        operating[
            "experiment_id"
        ].nunique(),
        operating[
            "experiment_id"
        ].nunique()
        == EXPECTED_EXPERIMENTS,
    )

    add(
        "Participant prediction linkage coverage",
        EXPECTED_EXPERIMENTS,
        linkage[
            "experiment_id"
        ].nunique(),
        linkage[
            "experiment_id"
        ].nunique()
        == EXPECTED_EXPERIMENTS,
    )

    add(
        "Experiments with unmatched participants",
        0,
        int(
            (
                linkage[
                    "n_unmatched"
                ]
                > 0
            )
            .sum()
        ),
        not (
            linkage[
                "n_unmatched"
            ]
            > 0
        ).any(),
    )

    add(
        "Experiments with less than 100% outcome agreement",
        0,
        int(
            (
                linkage[
                    "outcome_agreement"
                ]
                < 1.0
            )
            .sum()
        ),
        (
            linkage[
                "outcome_agreement"
            ]
            == 1.0
        ).all(),
    )

    add(
        "Experiments with all 10 seeds",
        EXPECTED_EXPERIMENTS,
        int(
            (
                linkage[
                    "n_seeds"
                ]
                == len(
                    LOCKED_SEEDS
                )
            )
            .sum()
        ),
        (
            linkage[
                "n_seeds"
            ]
            == len(
                LOCKED_SEEDS
            )
        ).all(),
    )

    add(
        "Survey-weighted experiment coverage",
        EXPECTED_EXPERIMENTS,
        weighted[
            "experiment_id"
        ].nunique(),
        weighted[
            "experiment_id"
        ].nunique()
        == EXPECTED_EXPERIMENTS,
    )

    max_auroc_difference = (
        weighted[
            "recomputed_minus_stage06_AUROC"
        ]
        .abs()
        .max()
    )

    add(
        "Recomputed Stage06 AUROC agreement",
        "<=1e-6",
        max_auroc_difference,
        (
            max_auroc_difference
            <= 1e-6
        ),
    )

    max_auprc_difference = (
        weighted[
            "recomputed_minus_stage06_AUPRC"
        ]
        .abs()
        .max()
    )

    add(
        "Recomputed Stage06 AUPRC agreement",
        "<=1e-6",
        max_auprc_difference,
        (
            max_auprc_difference
            <= 1e-6
        ),
    )

    add(
        "Subgroup analysis non-empty",
        ">0",
        len(
            subgroup
        ),
        len(
            subgroup
        )
        > 0,
    )

    add(
        "Age 20-59 sensitivity experiment coverage",
        EXPECTED_EXPERIMENTS,
        age[
            "experiment_id"
        ].nunique(),
        age[
            "experiment_id"
        ].nunique()
        == EXPECTED_EXPERIMENTS,
    )

    add(
        "Survey descriptive output non-empty",
        ">0",
        len(
            survey
        ),
        len(
            survey
        )
        > 0,
    )

    add(
        "Six model families represented",
        6,
        model_robustness[
            "model"
        ].nunique(),
        model_robustness[
            "model"
        ].nunique()
        == 6,
    )

    hashes_unchanged = (
        stage06_hash_before
        ==
        stage06_hash_after
    )

    add(
        "Frozen Stage06 canonical files unchanged",
        True,
        hashes_unchanged,
        hashes_unchanged,
    )

    result = pd.DataFrame(
        checks
    )

    result.to_csv(
        INTEGRITY_PATH,
        index=False,
    )

    return result


# =============================================================================
# 30. DECISION LEDGER
# =============================================================================

def build_decision_ledger():

    rows = [
        {
            "decision_id":
                "D081",

            "topic":
                "Canonical Stage 08",

            "decision":
                (
                    "The earlier aggregate-only Stage 08 is "
                    "superseded by this comprehensive Stage 08."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },

        {
            "decision_id":
                "D082",

            "topic":
                "Participant predictions",

            "decision":
                (
                    "Sigmoid predictions are averaged across "
                    "the 10 locked Stage-06 seeds by SEQN."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },

        {
            "decision_id":
                "D083",

            "topic":
                "Linkage integrity",

            "decision":
                (
                    "Participant robustness analysis requires "
                    "100% SEQN linkage and 100% outcome agreement."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },

        {
            "decision_id":
                "D084",

            "topic":
                "Survey-weighted ML metrics",

            "decision":
                (
                    "MEC-weighted predictive metrics are treated "
                    "as population-weight sensitivity analyses, "
                    "not full complex-survey estimators."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },

        {
            "decision_id":
                "D085",

            "topic":
                "Subgroups",

            "decision":
                (
                    "Subgroup estimates require at least "
                    f"{MIN_SUBGROUP_N} participants, "
                    f"{MIN_SUBGROUP_CASES} cases, and "
                    f"{MIN_SUBGROUP_NONCASES} noncases."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },

        {
            "decision_id":
                "D086",

            "topic":
                "Fairness interpretation",

            "decision":
                (
                    "Sensitivity/FNR, discrimination and calibration "
                    "are reported descriptively by subgroup; "
                    "no causal fairness claim is made."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },

        {
            "decision_id":
                "D087",

            "topic":
                "Age 20-59 sensitivity",

            "decision":
                (
                    "Frozen 18-59 models are evaluated on ages "
                    "20-59 without retraining to assess the "
                    "adult-education structural issue."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },

        {
            "decision_id":
                "D088",

            "topic":
                "Clinical utility",

            "decision":
                (
                    "90%, 95%, and 97.5% targets remain "
                    "prespecified; no single target is chosen "
                    "post hoc as universally optimal."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },

        {
            "decision_id":
                "D089",

            "topic":
                "Resource interpretation",

            "decision":
                (
                    "Results are described as resource-efficient, "
                    "resource-adaptive or testing-efficient; "
                    "formal cost-effectiveness is not claimed."
                ),

            "status":
                "FROZEN_IF_INTEGRITY_PASS",
        },
    ]

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        DECISION_LEDGER_PATH,
        index=False,
    )

    return result


# =============================================================================
# 31. WRITE CONSOLIDATED WORKBOOK
# =============================================================================

def write_workbook(
    key_findings,
    integrity,
    dca_experiment,
    dca_strategy,
    dca_threshold,
    operating_summary,
    operating_target,
    model_robustness,
    model_consensus,
    linkage,
    weighted,
    weighted_summary,
    subgroup,
    subgroup_disparity,
    age,
    age_summary,
    survey,
    temporal_utility,
    final_robustness,
    threshold_audit,
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

        dca_strategy.to_excel(
            writer,
            sheet_name="DCA_Strategy",
            index=False,
        )

        dca_threshold.to_excel(
            writer,
            sheet_name="DCA_Thresholds",
            index=False,
        )

        dca_experiment.to_excel(
            writer,
            sheet_name="DCA_Experiment",
            index=False,
        )

        operating_summary.to_excel(
            writer,
            sheet_name="Operating_Summary",
            index=False,
        )

        operating_target.to_excel(
            writer,
            sheet_name="Target_Robustness",
            index=False,
        )

        model_robustness.to_excel(
            writer,
            sheet_name="Model_Tier_Robust",
            index=False,
        )

        model_consensus.to_excel(
            writer,
            sheet_name="Model_Consensus",
            index=False,
        )

        linkage.to_excel(
            writer,
            sheet_name="Prediction_Linkage",
            index=False,
        )

        weighted.to_excel(
            writer,
            sheet_name="Weighted_Performance",
            index=False,
        )

        weighted_summary.to_excel(
            writer,
            sheet_name="Weighted_vs_Unweighted",
            index=False,
        )

        subgroup.to_excel(
            writer,
            sheet_name="Subgroup_Performance",
            index=False,
        )

        subgroup_disparity.to_excel(
            writer,
            sheet_name="Subgroup_Disparity",
            index=False,
        )

        age.to_excel(
            writer,
            sheet_name="Age20_59_Detail",
            index=False,
        )

        age_summary.to_excel(
            writer,
            sheet_name="Age20_59_Summary",
            index=False,
        )

        survey.to_excel(
            writer,
            sheet_name="Survey_Descriptive",
            index=False,
        )

        temporal_utility.to_excel(
            writer,
            sheet_name="Temporal_Utility",
            index=False,
        )

        final_robustness.to_excel(
            writer,
            sheet_name="Final_Robustness",
            index=False,
        )

        threshold_audit.to_excel(
            writer,
            sheet_name="Threshold_Audit",
            index=False,
        )

        ledger.to_excel(
            writer,
            sheet_name="Decision_Ledger",
            index=False,
        )


# =============================================================================
# 32. MAIN
# =============================================================================

def main():

    warnings.filterwarnings(
        "ignore"
    )

    print("\n")
    print(
        "=" * 110
    )

    print(
        "STAGE 08 — FINAL CANONICAL RUN"
    )

    print(
        "Clinical Utility | Survey-Weighted Robustness | "
        "Subgroup Transportability | Structural Sensitivity"
    )

    print(
        "=" * 110
    )

    # -------------------------------------------------------------------------
    # Backup previous Stage 08
    # -------------------------------------------------------------------------

    backup_previous_stage08()

    # -------------------------------------------------------------------------
    # Protect Stage 06
    # -------------------------------------------------------------------------

    stage06_hash_before = (
        stage06_hash_snapshot()
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    data = load_inputs()

    experiments = (
        get_primary_experiments(
            data[
                "metrics"
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Thresholds
    # -------------------------------------------------------------------------

    print(
        "\nBuilding leakage-safe operating thresholds..."
    )

    thresholds = (
        build_threshold_table(
            burden=
                data[
                    "burden"
                ],
            experiments=
                experiments,
        )
    )

    # -------------------------------------------------------------------------
    # Decision curves
    # -------------------------------------------------------------------------

    print(
        "\nSynthesizing decision-curve analysis..."
    )

    (
        dca_experiment,
        dca_strategy,
        dca_threshold,
    ) = build_dca_analysis(
        data[
            "dca"
        ]
    )

    # -------------------------------------------------------------------------
    # Operating points
    # -------------------------------------------------------------------------

    print(
        "\nSynthesizing testing-burden operating points..."
    )

    (
        operating_experiment,
        operating_summary,
        operating_target,
    ) = build_operating_analysis(
        data[
            "burden"
        ]
    )

    # -------------------------------------------------------------------------
    # Model-family robustness
    # -------------------------------------------------------------------------

    print(
        "\nEvaluating model-family robustness..."
    )

    (
        model_robustness,
        model_consensus,
    ) = build_model_family_robustness(
        data[
            "tier"
        ]
    )

    # -------------------------------------------------------------------------
    # Participant-level linkage and robustness
    # -------------------------------------------------------------------------

    print(
        "\nLinking Stage-06 predictions to Stage-03 participants..."
    )

    (
        linkage,
        weighted,
        subgroup,
        age,
    ) = build_participant_robustness(
        experiments=
            experiments,
        cohorts=
            data[
                "cohorts"
            ],
        thresholds=
            thresholds,
    )

    # -------------------------------------------------------------------------
    # Weighted summary
    # -------------------------------------------------------------------------

    weighted_summary = (
        build_weighted_summary(
            weighted
        )
    )

    # -------------------------------------------------------------------------
    # Subgroup disparity
    # -------------------------------------------------------------------------

    subgroup_disparity = (
        build_subgroup_disparity(
            subgroup
        )
    )

    # -------------------------------------------------------------------------
    # Age sensitivity
    # -------------------------------------------------------------------------

    age_summary = (
        build_age_sensitivity_summary(
            age
        )
    )

    # -------------------------------------------------------------------------
    # Survey descriptive
    # -------------------------------------------------------------------------

    print(
        "\nComputing survey-design-aware descriptive prevalence..."
    )

    survey = (
        build_survey_descriptive(
            data[
                "cohorts"
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Temporal utility
    # -------------------------------------------------------------------------

    temporal_utility = (
        build_temporal_utility(
            operating_experiment
        )
    )

    # -------------------------------------------------------------------------
    # Final audit
    # -------------------------------------------------------------------------

    final_robustness = (
        build_final_robustness_audit()
    )

    # -------------------------------------------------------------------------
    # Key findings
    # -------------------------------------------------------------------------

    key_findings = (
        build_key_findings(
            model_consensus=
                model_consensus,
            operating_summary=
                operating_summary,
            dca_strategy=
                dca_strategy,
            weighted_summary=
                weighted_summary,
            age_summary=
                age_summary,
        )
    )

    # -------------------------------------------------------------------------
    # Confirm Stage 06 untouched
    # -------------------------------------------------------------------------

    stage06_hash_after = (
        stage06_hash_snapshot()
    )

    # -------------------------------------------------------------------------
    # Integrity
    # -------------------------------------------------------------------------

    integrity = (
        build_integrity_check(
            experiments=
                experiments,
            dca_experiment=
                dca_experiment,
            operating=
                operating_experiment,
            model_robustness=
                model_robustness,
            linkage=
                linkage,
            weighted=
                weighted,
            subgroup=
                subgroup,
            age=
                age,
            survey=
                survey,
            stage06_hash_before=
                stage06_hash_before,
            stage06_hash_after=
                stage06_hash_after,
        )
    )

    ledger = (
        build_decision_ledger()
    )

    # -------------------------------------------------------------------------
    # Workbook
    # -------------------------------------------------------------------------

    write_workbook(
        key_findings=
            key_findings,
        integrity=
            integrity,
        dca_experiment=
            dca_experiment,
        dca_strategy=
            dca_strategy,
        dca_threshold=
            dca_threshold,
        operating_summary=
            operating_summary,
        operating_target=
            operating_target,
        model_robustness=
            model_robustness,
        model_consensus=
            model_consensus,
        linkage=
            linkage,
        weighted=
            weighted,
        weighted_summary=
            weighted_summary,
        subgroup=
            subgroup,
        subgroup_disparity=
            subgroup_disparity,
        age=
            age,
        age_summary=
            age_summary,
        survey=
            survey,
        temporal_utility=
            temporal_utility,
        final_robustness=
            final_robustness,
        threshold_audit=
            thresholds,
        ledger=
            ledger,
    )

    # =========================================================================
    # CONSOLE REPORT
    # =========================================================================

    print("\n")
    print(
        "-" * 110
    )

    print(
        "MODEL-FAMILY CONSENSUS"
    )

    print(
        "-" * 110
    )

    print(
        model_consensus
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print("\n")
    print(
        "-" * 110
    )

    print(
        "WEIGHTED VS UNWEIGHTED"
    )

    print(
        "-" * 110
    )

    display_columns = [
        column
        for column in [
            "study",
            "feature_tier",
            "mean_unweighted_AUROC",
            "mean_weighted_AUROC",
            "mean_weighted_minus_unweighted_AUROC",
            "mean_unweighted_AUPRC",
            "mean_weighted_AUPRC",
            "mean_weighted_minus_unweighted_AUPRC",
        ]
        if column
        in weighted_summary.columns
    ]

    print(
        weighted_summary[
            display_columns
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print("\n")
    print(
        "-" * 110
    )

    print(
        "AGE 20–59 SENSITIVITY"
    )

    print(
        "-" * 110
    )

    age_display = [
        column
        for column in [
            "study",
            "feature_tier",
            "mean_fraction_retained",
            "mean_delta_age20_59_minus_full_unweighted_AUROC",
            "mean_delta_age20_59_minus_full_weighted_AUROC",
        ]
        if column
        in age_summary.columns
    ]

    print(
        age_summary[
            age_display
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print("\n")
    print(
        "-" * 110
    )

    print(
        "INTEGRITY CHECK"
    )

    print(
        "-" * 110
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
        "=" * 110
    )

    if all_pass:

        print(
            "STAGE 08 EXECUTION: PASS"
        )

        print(
            "All aggregate and participant-level robustness "
            "analyses were generated successfully."
        )

        print(
            "Stage 08 is ready for scientific review before freezing."
        )

    else:

        print(
            "STAGE 08 EXECUTION: FAIL"
        )

        print(
            "DO NOT FREEZE STAGE 08."
        )

        print(
            "Review failed integrity checks before interpretation."
        )

    print(
        "=" * 110
    )

    print(
        f"\nCanonical workbook:\n"
        f"  {WORKBOOK_PATH}"
    )


# =============================================================================
# 33. ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()