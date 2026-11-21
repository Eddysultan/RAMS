"""
===============================================================================
STAGE 05
Locked Temporal Modeling Design,
Leakage-Safe Preprocessing Specification,
and Experiment Manifest
===============================================================================

===============================================================================
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "processed"
    / "04_predictor_harmonization"
    / "02B_final_feature_set.csv"
)

VITD_PATH = (
    PROJECT_ROOT
    / "processed"
    / "05_analytic_cohorts"
    / "vitamin_d"
    / "NHANES_vitaminD_primary_multicycle.csv"
)

IRON_OBSERVED_PATH = (
    PROJECT_ROOT
    / "processed"
    / "05_analytic_cohorts"
    / "iron"
    / "observed"
    / "NHANES_iron_observed_multicycle.csv"
)

IRON_MATCHED_PATH = (
    PROJECT_ROOT
    / "processed"
    / "05_analytic_cohorts"
    / "iron"
    / "matched"
    / "NHANES_iron_matched_multicycle.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed"
    / "07_model_design"
)

LOG_DIR = (
    PROJECT_ROOT
    / "processed"
    / "logs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. LOGGING
# =============================================================================

LOG_FILE = (
    LOG_DIR
    / "stage05_locked_modeling_design.log"
)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            mode="w",
            encoding="utf-8",
        ),
        logging.StreamHandler(
            sys.stdout
        ),
    ],
)

logger = logging.getLogger(__name__)


# =============================================================================
# 3. CYCLES
# =============================================================================

VITD_CYCLES = [
    "2009_2010",
    "2011_2012",
    "2013_2014",
    "2015_2016",
    "2017_2018",
    "2021_2023",
]

IRON_CYCLES = [
    "2009_2010",
    "2015_2016",
    "2017_2018",
    "2021_2023",
]

CYCLE_ORDER = {
    "2009_2010": 1,
    "2011_2012": 2,
    "2013_2014": 3,
    "2015_2016": 4,
    "2017_2018": 5,
    "2021_2023": 6,
}

CYCLE_LABEL = {
    "2009_2010": "2009–2010",
    "2011_2012": "2011–2012",
    "2013_2014": "2013–2014",
    "2015_2016": "2015–2016",
    "2017_2018": "2017–2018",
    "2021_2023": "2021–2023",
}


# =============================================================================
# 4. OUTCOME CONFIGURATION
# =============================================================================

OUTCOME_CONFIG = {
    "vitamin_d": {
        "population":
            "primary",
        "primary_outcome":
            "outcome_vitd_lt50",
        "sensitivity_outcome":
            "outcome_vitd_lt30",
        "cycles":
            VITD_CYCLES,
    },

    "iron_observed": {
        "population":
            "observed",
        "primary_outcome":
            "outcome_iron_lt15",
        "sensitivity_outcome":
            "outcome_iron_lt25",
        "cycles":
            IRON_CYCLES,
    },

    "iron_matched": {
        "population":
            "matched",
        "primary_outcome":
            "outcome_iron_lt15",
        "sensitivity_outcome":
            "outcome_iron_lt25",
        "cycles":
            IRON_CYCLES,
    },
}


# =============================================================================
# 5. FEATURE TYPES
# =============================================================================

CATEGORICAL_FEATURES = {
    "sex",
    "race_ethnicity_5cat",
    "education_adult_5cat",
    "health_insurance",
    "has_routine_place_for_care",
    "adult_food_security_4cat",
    "smoking_status_3cat",
    "doctor_diagnosed_diabetes",
    "doctor_diagnosed_hypertension",
}


# =============================================================================
# 6. RANDOM SEEDS
# =============================================================================

RANDOM_SEEDS = [
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

PRIMARY_TUNING_SEED = 1107


# =============================================================================
# 7. CV / TUNING DESIGN
# =============================================================================

CV_FOLDS = 5

HYPERPARAMETER_SEARCH_ITERATIONS = {
    "logistic_regression": 1,
    "elastic_net_logistic": 25,
    "random_forest": 30,
    "xgboost": 40,
    "lightgbm": 40,
    "catboost": 40,
}


# =============================================================================
# 8. MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY = {

    "logistic_regression": {
        "family": "linear",
        "primary": True,
        "calibration_allowed": True,
        "scaling_required": True,
        "categorical_strategy":
            "one_hot",
        "class_weight_strategy":
            "balanced",
    },

    "elastic_net_logistic": {
        "family": "linear",
        "primary": True,
        "calibration_allowed": True,
        "scaling_required": True,
        "categorical_strategy":
            "one_hot",
        "class_weight_strategy":
            "balanced",
    },

    "random_forest": {
        "family": "tree",
        "primary": True,
        "calibration_allowed": True,
        "scaling_required": False,
        "categorical_strategy":
            "one_hot",
        "class_weight_strategy":
            "balanced",
    },

    "xgboost": {
        "family": "boosting",
        "primary": True,
        "calibration_allowed": True,
        "scaling_required": False,
        "categorical_strategy":
            "one_hot",
        "class_weight_strategy":
            "scale_pos_weight_training_only",
    },

    "lightgbm": {
        "family": "boosting",
        "primary": True,
        "calibration_allowed": True,
        "scaling_required": False,
        "categorical_strategy":
            "one_hot",
        "class_weight_strategy":
            "balanced_or_scale_pos_weight_training_only",
    },

    "catboost": {
        "family": "boosting",
        "primary": True,
        "calibration_allowed": True,
        "scaling_required": False,
        "categorical_strategy":
            "native_or_locked_encoded",
        "class_weight_strategy":
            "training_only",
    },
}


# =============================================================================
# 9. HYPERPARAMETER SEARCH SPACES
# =============================================================================
#
# These are intentionally modest.
#
# We want comparable tuning budgets rather than an architecture contest.
#
# =============================================================================

HYPERPARAMETER_SPACES = {

    "logistic_regression": {
        "C": [1.0],
        "penalty": ["l2"],
        "solver": ["liblinear"],
    },

    "elastic_net_logistic": {
        "C": [
            0.01,
            0.03,
            0.1,
            0.3,
            1.0,
            3.0,
            10.0,
        ],
        "l1_ratio": [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        ],
        "solver": [
            "saga",
        ],
    },

    "random_forest": {
        "n_estimators": [
            300,
            500,
            800,
        ],
        "max_depth": [
            None,
            5,
            10,
            20,
        ],
        "min_samples_split": [
            2,
            5,
            10,
        ],
        "min_samples_leaf": [
            1,
            2,
            5,
            10,
        ],
        "max_features": [
            "sqrt",
            "log2",
            0.5,
        ],
    },

    "xgboost": {
        "n_estimators": [
            200,
            400,
            700,
        ],
        "max_depth": [
            2,
            3,
            4,
            6,
        ],
        "learning_rate": [
            0.01,
            0.03,
            0.05,
            0.1,
        ],
        "subsample": [
            0.7,
            0.85,
            1.0,
        ],
        "colsample_bytree": [
            0.7,
            0.85,
            1.0,
        ],
        "min_child_weight": [
            1,
            3,
            5,
        ],
        "reg_lambda": [
            1.0,
            3.0,
            10.0,
        ],
        "reg_alpha": [
            0.0,
            0.1,
            1.0,
        ],
    },

    "lightgbm": {
        "n_estimators": [
            200,
            400,
            700,
        ],
        "num_leaves": [
            7,
            15,
            31,
            63,
        ],
        "max_depth": [
            -1,
            3,
            5,
            8,
        ],
        "learning_rate": [
            0.01,
            0.03,
            0.05,
            0.1,
        ],
        "subsample": [
            0.7,
            0.85,
            1.0,
        ],
        "colsample_bytree": [
            0.7,
            0.85,
            1.0,
        ],
        "reg_lambda": [
            0.0,
            1.0,
            5.0,
            10.0,
        ],
        "reg_alpha": [
            0.0,
            0.1,
            1.0,
        ],
    },

    "catboost": {
        "iterations": [
            300,
            500,
            800,
        ],
        "depth": [
            4,
            6,
            8,
        ],
        "learning_rate": [
            0.01,
            0.03,
            0.05,
            0.1,
        ],
        "l2_leaf_reg": [
            1,
            3,
            5,
            10,
        ],
        "random_strength": [
            0.0,
            0.5,
            1.0,
        ],
    },
}


# =============================================================================
# 10. CALIBRATION REGISTRY
# =============================================================================

CALIBRATION_REGISTRY = [
    {
        "calibration_method":
            "none",
        "primary":
            False,
        "description":
            "Raw model probability.",
    },

    {
        "calibration_method":
            "sigmoid",
        "primary":
            True,
        "description":
            "Platt-style logistic calibration learned from training data only.",
    },

    {
        "calibration_method":
            "isotonic",
        "primary":
            False,
        "description":
            "Non-parametric isotonic calibration; predefined sensitivity method.",
    },
]


# =============================================================================
# 11. METRIC REGISTRY
# =============================================================================

METRIC_REGISTRY = [

    # -------------------------------------------------------------------------
    # Discrimination
    # -------------------------------------------------------------------------

    {
        "metric":
            "AUROC",
        "domain":
            "discrimination",
        "primary":
            True,
        "direction":
            "higher_better",
    },

    {
        "metric":
            "AUPRC",
        "domain":
            "discrimination",
        "primary":
            True,
        "direction":
            "higher_better",
    },

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

    {
        "metric":
            "Brier",
        "domain":
            "calibration",
        "primary":
            True,
        "direction":
            "lower_better",
    },

    {
        "metric":
            "CalibrationIntercept",
        "domain":
            "calibration",
        "primary":
            True,
        "direction":
            "target_0",
    },

    {
        "metric":
            "CalibrationSlope",
        "domain":
            "calibration",
        "primary":
            True,
        "direction":
            "target_1",
    },

    {
        "metric":
            "ECE",
        "domain":
            "calibration",
        "primary":
            True,
        "direction":
            "lower_better",
    },

    # -------------------------------------------------------------------------
    # Classification threshold metrics
    # -------------------------------------------------------------------------

    {
        "metric":
            "Sensitivity",
        "domain":
            "classification",
        "primary":
            False,
        "direction":
            "higher_better",
    },

    {
        "metric":
            "Specificity",
        "domain":
            "classification",
        "primary":
            False,
        "direction":
            "higher_better",
    },

    {
        "metric":
            "PPV",
        "domain":
            "classification",
        "primary":
            False,
        "direction":
            "higher_better",
    },

    {
        "metric":
            "NPV",
        "domain":
            "classification",
        "primary":
            False,
        "direction":
            "higher_better",
    },

    {
        "metric":
            "F1",
        "domain":
            "classification",
        "primary":
            False,
        "direction":
            "higher_better",
    },

    # -------------------------------------------------------------------------
    # Resource-oriented endpoints
    # -------------------------------------------------------------------------

    {
        "metric":
            "ReferralRate",
        "domain":
            "testing_burden",
        "primary":
            True,
        "direction":
            "context",
    },

    {
        "metric":
            "TestsAvoided",
        "domain":
            "testing_burden",
        "primary":
            True,
        "direction":
            "higher_better_given_sensitivity_target",
    },

    {
        "metric":
            "CasesDetected",
        "domain":
            "testing_burden",
        "primary":
            True,
        "direction":
            "higher_better",
    },

    {
        "metric":
            "CasesMissed",
        "domain":
            "testing_burden",
        "primary":
            True,
        "direction":
            "lower_better",
    },

    {
        "metric":
            "TestsPerDetectedCase",
        "domain":
            "testing_burden",
        "primary":
            True,
        "direction":
            "lower_better",
    },

    {
        "metric":
            "NetBenefit",
        "domain":
            "decision_curve",
        "primary":
            True,
        "direction":
            "higher_better",
    },
]


# =============================================================================
# 12. SENSITIVITY TARGETS
# =============================================================================

SENSITIVITY_TARGETS = [
    0.90,
    0.95,
    0.975,
]


# =============================================================================
# 13. LOAD HELPERS
# =============================================================================

def load_csv(
    path: Path,
    label: str,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    logger.info(
        f"{label}: {df.shape[0]:,} rows × {df.shape[1]:,} columns"
    )

    return df


# =============================================================================
# 14. FEATURE BLOCKS
# =============================================================================

def build_locked_feature_manifest(
    manifest: pd.DataFrame,
) -> pd.DataFrame:

    manifest = (
        manifest.copy()
    )

    manifest[
        "locked_stage"
    ] = "05"

    manifest[
        "locked"
    ] = True

    manifest[
        "feature_type"
    ] = np.where(
        manifest[
            "feature"
        ].isin(
            CATEGORICAL_FEATURES
        ),
        "categorical",
        "continuous",
    )

    return manifest


def extract_feature_blocks(
    manifest: pd.DataFrame,
) -> Dict[str, List[str]]:

    tier1 = (
        manifest.loc[
            manifest[
                "inclusion_role"
            ]
            == "PRIMARY_T1",
            "feature",
        ]
        .tolist()
    )

    cbc = (
        manifest.loc[
            manifest[
                "inclusion_role"
            ]
            == "PRIMARY_T2_CBC",
            "feature",
        ]
        .tolist()
    )

    chemistry = (
        manifest.loc[
            manifest[
                "inclusion_role"
            ]
            == "T2_SENSITIVITY",
            "feature",
        ]
        .tolist()
    )

    return {
        "tier1":
            tier1,
        "tier2":
            tier1 + cbc,
        "tier2_extended":
            tier1 + cbc + chemistry,
    }


# =============================================================================
# 15. PREPROCESSING SPECIFICATION
# =============================================================================

def build_preprocessing_specification(
    locked_manifest: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, row in locked_manifest.iterrows():

        feature = (
            row[
                "feature"
            ]
        )

        role = (
            row[
                "inclusion_role"
            ]
        )

        if role not in {
            "PRIMARY_T1",
            "PRIMARY_T2_CBC",
            "T2_SENSITIVITY",
        }:
            continue

        feature_type = (
            row[
                "feature_type"
            ]
        )

        if feature_type == "continuous":

            imputation = (
                "median_fit_on_training_only"
            )

            linear_transform = (
                "standardize_fit_on_training_only"
            )

            tree_transform = (
                "no_scaling"
            )

        else:

            imputation = (
                "most_frequent_fit_on_training_only"
            )

            linear_transform = (
                "one_hot_fit_on_training_only"
            )

            tree_transform = (
                "one_hot_or_locked_native_category"
            )

        rows.append(
            {
                "feature":
                    feature,
                "feature_type":
                    feature_type,
                "inclusion_role":
                    role,
                "imputation":
                    imputation,
                "linear_model_transform":
                    linear_transform,
                "tree_model_transform":
                    tree_transform,
                "future_data_used":
                    False,
                "outcome_used":
                    False,
                "fit_scope":
                    "training_fold_only",
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 16. MODEL TABLE
# =============================================================================

def build_model_registry_table() -> pd.DataFrame:

    rows = []

    for model_name, spec in MODEL_REGISTRY.items():

        rows.append(
            {
                "model":
                    model_name,
                **spec,
                "tuning_iterations":
                    HYPERPARAMETER_SEARCH_ITERATIONS[
                        model_name
                    ],
                "cv_folds":
                    CV_FOLDS,
                "tuning_seed":
                    PRIMARY_TUNING_SEED,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 17. HYPERPARAMETER TABLE
# =============================================================================

def build_hyperparameter_table() -> pd.DataFrame:

    rows = []

    for model, parameters in HYPERPARAMETER_SPACES.items():

        for parameter, values in parameters.items():

            rows.append(
                {
                    "model":
                        model,
                    "parameter":
                        parameter,
                    "allowed_values_json":
                        json.dumps(
                            values
                        ),
                    "search_space_locked":
                        True,
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 18. TEMPORAL SPLIT MANIFEST
# =============================================================================

def add_split(
    rows: List[Dict],
    study: str,
    design: str,
    train_cycles: List[str],
    test_cycle: str,
    split_role: str,
):

    rows.append(
        {
            "study":
                study,
            "design":
                design,
            "train_cycles":
                "|".join(
                    train_cycles
                ),
            "test_cycle":
                test_cycle,
            "split_role":
                split_role,
            "n_train_cycles":
                len(
                    train_cycles
                ),
            "temporal_direction":
                (
                    "forward"
                    if max(
                        CYCLE_ORDER[
                            cycle
                        ]
                        for cycle
                        in train_cycles
                    )
                    <
                    CYCLE_ORDER[
                        test_cycle
                    ]
                    else
                    "backward"
                ),
            "locked":
                True,
        }
    )


def build_temporal_split_manifest() -> pd.DataFrame:

    rows = []

    # =========================================================================
    # VITAMIN D — FIXED ORIGIN
    # =========================================================================

    origin = (
        VITD_CYCLES[0]
    )

    for test_cycle in VITD_CYCLES[1:]:

        add_split(
            rows,
            "vitamin_d",
            "fixed_origin",
            [origin],
            test_cycle,
            "primary",
        )

    # =========================================================================
    # VITAMIN D — ROLLING ORIGIN
    # =========================================================================
    #
    # Expanding training set.
    #
    # Example:
    #   F -> G
    #   F+G -> H
    #   F+G+H -> I
    # =========================================================================

    for index in range(
        1,
        len(
            VITD_CYCLES
        ),
    ):

        add_split(
            rows,
            "vitamin_d",
            "rolling_expanding",
            VITD_CYCLES[
                :index
            ],
            VITD_CYCLES[
                index
            ],
            "primary",
        )

    # =========================================================================
    # VITAMIN D — J BRIDGE
    # =========================================================================

    add_split(
        rows,
        "vitamin_d",
        "bridge_J_to_I",
        ["2017_2018"],
        "2015_2016",
        "secondary",
    )

    add_split(
        rows,
        "vitamin_d",
        "bridge_J_to_L",
        ["2017_2018"],
        "2021_2023",
        "secondary",
    )

    # =========================================================================
    # IRON — BOTH OBSERVED AND MATCHED
    # =========================================================================

    for study in [
        "iron_observed",
        "iron_matched",
    ]:

        origin = (
            IRON_CYCLES[0]
        )

        # Fixed-origin
        for test_cycle in IRON_CYCLES[1:]:

            add_split(
                rows,
                study,
                "fixed_origin",
                [origin],
                test_cycle,
                "primary",
            )

        # Rolling expanding
        for index in range(
            1,
            len(
                IRON_CYCLES
            ),
        ):

            add_split(
                rows,
                study,
                "rolling_expanding",
                IRON_CYCLES[
                    :index
                ],
                IRON_CYCLES[
                    index
                ],
                "primary",
            )

        # Original J bridge
        add_split(
            rows,
            study,
            "bridge_J_to_I",
            ["2017_2018"],
            "2015_2016",
            "secondary",
        )

        add_split(
            rows,
            study,
            "bridge_J_to_L",
            ["2017_2018"],
            "2021_2023",
            "secondary",
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 19. EXPERIMENT MANIFEST
# =============================================================================

def build_experiment_manifest(
    split_manifest: pd.DataFrame,
    feature_blocks: Dict[str, List[str]],
) -> pd.DataFrame:

    rows = []

    experiment_counter = 1

    tier_map = {
        "T1":
            feature_blocks[
                "tier1"
            ],

        "T2":
            feature_blocks[
                "tier2"
            ],
    }

    for _, split in split_manifest.iterrows():

        study = (
            split[
                "study"
            ]
        )

        outcome = (
            OUTCOME_CONFIG[
                study
            ][
                "primary_outcome"
            ]
        )

        for tier_name, tier_features in tier_map.items():

            for model_name in MODEL_REGISTRY.keys():

                experiment_id = (
                    f"EXP_{experiment_counter:04d}"
                )

                rows.append(
                    {
                        "experiment_id":
                            experiment_id,

                        "study":
                            study,

                        "population":
                            OUTCOME_CONFIG[
                                study
                            ][
                                "population"
                            ],

                        "outcome":
                            outcome,

                        "outcome_role":
                            "primary",

                        "temporal_design":
                            split[
                                "design"
                            ],

                        "split_role":
                            split[
                                "split_role"
                            ],

                        "train_cycles":
                            split[
                                "train_cycles"
                            ],

                        "test_cycle":
                            split[
                                "test_cycle"
                            ],

                        "feature_tier":
                            tier_name,

                        "n_features":
                            len(
                                tier_features
                            ),

                        "feature_list":
                            "|".join(
                                tier_features
                            ),

                        "model":
                            model_name,

                        "primary_calibration":
                            "sigmoid",

                        "calibration_sensitivity":
                            "none|isotonic",

                        "cv_folds":
                            CV_FOLDS,

                        "tuning_iterations":
                            HYPERPARAMETER_SEARCH_ITERATIONS[
                                model_name
                            ],

                        "n_random_seeds":
                            len(
                                RANDOM_SEEDS
                            ),

                        "bootstrap_ci":
                            True,

                        "bootstrap_iterations":
                            1000,

                        "locked_before_modeling":
                            True,
                    }
                )

                experiment_counter += 1

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 20. SENSITIVITY EXPERIMENT MANIFEST
# =============================================================================

def build_sensitivity_experiments(
    feature_blocks: Dict[str, List[str]],
) -> pd.DataFrame:

    rows = []

    experiment_id = 1

    # -------------------------------------------------------------------------
    # Outcome-threshold sensitivity
    # -------------------------------------------------------------------------

    for study, config in OUTCOME_CONFIG.items():

        rows.append(
            {
                "sensitivity_id":
                    f"SENS_{experiment_id:03d}",
                "study":
                    study,
                "sensitivity_type":
                    "outcome_threshold",
                "primary_definition":
                    config[
                        "primary_outcome"
                    ],
                "alternative_definition":
                    config[
                        "sensitivity_outcome"
                    ],
                "scope":
                    "all_primary_temporal_experiments",
            }
        )

        experiment_id += 1

    # -------------------------------------------------------------------------
    # Tier-2 chemistry extension
    # -------------------------------------------------------------------------

    for study in OUTCOME_CONFIG.keys():

        rows.append(
            {
                "sensitivity_id":
                    f"SENS_{experiment_id:03d}",
                "study":
                    study,
                "sensitivity_type":
                    "expanded_routine_labs",
                "primary_definition":
                    "Tier2_CBC",
                "alternative_definition":
                    "Tier2_CBC_plus_albumin_creatinine_glucose",
                "scope":
                    "selected_primary_models",
            }
        )

        experiment_id += 1

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 21. RANDOM SEED TABLE
# =============================================================================

def build_seed_registry() -> pd.DataFrame:

    rows = []

    for index, seed in enumerate(
        RANDOM_SEEDS,
        start=1,
    ):

        rows.append(
            {
                "seed_number":
                    index,
                "random_seed":
                    seed,
                "purpose":
                    (
                        "Repeated model fitting / "
                        "stochastic algorithm stability"
                    ),
                "locked":
                    True,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 22. CALIBRATION TABLE
# =============================================================================

def build_calibration_table() -> pd.DataFrame:

    rows = []

    for item in CALIBRATION_REGISTRY:

        rows.append(
            {
                **item,
                "fit_scope":
                    "training_data_only",
                "test_labels_used":
                    False,
                "locked":
                    True,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 23. METRIC TABLE
# =============================================================================

def build_metric_table() -> pd.DataFrame:

    df = pd.DataFrame(
        METRIC_REGISTRY
    )

    df[
        "locked"
    ] = True

    return df


# =============================================================================
# 24. MODELING GUARDRAILS
# =============================================================================

def build_guardrails() -> pd.DataFrame:

    rules = [

        (
            "G001",
            "No biomarker leakage",
            "Ferritin and 25(OH)D must never appear in predictor matrices.",
        ),

        (
            "G002",
            "Training-only preprocessing",
            "Imputation, scaling and encoding must be fit only on training data.",
        ),

        (
            "G003",
            "Training-only class weighting",
            "Class weights or scale_pos_weight must be calculated using training labels only.",
        ),

        (
            "G004",
            "No test-driven tuning",
            "Temporal test cycles must never participate in hyperparameter selection.",
        ),

        (
            "G005",
            "No test-driven calibration",
            "Calibration functions must be estimated without using temporal test labels.",
        ),

        (
            "G006",
            "No post-hoc predictor addition",
            "Primary predictors are frozen from Stage 02B before model results.",
        ),

        (
            "G007",
            "No post-hoc model deletion",
            "All predefined primary model families must be reported.",
        ),

        (
            "G008",
            "Matched iron retained",
            "Observed and matched iron results must be analyzed in parallel.",
        ),

        (
            "G009",
            "P remains supplementary",
            "2017–March 2020 must not be treated as an independent sequential cycle alongside J.",
        ),

        (
            "G010",
            "Multiple seeds",
            "Stochastic algorithms must be evaluated across all locked seeds.",
        ),

        (
            "G011",
            "Bootstrap uncertainty",
            "Primary test metrics require 95% bootstrap confidence intervals.",
        ),

        (
            "G012",
            "Calibration reported",
            "Discrimination alone is insufficient; Brier, calibration slope/intercept and ECE are required.",
        ),

        (
            "G013",
            "Resource endpoints reported",
            "Testing burden must be evaluated at predefined sensitivity targets.",
        ),

        (
            "G014",
            "Survey weights separate from primary ML objective",
            "Primary predictive training remains participant-level unless a later locked sensitivity analysis explicitly adds survey weighting.",
        ),
    ]

    return pd.DataFrame(
        [
            {
                "guardrail_id":
                    rule_id,
                "name":
                    name,
                "rule":
                    rule,
                "status":
                    "LOCKED",
            }
            for (
                rule_id,
                name,
                rule,
            ) in rules
        ]
    )


# =============================================================================
# 25. DATA VALIDATION
# =============================================================================

def validate_input_dataset(
    df: pd.DataFrame,
    study: str,
    feature_blocks: Dict[str, List[str]],
):

    required = {
        "SEQN",
        "survey_cycle",
        "primary_outcome",
    }

    missing_required = (
        required
        - set(
            df.columns
        )
    )

    if missing_required:

        raise RuntimeError(
            f"{study}: missing required columns: "
            f"{sorted(missing_required)}"
        )

    primary_features = (
        feature_blocks[
            "tier2"
        ]
    )

    missing_features = [
        feature
        for feature
        in primary_features
        if feature
        not in df.columns
    ]

    if missing_features:

        raise RuntimeError(
            f"{study}: missing frozen primary features:\n"
            f"{missing_features}"
        )

    if df[
        [
            "survey_cycle",
            "SEQN",
        ]
    ].duplicated().any():

        raise RuntimeError(
            f"{study}: duplicate cycle-SEQN detected."
        )

    expected_outcome = (
        OUTCOME_CONFIG[
            study
        ][
            "primary_outcome"
        ]
    )

    if expected_outcome not in df.columns:

        raise RuntimeError(
            f"{study}: expected outcome "
            f"{expected_outcome} missing."
        )

    # -------------------------------------------------------------------------
    # Biomarker leakage check
    # -------------------------------------------------------------------------

    forbidden_tokens = [
        "ferritin_ug_L",
        "vitamin_d_nmol_L",
        "LBXFER",
        "LBXVIDMS",
    ]

    for feature in primary_features:

        if any(
            token.lower()
            in feature.lower()
            for token
            in forbidden_tokens
        ):

            raise RuntimeError(
                f"Biomarker leakage detected: {feature}"
            )


# =============================================================================
# 26. EXPERIMENT SAMPLE SIZE AUDIT
# =============================================================================

def build_experiment_sample_size_audit(
    experiment_manifest: pd.DataFrame,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:

    rows = []

    for _, experiment in experiment_manifest.iterrows():

        study = (
            experiment[
                "study"
            ]
        )

        df = (
            datasets[
                study
            ]
        )

        train_cycles = (
            str(
                experiment[
                    "train_cycles"
                ]
            )
            .split("|")
        )

        test_cycle = (
            experiment[
                "test_cycle"
            ]
        )

        outcome = (
            experiment[
                "outcome"
            ]
        )

        train = (
            df[
                df[
                    "survey_cycle"
                ]
                .isin(
                    train_cycles
                )
            ]
        )

        test = (
            df[
                df[
                    "survey_cycle"
                ]
                == test_cycle
            ]
        )

        train_y = (
            train[
                outcome
            ]
            if outcome
            in train.columns
            else pd.Series(
                dtype=float
            )
        )

        test_y = (
            test[
                outcome
            ]
            if outcome
            in test.columns
            else pd.Series(
                dtype=float
            )
        )

        rows.append(
            {
                "experiment_id":
                    experiment[
                        "experiment_id"
                    ],

                "study":
                    study,

                "train_cycles":
                    experiment[
                        "train_cycles"
                    ],

                "test_cycle":
                    test_cycle,

                "outcome":
                    outcome,

                "n_train":
                    len(
                        train
                    ),

                "train_cases":
                    int(
                        train_y.sum()
                    )
                    if len(
                        train_y
                    ) > 0
                    else 0,

                "train_case_pct":
                    (
                        100
                        * train_y.mean()
                        if len(
                            train_y
                        ) > 0
                        else np.nan
                    ),

                "n_test":
                    len(
                        test
                    ),

                "test_cases":
                    int(
                        test_y.sum()
                    )
                    if len(
                        test_y
                    ) > 0
                    else 0,

                "test_case_pct":
                    (
                        100
                        * test_y.mean()
                        if len(
                            test_y
                        ) > 0
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 27. DECISION LEDGER
# =============================================================================

def build_decision_ledger() -> pd.DataFrame:

    today = (
        datetime.now()
        .date()
        .isoformat()
    )

    decisions = [

        {
            "decision_id":
                "D040",
            "stage":
                "05",
            "topic":
                "Model suite",
            "decision":
                (
                    "Primary model suite is LR, Elastic-Net LR, RF, "
                    "XGBoost, LightGBM and CatBoost."
                ),
            "reason":
                (
                    "Provides interpretable linear baselines and strong "
                    "tabular nonlinear comparators without architecture sprawl."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id":
                "D041",
            "stage":
                "05",
            "topic":
                "Primary resource tiers",
            "decision":
                "Primary comparisons use Tier 1 and Tier 2 CBC.",
            "reason":
                (
                    "Directly tests the incremental value of routine CBC."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id":
                "D042",
            "stage":
                "05",
            "topic":
                "Chemistry extension",
            "decision":
                (
                    "Albumin, creatinine and glucose remain a predefined "
                    "sensitivity extension only."
                ),
            "reason":
                (
                    "Preserves the primary CBC-centered resource hierarchy."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id":
                "D043",
            "stage":
                "05",
            "topic":
                "Preprocessing",
            "decision":
                (
                    "All imputation, encoding and scaling parameters are "
                    "fit within training data only."
                ),
            "reason":
                "Prevents temporal leakage.",
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id":
                "D044",
            "stage":
                "05",
            "topic":
                "Hyperparameter tuning",
            "decision":
                (
                    "Hyperparameters are selected by 5-fold CV entirely "
                    "within the training cohort."
                ),
            "reason":
                (
                    "Temporal test cohorts remain locked external evaluations."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id":
                "D045",
            "stage":
                "05",
            "topic":
                "Temporal experiments",
            "decision":
                (
                    "Fixed-origin, rolling-expanding and J-to-I/J-to-L "
                    "bridge experiments are predefined."
                ),
            "reason":
                (
                    "Captures long-range, local and original-study temporal transport."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id":
                "D046",
            "stage":
                "05",
            "topic":
                "Random seeds",
            "decision":
                "Ten predefined seeds are used.",
            "reason":
                (
                    "Quantifies stochastic model variation."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id":
                "D047",
            "stage":
                "05",
            "topic":
                "Calibration",
            "decision":
                (
                    "Sigmoid calibration is primary; uncalibrated and "
                    "isotonic outputs are predefined sensitivity analyses."
                ),
            "reason":
                (
                    "Calibration reliability is a central deployment outcome."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id":
                "D048",
            "stage":
                "05",
            "topic":
                "Evaluation metrics",
            "decision":
                (
                    "AUROC, AUPRC, Brier, calibration slope/intercept, ECE "
                    "and operational testing-burden endpoints are locked."
                ),
            "reason":
                (
                    "Discrimination alone is insufficient for clinical triage."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id":
                "D049",
            "stage":
                "05",
            "topic":
                "Sensitivity targets",
            "decision":
                (
                    "Testing-burden endpoints are evaluated at sensitivity "
                    "targets 90%, 95% and 97.5%."
                ),
            "reason":
                (
                    "Allows clinically interpretable comparison of targeted "
                    "confirmatory testing strategies."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id":
                "D050",
            "stage":
                "05",
            "topic":
                "Iron natural experiment",
            "decision":
                (
                    "All key temporal iron experiments are repeated in "
                    "observed and matched cohorts."
                ),
            "reason":
                (
                    "Tests whether apparent model drift attenuates when "
                    "eligibility/population shift is controlled."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },
    ]

    return pd.DataFrame(
        decisions
    )


# =============================================================================
# 28. MAIN
# =============================================================================

def main():

    logger.info(
        "=" * 88
    )

    logger.info(
        "STAGE 05 STARTED"
    )

    logger.info(
        "=" * 88
    )

    # =========================================================================
    # LOAD
    # =========================================================================

    raw_manifest = (
        load_csv(
            FEATURE_MANIFEST_PATH,
            "Stage-02B feature manifest",
        )
    )

    vitamin_d = (
        load_csv(
            VITD_PATH,
            "Vitamin-D analytic master",
        )
    )

    iron_observed = (
        load_csv(
            IRON_OBSERVED_PATH,
            "Iron observed analytic master",
        )
    )

    iron_matched = (
        load_csv(
            IRON_MATCHED_PATH,
            "Iron matched analytic master",
        )
    )

    # =========================================================================
    # FEATURE MANIFEST
    # =========================================================================

    locked_feature_manifest = (
        build_locked_feature_manifest(
            raw_manifest
        )
    )

    feature_blocks = (
        extract_feature_blocks(
            locked_feature_manifest
        )
    )

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    datasets = {
        "vitamin_d":
            vitamin_d,

        "iron_observed":
            iron_observed,

        "iron_matched":
            iron_matched,
    }

    for study, df in datasets.items():

        validate_input_dataset(
            df=df,
            study=study,
            feature_blocks=
                feature_blocks,
        )

    # =========================================================================
    # DESIGN TABLES
    # =========================================================================

    preprocessing_df = (
        build_preprocessing_specification(
            locked_feature_manifest
        )
    )

    model_registry_df = (
        build_model_registry_table()
    )

    hyperparameter_df = (
        build_hyperparameter_table()
    )

    temporal_split_df = (
        build_temporal_split_manifest()
    )

    experiment_manifest_df = (
        build_experiment_manifest(
            temporal_split_df,
            feature_blocks,
        )
    )

    sensitivity_df = (
        build_sensitivity_experiments(
            feature_blocks
        )
    )

    seed_registry_df = (
        build_seed_registry()
    )

    calibration_df = (
        build_calibration_table()
    )

    metric_df = (
        build_metric_table()
    )

    guardrails_df = (
        build_guardrails()
    )

    decision_ledger_df = (
        build_decision_ledger()
    )

    sample_size_audit_df = (
        build_experiment_sample_size_audit(
            experiment_manifest=
                experiment_manifest_df,
            datasets=
                datasets,
        )
    )

    # =========================================================================
    # VALIDATE TEMPORAL SPLITS
    # =========================================================================

    invalid_overlap = []

    for _, row in temporal_split_df.iterrows():

        train_cycles = (
            str(
                row[
                    "train_cycles"
                ]
            )
            .split("|")
        )

        test_cycle = (
            row[
                "test_cycle"
            ]
        )

        if (
            test_cycle
            in train_cycles
        ):

            invalid_overlap.append(
                row.to_dict()
            )

    if invalid_overlap:

        raise RuntimeError(
            "Training/test cycle overlap detected."
        )

    # =========================================================================
    # SAVE CSVs
    # =========================================================================

    outputs = {

        "05_feature_manifest_locked.csv":
            locked_feature_manifest,

        "05_preprocessing_specification.csv":
            preprocessing_df,

        "05_model_registry.csv":
            model_registry_df,

        "05_hyperparameter_search_space.csv":
            hyperparameter_df,

        "05_temporal_split_manifest.csv":
            temporal_split_df,

        "05_experiment_manifest.csv":
            experiment_manifest_df,

        "05_sensitivity_experiment_manifest.csv":
            sensitivity_df,

        "05_metric_registry.csv":
            metric_df,

        "05_calibration_registry.csv":
            calibration_df,

        "05_random_seed_registry.csv":
            seed_registry_df,

        "05_modeling_guardrails.csv":
            guardrails_df,

        "05_experiment_sample_size_audit.csv":
            sample_size_audit_df,

        "decision_ledger_stage05.csv":
            decision_ledger_df,
    }

    for filename, df in outputs.items():

        df.to_csv(
            OUTPUT_DIR
            / filename,
            index=False,
        )

    # =========================================================================
    # EXCEL WORKBOOK
    # =========================================================================

    workbook_path = (
        OUTPUT_DIR
        / "05_stage05_summary.xlsx"
    )

    with pd.ExcelWriter(
        workbook_path,
        engine="openpyxl",
    ) as writer:

        locked_feature_manifest.to_excel(
            writer,
            sheet_name="Locked_Features",
            index=False,
        )

        preprocessing_df.to_excel(
            writer,
            sheet_name="Preprocessing",
            index=False,
        )

        model_registry_df.to_excel(
            writer,
            sheet_name="Models",
            index=False,
        )

        hyperparameter_df.to_excel(
            writer,
            sheet_name="Hyperparameters",
            index=False,
        )

        temporal_split_df.to_excel(
            writer,
            sheet_name="Temporal_Splits",
            index=False,
        )

        experiment_manifest_df.to_excel(
            writer,
            sheet_name="Experiments",
            index=False,
        )

        sample_size_audit_df.to_excel(
            writer,
            sheet_name="Experiment_Samples",
            index=False,
        )

        sensitivity_df.to_excel(
            writer,
            sheet_name="Sensitivity",
            index=False,
        )

        metric_df.to_excel(
            writer,
            sheet_name="Metrics",
            index=False,
        )

        calibration_df.to_excel(
            writer,
            sheet_name="Calibration",
            index=False,
        )

        seed_registry_df.to_excel(
            writer,
            sheet_name="Seeds",
            index=False,
        )

        guardrails_df.to_excel(
            writer,
            sheet_name="Guardrails",
            index=False,
        )

        decision_ledger_df.to_excel(
            writer,
            sheet_name="Decision_Ledger",
            index=False,
        )

    # =========================================================================
    # CONSOLE SUMMARY
    # =========================================================================

    print("\n")
    print(
        "=" * 90
    )
    print(
        "STAGE 05 COMPLETE"
    )
    print(
        "=" * 90
    )

    print(
        "\nLOCKED FEATURE TIERS"
    )
    print(
        "-" * 90
    )

    print(
        f"Tier 1: "
        f"{len(feature_blocks['tier1'])} features"
    )

    print(
        f"Tier 2 primary: "
        f"{len(feature_blocks['tier2'])} features"
    )

    print(
        f"Tier 2 extended sensitivity: "
        f"{len(feature_blocks['tier2_extended'])} features"
    )

    print(
        "\nMODEL SUITE"
    )
    print(
        "-" * 90
    )

    for model in MODEL_REGISTRY:

        print(
            f"  • {model}"
        )

    print(
        "\nTEMPORAL SPLITS"
    )
    print(
        "-" * 90
    )

    print(
        temporal_split_df[
            [
                "study",
                "design",
                "train_cycles",
                "test_cycle",
                "split_role",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nEXPERIMENT COUNT"
    )
    print(
        "-" * 90
    )

    print(
        f"Primary locked experiments: "
        f"{len(experiment_manifest_df):,}"
    )

    print(
        f"Sensitivity design entries: "
        f"{len(sensitivity_df):,}"
    )

    print(
        "\nSAMPLE SIZE RANGE"
    )
    print(
        "-" * 90
    )

    print(
        f"Training N: "
        f"{sample_size_audit_df['n_train'].min():,}"
        f"–"
        f"{sample_size_audit_df['n_train'].max():,}"
    )

    print(
        f"Test N: "
        f"{sample_size_audit_df['n_test'].min():,}"
        f"–"
        f"{sample_size_audit_df['n_test'].max():,}"
    )

    print(
        "\nPRIMARY WORKBOOK:"
    )

    print(
        f"  {workbook_path}"
    )

    print(
        "\nSTAGE 05 STATUS:"
    )

    print(
        "  • Feature sets frozen."
    )

    print(
        "  • Models frozen."
    )

    print(
        "  • Hyperparameter search spaces frozen."
    )

    print(
        "  • Temporal splits frozen."
    )

    print(
        "  • Seeds frozen."
    )

    print(
        "  • Metrics frozen."
    )

    print(
        "  • Calibration strategy frozen."
    )

    print(
        "  • No predictive model has yet been fitted."
    )

    print(
        "\nNEXT:"
    )

    print(
        "  Stage 06 — Leakage-Safe Model Training, Nested Tuning, "
        "Temporal Transport Evaluation, and Calibration."
    )

    print(
        "\n" + "=" * 90
    )


# =============================================================================
# 29. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()