"""
===============================================================================
STAGE 06
Leakage-Safe Model Training, Hyperparameter Tuning,
Temporal Transport Evaluation, and Calibration
===============================================================================

===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import time
import traceback
import warnings

from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# =============================================================================
# 1. OPTIONAL ML LIBRARIES
# =============================================================================

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False


try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False


# =============================================================================
# 2. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAGE03_DIR = (
    PROJECT_ROOT
    / "processed"
    / "05_analytic_cohorts"
)

STAGE05_DIR = (
    PROJECT_ROOT
    / "processed"
    / "07_model_design"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed"
    / "08_model_results"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "checkpoints"
)

TUNING_CHECKPOINT_DIR = (
    CHECKPOINT_DIR
    / "tuning"
)

TEST_PREDICTION_DIR = (
    CHECKPOINT_DIR
    / "seed_test_predictions"
)

OOF_PREDICTION_DIR = (
    CHECKPOINT_DIR
    / "seed_oof_predictions"
)

LOG_DIR = (
    PROJECT_ROOT
    / "processed"
    / "logs"
)


for folder in [
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    TUNING_CHECKPOINT_DIR,
    TEST_PREDICTION_DIR,
    OOF_PREDICTION_DIR,
    LOG_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# 3. INPUT FILES
# =============================================================================

DATA_PATHS = {

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


EXPERIMENT_MANIFEST_PATH = (
    STAGE05_DIR
    / "05_experiment_manifest.csv"
)

SEED_REGISTRY_PATH = (
    STAGE05_DIR
    / "05_random_seed_registry.csv"
)

MODEL_REGISTRY_PATH = (
    STAGE05_DIR
    / "05_model_registry.csv"
)

FEATURE_MANIFEST_PATH = (
    STAGE05_DIR
    / "05_feature_manifest_locked.csv"
)

HYPERPARAMETER_PATH = (
    STAGE05_DIR
    / "05_hyperparameter_search_space.csv"
)


# =============================================================================
# 4. OUTPUT FILES
# =============================================================================

TUNING_CACHE_PATH = (
    CHECKPOINT_DIR
    / "tuning_cache.json"
)

TUNING_RESULTS_PATH = (
    OUTPUT_DIR
    / "06_tuning_results.csv"
)

SEED_METRICS_PATH = (
    OUTPUT_DIR
    / "06_seed_metrics.csv"
)

EXPERIMENT_METRICS_PATH = (
    OUTPUT_DIR
    / "06_experiment_metrics.csv"
)

BOOTSTRAP_PATH = (
    OUTPUT_DIR
    / "06_bootstrap_ci.csv"
)

TESTING_BURDEN_PATH = (
    OUTPUT_DIR
    / "06_testing_burden.csv"
)

DCA_PATH = (
    OUTPUT_DIR
    / "06_decision_curve.csv"
)

STATUS_PATH = (
    OUTPUT_DIR
    / "06_execution_status.csv"
)

DECISION_LEDGER_PATH = (
    OUTPUT_DIR
    / "decision_ledger_stage06.csv"
)

WORKBOOK_PATH = (
    OUTPUT_DIR
    / "06_stage06_summary.xlsx"
)


# =============================================================================
# 5. LOGGING
# =============================================================================

LOG_FILE = (
    LOG_DIR
    / "stage06_model_training.log"
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
            mode="a",
            encoding="utf-8",
        ),
        logging.StreamHandler(
            sys.stdout
        ),
    ],
)

logger = logging.getLogger(__name__)


# =============================================================================
# 6. GLOBAL DESIGN CONSTANTS
# =============================================================================

CV_FOLDS = 5

CALIBRATION_FOLDS = 5

TUNING_SCORING = "average_precision"

PRIMARY_CALIBRATION = "sigmoid"

CALIBRATION_METHODS = [
    "raw",
    "sigmoid",
    "isotonic",
]

SENSITIVITY_TARGETS = [
    0.90,
    0.95,
    0.975,
]

ECE_BINS = 10

BOOTSTRAP_ITERATIONS = 1000

BOOTSTRAP_ALPHA = 0.05

DCA_THRESHOLDS = np.arange(
    0.01,
    0.51,
    0.01,
)

EPSILON = 1e-6


# =============================================================================
# 7. OPTIONAL EXECUTION CONTROLS
# =============================================================================
#
# Examples:
#
# Run only 5 experiments:
#
#   Windows PowerShell:
#       $env:STAGE06_MAX_EXPERIMENTS="5"
#
# Run specific experiments:
#
#       $env:STAGE06_EXPERIMENT_IDS="EXP_0001,EXP_0002"
#
# Force rerun:
#
#       $env:STAGE06_FORCE_RETRAIN="1"
#
# =============================================================================

MAX_EXPERIMENTS = int(
    os.environ.get(
        "STAGE06_MAX_EXPERIMENTS",
        "0",
    )
)

EXPERIMENT_IDS_FILTER = (
    os.environ.get(
        "STAGE06_EXPERIMENT_IDS",
        "",
    )
    .strip()
)

FORCE_RETRAIN = (
    os.environ.get(
        "STAGE06_FORCE_RETRAIN",
        "0",
    )
    == "1"
)


# =============================================================================
# 8. HELPERS
# =============================================================================

def safe_numeric(
    series: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def json_safe(
    obj: Any,
) -> Any:

    if isinstance(
        obj,
        dict,
    ):
        return {
            str(k):
                json_safe(v)
            for k, v
            in obj.items()
        }

    if isinstance(
        obj,
        list,
    ):
        return [
            json_safe(v)
            for v
            in obj
        ]

    if isinstance(
        obj,
        tuple,
    ):
        return [
            json_safe(v)
            for v
            in obj
        ]

    if isinstance(
        obj,
        (
            np.integer,
            np.int32,
            np.int64,
        ),
    ):
        return int(obj)

    if isinstance(
        obj,
        (
            np.floating,
            np.float32,
            np.float64,
        ),
    ):
        return float(obj)

    if isinstance(
        obj,
        np.bool_,
    ):
        return bool(obj)

    return obj


def stable_integer_seed(
    text: str,
) -> int:

    digest = hashlib.md5(
        text.encode(
            "utf-8"
        )
    ).hexdigest()

    return int(
        digest[:8],
        16,
    )


def append_csv(
    path: Path,
    rows: List[Dict],
):

    if not rows:
        return

    df = pd.DataFrame(
        rows
    )

    header = (
        not path.exists()
    )

    df.to_csv(
        path,
        mode="a",
        header=header,
        index=False,
    )


# =============================================================================
# 9. VALIDATE DEPENDENCIES
# =============================================================================

def validate_ml_dependencies(
    experiments: pd.DataFrame,
):

    models = set(
        experiments[
            "model"
        ].unique()
    )

    missing = []

    if (
        "xgboost" in models
        and not XGBOOST_AVAILABLE
    ):
        missing.append(
            "xgboost"
        )

    if (
        "lightgbm" in models
        and not LIGHTGBM_AVAILABLE
    ):
        missing.append(
            "lightgbm"
        )

    if (
        "catboost" in models
        and not CATBOOST_AVAILABLE
    ):
        missing.append(
            "catboost"
        )

    if missing:

        raise ImportError(
            "\nMissing required model libraries:\n\n"
            + "\n".join(
                f"  - {x}"
                for x in missing
            )
            + "\n\nInstall them with:\n\n"
            + "pip install "
            + " ".join(
                missing
            )
        )


# =============================================================================
# 10. LOAD DATA
# =============================================================================

def load_inputs():

    logger.info(
        "Loading Stage 05 design..."
    )

    experiments = pd.read_csv(
        EXPERIMENT_MANIFEST_PATH,
        low_memory=False,
    )

    seeds_df = pd.read_csv(
        SEED_REGISTRY_PATH,
        low_memory=False,
    )

    models_df = pd.read_csv(
        MODEL_REGISTRY_PATH,
        low_memory=False,
    )

    features_df = pd.read_csv(
        FEATURE_MANIFEST_PATH,
        low_memory=False,
    )

    hyper_df = pd.read_csv(
        HYPERPARAMETER_PATH,
        low_memory=False,
    )

    datasets = {}

    for study, path in DATA_PATHS.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Missing analytic dataset:\n{path}"
            )

        datasets[
            study
        ] = pd.read_csv(
            path,
            low_memory=False,
        )

        logger.info(
            f"{study}: "
            f"{datasets[study].shape}"
        )

    return (
        experiments,
        seeds_df,
        models_df,
        features_df,
        hyper_df,
        datasets,
    )


# =============================================================================
# 11. EXECUTION FILTER
# =============================================================================

def filter_experiments(
    experiments: pd.DataFrame,
) -> pd.DataFrame:

    out = experiments.copy()

    if EXPERIMENT_IDS_FILTER:

        requested = {
            x.strip()
            for x
            in EXPERIMENT_IDS_FILTER.split(",")
            if x.strip()
        }

        out = out[
            out[
                "experiment_id"
            ].isin(
                requested
            )
        ].copy()

    if MAX_EXPERIMENTS > 0:

        out = out.head(
            MAX_EXPERIMENTS
        ).copy()

    return out.reset_index(
        drop=True
    )


# =============================================================================
# 12. FEATURE TYPE RESOLUTION
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


def split_feature_types(
    features: List[str],
) -> Tuple[
    List[str],
    List[str],
]:

    categorical = [
        feature
        for feature
        in features
        if feature
        in CATEGORICAL_FEATURES
    ]

    continuous = [
        feature
        for feature
        in features
        if feature
        not in categorical
    ]

    return (
        continuous,
        categorical,
    )


# =============================================================================
# 13. ONE-HOT ENCODER VERSION COMPATIBILITY
# =============================================================================

def make_one_hot_encoder():

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )


# =============================================================================
# 14. PREPROCESSOR
# =============================================================================

def build_preprocessor(
    features: List[str],
    scale_continuous: bool,
) -> ColumnTransformer:

    continuous, categorical = (
        split_feature_types(
            features
        )
    )

    continuous_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        )
    ]

    if scale_continuous:

        continuous_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    continuous_pipeline = Pipeline(
        steps=
            continuous_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                make_one_hot_encoder(),
            ),
        ]
    )

    transformers = []

    if continuous:

        transformers.append(
            (
                "continuous",
                continuous_pipeline,
                continuous,
            )
        )

    if categorical:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical,
            )
        )

    return ColumnTransformer(
        transformers=
            transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )


# =============================================================================
# 15. CLASS BALANCE
# =============================================================================

def positive_class_ratio(
    y: pd.Series,
) -> float:

    y = np.asarray(
        y,
        dtype=int,
    )

    positives = int(
        np.sum(
            y == 1
        )
    )

    negatives = int(
        np.sum(
            y == 0
        )
    )

    if positives == 0:
        return 1.0

    return (
        negatives
        / positives
    )


# =============================================================================
# 16. ESTIMATOR FACTORY
# =============================================================================

def build_estimator(
    model_name: str,
    seed: int,
    y_training: pd.Series,
):

    ratio = positive_class_ratio(
        y_training
    )

    if model_name == "logistic_regression":

        return LogisticRegression(
            class_weight="balanced",
            random_state=seed,
            max_iter=5000,
            solver="liblinear",
        )

    if model_name == "elastic_net_logistic":

        return LogisticRegression(
            class_weight="balanced",
            random_state=seed,
            max_iter=10000,
            solver="saga",
            penalty="elasticnet",
        )

    if model_name == "random_forest":

        return RandomForestClassifier(
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )

    if model_name == "xgboost":

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            scale_pos_weight=ratio,
            tree_method="hist",
        )

    if model_name == "lightgbm":

        return LGBMClassifier(
            objective="binary",
            random_state=seed,
            n_jobs=1,
            class_weight="balanced",
            verbosity=-1,
        )

    if model_name == "catboost":

        return CatBoostClassifier(
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            class_weights=[
                1.0,
                ratio,
            ],
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# =============================================================================
# 17. PIPELINE FACTORY
# =============================================================================

def build_model_pipeline(
    model_name: str,
    seed: int,
    y_training: pd.Series,
    features: List[str],
    model_params: Optional[Dict] = None,
) -> Pipeline:

    linear_model = (
        model_name
        in {
            "logistic_regression",
            "elastic_net_logistic",
        }
    )

    preprocessor = build_preprocessor(
        features=
            features,
        scale_continuous=
            linear_model,
    )

    estimator = build_estimator(
        model_name=
            model_name,
        seed=
            seed,
        y_training=
            y_training,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                estimator,
            ),
        ]
    )

    if model_params:

        prefixed = {
            (
                key
                if key.startswith(
                    "model__"
                )
                else
                f"model__{key}"
            ):
                value
            for key, value
            in model_params.items()
        }

        pipeline.set_params(
            **prefixed
        )

    return pipeline


# =============================================================================
# 18. LOAD SEARCH SPACE
# =============================================================================

def search_space_for_model(
    hyper_df: pd.DataFrame,
    model_name: str,
) -> Dict[str, List]:

    subset = hyper_df[
        hyper_df[
            "model"
        ]
        == model_name
    ]

    search_space = {}

    for _, row in subset.iterrows():

        values = json.loads(
            row[
                "allowed_values_json"
            ]
        )

        search_space[
            f"model__{row['parameter']}"
        ] = values

    return search_space


# =============================================================================
# 19. TOTAL SEARCH COMBINATIONS
# =============================================================================

def count_search_combinations(
    search_space: Dict[
        str,
        List,
    ],
) -> int:

    if not search_space:
        return 1

    total = 1

    for values in search_space.values():

        total *= len(
            values
        )

    return total


# =============================================================================
# 20. TUNING CACHE
# =============================================================================

def load_tuning_cache() -> Dict:

    if not TUNING_CACHE_PATH.exists():
        return {}

    try:

        with open(
            TUNING_CACHE_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(
                f
            )

    except Exception:

        return {}


def save_tuning_cache(
    cache: Dict,
):

    with open(
        TUNING_CACHE_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_safe(
                cache
            ),
            f,
            indent=2,
        )


# =============================================================================
# 21. TUNING KEY
# =============================================================================

def make_tuning_key(
    study: str,
    train_cycles: str,
    feature_tier: str,
    model_name: str,
    outcome: str,
) -> str:

    raw = (
        f"{study}"
        f"__{train_cycles}"
        f"__{feature_tier}"
        f"__{model_name}"
        f"__{outcome}"
    )

    return raw.replace(
        "|",
        "+"
    )


# =============================================================================
# 22. HYPERPARAMETER TUNING
# =============================================================================

def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: List[str],
    model_name: str,
    search_space: Dict,
    tuning_iterations: int,
    tuning_seed: int,
) -> Tuple[
    Dict,
    float,
    float,
]:

    start = time.time()

    base_pipeline = (
        build_model_pipeline(
            model_name=
                model_name,
            seed=
                tuning_seed,
            y_training=
                y_train,
            features=
                features,
        )
    )

    # -------------------------------------------------------------------------
    # Fixed one-combination model
    # -------------------------------------------------------------------------

    total_combinations = (
        count_search_combinations(
            search_space
        )
    )

    if (
        total_combinations == 1
        and search_space
    ):

        params = {
            key:
                values[0]
            for key, values
            in search_space.items()
        }

        base_pipeline.set_params(
            **params
        )

        cv = StratifiedKFold(
            n_splits=
                CV_FOLDS,
            shuffle=True,
            random_state=
                tuning_seed,
        )

        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(
            base_pipeline,
            X_train[
                features
            ],
            y_train,
            scoring=
                TUNING_SCORING,
            cv=
                cv,
            n_jobs=-1,
        )

        best_score = float(
            np.mean(
                scores
            )
        )

        best_params = {
            key.replace(
                "model__",
                ""
            ):
                value
            for key, value
            in params.items()
        }

    else:

        n_iter = min(
            int(
                tuning_iterations
            ),
            int(
                total_combinations
            ),
        )

        cv = StratifiedKFold(
            n_splits=
                CV_FOLDS,
            shuffle=True,
            random_state=
                tuning_seed,
        )

        search = RandomizedSearchCV(
            estimator=
                base_pipeline,
            param_distributions=
                search_space,
            n_iter=
                n_iter,
            scoring=
                TUNING_SCORING,
            cv=
                cv,
            random_state=
                tuning_seed,
            n_jobs=-1,
            refit=False,
            verbose=0,
            error_score="raise",
        )

        search.fit(
            X_train[
                features
            ],
            y_train,
        )

        best_score = float(
            search.best_score_
        )

        best_params = {
            key.replace(
                "model__",
                ""
            ):
                value
            for key, value
            in search.best_params_.items()
        }

    elapsed = (
        time.time()
        - start
    )

    return (
        json_safe(
            best_params
        ),
        best_score,
        elapsed,
    )


# =============================================================================
# 23. OUT-OF-FOLD RAW PROBABILITIES
# =============================================================================

def generate_raw_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    model_name: str,
    model_params: Dict,
    seed: int,
) -> np.ndarray:

    cv = StratifiedKFold(
        n_splits=
            CALIBRATION_FOLDS,
        shuffle=True,
        random_state=
            seed,
    )

    oof = np.full(
        len(
            y
        ),
        np.nan,
        dtype=float,
    )

    y_array = np.asarray(
        y,
        dtype=int,
    )

    for (
        train_index,
        valid_index,
    ) in cv.split(
        X,
        y_array,
    ):

        X_fold_train = X.iloc[
            train_index
        ]

        X_fold_valid = X.iloc[
            valid_index
        ]

        y_fold_train = y.iloc[
            train_index
        ]

        pipeline = (
            build_model_pipeline(
                model_name=
                    model_name,
                seed=
                    seed,
                y_training=
                    y_fold_train,
                features=
                    features,
                model_params=
                    model_params,
            )
        )

        pipeline.fit(
            X_fold_train[
                features
            ],
            y_fold_train,
        )

        oof[
            valid_index
        ] = (
            pipeline.predict_proba(
                X_fold_valid[
                    features
                ]
            )[
                :,
                1,
            ]
        )

    if np.isnan(
        oof
    ).any():

        raise RuntimeError(
            "OOF probability generation incomplete."
        )

    return np.clip(
        oof,
        EPSILON,
        1.0
        - EPSILON,
    )


# =============================================================================
# 24. CALIBRATOR FITTING
# =============================================================================

def fit_sigmoid_calibrator(
    raw_prob: np.ndarray,
    y: np.ndarray,
):

    x = logit(
        np.clip(
            raw_prob,
            EPSILON,
            1.0 - EPSILON,
        )
    ).reshape(
        -1,
        1,
    )

    calibrator = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
    )

    try:

        calibrator.fit(
            x,
            y,
        )

    except Exception:

        calibrator = LogisticRegression(
            C=1e6,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
        )

        calibrator.fit(
            x,
            y,
        )

    return calibrator


def apply_sigmoid_calibrator(
    calibrator,
    raw_prob: np.ndarray,
) -> np.ndarray:

    x = logit(
        np.clip(
            raw_prob,
            EPSILON,
            1.0 - EPSILON,
        )
    ).reshape(
        -1,
        1,
    )

    return calibrator.predict_proba(
        x
    )[
        :,
        1,
    ]


def fit_isotonic_calibrator(
    raw_prob: np.ndarray,
    y: np.ndarray,
):

    calibrator = IsotonicRegression(
        out_of_bounds="clip",
        y_min=0.0,
        y_max=1.0,
    )

    calibrator.fit(
        raw_prob,
        y,
    )

    return calibrator


# =============================================================================
# 25. CROSS-FITTED CALIBRATED OOF
# =============================================================================

def crossfit_calibrated_oof(
    raw_oof: np.ndarray,
    y: np.ndarray,
    method: str,
    seed: int,
) -> np.ndarray:

    if method == "raw":

        return raw_oof.copy()

    cv = StratifiedKFold(
        n_splits=
            CALIBRATION_FOLDS,
        shuffle=True,
        random_state=
            seed + 7919,
    )

    calibrated = np.full(
        len(
            y
        ),
        np.nan,
    )

    dummy_X = np.zeros(
        (
            len(
                y
            ),
            1,
        )
    )

    for (
        fit_index,
        validation_index,
    ) in cv.split(
        dummy_X,
        y,
    ):

        train_prob = raw_oof[
            fit_index
        ]

        train_y = y[
            fit_index
        ]

        validation_prob = raw_oof[
            validation_index
        ]

        if method == "sigmoid":

            calibrator = (
                fit_sigmoid_calibrator(
                    train_prob,
                    train_y,
                )
            )

            calibrated[
                validation_index
            ] = (
                apply_sigmoid_calibrator(
                    calibrator,
                    validation_prob,
                )
            )

        elif method == "isotonic":

            calibrator = (
                fit_isotonic_calibrator(
                    train_prob,
                    train_y,
                )
            )

            calibrated[
                validation_index
            ] = (
                calibrator.predict(
                    validation_prob
                )
            )

        else:

            raise ValueError(
                method
            )

    return np.clip(
        calibrated,
        EPSILON,
        1.0
        - EPSILON,
    )


# =============================================================================
# 26. FINAL TRAINING + TEST PREDICTION
# =============================================================================

def fit_final_model_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    features: List[str],
    model_name: str,
    model_params: Dict,
    seed: int,
) -> np.ndarray:

    pipeline = build_model_pipeline(
        model_name=
            model_name,
        seed=
            seed,
        y_training=
            y_train,
        features=
            features,
        model_params=
            model_params,
    )

    pipeline.fit(
        X_train[
            features
        ],
        y_train,
    )

    prob = pipeline.predict_proba(
        X_test[
            features
        ]
    )[
        :,
        1,
    ]

    return np.clip(
        prob,
        EPSILON,
        1.0
        - EPSILON,
    )


# =============================================================================
# 27. FINAL CALIBRATORS
# =============================================================================

def fit_final_calibrators(
    raw_oof: np.ndarray,
    y_train: np.ndarray,
):

    sigmoid = (
        fit_sigmoid_calibrator(
            raw_oof,
            y_train,
        )
    )

    isotonic = (
        fit_isotonic_calibrator(
            raw_oof,
            y_train,
        )
    )

    return (
        sigmoid,
        isotonic,
    )


# =============================================================================
# 28. ECE
# =============================================================================

def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = ECE_BINS,
) -> float:

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ids = np.digitize(
        probability,
        bins,
        right=True,
    )

    ids = np.clip(
        ids,
        1,
        n_bins,
    )

    ece = 0.0

    n = len(
        y_true
    )

    for bin_id in range(
        1,
        n_bins + 1,
    ):

        mask = (
            ids
            == bin_id
        )

        if not mask.any():
            continue

        observed = np.mean(
            y_true[
                mask
            ]
        )

        predicted = np.mean(
            probability[
                mask
            ]
        )

        ece += (
            np.sum(
                mask
            )
            / n
            * abs(
                observed
                - predicted
            )
        )

    return float(
        ece
    )


# =============================================================================
# 29. CALIBRATION INTERCEPT / SLOPE
# =============================================================================

def calibration_intercept_slope(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> Tuple[
    float,
    float,
]:

    probability = np.clip(
        probability,
        EPSILON,
        1.0 - EPSILON,
    )

    lp = logit(
        probability
    ).reshape(
        -1,
        1,
    )

    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
    )

    try:

        model.fit(
            lp,
            y_true,
        )

    except Exception:

        model = LogisticRegression(
            C=1e6,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
        )

        model.fit(
            lp,
            y_true,
        )

    intercept = float(
        model.intercept_[0]
    )

    slope = float(
        model.coef_[0][0]
    )

    return (
        intercept,
        slope,
    )


# =============================================================================
# 30. PROBABILITY METRICS
# =============================================================================

def compute_probability_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> Dict:

    probability = np.clip(
        probability,
        EPSILON,
        1.0 - EPSILON,
    )

    if len(
        np.unique(
            y_true
        )
    ) < 2:

        auroc = np.nan
        auprc = np.nan

    else:

        auroc = roc_auc_score(
            y_true,
            probability,
        )

        auprc = average_precision_score(
            y_true,
            probability,
        )

    brier = brier_score_loss(
        y_true,
        probability,
    )

    ece = expected_calibration_error(
        y_true,
        probability,
    )

    try:

        (
            calibration_intercept,
            calibration_slope,
        ) = calibration_intercept_slope(
            y_true,
            probability,
        )

    except Exception:

        calibration_intercept = np.nan
        calibration_slope = np.nan

    return {
        "AUROC":
            auroc,

        "AUPRC":
            auprc,

        "Brier":
            brier,

        "ECE":
            ece,

        "CalibrationIntercept":
            calibration_intercept,

        "CalibrationSlope":
            calibration_slope,
    }


# =============================================================================
# 31. DEFAULT 0.5 THRESHOLD METRICS
# =============================================================================

def compute_threshold_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> Dict:

    predicted = (
        probability
        >= threshold
    ).astype(
        int
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predicted,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )

    sensitivity = (
        tp
        / (
            tp + fn
        )
        if (
            tp + fn
        ) > 0
        else np.nan
    )

    specificity = (
        tn
        / (
            tn + fp
        )
        if (
            tn + fp
        ) > 0
        else np.nan
    )

    ppv = (
        tp
        / (
            tp + fp
        )
        if (
            tp + fp
        ) > 0
        else np.nan
    )

    npv = (
        tn
        / (
            tn + fn
        )
        if (
            tn + fn
        ) > 0
        else np.nan
    )

    f1 = f1_score(
        y_true,
        predicted,
        zero_division=0,
    )

    return {
        "threshold":
            threshold,

        "Sensitivity":
            sensitivity,

        "Specificity":
            specificity,

        "PPV":
            ppv,

        "NPV":
            npv,

        "F1":
            f1,

        "TP":
            int(
                tp
            ),

        "FP":
            int(
                fp
            ),

        "TN":
            int(
                tn
            ),

        "FN":
            int(
                fn
            ),
    }


# =============================================================================
# 32. TRAINING THRESHOLD FOR TARGET SENSITIVITY
# =============================================================================

def threshold_for_target_sensitivity(
    y_train: np.ndarray,
    train_probability: np.ndarray,
    target_sensitivity: float,
) -> float:

    fpr, tpr, thresholds = roc_curve(
        y_train,
        train_probability,
    )

    valid = (
        tpr
        >= target_sensitivity
    )

    if not np.any(
        valid
    ):

        return 0.0

    candidate_thresholds = (
        thresholds[
            valid
        ]
    )

    finite = candidate_thresholds[
        np.isfinite(
            candidate_thresholds
        )
    ]

    if len(
        finite
    ) == 0:

        return 0.0

    # Largest threshold satisfying target sensitivity:
    # minimizes referral burden under the training distribution.
    return float(
        np.max(
            finite
        )
    )


# =============================================================================
# 33. TESTING BURDEN
# =============================================================================

def compute_testing_burden(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> Dict:

    predicted = (
        probability
        >= threshold
    ).astype(
        int
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predicted,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )

    n = len(
        y_true
    )

    referred = (
        tp + fp
    )

    referral_rate = (
        referred
        / n
        if n > 0
        else np.nan
    )

    tests_avoided = (
        1.0
        - referral_rate
        if not np.isnan(
            referral_rate
        )
        else np.nan
    )

    sensitivity = (
        tp
        / (
            tp + fn
        )
        if (
            tp + fn
        ) > 0
        else np.nan
    )

    specificity = (
        tn
        / (
            tn + fp
        )
        if (
            tn + fp
        ) > 0
        else np.nan
    )

    ppv = (
        tp
        / (
            tp + fp
        )
        if (
            tp + fp
        ) > 0
        else np.nan
    )

    npv = (
        tn
        / (
            tn + fn
        )
        if (
            tn + fn
        ) > 0
        else np.nan
    )

    tests_per_detected_case = (
        referred
        / tp
        if tp > 0
        else np.nan
    )

    return {
        "threshold":
            threshold,

        "ReferralRate":
            referral_rate,

        "TestsAvoided":
            tests_avoided,

        "CasesDetected":
            int(
                tp
            ),

        "CasesMissed":
            int(
                fn
            ),

        "TestsPerDetectedCase":
            tests_per_detected_case,

        "Sensitivity":
            sensitivity,

        "Specificity":
            specificity,

        "PPV":
            ppv,

        "NPV":
            npv,

        "TP":
            int(
                tp
            ),

        "FP":
            int(
                fp
            ),

        "TN":
            int(
                tn
            ),

        "FN":
            int(
                fn
            ),
    }


# =============================================================================
# 34. DECISION CURVE
# =============================================================================

def decision_curve_rows(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> List[Dict]:

    rows = []

    n = len(
        y_true
    )

    prevalence = np.mean(
        y_true
    )

    for threshold in DCA_THRESHOLDS:

        prediction = (
            probability
            >= threshold
        ).astype(
            int
        )

        tn, fp, fn, tp = (
            confusion_matrix(
                y_true,
                prediction,
                labels=[
                    0,
                    1,
                ],
            )
            .ravel()
        )

        odds = (
            threshold
            / (
                1.0 - threshold
            )
        )

        model_nb = (
            tp
            / n
            -
            fp
            / n
            * odds
        )

        treat_all_nb = (
            prevalence
            -
            (
                1.0
                - prevalence
            )
            * odds
        )

        rows.append(
            {
                "threshold_probability":
                    threshold,

                "model_net_benefit":
                    model_nb,

                "test_all_net_benefit":
                    treat_all_nb,

                "test_none_net_benefit":
                    0.0,
            }
        )

    return rows


# =============================================================================
# 35. BOOTSTRAP CORE METRICS
# =============================================================================

BOOTSTRAP_METRICS = [
    "AUROC",
    "AUPRC",
    "Brier",
    "ECE",
]


def bootstrap_primary_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    bootstrap_seed: int,
) -> List[Dict]:

    rng = np.random.default_rng(
        bootstrap_seed
    )

    n = len(
        y_true
    )

    values = {
        metric:
            []
        for metric
        in BOOTSTRAP_METRICS
    }

    successful = 0

    attempts = 0

    max_attempts = (
        BOOTSTRAP_ITERATIONS
        * 3
    )

    while (
        successful
        < BOOTSTRAP_ITERATIONS
        and attempts
        < max_attempts
    ):

        attempts += 1

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        y_b = y_true[
            idx
        ]

        p_b = probability[
            idx
        ]

        if len(
            np.unique(
                y_b
            )
        ) < 2:
            continue

        metrics = (
            compute_probability_metrics(
                y_b,
                p_b,
            )
        )

        for metric in BOOTSTRAP_METRICS:

            values[
                metric
            ].append(
                metrics[
                    metric
                ]
            )

        successful += 1

    rows = []

    for metric, metric_values in values.items():

        array = np.asarray(
            metric_values,
            dtype=float,
        )

        array = array[
            np.isfinite(
                array
            )
        ]

        if len(
            array
        ) == 0:

            lower = np.nan
            upper = np.nan

        else:

            lower = np.quantile(
                array,
                BOOTSTRAP_ALPHA / 2
            )

            upper = np.quantile(
                array,
                1.0
                - BOOTSTRAP_ALPHA / 2,
            )

        rows.append(
            {
                "metric":
                    metric,

                "ci95_lower":
                    lower,

                "ci95_upper":
                    upper,

                "bootstrap_successful":
                    successful,

                "bootstrap_requested":
                    BOOTSTRAP_ITERATIONS,
            }
        )

    return rows


# =============================================================================
# 36. CHECKPOINT PATHS
# =============================================================================

def test_prediction_path(
    experiment_id: str,
    seed: int,
) -> Path:

    return (
        TEST_PREDICTION_DIR
        / (
            f"{experiment_id}"
            f"__seed_{seed}"
            ".npz"
        )
    )


def oof_prediction_path(
    experiment_id: str,
    seed: int,
) -> Path:

    return (
        OOF_PREDICTION_DIR
        / (
            f"{experiment_id}"
            f"__seed_{seed}"
            ".npz"
        )
    )


# =============================================================================
# 37. SAVE / LOAD NPZ CHECKPOINT
# =============================================================================

def save_prediction_npz(
    path: Path,
    seqn: np.ndarray,
    y: np.ndarray,
    raw: np.ndarray,
    sigmoid: np.ndarray,
    isotonic: np.ndarray,
):

    np.savez_compressed(
        path,
        seqn=
            seqn,
        y=
            y,
        raw=
            raw,
        sigmoid=
            sigmoid,
        isotonic=
            isotonic,
    )


def load_prediction_npz(
    path: Path,
):

    data = np.load(
        path,
        allow_pickle=False,
    )

    return {
        "seqn":
            data[
                "seqn"
            ],

        "y":
            data[
                "y"
            ],

        "raw":
            data[
                "raw"
            ],

        "sigmoid":
            data[
                "sigmoid"
            ],

        "isotonic":
            data[
                "isotonic"
            ],
    }


# =============================================================================
# 38. EXPERIMENT DATA
# =============================================================================

def get_experiment_data(
    experiment: pd.Series,
    datasets: Dict[
        str,
        pd.DataFrame,
    ],
):

    study = (
        experiment[
            "study"
        ]
    )

    df = datasets[
        study
    ]

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

    features = (
        str(
            experiment[
                "feature_list"
            ]
        )
        .split("|")
    )

    train_df = (
        df[
            df[
                "survey_cycle"
            ]
            .isin(
                train_cycles
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    test_df = (
        df[
            df[
                "survey_cycle"
            ]
            == test_cycle
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if train_df.empty:

        raise RuntimeError(
            "Training dataframe is empty."
        )

    if test_df.empty:

        raise RuntimeError(
            "Test dataframe is empty."
        )

    if outcome not in train_df.columns:

        raise RuntimeError(
            f"Outcome missing: {outcome}"
        )

    missing_features = [
        x
        for x
        in features
        if x
        not in df.columns
    ]

    if missing_features:

        raise RuntimeError(
            f"Missing features: {missing_features}"
        )

    y_train = (
        safe_numeric(
            train_df[
                outcome
            ]
        )
        .astype(
            int
        )
    )

    y_test = (
        safe_numeric(
            test_df[
                outcome
            ]
        )
        .astype(
            int
        )
    )

    if y_train.nunique() < 2:

        raise RuntimeError(
            "Training outcome contains <2 classes."
        )

    if y_test.nunique() < 2:

        raise RuntimeError(
            "Test outcome contains <2 classes."
        )

    return (
        train_df,
        test_df,
        y_train,
        y_test,
        features,
    )


# =============================================================================
# 39. ENSEMBLE SEED PREDICTIONS
# =============================================================================

def aggregate_seed_predictions(
    experiment_id: str,
    seeds: List[int],
    directory: Path,
) -> Dict:

    loaded = []

    for seed in seeds:

        if directory == TEST_PREDICTION_DIR:

            path = test_prediction_path(
                experiment_id,
                seed,
            )

        else:

            path = oof_prediction_path(
                experiment_id,
                seed,
            )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing prediction checkpoint:\n{path}"
            )

        loaded.append(
            load_prediction_npz(
                path
            )
        )

    base_seqn = loaded[0][
        "seqn"
    ]

    base_y = loaded[0][
        "y"
    ]

    for item in loaded[1:]:

        if not np.array_equal(
            item[
                "seqn"
            ],
            base_seqn,
        ):

            raise RuntimeError(
                "SEQN order differs across seeds."
            )

        if not np.array_equal(
            item[
                "y"
            ],
            base_y,
        ):

            raise RuntimeError(
                "Outcome differs across seeds."
            )

    return {
        "seqn":
            base_seqn,

        "y":
            base_y,

        "raw":
            np.mean(
                np.vstack(
                    [
                        item[
                            "raw"
                        ]
                        for item
                        in loaded
                    ]
                ),
                axis=0,
            ),

        "sigmoid":
            np.mean(
                np.vstack(
                    [
                        item[
                            "sigmoid"
                        ]
                        for item
                        in loaded
                    ]
                ),
                axis=0,
            ),

        "isotonic":
            np.mean(
                np.vstack(
                    [
                        item[
                            "isotonic"
                        ]
                        for item
                        in loaded
                    ]
                ),
                axis=0,
            ),
    }


# =============================================================================
# 40. TUNING RESULT LOOKUP / CREATION
# =============================================================================

def get_or_create_tuned_parameters(
    experiment: pd.Series,
    train_df: pd.DataFrame,
    y_train: pd.Series,
    features: List[str],
    hyper_df: pd.DataFrame,
    tuning_cache: Dict,
) -> Tuple[
    Dict,
    Optional[Dict],
]:

    study = (
        experiment[
            "study"
        ]
    )

    train_cycles = (
        experiment[
            "train_cycles"
        ]
    )

    feature_tier = (
        experiment[
            "feature_tier"
        ]
    )

    model_name = (
        experiment[
            "model"
        ]
    )

    outcome = (
        experiment[
            "outcome"
        ]
    )

    tuning_key = make_tuning_key(
        study=
            study,
        train_cycles=
            train_cycles,
        feature_tier=
            feature_tier,
        model_name=
            model_name,
        outcome=
            outcome,
    )

    if (
        tuning_key
        in tuning_cache
        and not FORCE_RETRAIN
    ):

        return (
            tuning_cache[
                tuning_key
            ][
                "best_params"
            ],
            None,
        )

    search_space = (
        search_space_for_model(
            hyper_df,
            model_name,
        )
    )

    tuning_iterations = int(
        experiment[
            "tuning_iterations"
        ]
    )

    logger.info(
        f"Tuning {tuning_key}"
    )

    (
        best_params,
        best_score,
        elapsed,
    ) = tune_model(
        X_train=
            train_df,
        y_train=
            y_train,
        features=
            features,
        model_name=
            model_name,
        search_space=
            search_space,
        tuning_iterations=
            tuning_iterations,
        tuning_seed=
            1107,
    )

    cache_entry = {
        "best_params":
            best_params,

        "best_cv_score":
            best_score,

        "scoring":
            TUNING_SCORING,

        "elapsed_seconds":
            elapsed,

        "created":
            datetime.now()
            .isoformat(),

        "study":
            study,

        "train_cycles":
            train_cycles,

        "feature_tier":
            feature_tier,

        "model":
            model_name,

        "outcome":
            outcome,
    }

    tuning_cache[
        tuning_key
    ] = cache_entry

    save_tuning_cache(
        tuning_cache
    )

    tuning_row = {
        "tuning_key":
            tuning_key,

        **cache_entry,

        "best_params":
            json.dumps(
                best_params,
                sort_keys=True,
            ),
    }

    return (
        best_params,
        tuning_row,
    )


# =============================================================================
# 41. RUN ONE SEED
# =============================================================================

def run_seed(
    experiment: pd.Series,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    features: List[str],
    best_params: Dict,
    seed: int,
) -> Tuple[
    Dict,
    List[Dict],
]:

    experiment_id = (
        experiment[
            "experiment_id"
        ]
    )

    test_path = test_prediction_path(
        experiment_id,
        seed,
    )

    oof_path = oof_prediction_path(
        experiment_id,
        seed,
    )

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------

    if (
        test_path.exists()
        and oof_path.exists()
        and not FORCE_RETRAIN
    ):

        logger.info(
            f"{experiment_id} seed {seed}: checkpoint exists; loading."
        )

        test_predictions = (
            load_prediction_npz(
                test_path
            )
        )

        oof_predictions = (
            load_prediction_npz(
                oof_path
            )
        )

    else:

        # ---------------------------------------------------------------------
        # Cross-validated raw predictions in training data
        # ---------------------------------------------------------------------

        raw_oof = (
            generate_raw_oof_predictions(
                X=
                    train_df,
                y=
                    y_train,
                features=
                    features,
                model_name=
                    experiment[
                        "model"
                    ],
                model_params=
                    best_params,
                seed=
                    seed,
            )
        )

        y_train_array = (
            y_train
            .to_numpy(
                dtype=int
            )
        )

        # ---------------------------------------------------------------------
        # Cross-fitted calibration for training-threshold selection
        # ---------------------------------------------------------------------

        sigmoid_oof = (
            crossfit_calibrated_oof(
                raw_oof=
                    raw_oof,
                y=
                    y_train_array,
                method=
                    "sigmoid",
                seed=
                    seed,
            )
        )

        isotonic_oof = (
            crossfit_calibrated_oof(
                raw_oof=
                    raw_oof,
                y=
                    y_train_array,
                method=
                    "isotonic",
                seed=
                    seed,
            )
        )

        # ---------------------------------------------------------------------
        # Final calibrators trained only from training OOF predictions
        # ---------------------------------------------------------------------

        (
            sigmoid_calibrator,
            isotonic_calibrator,
        ) = fit_final_calibrators(
            raw_oof,
            y_train_array,
        )

        # ---------------------------------------------------------------------
        # Final model on full training cohort
        # ---------------------------------------------------------------------

        raw_test = (
            fit_final_model_and_predict(
                X_train=
                    train_df,
                y_train=
                    y_train,
                X_test=
                    test_df,
                features=
                    features,
                model_name=
                    experiment[
                        "model"
                    ],
                model_params=
                    best_params,
                seed=
                    seed,
            )
        )

        sigmoid_test = (
            apply_sigmoid_calibrator(
                sigmoid_calibrator,
                raw_test,
            )
        )

        isotonic_test = (
            isotonic_calibrator.predict(
                raw_test
            )
        )

        sigmoid_test = np.clip(
            sigmoid_test,
            EPSILON,
            1.0 - EPSILON,
        )

        isotonic_test = np.clip(
            isotonic_test,
            EPSILON,
            1.0 - EPSILON,
        )

        # ---------------------------------------------------------------------
        # Save
        # ---------------------------------------------------------------------

        save_prediction_npz(
            path=
                oof_path,

            seqn=
                train_df[
                    "SEQN"
                ]
                .to_numpy(),

            y=
                y_train_array,

            raw=
                raw_oof,

            sigmoid=
                sigmoid_oof,

            isotonic=
                isotonic_oof,
        )

        save_prediction_npz(
            path=
                test_path,

            seqn=
                test_df[
                    "SEQN"
                ]
                .to_numpy(),

            y=
                y_test
                .to_numpy(
                    dtype=int
                ),

            raw=
                raw_test,

            sigmoid=
                sigmoid_test,

            isotonic=
                isotonic_test,
        )

        test_predictions = {
            "seqn":
                test_df[
                    "SEQN"
                ]
                .to_numpy(),

            "y":
                y_test
                .to_numpy(
                    dtype=int
                ),

            "raw":
                raw_test,

            "sigmoid":
                sigmoid_test,

            "isotonic":
                isotonic_test,
        }

        oof_predictions = {
            "seqn":
                train_df[
                    "SEQN"
                ]
                .to_numpy(),

            "y":
                y_train_array,

            "raw":
                raw_oof,

            "sigmoid":
                sigmoid_oof,

            "isotonic":
                isotonic_oof,
        }

    # =========================================================================
    # SEED METRICS
    # =========================================================================

    metric_rows = []

    for calibration in CALIBRATION_METHODS:

        metrics = (
            compute_probability_metrics(
                test_predictions[
                    "y"
                ],
                test_predictions[
                    calibration
                ],
            )
        )

        default_metrics = (
            compute_threshold_metrics(
                test_predictions[
                    "y"
                ],
                test_predictions[
                    calibration
                ],
                threshold=0.50,
            )
        )

        metric_rows.append(
            {
                "experiment_id":
                    experiment_id,

                "seed":
                    seed,

                "calibration":
                    calibration,

                **metrics,

                "threshold_0_5_sensitivity":
                    default_metrics[
                        "Sensitivity"
                    ],

                "threshold_0_5_specificity":
                    default_metrics[
                        "Specificity"
                    ],

                "threshold_0_5_ppv":
                    default_metrics[
                        "PPV"
                    ],

                "threshold_0_5_npv":
                    default_metrics[
                        "NPV"
                    ],

                "threshold_0_5_f1":
                    default_metrics[
                        "F1"
                    ],
            }
        )

    return (
        test_predictions,
        metric_rows,
    )


# =============================================================================
# 42. EXPERIMENT ENSEMBLE METRICS
# =============================================================================

def summarize_experiment_predictions(
    experiment: pd.Series,
    seeds: List[int],
) -> Tuple[
    List[Dict],
    List[Dict],
    List[Dict],
    List[Dict],
]:

    experiment_id = (
        experiment[
            "experiment_id"
        ]
    )

    test = aggregate_seed_predictions(
        experiment_id=
            experiment_id,
        seeds=
            seeds,
        directory=
            TEST_PREDICTION_DIR,
    )

    train_oof = aggregate_seed_predictions(
        experiment_id=
            experiment_id,
        seeds=
            seeds,
        directory=
            OOF_PREDICTION_DIR,
    )

    metric_rows = []
    burden_rows = []
    dca_rows = []
    bootstrap_rows = []

    # =========================================================================
    # METRICS FOR ALL CALIBRATION MODES
    # =========================================================================

    for calibration in CALIBRATION_METHODS:

        probability = test[
            calibration
        ]

        metrics = (
            compute_probability_metrics(
                test[
                    "y"
                ],
                probability,
            )
        )

        default_metrics = (
            compute_threshold_metrics(
                test[
                    "y"
                ],
                probability,
                threshold=0.50,
            )
        )

        metric_rows.append(
            {
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

                "split_role":
                    experiment[
                        "split_role"
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

                "calibration":
                    calibration,

                "n_test":
                    len(
                        test[
                            "y"
                        ]
                    ),

                "test_cases":
                    int(
                        np.sum(
                            test[
                                "y"
                            ]
                        )
                    ),

                **metrics,

                "threshold_0_5_sensitivity":
                    default_metrics[
                        "Sensitivity"
                    ],

                "threshold_0_5_specificity":
                    default_metrics[
                        "Specificity"
                    ],

                "threshold_0_5_ppv":
                    default_metrics[
                        "PPV"
                    ],

                "threshold_0_5_npv":
                    default_metrics[
                        "NPV"
                    ],

                "threshold_0_5_f1":
                    default_metrics[
                        "F1"
                    ],
            }
        )

    # =========================================================================
    # TESTING BURDEN — PRIMARY SIGMOID CALIBRATION
    # =========================================================================

    for target in SENSITIVITY_TARGETS:

        threshold = (
            threshold_for_target_sensitivity(
                y_train=
                    train_oof[
                        "y"
                    ],
                train_probability=
                    train_oof[
                        PRIMARY_CALIBRATION
                    ],
                target_sensitivity=
                    target,
            )
        )

        burden = (
            compute_testing_burden(
                y_true=
                    test[
                        "y"
                    ],
                probability=
                    test[
                        PRIMARY_CALIBRATION
                    ],
                threshold=
                    threshold,
            )
        )

        burden_rows.append(
            {
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

                "calibration":
                    PRIMARY_CALIBRATION,

                "target_training_sensitivity":
                    target,

                **burden,
            }
        )

    # =========================================================================
    # DECISION CURVE — PRIMARY SIGMOID
    # =========================================================================

    for row in decision_curve_rows(
        test[
            "y"
        ],
        test[
            PRIMARY_CALIBRATION
        ],
    ):

        dca_rows.append(
            {
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

                **row,
            }
        )

    # =========================================================================
    # BOOTSTRAP CI — PRIMARY SIGMOID
    # =========================================================================

    bootstrap_seed = stable_integer_seed(
        experiment_id
        + "_bootstrap"
    )

    point_metrics = (
        compute_probability_metrics(
            test[
                "y"
            ],
            test[
                PRIMARY_CALIBRATION
            ],
        )
    )

    bootstrap = (
        bootstrap_primary_metrics(
            y_true=
                test[
                    "y"
                ],
            probability=
                test[
                    PRIMARY_CALIBRATION
                ],
            bootstrap_seed=
                bootstrap_seed,
        )
    )

    for row in bootstrap:

        bootstrap_rows.append(
            {
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

                "calibration":
                    PRIMARY_CALIBRATION,

                "metric":
                    row[
                        "metric"
                    ],

                "point_estimate":
                    point_metrics[
                        row[
                            "metric"
                        ]
                    ],

                "ci95_lower":
                    row[
                        "ci95_lower"
                    ],

                "ci95_upper":
                    row[
                        "ci95_upper"
                    ],

                "bootstrap_successful":
                    row[
                        "bootstrap_successful"
                    ],

                "bootstrap_requested":
                    row[
                        "bootstrap_requested"
                    ],
            }
        )

    return (
        metric_rows,
        burden_rows,
        dca_rows,
        bootstrap_rows,
    )


# =============================================================================
# 43. DECISION LEDGER
# =============================================================================

def build_decision_ledger():

    today = (
        datetime.now()
        .date()
        .isoformat()
    )

    rows = [

        {
            "decision_id":
                "D051",

            "stage":
                "06",

            "topic":
                "Tuning objective",

            "decision":
                (
                    "Hyperparameter optimization uses average precision "
                    "within training-only stratified cross-validation."
                ),

            "reason":
                (
                    "AUPRC is prevalence-sensitive and appropriate for "
                    "the relatively uncommon deficiency outcomes."
                ),

            "status":
                "FROZEN",

            "date":
                today,
        },

        {
            "decision_id":
                "D052",

            "stage":
                "06",

            "topic":
                "Tuning reuse",

            "decision":
                (
                    "Tuning occurs once for each unique study × training "
                    "cycles × feature tier × model × outcome configuration."
                ),

            "reason":
                (
                    "Identical training problems should not be re-tuned "
                    "for each external temporal test cycle or random seed."
                ),

            "status":
                "FROZEN",

            "date":
                today,
        },

        {
            "decision_id":
                "D053",

            "stage":
                "06",

            "topic":
                "Calibration construction",

            "decision":
                (
                    "Calibration models are fit from out-of-fold training "
                    "predictions and never from temporal test labels."
                ),

            "reason":
                "Prevents calibration leakage.",

            "status":
                "FROZEN",

            "date":
                today,
        },

        {
            "decision_id":
                "D054",

            "stage":
                "06",

            "topic":
                "Operating thresholds",

            "decision":
                (
                    "Sensitivity-target thresholds are selected from "
                    "cross-fitted training probabilities."
                ),

            "reason":
                (
                    "Operational test-cycle thresholds must not be chosen "
                    "using external test outcomes."
                ),

            "status":
                "FROZEN",

            "date":
                today,
        },

        {
            "decision_id":
                "D055",

            "stage":
                "06",

            "topic":
                "Seed aggregation",

            "decision":
                (
                    "Primary experiment-level probabilities are the mean "
                    "predicted probabilities across the ten locked seeds."
                ),

            "reason":
                (
                    "Preserves seed-level results while producing one "
                    "stable probability estimate per participant and experiment."
                ),

            "status":
                "FROZEN",

            "date":
                today,
        },

        {
            "decision_id":
                "D056",

            "stage":
                "06",

            "topic":
                "Bootstrap inference",

            "decision":
                (
                    "Primary sigmoid-calibrated AUROC, AUPRC, Brier and "
                    "ECE receive 1,000 participant-level bootstrap replicates."
                ),

            "reason":
                (
                    "Provides uncertainty around external temporal test "
                    "performance while limiting computational duplication."
                ),

            "status":
                "FROZEN",

            "date":
                today,
        },
    ]

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 44. MAIN
# =============================================================================

def main():

    warnings.filterwarnings(
        "ignore"
    )

    logger.info("")
    logger.info(
        "=" * 100
    )
    logger.info(
        "STAGE 06 STARTED"
    )
    logger.info(
        "=" * 100
    )

    (
        experiments,
        seeds_df,
        models_df,
        features_df,
        hyper_df,
        datasets,
    ) = load_inputs()

    experiments = filter_experiments(
        experiments
    )

    if experiments.empty:

        raise RuntimeError(
            "No experiments selected."
        )

    validate_ml_dependencies(
        experiments
    )

    seeds = (
        seeds_df[
            "random_seed"
        ]
        .astype(
            int
        )
        .tolist()
    )

    logger.info(
        f"Experiments selected: "
        f"{len(experiments):,}"
    )

    logger.info(
        f"Seeds: {seeds}"
    )

    tuning_cache = (
        load_tuning_cache()
    )

    # -------------------------------------------------------------------------
    # Fresh consolidated outputs
    #
    # Checkpoint NPZ files remain intact.
    # -------------------------------------------------------------------------

    for path in [
        TUNING_RESULTS_PATH,
        SEED_METRICS_PATH,
        EXPERIMENT_METRICS_PATH,
        BOOTSTRAP_PATH,
        TESTING_BURDEN_PATH,
        DCA_PATH,
        STATUS_PATH,
    ]:

        if path.exists():

            path.unlink()

    all_tuning_rows = []
    all_seed_metric_rows = []
    all_experiment_metric_rows = []
    all_bootstrap_rows = []
    all_burden_rows = []
    all_dca_rows = []
    all_status_rows = []

    # =========================================================================
    # EXPERIMENT LOOP
    # =========================================================================

    for experiment_number, (
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

        logger.info("")
        logger.info(
            "-" * 100
        )
        logger.info(
            f"[{experiment_number}/{len(experiments)}] "
            f"{experiment_id}"
        )
        logger.info(
            (
                f"{experiment['study']} | "
                f"{experiment['temporal_design']} | "
                f"{experiment['train_cycles']} -> "
                f"{experiment['test_cycle']} | "
                f"{experiment['feature_tier']} | "
                f"{experiment['model']}"
            )
        )
        logger.info(
            "-" * 100
        )

        experiment_start = (
            time.time()
        )

        try:

            (
                train_df,
                test_df,
                y_train,
                y_test,
                features,
            ) = get_experiment_data(
                experiment=
                    experiment,
                datasets=
                    datasets,
            )

            # =============================================================
            # TUNING
            # =============================================================

            (
                best_params,
                tuning_row,
            ) = get_or_create_tuned_parameters(
                experiment=
                    experiment,
                train_df=
                    train_df,
                y_train=
                    y_train,
                features=
                    features,
                hyper_df=
                    hyper_df,
                tuning_cache=
                    tuning_cache,
            )

            if tuning_row is not None:

                all_tuning_rows.append(
                    tuning_row
                )

                append_csv(
                    TUNING_RESULTS_PATH,
                    [
                        tuning_row
                    ],
                )

            logger.info(
                "Best parameters: "
                + json.dumps(
                    best_params,
                    sort_keys=True,
                )
            )

            # =============================================================
            # SEEDS
            # =============================================================

            experiment_seed_metrics = []

            for seed_index, seed in enumerate(
                seeds,
                start=1,
            ):

                logger.info(
                    f"  Seed "
                    f"{seed_index}/{len(seeds)}: "
                    f"{seed}"
                )

                (
                    _,
                    seed_metric_rows,
                ) = run_seed(
                    experiment=
                        experiment,
                    train_df=
                        train_df,
                    test_df=
                        test_df,
                    y_train=
                        y_train,
                    y_test=
                        y_test,
                    features=
                        features,
                    best_params=
                        best_params,
                    seed=
                        seed,
                )

                experiment_seed_metrics.extend(
                    seed_metric_rows
                )

                append_csv(
                    SEED_METRICS_PATH,
                    seed_metric_rows,
                )

            all_seed_metric_rows.extend(
                experiment_seed_metrics
            )

            # =============================================================
            # ENSEMBLE / EXPERIMENT-LEVEL SUMMARY
            # =============================================================

            (
                experiment_metrics,
                burden_rows,
                dca_rows,
                bootstrap_rows,
            ) = summarize_experiment_predictions(
                experiment=
                    experiment,
                seeds=
                    seeds,
            )

            all_experiment_metric_rows.extend(
                experiment_metrics
            )

            all_burden_rows.extend(
                burden_rows
            )

            all_dca_rows.extend(
                dca_rows
            )

            all_bootstrap_rows.extend(
                bootstrap_rows
            )

            append_csv(
                EXPERIMENT_METRICS_PATH,
                experiment_metrics,
            )

            append_csv(
                TESTING_BURDEN_PATH,
                burden_rows,
            )

            append_csv(
                DCA_PATH,
                dca_rows,
            )

            append_csv(
                BOOTSTRAP_PATH,
                bootstrap_rows,
            )

            elapsed = (
                time.time()
                - experiment_start
            )

            status_row = {
                "experiment_id":
                    experiment_id,

                "status":
                    "SUCCESS",

                "elapsed_seconds":
                    elapsed,

                "error":
                    "",
            }

            all_status_rows.append(
                status_row
            )

            append_csv(
                STATUS_PATH,
                [
                    status_row
                ],
            )

            logger.info(
                f"{experiment_id} complete "
                f"in {elapsed / 60:.2f} minutes."
            )

        except Exception as exc:

            elapsed = (
                time.time()
                - experiment_start
            )

            error_text = (
                f"{type(exc).__name__}: "
                f"{str(exc)}"
            )

            logger.error(
                f"{experiment_id} FAILED:"
            )

            logger.error(
                error_text
            )

            logger.error(
                traceback.format_exc()
            )

            status_row = {
                "experiment_id":
                    experiment_id,

                "status":
                    "FAILED",

                "elapsed_seconds":
                    elapsed,

                "error":
                    error_text,
            }

            all_status_rows.append(
                status_row
            )

            append_csv(
                STATUS_PATH,
                [
                    status_row
                ],
            )

            # Continue to next experiment.
            continue

    # =========================================================================
    # LOAD COMPLETE OUTPUTS FROM DISK
    # =========================================================================

    def read_if_exists(
        path: Path,
    ):

        if path.exists():

            return pd.read_csv(
                path,
                low_memory=False,
            )

        return pd.DataFrame()

    tuning_df = read_if_exists(
        TUNING_RESULTS_PATH
    )

    seed_metrics_df = read_if_exists(
        SEED_METRICS_PATH
    )

    experiment_metrics_df = read_if_exists(
        EXPERIMENT_METRICS_PATH
    )

    bootstrap_df = read_if_exists(
        BOOTSTRAP_PATH
    )

    burden_df = read_if_exists(
        TESTING_BURDEN_PATH
    )

    dca_df = read_if_exists(
        DCA_PATH
    )

    status_df = read_if_exists(
        STATUS_PATH
    )

    decision_ledger_df = (
        build_decision_ledger()
    )

    decision_ledger_df.to_csv(
        DECISION_LEDGER_PATH,
        index=False,
    )

    # =========================================================================
    # HIGH-LEVEL MODEL SUMMARY
    # =========================================================================

    primary_summary = pd.DataFrame()

    if not experiment_metrics_df.empty:

        primary_summary = (
            experiment_metrics_df[
                experiment_metrics_df[
                    "calibration"
                ]
                == PRIMARY_CALIBRATION
            ]
            .copy()
        )

        primary_summary = (
            primary_summary
            .sort_values(
                [
                    "study",
                    "temporal_design",
                    "test_cycle",
                    "feature_tier",
                    "model",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # =========================================================================
    # TIER COMPARISON
    # =========================================================================

    tier_comparison_df = (
        pd.DataFrame()
    )

    if not primary_summary.empty:

        pivot_cols = [
            "experiment_id",
            "study",
            "temporal_design",
            "split_role",
            "train_cycles",
            "test_cycle",
            "feature_tier",
            "model",
            "AUROC",
            "AUPRC",
            "Brier",
            "ECE",
        ]

        temp = (
            primary_summary[
                pivot_cols
            ]
            .copy()
        )

        merge_keys = [
            "study",
            "temporal_design",
            "split_role",
            "train_cycles",
            "test_cycle",
            "model",
        ]

        t1 = (
            temp[
                temp[
                    "feature_tier"
                ]
                == "T1"
            ]
            .copy()
        )

        t2 = (
            temp[
                temp[
                    "feature_tier"
                ]
                == "T2"
            ]
            .copy()
        )

        tier_comparison_df = (
            t1.merge(
                t2,
                on=
                    merge_keys,
                how=
                    "inner",
                suffixes=(
                    "_T1",
                    "_T2",
                ),
            )
        )

        if not tier_comparison_df.empty:

            tier_comparison_df[
                "delta_AUROC_T2_minus_T1"
            ] = (
                tier_comparison_df[
                    "AUROC_T2"
                ]
                -
                tier_comparison_df[
                    "AUROC_T1"
                ]
            )

            tier_comparison_df[
                "delta_AUPRC_T2_minus_T1"
            ] = (
                tier_comparison_df[
                    "AUPRC_T2"
                ]
                -
                tier_comparison_df[
                    "AUPRC_T1"
                ]
            )

            tier_comparison_df[
                "delta_Brier_T2_minus_T1"
            ] = (
                tier_comparison_df[
                    "Brier_T2"
                ]
                -
                tier_comparison_df[
                    "Brier_T1"
                ]
            )

            tier_comparison_df[
                "delta_ECE_T2_minus_T1"
            ] = (
                tier_comparison_df[
                    "ECE_T2"
                ]
                -
                tier_comparison_df[
                    "ECE_T1"
                ]
            )

            tier_comparison_df.to_csv(
                OUTPUT_DIR
                / "06_tier1_vs_tier2_comparison.csv",
                index=False,
            )

    # =========================================================================
    # OBSERVED VS MATCHED IRON PERFORMANCE
    # =========================================================================

    iron_comparison_df = (
        pd.DataFrame()
    )

    if not primary_summary.empty:

        iron_observed = (
            primary_summary[
                primary_summary[
                    "study"
                ]
                == "iron_observed"
            ]
            .copy()
        )

        iron_matched = (
            primary_summary[
                primary_summary[
                    "study"
                ]
                == "iron_matched"
            ]
            .copy()
        )

        keys = [
            "temporal_design",
            "split_role",
            "train_cycles",
            "test_cycle",
            "feature_tier",
            "model",
            "calibration",
        ]

        iron_comparison_df = (
            iron_observed.merge(
                iron_matched,
                on=
                    keys,
                how=
                    "inner",
                suffixes=(
                    "_observed",
                    "_matched",
                ),
            )
        )

        if not iron_comparison_df.empty:

            for metric in [
                "AUROC",
                "AUPRC",
                "Brier",
                "ECE",
            ]:

                iron_comparison_df[
                    f"{metric}_matched_minus_observed"
                ] = (
                    iron_comparison_df[
                        f"{metric}_matched"
                    ]
                    -
                    iron_comparison_df[
                        f"{metric}_observed"
                    ]
                )

            iron_comparison_df.to_csv(
                OUTPUT_DIR
                / "06_iron_observed_vs_matched_performance.csv",
                index=False,
            )

    # =========================================================================
    # EXCEL SUMMARY
    # =========================================================================

    with pd.ExcelWriter(
        WORKBOOK_PATH,
        engine="openpyxl",
    ) as writer:

        status_df.to_excel(
            writer,
            sheet_name="Execution_Status",
            index=False,
        )

        tuning_df.to_excel(
            writer,
            sheet_name="Tuning",
            index=False,
        )

        experiment_metrics_df.to_excel(
            writer,
            sheet_name="Experiment_Metrics",
            index=False,
        )

        primary_summary.to_excel(
            writer,
            sheet_name="Primary_Sigmoid",
            index=False,
        )

        seed_metrics_df.to_excel(
            writer,
            sheet_name="Seed_Metrics",
            index=False,
        )

        bootstrap_df.to_excel(
            writer,
            sheet_name="Bootstrap_CI",
            index=False,
        )

        burden_df.to_excel(
            writer,
            sheet_name="Testing_Burden",
            index=False,
        )

        if not tier_comparison_df.empty:

            tier_comparison_df.to_excel(
                writer,
                sheet_name="Tier1_vs_Tier2",
                index=False,
            )

        if not iron_comparison_df.empty:

            iron_comparison_df.to_excel(
                writer,
                sheet_name="Iron_Observed_Matched",
                index=False,
            )

        decision_ledger_df.to_excel(
            writer,
            sheet_name="Decision_Ledger",
            index=False,
        )

    # =========================================================================
    # FINAL CONSOLE SUMMARY
    # =========================================================================

    print("\n")
    print(
        "=" * 100
    )
    print(
        "STAGE 06 COMPLETE"
    )
    print(
        "=" * 100
    )

    if not status_df.empty:

        success_count = int(
            (
                status_df[
                    "status"
                ]
                == "SUCCESS"
            )
            .sum()
        )

        failure_count = int(
            (
                status_df[
                    "status"
                ]
                == "FAILED"
            )
            .sum()
        )

        print(
            f"\nSuccessful experiments: "
            f"{success_count:,}"
        )

        print(
            f"Failed experiments: "
            f"{failure_count:,}"
        )

    if not primary_summary.empty:

        print(
            "\nPRIMARY SIGMOID PERFORMANCE RANGE"
        )

        print(
            "-" * 100
        )

        for metric in [
            "AUROC",
            "AUPRC",
            "Brier",
            "ECE",
        ]:

            print(
                f"{metric}: "
                f"{primary_summary[metric].min():.4f}"
                f" – "
                f"{primary_summary[metric].max():.4f}"
            )

    print(
        "\nPRIMARY WORKBOOK:"
    )

    print(
        f"  {WORKBOOK_PATH}"
    )

    print(
        "\nCHECKPOINT DIRECTORY:"
    )

    print(
        f"  {CHECKPOINT_DIR}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  • Hyperparameter tuning used training data only."
    )

    print(
        "  • Test cycles never participated in tuning."
    )

    print(
        "  • Calibration used OOF training predictions only."
    )

    print(
        "  • Sensitivity-target thresholds were derived from training OOF predictions."
    )

    print(
        "  • All 10 locked seeds were retained."
    )

    print(
        "  • Raw, sigmoid, and isotonic probabilities were preserved."
    )

    print(
        "  • Participant-level checkpoints permit restart without losing completed work."
    )

    print(
        "\nNEXT AFTER REVIEW:"
    )

    print(
        "  Stage 07 — Temporal Degradation Profiles, "
        "Recalibration-vs-Retraining Analysis, "
        "Tier Contribution, and Eligibility-Shift Decomposition."
    )

    print(
        "\n" + "=" * 100
    )


# =============================================================================
# 45. ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()