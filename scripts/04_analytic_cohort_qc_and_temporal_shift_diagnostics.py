"""
===============================================================================
STAGE 04
Analytic Cohort QC, Survey-Weighted Descriptives,
and Temporal Shift Diagnostics
===============================================================================


===============================================================================
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "processed"
    / "04_predictor_harmonization"
    / "02B_final_feature_set.csv"
)

STAGE03_DIR = (
    PROJECT_ROOT
    / "processed"
    / "05_analytic_cohorts"
)

VITD_PATH = (
    STAGE03_DIR
    / "vitamin_d"
    / "NHANES_vitaminD_primary_multicycle.csv"
)

IRON_OBSERVED_PATH = (
    STAGE03_DIR
    / "iron"
    / "observed"
    / "NHANES_iron_observed_multicycle.csv"
)

IRON_MATCHED_PATH = (
    STAGE03_DIR
    / "iron"
    / "matched"
    / "NHANES_iron_matched_multicycle.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed"
    / "06_shift_diagnostics"
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
# 2. CYCLES
# =============================================================================

MAIN_CYCLES = [
    "2009_2010",
    "2011_2012",
    "2013_2014",
    "2015_2016",
    "2017_2018",
    "2021_2023",
]

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

CYCLE_MIDPOINT_YEAR = {
    "2009_2010": 2009.5,
    "2011_2012": 2011.5,
    "2013_2014": 2013.5,
    "2015_2016": 2015.5,
    "2017_2018": 2017.5,
    "2021_2023": 2022.0,
}

CYCLE_LABELS = {
    "2009_2010": "2009–2010",
    "2011_2012": "2011–2012",
    "2013_2014": "2013–2014",
    "2015_2016": "2015–2016",
    "2017_2018": "2017–2018",
    "2021_2023": "2021–2023",
}


# =============================================================================
# 3. LOGGING
# =============================================================================

LOG_FILE = (
    LOG_DIR
    / "stage04_shift_diagnostics.log"
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
# 4. FEATURE TYPE DEFINITIONS
# =============================================================================
#
# These are conceptually categorical even though harmonized as numeric codes.
#
# They must NOT be treated as continuous for temporal shift analysis.
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

CONTINUOUS_FEATURES_HINT = {
    "age_years",
    "poverty_income_ratio",
    "bmi",
    "waist_cm",
    "day1_energy_kcal",
    "day1_dietary_iron_mg",
    "day1_dietary_vitamin_d",
    "cbc_wbc",
    "cbc_rbc",
    "cbc_hemoglobin",
    "cbc_hematocrit",
    "cbc_mcv",
    "cbc_mch",
    "cbc_mchc",
    "cbc_rdw",
    "cbc_platelets",
    "serum_albumin",
    "serum_creatinine",
    "serum_glucose",
}


# =============================================================================
# 5. KNOWN PROTOCOL / ASCERTAINMENT SHIFTS
# =============================================================================
#
# These indicators are based on Stage 02A documentation review.
#
# Do not infer assay changes from numeric distributions.
# =============================================================================

KNOWN_PROTOCOL_SHIFTS = [
    {
        "outcome_family": "iron",
        "cycle": "2009_2010",
        "shift_type": "ferritin_eligibility",
        "indicator": 1,
        "description":
            "Ferritin eligibility restricted primarily to children "
            "and females age 12–49.",
    },
    {
        "outcome_family": "iron",
        "cycle": "2015_2016",
        "shift_type": "ferritin_eligibility",
        "indicator": 1,
        "description":
            "Ferritin eligibility restricted primarily to children "
            "and females age 12–49.",
    },
    {
        "outcome_family": "iron",
        "cycle": "2015_2016",
        "shift_type": "ferritin_instrument_change",
        "indicator": 1,
        "description":
            "Ferritin instrumentation changed during 2016 from "
            "Roche Elecsys-170 to Roche e601 according to NHANES documentation.",
    },
    {
        "outcome_family": "iron",
        "cycle": "2017_2018",
        "shift_type": "ferritin_eligibility_expansion",
        "indicator": 1,
        "description":
            "Ferritin eligibility expanded to examined participants "
            "age 1–5 and age 12+, including adult men.",
    },
    {
        "outcome_family": "iron",
        "cycle": "2021_2023",
        "shift_type": "ferritin_eligibility_restriction",
        "indicator": 1,
        "description":
            "Ferritin eligibility returned to children age 1–5 "
            "and females age 12–49.",
    },
]


# =============================================================================
# 6. GENERAL HELPERS
# =============================================================================

def safe_numeric(
    series: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def safe_pct(
    numerator: float,
    denominator: float,
) -> float:

    if (
        denominator is None
        or pd.isna(denominator)
        or denominator == 0
    ):
        return np.nan

    return (
        100.0
        * float(numerator)
        / float(denominator)
    )


def find_weight_column(
    df: pd.DataFrame,
) -> Optional[str]:

    candidates = [
        "WTMEC2YR",
        "WTMECPRP",
    ]

    for col in candidates:

        if col in df.columns:
            return col

    return None


def find_stratum_column(
    df: pd.DataFrame,
) -> Optional[str]:

    if "SDMVSTRA" in df.columns:
        return "SDMVSTRA"

    return None


def find_psu_column(
    df: pd.DataFrame,
) -> Optional[str]:

    if "SDMVPSU" in df.columns:
        return "SDMVPSU"

    return None


# =============================================================================
# 7. LOAD DATA
# =============================================================================

def load_required_csv(
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
        f"{label}: {df.shape[0]:,} × {df.shape[1]:,}"
    )

    return df


def load_feature_manifest() -> pd.DataFrame:

    return load_required_csv(
        FEATURE_MANIFEST_PATH,
        "Stage-02B feature manifest",
    )


# =============================================================================
# 8. FEATURE BLOCKS
# =============================================================================

def feature_blocks_from_manifest(
    manifest: pd.DataFrame,
) -> Dict[str, List[str]]:

    t1 = manifest.loc[
        manifest[
            "inclusion_role"
        ] == "PRIMARY_T1",
        "feature",
    ].tolist()

    cbc = manifest.loc[
        manifest[
            "inclusion_role"
        ] == "PRIMARY_T2_CBC",
        "feature",
    ].tolist()

    chemistry = manifest.loc[
        manifest[
            "inclusion_role"
        ] == "T2_SENSITIVITY",
        "feature",
    ].tolist()

    primary = (
        t1 + cbc
    )

    extended = (
        primary + chemistry
    )

    return {
        "tier1": t1,
        "cbc": cbc,
        "chemistry": chemistry,
        "primary": primary,
        "extended": extended,
    }


def classify_feature_type(
    feature: str,
) -> str:

    if feature in CATEGORICAL_FEATURES:
        return "categorical"

    if feature in CONTINUOUS_FEATURES_HINT:
        return "continuous"

    return "unknown"


# =============================================================================
# 9. COHORT QC
# =============================================================================

def build_cohort_qc(
    df: pd.DataFrame,
    study: str,
    features: List[str],
) -> List[Dict]:

    rows = []

    for cycle, cycle_df in df.groupby(
        "survey_cycle",
        sort=False,
    ):

        n = len(
            cycle_df
        )

        n_unique = (
            cycle_df[
                "SEQN"
            ]
            .nunique()
        )

        duplicates = (
            n - n_unique
        )

        missing_primary_outcome = (
            int(
                cycle_df[
                    "primary_outcome"
                ]
                .isna()
                .sum()
            )
            if "primary_outcome"
            in cycle_df.columns
            else np.nan
        )

        feature_cols = [
            feature
            for feature
            in features
            if feature
            in cycle_df.columns
        ]

        completely_missing_features = [
            feature
            for feature
            in feature_cols
            if cycle_df[
                feature
            ]
            .notna()
            .sum()
            == 0
        ]

        rows.append(
            {
                "study": study,
                "cycle": cycle,
                "cycle_label":
                    CYCLE_LABELS.get(
                        cycle,
                        cycle,
                    ),
                "n_rows": n,
                "n_unique_seqn":
                    n_unique,
                "duplicate_seqn":
                    duplicates,
                "missing_primary_outcome":
                    missing_primary_outcome,
                "n_expected_features":
                    len(features),
                "n_feature_columns_present":
                    len(
                        feature_cols
                    ),
                "n_completely_missing_features":
                    len(
                        completely_missing_features
                    ),
                "completely_missing_features":
                    "|".join(
                        completely_missing_features
                    ),
            }
        )

    return rows


# =============================================================================
# 10. SURVEY-WEIGHTED MEANS
# =============================================================================

def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float:

    x = safe_numeric(
        values
    )

    w = safe_numeric(
        weights
    )

    valid = (
        x.notna()
        & w.notna()
        & (w > 0)
    )

    if valid.sum() == 0:
        return np.nan

    return float(
        np.average(
            x.loc[valid],
            weights=w.loc[valid],
        )
    )


# =============================================================================
# 11. APPROXIMATE DESIGN-BASED RATIO-MEAN VARIANCE
# =============================================================================
#
# This implements a first-order PSU/stratum linearization for a weighted mean.
#
# It is adequate for descriptive QC and approximate confidence intervals.
# It is NOT intended to replace a dedicated survey-analysis package for final
# publication inference.
#
# Final manuscript tables should still be validated against a survey package
# if formal NHANES design-based inference is required.
# =============================================================================

def survey_weighted_mean_and_se(
    values: pd.Series,
    weights: pd.Series,
    strata: Optional[pd.Series],
    psu: Optional[pd.Series],
) -> Tuple[
    float,
    float,
    float,
    float,
    int,
]:

    x = safe_numeric(
        values
    )

    w = safe_numeric(
        weights
    )

    valid = (
        x.notna()
        & w.notna()
        & (w > 0)
    )

    if strata is not None:
        valid = (
            valid
            & strata.notna()
        )

    if psu is not None:
        valid = (
            valid
            & psu.notna()
        )

    if valid.sum() == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0,
        )

    xv = x.loc[
        valid
    ]

    wv = w.loc[
        valid
    ]

    mean = float(
        np.sum(
            wv * xv
        )
        / np.sum(
            wv
        )
    )

    # -------------------------------------------------------------------------
    # If design variables unavailable, fall back to naive weighted approximation.
    # -------------------------------------------------------------------------

    if (
        strata is None
        or psu is None
    ):

        variance = (
            np.average(
                (
                    xv
                    - mean
                ) ** 2,
                weights=wv,
            )
        )

        effective_n = (
            (
                wv.sum()
                ** 2
            )
            /
            (
                (
                    wv ** 2
                )
                .sum()
            )
        )

        se = (
            math.sqrt(
                variance
                / effective_n
            )
            if effective_n > 0
            else np.nan
        )

        return (
            mean,
            se,
            mean - 1.96 * se,
            mean + 1.96 * se,
            int(
                valid.sum()
            ),
        )

    strata_v = strata.loc[
        valid
    ]

    psu_v = psu.loc[
        valid
    ]

    temp = pd.DataFrame(
        {
            "x": xv,
            "w": wv,
            "strata":
                strata_v,
            "psu":
                psu_v,
        }
    )

    total_weight = (
        temp[
            "w"
        ]
        .sum()
    )

    # Linearized weighted residual contribution.
    temp[
        "linearized"
    ] = (
        temp[
            "w"
        ]
        * (
            temp[
                "x"
            ]
            - mean
        )
        / total_weight
    )

    psu_totals = (
        temp
        .groupby(
            [
                "strata",
                "psu",
            ],
            dropna=False,
        )[
            "linearized"
        ]
        .sum()
        .reset_index()
    )

    variance = 0.0
    usable_strata = 0

    for (
        stratum_value,
        group,
    ) in psu_totals.groupby(
        "strata"
    ):

        m = len(
            group
        )

        if m < 2:
            continue

        z = (
            group[
                "linearized"
            ]
            .to_numpy(
                dtype=float
            )
        )

        z_bar = (
            z.mean()
        )

        variance_h = (
            m
            / (
                m - 1
            )
            * np.sum(
                (
                    z
                    - z_bar
                ) ** 2
            )
        )

        variance += (
            variance_h
        )

        usable_strata += 1

    if usable_strata == 0:

        se = np.nan

    else:

        se = math.sqrt(
            max(
                variance,
                0.0,
            )
        )

    lower = (
        mean
        - 1.96 * se
        if not pd.isna(
            se
        )
        else np.nan
    )

    upper = (
        mean
        + 1.96 * se
        if not pd.isna(
            se
        )
        else np.nan
    )

    return (
        mean,
        se,
        lower,
        upper,
        int(
            valid.sum()
        ),
    )


# =============================================================================
# 12. OUTCOME PREVALENCE
# =============================================================================

def build_weighted_prevalence(
    df: pd.DataFrame,
    study: str,
) -> List[Dict]:

    rows = []

    for cycle, cycle_df in df.groupby(
        "survey_cycle",
        sort=False,
    ):

        weight_col = (
            find_weight_column(
                cycle_df
            )
        )

        strata_col = (
            find_stratum_column(
                cycle_df
            )
        )

        psu_col = (
            find_psu_column(
                cycle_df
            )
        )

        if (
            "primary_outcome"
            not in cycle_df.columns
        ):
            continue

        y = safe_numeric(
            cycle_df[
                "primary_outcome"
            ]
        )

        n_valid = int(
            y.notna()
            .sum()
        )

        n_cases = int(
            (
                y == 1
            )
            .sum()
        )

        raw_prev = (
            n_cases
            / n_valid
            if n_valid > 0
            else np.nan
        )

        if (
            weight_col is None
        ):

            weighted_prev = np.nan
            weighted_se = np.nan
            lower = np.nan
            upper = np.nan
            n_design = 0

        else:

            (
                weighted_prev,
                weighted_se,
                lower,
                upper,
                n_design,
            ) = survey_weighted_mean_and_se(
                values=y,
                weights=
                    cycle_df[
                        weight_col
                    ],
                strata=(
                    cycle_df[
                        strata_col
                    ]
                    if strata_col
                    is not None
                    else None
                ),
                psu=(
                    cycle_df[
                        psu_col
                    ]
                    if psu_col
                    is not None
                    else None
                ),
            )

        rows.append(
            {
                "study": study,
                "cycle": cycle,
                "cycle_label":
                    CYCLE_LABELS.get(
                        cycle,
                        cycle,
                    ),
                "n":
                    len(
                        cycle_df
                    ),
                "n_valid_outcome":
                    n_valid,
                "cases":
                    n_cases,
                "unweighted_prevalence_pct":
                    100 * raw_prev
                    if not pd.isna(
                        raw_prev
                    )
                    else np.nan,
                "weighted_prevalence_pct":
                    100 * weighted_prev
                    if not pd.isna(
                        weighted_prev
                    )
                    else np.nan,
                "weighted_se_pct":
                    100 * weighted_se
                    if not pd.isna(
                        weighted_se
                    )
                    else np.nan,
                "weighted_ci95_lower_pct":
                    max(
                        0.0,
                        100 * lower,
                    )
                    if not pd.isna(
                        lower
                    )
                    else np.nan,
                "weighted_ci95_upper_pct":
                    min(
                        100.0,
                        100 * upper,
                    )
                    if not pd.isna(
                        upper
                    )
                    else np.nan,
                "weight_variable":
                    weight_col,
                "stratum_variable":
                    strata_col,
                "psu_variable":
                    psu_col,
                "n_design_valid":
                    n_design,
                "ci_method":
                    (
                        "approximate_stratified_PSU_linearization"
                        if strata_col
                        and psu_col
                        else
                        "weighted_naive_approximation"
                    ),
            }
        )

    return rows


# =============================================================================
# 13. CONTINUOUS DESCRIPTIVES
# =============================================================================

def build_weighted_continuous_descriptives(
    df: pd.DataFrame,
    study: str,
    continuous_features: List[str],
) -> List[Dict]:

    rows = []

    for cycle, cycle_df in df.groupby(
        "survey_cycle",
        sort=False,
    ):

        weight_col = (
            find_weight_column(
                cycle_df
            )
        )

        strata_col = (
            find_stratum_column(
                cycle_df
            )
        )

        psu_col = (
            find_psu_column(
                cycle_df
            )
        )

        for feature in continuous_features:

            if feature not in cycle_df.columns:
                continue

            x = safe_numeric(
                cycle_df[
                    feature
                ]
            )

            valid = x.dropna()

            if len(valid) == 0:
                continue

            unweighted_mean = (
                valid.mean()
            )

            unweighted_sd = (
                valid.std(
                    ddof=1
                )
            )

            median = (
                valid.median()
            )

            q1 = (
                valid.quantile(
                    0.25
                )
            )

            q3 = (
                valid.quantile(
                    0.75
                )
            )

            if weight_col is not None:

                (
                    weighted_m,
                    weighted_se,
                    ci_lower,
                    ci_upper,
                    n_design,
                ) = survey_weighted_mean_and_se(
                    values=
                        cycle_df[
                            feature
                        ],
                    weights=
                        cycle_df[
                            weight_col
                        ],
                    strata=(
                        cycle_df[
                            strata_col
                        ]
                        if strata_col
                        else None
                    ),
                    psu=(
                        cycle_df[
                            psu_col
                        ]
                        if psu_col
                        else None
                    ),
                )

            else:

                weighted_m = np.nan
                weighted_se = np.nan
                ci_lower = np.nan
                ci_upper = np.nan
                n_design = 0

            rows.append(
                {
                    "study": study,
                    "cycle": cycle,
                    "feature":
                        feature,
                    "n_nonmissing":
                        int(
                            x.notna()
                            .sum()
                        ),
                    "missing_pct":
                        safe_pct(
                            x.isna()
                            .sum(),
                            len(
                                x
                            ),
                        ),
                    "mean_unweighted":
                        unweighted_mean,
                    "sd_unweighted":
                        unweighted_sd,
                    "median_unweighted":
                        median,
                    "q1_unweighted":
                        q1,
                    "q3_unweighted":
                        q3,
                    "mean_weighted":
                        weighted_m,
                    "weighted_se":
                        weighted_se,
                    "weighted_ci95_lower":
                        ci_lower,
                    "weighted_ci95_upper":
                        ci_upper,
                    "n_design_valid":
                        n_design,
                }
            )

    return rows


# =============================================================================
# 14. CATEGORICAL DESCRIPTIVES
# =============================================================================

def weighted_category_proportions(
    values: pd.Series,
    weights: Optional[pd.Series],
) -> pd.DataFrame:

    x = values.copy()

    valid = x.notna()

    if weights is None:

        counts = (
            x.loc[
                valid
            ]
            .value_counts(
                normalize=True,
                dropna=False,
            )
        )

        return pd.DataFrame(
            {
                "category":
                    counts.index,
                "proportion":
                    counts.values,
            }
        )

    w = safe_numeric(
        weights
    )

    valid = (
        valid
        & w.notna()
        & (w > 0)
    )

    if valid.sum() == 0:

        return pd.DataFrame(
            columns=[
                "category",
                "proportion",
            ]
        )

    temp = pd.DataFrame(
        {
            "category":
                x.loc[
                    valid
                ],
            "weight":
                w.loc[
                    valid
                ],
        }
    )

    grouped = (
        temp
        .groupby(
            "category",
            dropna=False,
        )[
            "weight"
        ]
        .sum()
    )

    proportions = (
        grouped
        / grouped.sum()
    )

    return pd.DataFrame(
        {
            "category":
                proportions.index,
            "proportion":
                proportions.values,
        }
    )


def build_categorical_descriptives(
    df: pd.DataFrame,
    study: str,
    categorical_features: List[str],
) -> List[Dict]:

    rows = []

    for cycle, cycle_df in df.groupby(
        "survey_cycle",
        sort=False,
    ):

        weight_col = (
            find_weight_column(
                cycle_df
            )
        )

        weights = (
            cycle_df[
                weight_col
            ]
            if weight_col
            is not None
            else None
        )

        for feature in categorical_features:

            if feature not in cycle_df.columns:
                continue

            series = (
                cycle_df[
                    feature
                ]
            )

            weighted_props = (
                weighted_category_proportions(
                    series,
                    weights,
                )
            )

            raw_counts = (
                series
                .value_counts(
                    dropna=False
                )
            )

            for _, row in weighted_props.iterrows():

                category = (
                    row[
                        "category"
                    ]
                )

                raw_n = int(
                    (
                        series
                        == category
                    )
                    .sum()
                )

                rows.append(
                    {
                        "study":
                            study,
                        "cycle":
                            cycle,
                        "feature":
                            feature,
                        "category":
                            category,
                        "raw_n":
                            raw_n,
                        "raw_pct":
                            safe_pct(
                                raw_n,
                                series.notna()
                                .sum(),
                            ),
                        "weighted_pct":
                            100
                            * row[
                                "proportion"
                            ],
                    }
                )

    return rows


# =============================================================================
# 15. PREDICTOR MISSINGNESS
# =============================================================================

def build_predictor_missingness(
    df: pd.DataFrame,
    study: str,
    features: List[str],
) -> List[Dict]:

    rows = []

    for cycle, cycle_df in df.groupby(
        "survey_cycle",
        sort=False,
    ):

        for feature in features:

            if feature not in cycle_df.columns:
                continue

            missing_n = int(
                cycle_df[
                    feature
                ]
                .isna()
                .sum()
            )

            rows.append(
                {
                    "study":
                        study,
                    "cycle":
                        cycle,
                    "feature":
                        feature,
                    "n":
                        len(
                            cycle_df
                        ),
                    "n_missing":
                        missing_n,
                    "missing_pct":
                        safe_pct(
                            missing_n,
                            len(
                                cycle_df
                            ),
                        ),
                }
            )

    return rows


# =============================================================================
# 16. STANDARDIZED MEAN DIFFERENCE
# =============================================================================

def standardized_mean_difference(
    x1: Iterable,
    x2: Iterable,
) -> float:

    x1 = pd.Series(
        x1
    ).dropna()

    x2 = pd.Series(
        x2
    ).dropna()

    if (
        len(x1) < 2
        or len(x2) < 2
    ):
        return np.nan

    mean1 = (
        x1.mean()
    )

    mean2 = (
        x2.mean()
    )

    var1 = (
        x1.var(
            ddof=1
        )
    )

    var2 = (
        x2.var(
            ddof=1
        )
    )

    pooled_sd = math.sqrt(
        (
            var1
            + var2
        )
        / 2.0
    )

    if pooled_sd == 0:
        return 0.0

    return (
        mean2
        - mean1
    ) / pooled_sd


# =============================================================================
# 17. PSI
# =============================================================================

def population_stability_index(
    reference: pd.Series,
    comparison: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    PSI based on quantile bins estimated ONLY from the reference distribution.

    This matters because comparison-cycle information must not define the
    reference bins.
    """

    ref = safe_numeric(
        reference
    ).dropna()

    comp = safe_numeric(
        comparison
    ).dropna()

    if (
        len(ref) < 20
        or len(comp) < 20
    ):
        return np.nan

    try:

        _, bin_edges = pd.qcut(
            ref,
            q=bins,
            retbins=True,
            duplicates="drop",
        )

    except ValueError:

        return np.nan

    bin_edges = np.unique(
        bin_edges
    )

    if len(
        bin_edges
    ) < 3:
        return np.nan

    # Extend limits to capture comparison values outside reference range.
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_bins = pd.cut(
        ref,
        bins=bin_edges,
        include_lowest=True,
    )

    comp_bins = pd.cut(
        comp,
        bins=bin_edges,
        include_lowest=True,
    )

    categories = (
        ref_bins
        .cat
        .categories
    )

    ref_prop = (
        ref_bins
        .value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0,
        )
        .to_numpy(
            dtype=float
        )
    )

    comp_prop = (
        comp_bins
        .value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0,
        )
        .to_numpy(
            dtype=float
        )
    )

    ref_prop = np.clip(
        ref_prop,
        epsilon,
        None,
    )

    comp_prop = np.clip(
        comp_prop,
        epsilon,
        None,
    )

    psi = np.sum(
        (
            comp_prop
            - ref_prop
        )
        * np.log(
            comp_prop
            / ref_prop
        )
    )

    return float(
        psi
    )


# =============================================================================
# 18. CONTINUOUS SHIFT
# =============================================================================

def compare_continuous_cycles(
    df: pd.DataFrame,
    study: str,
    cycles: List[str],
    features: List[str],
) -> List[Dict]:

    rows = []

    available_cycles = [
        cycle
        for cycle in cycles
        if cycle
        in set(
            df[
                "survey_cycle"
            ]
        )
    ]

    if len(
        available_cycles
    ) < 2:
        return rows

    reference_cycle = (
        available_cycles[0]
    )

    reference_df = (
        df[
            df[
                "survey_cycle"
            ]
            == reference_cycle
        ]
    )

    # -------------------------------------------------------------------------
    # Fixed-origin comparisons
    # -------------------------------------------------------------------------

    for comparison_cycle in available_cycles[1:]:

        comparison_df = (
            df[
                df[
                    "survey_cycle"
                ]
                == comparison_cycle
            ]
        )

        for feature in features:

            if (
                feature
                not in df.columns
            ):
                continue

            ref = safe_numeric(
                reference_df[
                    feature
                ]
            ).dropna()

            comp = safe_numeric(
                comparison_df[
                    feature
                ]
            ).dropna()

            if (
                len(ref) < 2
                or len(comp) < 2
            ):
                continue

            try:

                ks = (
                    ks_2samp(
                        ref,
                        comp,
                    )
                    .statistic
                )

            except Exception:

                ks = np.nan

            try:

                wass = (
                    wasserstein_distance(
                        ref,
                        comp,
                    )
                )

            except Exception:

                wass = np.nan

            rows.append(
                {
                    "study":
                        study,
                    "comparison_design":
                        "fixed_origin",
                    "reference_cycle":
                        reference_cycle,
                    "comparison_cycle":
                        comparison_cycle,
                    "temporal_distance_years":
                        abs(
                            CYCLE_MIDPOINT_YEAR[
                                comparison_cycle
                            ]
                            -
                            CYCLE_MIDPOINT_YEAR[
                                reference_cycle
                            ]
                        ),
                    "feature":
                        feature,
                    "n_reference":
                        len(ref),
                    "n_comparison":
                        len(comp),
                    "smd":
                        standardized_mean_difference(
                            ref,
                            comp,
                        ),
                    "abs_smd":
                        abs(
                            standardized_mean_difference(
                                ref,
                                comp,
                            )
                        ),
                    "ks_statistic":
                        ks,
                    "wasserstein":
                        wass,
                    "psi":
                        population_stability_index(
                            ref,
                            comp,
                        ),
                }
            )

    # -------------------------------------------------------------------------
    # Rolling comparisons
    # -------------------------------------------------------------------------

    for i in range(
        len(
            available_cycles
        )
        - 1
    ):

        ref_cycle = (
            available_cycles[i]
        )

        comp_cycle = (
            available_cycles[
                i + 1
            ]
        )

        ref_df = (
            df[
                df[
                    "survey_cycle"
                ]
                == ref_cycle
            ]
        )

        comp_df = (
            df[
                df[
                    "survey_cycle"
                ]
                == comp_cycle
            ]
        )

        for feature in features:

            if feature not in df.columns:
                continue

            ref = safe_numeric(
                ref_df[
                    feature
                ]
            ).dropna()

            comp = safe_numeric(
                comp_df[
                    feature
                ]
            ).dropna()

            if (
                len(ref) < 2
                or len(comp) < 2
            ):
                continue

            try:

                ks = (
                    ks_2samp(
                        ref,
                        comp,
                    )
                    .statistic
                )

            except Exception:

                ks = np.nan

            try:

                wass = (
                    wasserstein_distance(
                        ref,
                        comp,
                    )
                )

            except Exception:

                wass = np.nan

            smd = (
                standardized_mean_difference(
                    ref,
                    comp,
                )
            )

            rows.append(
                {
                    "study":
                        study,
                    "comparison_design":
                        "rolling",
                    "reference_cycle":
                        ref_cycle,
                    "comparison_cycle":
                        comp_cycle,
                    "temporal_distance_years":
                        abs(
                            CYCLE_MIDPOINT_YEAR[
                                comp_cycle
                            ]
                            -
                            CYCLE_MIDPOINT_YEAR[
                                ref_cycle
                            ]
                        ),
                    "feature":
                        feature,
                    "n_reference":
                        len(ref),
                    "n_comparison":
                        len(comp),
                    "smd":
                        smd,
                    "abs_smd":
                        abs(
                            smd
                        )
                        if not pd.isna(
                            smd
                        )
                        else np.nan,
                    "ks_statistic":
                        ks,
                    "wasserstein":
                        wass,
                    "psi":
                        population_stability_index(
                            ref,
                            comp,
                        ),
                }
            )

    return rows


# =============================================================================
# 19. CATEGORICAL SHIFT
# =============================================================================

def categorical_distribution(
    series: pd.Series,
) -> Dict:

    valid = (
        series
        .dropna()
    )

    if len(valid) == 0:
        return {}

    proportions = (
        valid
        .value_counts(
            normalize=True
        )
    )

    return (
        proportions
        .to_dict()
    )


def compare_categorical_distribution(
    reference: pd.Series,
    comparison: pd.Series,
) -> Tuple[
    float,
    float,
]:
    """
    Returns:
        Jensen-Shannon divergence
        max absolute category proportion difference
    """

    ref_dist = (
        categorical_distribution(
            reference
        )
    )

    comp_dist = (
        categorical_distribution(
            comparison
        )
    )

    categories = sorted(
        set(
            ref_dist.keys()
        )
        |
        set(
            comp_dist.keys()
        ),
        key=lambda x: str(x),
    )

    if len(categories) == 0:

        return (
            np.nan,
            np.nan,
        )

    p = np.array(
        [
            ref_dist.get(
                cat,
                0.0,
            )
            for cat
            in categories
        ],
        dtype=float,
    )

    q = np.array(
        [
            comp_dist.get(
                cat,
                0.0,
            )
            for cat
            in categories
        ],
        dtype=float,
    )

    # scipy returns JS distance.
    # Square it to obtain Jensen-Shannon divergence.
    js_distance = (
        jensenshannon(
            p,
            q,
            base=2,
        )
    )

    js_divergence = (
        js_distance ** 2
    )

    max_diff = np.max(
        np.abs(
            p - q
        )
    )

    return (
        float(
            js_divergence
        ),
        float(
            max_diff
        ),
    )


def compare_categorical_cycles(
    df: pd.DataFrame,
    study: str,
    cycles: List[str],
    features: List[str],
) -> List[Dict]:

    rows = []

    available_cycles = [
        cycle
        for cycle
        in cycles
        if cycle
        in set(
            df[
                "survey_cycle"
            ]
        )
    ]

    if len(
        available_cycles
    ) < 2:
        return rows

    # -------------------------------------------------------------------------
    # Fixed-origin
    # -------------------------------------------------------------------------

    origin = (
        available_cycles[0]
    )

    origin_df = (
        df[
            df[
                "survey_cycle"
            ]
            == origin
        ]
    )

    for comp_cycle in available_cycles[1:]:

        comp_df = (
            df[
                df[
                    "survey_cycle"
                ]
                == comp_cycle
            ]
        )

        for feature in features:

            if feature not in df.columns:
                continue

            (
                js,
                max_diff,
            ) = compare_categorical_distribution(
                origin_df[
                    feature
                ],
                comp_df[
                    feature
                ],
            )

            rows.append(
                {
                    "study":
                        study,
                    "comparison_design":
                        "fixed_origin",
                    "reference_cycle":
                        origin,
                    "comparison_cycle":
                        comp_cycle,
                    "temporal_distance_years":
                        abs(
                            CYCLE_MIDPOINT_YEAR[
                                comp_cycle
                            ]
                            -
                            CYCLE_MIDPOINT_YEAR[
                                origin
                            ]
                        ),
                    "feature":
                        feature,
                    "js_divergence":
                        js,
                    "max_abs_category_difference":
                        max_diff,
                }
            )

    # -------------------------------------------------------------------------
    # Rolling
    # -------------------------------------------------------------------------

    for i in range(
        len(
            available_cycles
        )
        - 1
    ):

        ref_cycle = (
            available_cycles[i]
        )

        comp_cycle = (
            available_cycles[
                i + 1
            ]
        )

        ref_df = (
            df[
                df[
                    "survey_cycle"
                ]
                == ref_cycle
            ]
        )

        comp_df = (
            df[
                df[
                    "survey_cycle"
                ]
                == comp_cycle
            ]
        )

        for feature in features:

            if feature not in df.columns:
                continue

            (
                js,
                max_diff,
            ) = compare_categorical_distribution(
                ref_df[
                    feature
                ],
                comp_df[
                    feature
                ],
            )

            rows.append(
                {
                    "study":
                        study,
                    "comparison_design":
                        "rolling",
                    "reference_cycle":
                        ref_cycle,
                    "comparison_cycle":
                        comp_cycle,
                    "temporal_distance_years":
                        abs(
                            CYCLE_MIDPOINT_YEAR[
                                comp_cycle
                            ]
                            -
                            CYCLE_MIDPOINT_YEAR[
                                ref_cycle
                            ]
                        ),
                    "feature":
                        feature,
                    "js_divergence":
                        js,
                    "max_abs_category_difference":
                        max_diff,
                }
            )

    return rows


# =============================================================================
# 20. MISSINGNESS SHIFT
# =============================================================================

def compare_missingness_cycles(
    df: pd.DataFrame,
    study: str,
    cycles: List[str],
    features: List[str],
) -> List[Dict]:

    rows = []

    available_cycles = [
        cycle
        for cycle
        in cycles
        if cycle
        in set(
            df[
                "survey_cycle"
            ]
        )
    ]

    if len(
        available_cycles
    ) < 2:
        return rows

    origin = (
        available_cycles[0]
    )

    origin_df = (
        df[
            df[
                "survey_cycle"
            ]
            == origin
        ]
    )

    for comp_cycle in available_cycles[1:]:

        comp_df = (
            df[
                df[
                    "survey_cycle"
                ]
                == comp_cycle
            ]
        )

        for feature in features:

            if feature not in df.columns:
                continue

            ref_missing = (
                origin_df[
                    feature
                ]
                .isna()
                .mean()
            )

            comp_missing = (
                comp_df[
                    feature
                ]
                .isna()
                .mean()
            )

            rows.append(
                {
                    "study":
                        study,
                    "comparison_design":
                        "fixed_origin",
                    "reference_cycle":
                        origin,
                    "comparison_cycle":
                        comp_cycle,
                    "feature":
                        feature,
                    "reference_missing_pct":
                        100
                        * ref_missing,
                    "comparison_missing_pct":
                        100
                        * comp_missing,
                    "missingness_shift_pp":
                        100
                        * (
                            comp_missing
                            - ref_missing
                        ),
                    "abs_missingness_shift_pp":
                        abs(
                            100
                            * (
                                comp_missing
                                - ref_missing
                            )
                        ),
                }
            )

    return rows


# =============================================================================
# 21. TEMPORAL DISTANCE MATRIX
# =============================================================================

def build_temporal_distance_matrix(
    cycles: List[str],
    study: str,
) -> List[Dict]:

    rows = []

    for i, cycle_a in enumerate(
        cycles
    ):

        for cycle_b in cycles[
            i + 1:
        ]:

            rows.append(
                {
                    "study":
                        study,
                    "cycle_a":
                        cycle_a,
                    "cycle_b":
                        cycle_b,
                    "midpoint_a":
                        CYCLE_MIDPOINT_YEAR[
                            cycle_a
                        ],
                    "midpoint_b":
                        CYCLE_MIDPOINT_YEAR[
                            cycle_b
                        ],
                    "temporal_distance_years":
                        abs(
                            CYCLE_MIDPOINT_YEAR[
                                cycle_b
                            ]
                            -
                            CYCLE_MIDPOINT_YEAR[
                                cycle_a
                            ]
                        ),
                }
            )

    return rows


# =============================================================================
# 22. IRON OBSERVED VS MATCHED SHIFT COMPARISON
# =============================================================================

def summarize_shift_by_pair(
    continuous_shift: pd.DataFrame,
    categorical_shift: pd.DataFrame,
    study: str,
) -> pd.DataFrame:

    continuous = (
        continuous_shift[
            continuous_shift[
                "study"
            ]
            == study
        ]
        .copy()
    )

    categorical = (
        categorical_shift[
            categorical_shift[
                "study"
            ]
            == study
        ]
        .copy()
    )

    rows = []

    pair_keys = set()

    for _, row in continuous.iterrows():

        pair_keys.add(
            (
                row[
                    "comparison_design"
                ],
                row[
                    "reference_cycle"
                ],
                row[
                    "comparison_cycle"
                ],
            )
        )

    for _, row in categorical.iterrows():

        pair_keys.add(
            (
                row[
                    "comparison_design"
                ],
                row[
                    "reference_cycle"
                ],
                row[
                    "comparison_cycle"
                ],
            )
        )

    for (
        design,
        ref,
        comp,
    ) in sorted(
        pair_keys
    ):

        c = continuous[
            (
                continuous[
                    "comparison_design"
                ]
                == design
            )
            &
            (
                continuous[
                    "reference_cycle"
                ]
                == ref
            )
            &
            (
                continuous[
                    "comparison_cycle"
                ]
                == comp
            )
        ]

        cat = categorical[
            (
                categorical[
                    "comparison_design"
                ]
                == design
            )
            &
            (
                categorical[
                    "reference_cycle"
                ]
                == ref
            )
            &
            (
                categorical[
                    "comparison_cycle"
                ]
                == comp
            )
        ]

        rows.append(
            {
                "study":
                    study,
                "comparison_design":
                    design,
                "reference_cycle":
                    ref,
                "comparison_cycle":
                    comp,
                "mean_abs_smd":
                    c[
                        "abs_smd"
                    ].mean()
                    if not c.empty
                    else np.nan,
                "max_abs_smd":
                    c[
                        "abs_smd"
                    ].max()
                    if not c.empty
                    else np.nan,
                "mean_psi":
                    c[
                        "psi"
                    ].mean()
                    if not c.empty
                    else np.nan,
                "max_psi":
                    c[
                        "psi"
                    ].max()
                    if not c.empty
                    else np.nan,
                "mean_js_divergence":
                    cat[
                        "js_divergence"
                    ].mean()
                    if not cat.empty
                    else np.nan,
                "max_js_divergence":
                    cat[
                        "js_divergence"
                    ].max()
                    if not cat.empty
                    else np.nan,
            }
        )

    return pd.DataFrame(
        rows
    )


def compare_iron_shift_reduction(
    observed_summary: pd.DataFrame,
    matched_summary: pd.DataFrame,
) -> pd.DataFrame:

    merge_keys = [
        "comparison_design",
        "reference_cycle",
        "comparison_cycle",
    ]

    merged = (
        observed_summary
        .merge(
            matched_summary,
            on=merge_keys,
            how="inner",
            suffixes=(
                "_observed",
                "_matched",
            ),
        )
    )

    if merged.empty:
        return merged

    for metric in [
        "mean_abs_smd",
        "max_abs_smd",
        "mean_psi",
        "max_psi",
        "mean_js_divergence",
        "max_js_divergence",
    ]:

        obs = (
            metric
            + "_observed"
        )

        mat = (
            metric
            + "_matched"
        )

        merged[
            metric
            + "_reduction"
        ] = (
            merged[
                obs
            ]
            -
            merged[
                mat
            ]
        )

        merged[
            metric
            + "_reduction_pct"
        ] = np.where(
            merged[
                obs
            ]
            .abs()
            > 1e-12,
            100
            * (
                merged[
                    obs
                ]
                -
                merged[
                    mat
                ]
            )
            /
            merged[
                obs
            ],
            np.nan,
        )

    return merged


# =============================================================================
# 23. SHIFT SOURCE SUMMARY
# =============================================================================

def build_shift_source_summary(
    prevalence_df: pd.DataFrame,
    continuous_shift: pd.DataFrame,
    categorical_shift: pd.DataFrame,
    iron_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Descriptive summary only.

    This table does NOT claim causal decomposition.
    It provides observed indicators for later modeling.
    """

    rows = []

    studies = [
        "vitamin_d",
        "iron_observed",
        "iron_matched",
    ]

    for study in studies:

        prevalence = (
            prevalence_df[
                prevalence_df[
                    "study"
                ]
                == study
            ]
            .copy()
        )

        cont = (
            continuous_shift[
                continuous_shift[
                    "study"
                ]
                == study
            ]
        )

        cat = (
            categorical_shift[
                categorical_shift[
                    "study"
                ]
                == study
            ]
        )

        if prevalence.empty:
            continue

        prevalence = (
            prevalence
            .sort_values(
                "cycle"
            )
        )

        baseline = (
            prevalence.iloc[0][
                "weighted_prevalence_pct"
            ]
        )

        for _, p in prevalence.iterrows():

            cycle = (
                p[
                    "cycle"
                ]
            )

            cycle_cont = cont[
                cont[
                    "comparison_cycle"
                ]
                == cycle
            ]

            cycle_cat = cat[
                cat[
                    "comparison_cycle"
                ]
                == cycle
            ]

            protocol_indicator = 0

            if study.startswith(
                "iron"
            ):

                protocol_indicator = int(
                    any(
                        item[
                            "cycle"
                        ]
                        == cycle
                        for item
                        in KNOWN_PROTOCOL_SHIFTS
                    )
                )

            rows.append(
                {
                    "study":
                        study,
                    "cycle":
                        cycle,
                    "weighted_prevalence_pct":
                        p[
                            "weighted_prevalence_pct"
                        ],
                    "prevalence_shift_from_first_pp":
                        (
                            p[
                                "weighted_prevalence_pct"
                            ]
                            - baseline
                        )
                        if (
                            not pd.isna(
                                baseline
                            )
                            and not pd.isna(
                                p[
                                    "weighted_prevalence_pct"
                                ]
                            )
                        )
                        else np.nan,
                    "mean_abs_smd_fixed_origin":
                        (
                            cycle_cont[
                                cycle_cont[
                                    "comparison_design"
                                ]
                                == "fixed_origin"
                            ][
                                "abs_smd"
                            ]
                            .mean()
                        ),
                    "max_abs_smd_fixed_origin":
                        (
                            cycle_cont[
                                cycle_cont[
                                    "comparison_design"
                                ]
                                == "fixed_origin"
                            ][
                                "abs_smd"
                            ]
                            .max()
                        ),
                    "mean_psi_fixed_origin":
                        (
                            cycle_cont[
                                cycle_cont[
                                    "comparison_design"
                                ]
                                == "fixed_origin"
                            ][
                                "psi"
                            ]
                            .mean()
                        ),
                    "mean_js_fixed_origin":
                        (
                            cycle_cat[
                                cycle_cat[
                                    "comparison_design"
                                ]
                                == "fixed_origin"
                            ][
                                "js_divergence"
                            ]
                            .mean()
                        ),
                    "known_protocol_or_eligibility_shift":
                        protocol_indicator,
                    "interpretation":
                        (
                            "Descriptive shift indicators only; "
                            "not causal decomposition."
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 24. DECISION LEDGER
# =============================================================================

def build_decision_ledger() -> pd.DataFrame:

    today = (
        datetime.now()
        .date()
        .isoformat()
    )

    rows = [

        {
            "decision_id": "D031",
            "stage": "04",
            "topic":
                "Pre-model shift analysis",
            "decision":
                (
                    "Population, outcome and missingness shift are "
                    "quantified before model development."
                ),
            "reason":
                (
                    "Prevents post-hoc explanation of later model "
                    "performance changes."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D032",
            "stage": "04",
            "topic":
                "Continuous covariate shift",
            "decision":
                (
                    "Continuous predictors are assessed using absolute SMD, "
                    "KS statistic, Wasserstein distance and PSI."
                ),
            "reason":
                (
                    "No single shift statistic captures all forms of "
                    "distributional change."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D033",
            "stage": "04",
            "topic":
                "Categorical covariate shift",
            "decision":
                (
                    "Categorical predictors are assessed with category "
                    "proportion changes and Jensen-Shannon divergence."
                ),
            "reason":
                (
                    "Numeric coding of categorical variables must not be "
                    "mistaken for continuous measurement."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D034",
            "stage": "04",
            "topic":
                "PSI binning",
            "decision":
                (
                    "PSI cut-points are derived from the reference cycle only."
                ),
            "reason":
                (
                    "Avoids allowing the future/comparison distribution to "
                    "define reference bins."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D035",
            "stage": "04",
            "topic":
                "Temporal comparisons",
            "decision":
                (
                    "Both fixed-origin and rolling cycle comparisons are "
                    "calculated."
                ),
            "reason":
                (
                    "Supports long-range transport analysis and local "
                    "cycle-to-cycle change assessment."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D036",
            "stage": "04",
            "topic":
                "Iron natural experiment",
            "decision":
                (
                    "Population shift is quantified separately in observed "
                    "and matched iron cohorts."
                ),
            "reason":
                (
                    "Allows direct assessment of how much measured shift "
                    "shrinks after common-population restriction."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D037",
            "stage": "04",
            "topic":
                "Protocol changes",
            "decision":
                (
                    "Eligibility and laboratory protocol shifts are represented "
                    "using externally documented indicators rather than inferred "
                    "from observed distributions."
                ),
            "reason":
                (
                    "Observed numeric changes alone cannot establish a "
                    "measurement-method change."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D038",
            "stage": "04",
            "topic":
                "Survey estimates",
            "decision":
                (
                    "Survey-weighted descriptive statistics are produced, "
                    "with approximate PSU/stratum linearization for QC."
                ),
            "reason":
                (
                    "NHANES population interpretation requires survey-design "
                    "awareness."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D039",
            "stage": "04",
            "topic":
                "No modeling",
            "decision":
                (
                    "No predictive model, imputation model or feature-selection "
                    "procedure is fitted during Stage 04."
                ),
            "reason":
                (
                    "Stage 04 is a data-characterization stage."
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
# 25. MAIN
# =============================================================================

def main():

    logger.info(
        "=" * 90
    )

    logger.info(
        "STAGE 04 STARTED"
    )

    logger.info(
        "=" * 90
    )

    # =========================================================================
    # LOAD INPUTS
    # =========================================================================

    manifest = (
        load_feature_manifest()
    )

    feature_blocks = (
        feature_blocks_from_manifest(
            manifest
        )
    )

    vitd = (
        load_required_csv(
            VITD_PATH,
            "Vitamin-D analytic master",
        )
    )

    iron_observed = (
        load_required_csv(
            IRON_OBSERVED_PATH,
            "Iron observed analytic master",
        )
    )

    iron_matched = (
        load_required_csv(
            IRON_MATCHED_PATH,
            "Iron matched analytic master",
        )
    )

    # =========================================================================
    # FEATURE TYPE LISTS
    # =========================================================================

    primary_features = [
        feature
        for feature
        in feature_blocks[
            "primary"
        ]
        if classify_feature_type(
            feature
        )
        != "unknown"
    ]

    continuous_features = [
        feature
        for feature
        in primary_features
        if classify_feature_type(
            feature
        )
        == "continuous"
    ]

    categorical_features = [
        feature
        for feature
        in primary_features
        if classify_feature_type(
            feature
        )
        == "categorical"
    ]

    # =========================================================================
    # COHORT QC
    # =========================================================================

    cohort_qc_records = []

    cohort_qc_records.extend(
        build_cohort_qc(
            vitd,
            "vitamin_d",
            primary_features,
        )
    )

    cohort_qc_records.extend(
        build_cohort_qc(
            iron_observed,
            "iron_observed",
            primary_features,
        )
    )

    cohort_qc_records.extend(
        build_cohort_qc(
            iron_matched,
            "iron_matched",
            primary_features,
        )
    )

    cohort_qc_df = (
        pd.DataFrame(
            cohort_qc_records
        )
    )

    # =========================================================================
    # OUTCOME PREVALENCE
    # =========================================================================

    prevalence_records = []

    prevalence_records.extend(
        build_weighted_prevalence(
            vitd,
            "vitamin_d",
        )
    )

    prevalence_records.extend(
        build_weighted_prevalence(
            iron_observed,
            "iron_observed",
        )
    )

    prevalence_records.extend(
        build_weighted_prevalence(
            iron_matched,
            "iron_matched",
        )
    )

    prevalence_df = (
        pd.DataFrame(
            prevalence_records
        )
    )

    # =========================================================================
    # CONTINUOUS DESCRIPTIVES
    # =========================================================================

    continuous_desc_records = []

    for study, df in [
        (
            "vitamin_d",
            vitd,
        ),
        (
            "iron_observed",
            iron_observed,
        ),
        (
            "iron_matched",
            iron_matched,
        ),
    ]:

        continuous_desc_records.extend(
            build_weighted_continuous_descriptives(
                df=df,
                study=study,
                continuous_features=
                    continuous_features,
            )
        )

    continuous_desc_df = (
        pd.DataFrame(
            continuous_desc_records
        )
    )

    # =========================================================================
    # CATEGORICAL DESCRIPTIVES
    # =========================================================================

    categorical_desc_records = []

    for study, df in [
        (
            "vitamin_d",
            vitd,
        ),
        (
            "iron_observed",
            iron_observed,
        ),
        (
            "iron_matched",
            iron_matched,
        ),
    ]:

        categorical_desc_records.extend(
            build_categorical_descriptives(
                df=df,
                study=study,
                categorical_features=
                    categorical_features,
            )
        )

    categorical_desc_df = (
        pd.DataFrame(
            categorical_desc_records
        )
    )

    # =========================================================================
    # MISSINGNESS
    # =========================================================================

    missingness_records = []

    for study, df in [
        (
            "vitamin_d",
            vitd,
        ),
        (
            "iron_observed",
            iron_observed,
        ),
        (
            "iron_matched",
            iron_matched,
        ),
    ]:

        missingness_records.extend(
            build_predictor_missingness(
                df,
                study,
                primary_features,
            )
        )

    missingness_df = (
        pd.DataFrame(
            missingness_records
        )
    )

    # =========================================================================
    # CONTINUOUS SHIFT
    # =========================================================================

    continuous_shift_records = []

    continuous_shift_records.extend(
        compare_continuous_cycles(
            df=vitd,
            study="vitamin_d",
            cycles=VITD_CYCLES,
            features=
                continuous_features,
        )
    )

    continuous_shift_records.extend(
        compare_continuous_cycles(
            df=iron_observed,
            study="iron_observed",
            cycles=IRON_CYCLES,
            features=
                continuous_features,
        )
    )

    continuous_shift_records.extend(
        compare_continuous_cycles(
            df=iron_matched,
            study="iron_matched",
            cycles=IRON_CYCLES,
            features=
                continuous_features,
        )
    )

    continuous_shift_df = (
        pd.DataFrame(
            continuous_shift_records
        )
    )

    # =========================================================================
    # CATEGORICAL SHIFT
    # =========================================================================

    categorical_shift_records = []

    categorical_shift_records.extend(
        compare_categorical_cycles(
            df=vitd,
            study="vitamin_d",
            cycles=VITD_CYCLES,
            features=
                categorical_features,
        )
    )

    categorical_shift_records.extend(
        compare_categorical_cycles(
            df=iron_observed,
            study="iron_observed",
            cycles=IRON_CYCLES,
            features=
                categorical_features,
        )
    )

    categorical_shift_records.extend(
        compare_categorical_cycles(
            df=iron_matched,
            study="iron_matched",
            cycles=IRON_CYCLES,
            features=
                categorical_features,
        )
    )

    categorical_shift_df = (
        pd.DataFrame(
            categorical_shift_records
        )
    )

    # =========================================================================
    # MISSINGNESS SHIFT
    # =========================================================================

    missingness_shift_records = []

    missingness_shift_records.extend(
        compare_missingness_cycles(
            vitd,
            "vitamin_d",
            VITD_CYCLES,
            primary_features,
        )
    )

    missingness_shift_records.extend(
        compare_missingness_cycles(
            iron_observed,
            "iron_observed",
            IRON_CYCLES,
            primary_features,
        )
    )

    missingness_shift_records.extend(
        compare_missingness_cycles(
            iron_matched,
            "iron_matched",
            IRON_CYCLES,
            primary_features,
        )
    )

    missingness_shift_df = (
        pd.DataFrame(
            missingness_shift_records
        )
    )

    # =========================================================================
    # TEMPORAL DISTANCE
    # =========================================================================

    temporal_distance_records = []

    temporal_distance_records.extend(
        build_temporal_distance_matrix(
            VITD_CYCLES,
            "vitamin_d",
        )
    )

    temporal_distance_records.extend(
        build_temporal_distance_matrix(
            IRON_CYCLES,
            "iron",
        )
    )

    temporal_distance_df = (
        pd.DataFrame(
            temporal_distance_records
        )
    )

    # =========================================================================
    # IRON OBSERVED VS MATCHED SHIFT
    # =========================================================================

    observed_summary = (
        summarize_shift_by_pair(
            continuous_shift_df,
            categorical_shift_df,
            "iron_observed",
        )
    )

    matched_summary = (
        summarize_shift_by_pair(
            continuous_shift_df,
            categorical_shift_df,
            "iron_matched",
        )
    )

    iron_shift_comparison_df = (
        compare_iron_shift_reduction(
            observed_summary,
            matched_summary,
        )
    )

    # =========================================================================
    # KNOWN PROTOCOL INDICATORS
    # =========================================================================

    protocol_df = (
        pd.DataFrame(
            KNOWN_PROTOCOL_SHIFTS
        )
    )

    # =========================================================================
    # SHIFT SOURCE SUMMARY
    # =========================================================================

    shift_source_df = (
        build_shift_source_summary(
            prevalence_df=
                prevalence_df,
            continuous_shift=
                continuous_shift_df,
            categorical_shift=
                categorical_shift_df,
            iron_comparison=
                iron_shift_comparison_df,
        )
    )

    # =========================================================================
    # DECISION LEDGER
    # =========================================================================

    decision_ledger_df = (
        build_decision_ledger()
    )

    # =========================================================================
    # SAVE CSV OUTPUTS
    # =========================================================================

    output_tables = {
        "04_cohort_qc.csv":
            cohort_qc_df,

        "04_weighted_outcome_prevalence.csv":
            prevalence_df,

        "04_weighted_continuous_descriptives.csv":
            continuous_desc_df,

        "04_categorical_descriptives.csv":
            categorical_desc_df,

        "04_predictor_missingness_by_cycle.csv":
            missingness_df,

        "04_continuous_shift_metrics.csv":
            continuous_shift_df,

        "04_categorical_shift_metrics.csv":
            categorical_shift_df,

        "04_missingness_shift_metrics.csv":
            missingness_shift_df,

        "04_temporal_distance_matrix.csv":
            temporal_distance_df,

        "04_iron_observed_vs_matched_shift.csv":
            iron_shift_comparison_df,

        "04_shift_source_summary.csv":
            shift_source_df,

        "04_known_protocol_shift_indicators.csv":
            protocol_df,

        "decision_ledger_stage04.csv":
            decision_ledger_df,
    }

    for filename, table in output_tables.items():

        table.to_csv(
            OUTPUT_DIR
            / filename,
            index=False,
        )

    # =========================================================================
    # EXCEL SUMMARY
    # =========================================================================

    workbook_path = (
        OUTPUT_DIR
        / "04_stage04_summary.xlsx"
    )

    with pd.ExcelWriter(
        workbook_path,
        engine="openpyxl",
    ) as writer:

        cohort_qc_df.to_excel(
            writer,
            sheet_name="Cohort_QC",
            index=False,
        )

        prevalence_df.to_excel(
            writer,
            sheet_name="Outcome_Prevalence",
            index=False,
        )

        continuous_desc_df.to_excel(
            writer,
            sheet_name="Continuous_Desc",
            index=False,
        )

        categorical_desc_df.to_excel(
            writer,
            sheet_name="Categorical_Desc",
            index=False,
        )

        missingness_df.to_excel(
            writer,
            sheet_name="Missingness",
            index=False,
        )

        continuous_shift_df.to_excel(
            writer,
            sheet_name="Continuous_Shift",
            index=False,
        )

        categorical_shift_df.to_excel(
            writer,
            sheet_name="Categorical_Shift",
            index=False,
        )

        missingness_shift_df.to_excel(
            writer,
            sheet_name="Missingness_Shift",
            index=False,
        )

        iron_shift_comparison_df.to_excel(
            writer,
            sheet_name="Iron_Observed_vs_Matched",
            index=False,
        )

        shift_source_df.to_excel(
            writer,
            sheet_name="Shift_Source_Summary",
            index=False,
        )

        protocol_df.to_excel(
            writer,
            sheet_name="Protocol_Indicators",
            index=False,
        )

        temporal_distance_df.to_excel(
            writer,
            sheet_name="Temporal_Distance",
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
        "STAGE 04 COMPLETE"
    )
    print(
        "=" * 90
    )

    print(
        "\nCOHORT QC"
    )
    print(
        "-" * 90
    )

    if not cohort_qc_df.empty:

        cols = [
            "study",
            "cycle",
            "n_rows",
            "duplicate_seqn",
            "missing_primary_outcome",
            "n_feature_columns_present",
            "n_completely_missing_features",
        ]

        print(
            cohort_qc_df[
                cols
            ]
            .to_string(
                index=False
            )
        )

    print(
        "\nWEIGHTED OUTCOME PREVALENCE"
    )
    print(
        "-" * 90
    )

    if not prevalence_df.empty:

        display_cols = [
            "study",
            "cycle",
            "n",
            "cases",
            "unweighted_prevalence_pct",
            "weighted_prevalence_pct",
            "weighted_ci95_lower_pct",
            "weighted_ci95_upper_pct",
        ]

        print(
            prevalence_df[
                display_cols
            ]
            .round(2)
            .to_string(
                index=False
            )
        )

    print(
        "\nIRON OBSERVED VS MATCHED SHIFT"
    )
    print(
        "-" * 90
    )

    if not iron_shift_comparison_df.empty:

        display_cols = [
            "comparison_design",
            "reference_cycle",
            "comparison_cycle",
            "mean_abs_smd_observed",
            "mean_abs_smd_matched",
            "mean_abs_smd_reduction_pct",
            "mean_psi_observed",
            "mean_psi_matched",
            "mean_psi_reduction_pct",
        ]

        existing = [
            c
            for c in display_cols
            if c
            in iron_shift_comparison_df.columns
        ]

        print(
            iron_shift_comparison_df[
                existing
            ]
            .round(3)
            .to_string(
                index=False
            )
        )

    print(
        "\nPRIMARY WORKBOOK:"
    )

    print(
        f"  {workbook_path}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  • No models have been trained."
    )

    print(
        "  • No missing-value imputation has been performed."
    )

    print(
        "  • PSI bins are derived from reference cycles only."
    )

    print(
        "  • Categorical variables are not treated as continuous."
    )

    print(
        "  • Protocol-shift indicators come from documented Stage-02A evidence."
    )

    print(
        "  • Survey CIs in Stage 04 are approximate QC estimates."
    )

    print(
        "\nNEXT STEP AFTER REVIEW:"
    )

    print(
        "  Stage 05 — Locked Temporal Modeling Design, "
        "Preprocessing Pipelines, and Experiment Manifest."
    )

    print(
        "\n" + "=" * 90
    )


# =============================================================================
# 26. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()