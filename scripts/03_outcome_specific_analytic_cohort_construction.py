"""
===============================================================================
STAGE 03
NHANES Outcome-Specific Analytic Cohort Construction
and Eligibility Harmonization
===============================================================================



===============================================================================
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

JOINED_MAIN_DIR = (
    PROJECT_ROOT
    / "processed"
    / "02_joined_native"
    / "main_cycles"
)

JOINED_SUPP_DIR = (
    PROJECT_ROOT
    / "processed"
    / "02_joined_native"
    / "supplementary"
)

OUTCOME_AUDIT_DIR = (
    PROJECT_ROOT
    / "processed"
    / "03_outcome_audit"
)

PREDICTOR_DIR = (
    PROJECT_ROOT
    / "processed"
    / "04_predictor_harmonization"
)

PREDICTOR_CYCLE_DIR = (
    PREDICTOR_DIR
    / "harmonized_cycles"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed"
    / "05_analytic_cohorts"
)

AUDIT_DIR = OUTPUT_DIR / "audits"

VITD_DIR = OUTPUT_DIR / "vitamin_d"
VITD_CYCLE_DIR = VITD_DIR / "cycle_files"

IRON_DIR = OUTPUT_DIR / "iron"

IRON_OBSERVED_DIR = IRON_DIR / "observed"
IRON_OBSERVED_CYCLE_DIR = (
    IRON_OBSERVED_DIR
    / "cycle_files"
)

IRON_MATCHED_DIR = IRON_DIR / "matched"
IRON_MATCHED_CYCLE_DIR = (
    IRON_MATCHED_DIR
    / "cycle_files"
)

IRON_SUPP_DIR = IRON_DIR / "supplementary"

LOG_DIR = PROJECT_ROOT / "processed" / "logs"


for folder in [
    OUTPUT_DIR,
    AUDIT_DIR,
    VITD_DIR,
    VITD_CYCLE_DIR,
    IRON_DIR,
    IRON_OBSERVED_DIR,
    IRON_OBSERVED_CYCLE_DIR,
    IRON_MATCHED_DIR,
    IRON_MATCHED_CYCLE_DIR,
    IRON_SUPP_DIR,
    LOG_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# 2. CYCLE CONFIGURATION
# =============================================================================

MAIN_CYCLES = [
    "2009_2010",
    "2011_2012",
    "2013_2014",
    "2015_2016",
    "2017_2018",
    "2021_2023",
]

SUPPLEMENTARY_CYCLES = [
    "2017_2020_PrePandemic",
]

ALL_CYCLES = (
    MAIN_CYCLES
    + SUPPLEMENTARY_CYCLES
)

CYCLE_ORDER = {
    "2009_2010": 1,
    "2011_2012": 2,
    "2013_2014": 3,
    "2015_2016": 4,
    "2017_2018": 5,
    "2017_2020_PrePandemic": 6,
    "2021_2023": 7,
}

CYCLE_LABELS = {
    "2009_2010": "2009–2010",
    "2011_2012": "2011–2012",
    "2013_2014": "2013–2014",
    "2015_2016": "2015–2016",
    "2017_2018": "2017–2018",
    "2017_2020_PrePandemic":
        "2017–March 2020 pre-pandemic",
    "2021_2023": "2021–2023",
}


# =============================================================================
# 3. ANALYTIC AGE / OUTCOME DEFINITIONS
# =============================================================================

PRIMARY_MIN_AGE = 18
PRIMARY_MAX_AGE = 59

MATCHED_IRON_MIN_AGE = 18
MATCHED_IRON_MAX_AGE = 49

FEMALE_CODE = 2

PREGNANT_CODE = 1


# Vitamin D
VITD_PRIMARY_THRESHOLD_NMOL_L = 50.0
VITD_SENSITIVITY_THRESHOLD_NMOL_L = 30.0


# Ferritin
FERRITIN_PRIMARY_THRESHOLD_UG_L = 15.0
FERRITIN_SENSITIVITY_THRESHOLD_UG_L = 25.0


# Candidate sex-specific anemia thresholds
HGB_MALE_ANEMIA_G_DL = 13.0
HGB_FEMALE_ANEMIA_G_DL = 12.0


# =============================================================================
# 4. LOGGING
# =============================================================================

LOG_FILE = (
    LOG_DIR
    / "stage03_analytic_cohort_construction.log"
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
# 5. INPUT FILE HELPERS
# =============================================================================

def joined_file_path(
    cycle: str,
) -> Optional[Path]:

    filename = (
        f"NHANES_{cycle}_joined_native.csv"
    )

    if cycle in MAIN_CYCLES:
        path = JOINED_MAIN_DIR / filename
    else:
        path = JOINED_SUPP_DIR / filename

    if path.exists():
        return path

    return None


def predictor_file_path(
    cycle: str,
) -> Optional[Path]:

    path = (
        PREDICTOR_CYCLE_DIR
        / (
            f"NHANES_{cycle}"
            "_predictors_harmonized.csv"
        )
    )

    if path.exists():
        return path

    return None


# =============================================================================
# 6. BASIC HELPERS
# =============================================================================

def safe_numeric(
    series: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def pct(
    numerator: float,
    denominator: float,
) -> float:

    if (
        denominator is None
        or denominator == 0
        or pd.isna(denominator)
    ):
        return np.nan

    return (
        100.0
        * float(numerator)
        / float(denominator)
    )


def weighted_mean_binary(
    y: pd.Series,
    weights: pd.Series,
) -> float:
    """
    Descriptive weighted prevalence for a binary outcome.

    No design-based variance estimate is calculated here.
    """

    y = safe_numeric(y)
    w = safe_numeric(weights)

    valid = (
        y.notna()
        & w.notna()
        & (w > 0)
    )

    if valid.sum() == 0:
        return np.nan

    return float(
        np.average(
            y.loc[valid],
            weights=w.loc[valid],
        )
    )


# =============================================================================
# 7. LOAD STAGE 02 METADATA
# =============================================================================

def load_outcome_availability() -> pd.DataFrame:

    path = (
        OUTCOME_AUDIT_DIR
        / "02A_outcome_availability_matrix.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            "Stage 02A outcome availability file not found:\n"
            f"{path}"
        )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def load_final_feature_manifest() -> pd.DataFrame:

    path = (
        PREDICTOR_DIR
        / "02B_final_feature_set.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            "Stage 02B final feature manifest not found:\n"
            f"{path}"
        )

    return pd.read_csv(
        path,
        low_memory=False,
    )


# =============================================================================
# 8. FEATURE SET EXTRACTION
# =============================================================================

def get_feature_blocks(
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

    return {
        "tier1": t1,
        "cbc": cbc,
        "chemistry": chemistry,
        "tier2_primary": t1 + cbc,
        "tier2_extended":
            t1 + cbc + chemistry,
    }


# =============================================================================
# 9. OUTCOME VARIABLE RESOLUTION
# =============================================================================

def resolve_outcome_variables(
    cycle: str,
    outcome_availability: pd.DataFrame,
) -> Dict[str, Optional[str]]:

    subset = outcome_availability[
        outcome_availability[
            "cycle"
        ] == cycle
    ]

    if subset.empty:

        return {
            "vitamin_d": None,
            "ferritin": None,
            "hemoglobin": None,
            "pregnancy": None,
            "age": None,
            "sex": None,
        }

    row = subset.iloc[0]

    return {
        "vitamin_d":
            row.get(
                "vitamin_d_variable"
            ),
        "ferritin":
            row.get(
                "ferritin_variable"
            ),
        "hemoglobin":
            row.get(
                "hemoglobin_variable"
            ),
        "pregnancy":
            row.get(
                "pregnancy_variable"
            ),
        "age":
            row.get(
                "age_variable"
            ),
        "sex":
            row.get(
                "sex_variable"
            ),
    }


def normalize_optional_string(
    value,
) -> Optional[str]:

    if value is None:
        return None

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    return value


# =============================================================================
# 10. SURVEY DESIGN VARIABLE RESOLUTION
# =============================================================================

def resolve_survey_design_vars(
    cycle: str,
    columns: List[str],
) -> Dict[str, Optional[str]]:
    """
    Resolve survey design variables.

    Main F/G/H/I/J/L cycles:
        WTMEC2YR

    Supplementary P cycle:
        WTMECPRP preferred
        WTMEC2YR fallback only for audit if present

    Stratum:
        SDMVSTRA

    PSU:
        SDMVPSU
    """

    weight_candidates = []

    if cycle == "2017_2020_PrePandemic":

        weight_candidates = [
            "WTMECPRP",
            "WTMEC2YR",
        ]

    else:

        weight_candidates = [
            "WTMEC2YR",
        ]

    weight_var = next(
        (
            variable
            for variable
            in weight_candidates
            if variable in columns
        ),
        None,
    )

    stratum_var = (
        "SDMVSTRA"
        if "SDMVSTRA" in columns
        else None
    )

    psu_var = (
        "SDMVPSU"
        if "SDMVPSU" in columns
        else None
    )

    return {
        "weight_variable": weight_var,
        "stratum_variable": stratum_var,
        "psu_variable": psu_var,
    }


# =============================================================================
# 11. PREGNANCY HANDLING
# =============================================================================

def known_pregnant_mask(
    native: pd.DataFrame,
    pregnancy_var: Optional[str],
) -> pd.Series:
    """
    Exclude only participants explicitly coded pregnant.

    Missing pregnancy status is NOT automatically interpreted as pregnant.

    This is important because RIDEXPRG is only administered / defined for
    specific female age groups.
    """

    if (
        pregnancy_var is None
        or pregnancy_var not in native.columns
    ):

        return pd.Series(
            False,
            index=native.index,
        )

    pregnancy = safe_numeric(
        native[pregnancy_var]
    )

    return (
        pregnancy == PREGNANT_CODE
    )


# =============================================================================
# 12. ANEMIA CLASSIFICATION
# =============================================================================

def classify_candidate_anemia(
    sex: pd.Series,
    hemoglobin: pd.Series,
) -> pd.Series:
    """
    Candidate adult anemia classification.

    Male:
        Hgb < 13 g/dL

    Female:
        Hgb < 12 g/dL

    Missing sex or Hgb -> NA.
    """

    sex = safe_numeric(sex)
    hgb = safe_numeric(hemoglobin)

    anemia = pd.Series(
        pd.NA,
        index=sex.index,
        dtype="boolean",
    )

    male_valid = (
        (sex == 1)
        & hgb.notna()
    )

    female_valid = (
        (sex == 2)
        & hgb.notna()
    )

    anemia.loc[
        male_valid
    ] = (
        hgb.loc[
            male_valid
        ]
        < HGB_MALE_ANEMIA_G_DL
    )

    anemia.loc[
        female_valid
    ] = (
        hgb.loc[
            female_valid
        ]
        < HGB_FEMALE_ANEMIA_G_DL
    )

    return anemia


# =============================================================================
# 13. MERGE NATIVE + HARMONIZED PREDICTORS
# =============================================================================

def merge_native_predictors(
    cycle: str,
    native: pd.DataFrame,
    predictors: pd.DataFrame,
) -> pd.DataFrame:

    if native["SEQN"].duplicated().any():

        raise RuntimeError(
            f"{cycle}: duplicate SEQN in native file."
        )

    if predictors["SEQN"].duplicated().any():

        raise RuntimeError(
            f"{cycle}: duplicate SEQN in predictor file."
        )

    merged = native.merge(
        predictors,
        on="SEQN",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "__PRED",
        ),
    )

    if len(merged) != len(native):

        raise RuntimeError(
            f"{cycle}: participant count changed "
            "after predictor merge."
        )

    return merged


# =============================================================================
# 14. ANALYTIC OUTPUT COLUMN BUILDER
# =============================================================================

def build_output_columns(
    df: pd.DataFrame,
    feature_blocks: Dict[str, List[str]],
    survey_vars: Dict[str, Optional[str]],
    extra_columns: List[str],
) -> List[str]:

    columns = [
        "SEQN",
        "survey_cycle",
        "cycle_order",
        "dataset_role",
    ]

    design_columns = [
        survey_vars[
            "weight_variable"
        ],
        survey_vars[
            "stratum_variable"
        ],
        survey_vars[
            "psu_variable"
        ],
    ]

    columns.extend(
        [
            col
            for col in design_columns
            if col is not None
            and col in df.columns
        ]
    )

    columns.extend(
        [
            col
            for col
            in feature_blocks[
                "tier1"
            ]
            if col in df.columns
        ]
    )

    columns.extend(
        [
            col
            for col
            in feature_blocks[
                "cbc"
            ]
            if col in df.columns
        ]
    )

    columns.extend(
        [
            col
            for col
            in feature_blocks[
                "chemistry"
            ]
            if col in df.columns
        ]
    )

    columns.extend(
        [
            col
            for col in extra_columns
            if col in df.columns
        ]
    )

    # Deduplicate while preserving order.
    seen = set()
    ordered = []

    for col in columns:

        if col not in seen:
            seen.add(col)
            ordered.append(col)

    return ordered


# =============================================================================
# 15. COHORT FLOW AUDIT
# =============================================================================

def append_flow(
    records: List[Dict],
    cycle: str,
    outcome_family: str,
    cohort_type: str,
    step_number: int,
    step_name: str,
    mask: pd.Series,
    previous_n: Optional[int],
):

    n_remaining = int(
        mask.sum()
    )

    records.append(
        {
            "cycle": cycle,
            "cycle_label":
                CYCLE_LABELS.get(
                    cycle,
                    cycle,
                ),
            "dataset_role":
                (
                    "main"
                    if cycle in MAIN_CYCLES
                    else "supplementary"
                ),
            "outcome_family":
                outcome_family,
            "cohort_type":
                cohort_type,
            "step_number":
                step_number,
            "step_name":
                step_name,
            "n_remaining":
                n_remaining,
            "n_removed_from_previous":
                (
                    np.nan
                    if previous_n is None
                    else previous_n
                    - n_remaining
                ),
        }
    )


# =============================================================================
# 16. VITAMIN D COHORT
# =============================================================================

def construct_vitamin_d_cohort(
    cycle: str,
    merged: pd.DataFrame,
    outcome_vars: Dict[str, Optional[str]],
    feature_blocks: Dict[str, List[str]],
    flow_records: List[Dict],
) -> Tuple[
    Optional[pd.DataFrame],
    List[Dict],
    Dict,
]:

    vitd_var = normalize_optional_string(
        outcome_vars[
            "vitamin_d"
        ]
    )

    if (
        vitd_var is None
        or vitd_var not in merged.columns
    ):

        return (
            None,
            [],
            {
                "cycle": cycle,
                "status":
                    "vitamin_d_not_available",
            },
        )

    if "age_years" not in merged.columns:

        raise RuntimeError(
            f"{cycle}: harmonized age_years missing."
        )

    if "sex" not in merged.columns:

        raise RuntimeError(
            f"{cycle}: harmonized sex missing."
        )

    pregnancy_var = (
        normalize_optional_string(
            outcome_vars[
                "pregnancy"
            ]
        )
    )

    vitamin_d = safe_numeric(
        merged[vitd_var]
    )

    age = safe_numeric(
        merged["age_years"]
    )

    pregnant = known_pregnant_mask(
        merged,
        pregnancy_var,
    )

    # -------------------------------------------------------------------------
    # Cohort flow
    # -------------------------------------------------------------------------

    mask = pd.Series(
        True,
        index=merged.index,
    )

    previous_n = None

    append_flow(
        flow_records,
        cycle,
        "vitamin_d",
        "primary",
        0,
        "native_joined_population",
        mask,
        previous_n,
    )

    previous_n = int(
        mask.sum()
    )

    mask = (
        mask
        & age.notna()
        & (age >= PRIMARY_MIN_AGE)
        & (age <= PRIMARY_MAX_AGE)
    )

    append_flow(
        flow_records,
        cycle,
        "vitamin_d",
        "primary",
        1,
        "age_18_59",
        mask,
        previous_n,
    )

    previous_n = int(
        mask.sum()
    )

    mask = (
        mask
        & (~pregnant)
    )

    append_flow(
        flow_records,
        cycle,
        "vitamin_d",
        "primary",
        2,
        "exclude_known_pregnancy",
        mask,
        previous_n,
    )

    previous_n = int(
        mask.sum()
    )

    mask = (
        mask
        & vitamin_d.notna()
    )

    append_flow(
        flow_records,
        cycle,
        "vitamin_d",
        "primary",
        3,
        "valid_25ohd_measurement",
        mask,
        previous_n,
    )

    # -------------------------------------------------------------------------
    # Construct final cohort
    # -------------------------------------------------------------------------

    cohort = merged.loc[
        mask
    ].copy()

    cohort[
        "vitamin_d_nmol_L"
    ] = safe_numeric(
        cohort[vitd_var]
    )

    cohort[
        "outcome_vitd_lt50"
    ] = (
        cohort[
            "vitamin_d_nmol_L"
        ]
        < VITD_PRIMARY_THRESHOLD_NMOL_L
    ).astype(int)

    cohort[
        "outcome_vitd_lt30"
    ] = (
        cohort[
            "vitamin_d_nmol_L"
        ]
        < VITD_SENSITIVITY_THRESHOLD_NMOL_L
    ).astype(int)

    cohort[
        "primary_outcome"
    ] = cohort[
        "outcome_vitd_lt50"
    ]

    cohort[
        "outcome_family"
    ] = "vitamin_d"

    cohort[
        "cohort_type"
    ] = "primary"

    cohort[
        "dataset_role"
    ] = (
        "main"
        if cycle in MAIN_CYCLES
        else "supplementary"
    )

    cohort[
        "cycle_order"
    ] = CYCLE_ORDER[
        cycle
    ]

    cohort[
        "survey_cycle"
    ] = cycle

    survey_vars = (
        resolve_survey_design_vars(
            cycle,
            list(
                merged.columns
            ),
        )
    )

    extra_columns = [
        "vitamin_d_nmol_L",
        "outcome_vitd_lt50",
        "outcome_vitd_lt30",
        "primary_outcome",
        "outcome_family",
        "cohort_type",
    ]

    output_columns = (
        build_output_columns(
            cohort,
            feature_blocks,
            survey_vars,
            extra_columns,
        )
    )

    cohort = cohort[
        output_columns
    ].copy()

    prevalence_records = (
        build_prevalence_records(
            cohort=cohort,
            cycle=cycle,
            outcome_family=
                "vitamin_d",
            cohort_type=
                "primary",
            outcome_columns=[
                "outcome_vitd_lt50",
                "outcome_vitd_lt30",
            ],
            survey_vars=
                survey_vars,
        )
    )

    audit = {
        "cycle": cycle,
        "status": "constructed",
        "n_final":
            len(cohort),
        "vitamin_d_variable":
            vitd_var,
        "weight_variable":
            survey_vars[
                "weight_variable"
            ],
    }

    return (
        cohort,
        prevalence_records,
        audit,
    )


# =============================================================================
# 17. IRON COHORT CORE CONSTRUCTION
# =============================================================================

def prepare_iron_base(
    cycle: str,
    merged: pd.DataFrame,
    outcome_vars: Dict[str, Optional[str]],
) -> Optional[pd.DataFrame]:

    ferritin_var = (
        normalize_optional_string(
            outcome_vars[
                "ferritin"
            ]
        )
    )

    hgb_var = (
        normalize_optional_string(
            outcome_vars[
                "hemoglobin"
            ]
        )
    )

    if (
        ferritin_var is None
        or ferritin_var not in merged.columns
    ):
        return None

    if (
        hgb_var is None
        or hgb_var not in merged.columns
    ):
        return None

    if (
        "age_years"
        not in merged.columns
    ):

        raise RuntimeError(
            f"{cycle}: age_years unavailable."
        )

    if "sex" not in merged.columns:

        raise RuntimeError(
            f"{cycle}: sex unavailable."
        )

    out = merged.copy()

    out[
        "_iron_ferritin"
    ] = safe_numeric(
        out[ferritin_var]
    )

    out[
        "_iron_hgb"
    ] = safe_numeric(
        out[hgb_var]
    )

    out[
        "_iron_age"
    ] = safe_numeric(
        out["age_years"]
    )

    out[
        "_iron_sex"
    ] = safe_numeric(
        out["sex"]
    )

    out[
        "_iron_anemia"
    ] = classify_candidate_anemia(
        sex=out[
            "_iron_sex"
        ],
        hemoglobin=out[
            "_iron_hgb"
        ],
    )

    pregnancy_var = (
        normalize_optional_string(
            outcome_vars[
                "pregnancy"
            ]
        )
    )

    out[
        "_iron_known_pregnant"
    ] = known_pregnant_mask(
        out,
        pregnancy_var,
    )

    out.attrs[
        "ferritin_var"
    ] = ferritin_var

    out.attrs[
        "hgb_var"
    ] = hgb_var

    out.attrs[
        "pregnancy_var"
    ] = pregnancy_var

    return out


# =============================================================================
# 18. IRON OBSERVED COHORT
# =============================================================================

def construct_iron_observed_cohort(
    cycle: str,
    iron_base: pd.DataFrame,
    feature_blocks: Dict[str, List[str]],
    flow_records: List[Dict],
) -> Tuple[
    pd.DataFrame,
    List[Dict],
]:

    mask = pd.Series(
        True,
        index=iron_base.index,
    )

    previous_n = None

    append_flow(
        flow_records,
        cycle,
        "iron",
        "observed",
        0,
        "native_joined_population",
        mask,
        previous_n,
    )

    previous_n = int(
        mask.sum()
    )

    mask = (
        mask
        & iron_base[
            "_iron_age"
        ].notna()
        & (
            iron_base[
                "_iron_age"
            ]
            >= PRIMARY_MIN_AGE
        )
        & (
            iron_base[
                "_iron_age"
            ]
            <= PRIMARY_MAX_AGE
        )
    )

    append_flow(
        flow_records,
        cycle,
        "iron",
        "observed",
        1,
        "age_18_59",
        mask,
        previous_n,
    )

    previous_n = int(
        mask.sum()
    )

    mask = (
        mask
        & (
            ~iron_base[
                "_iron_known_pregnant"
            ]
        )
    )

    append_flow(
        flow_records,
        cycle,
        "iron",
        "observed",
        2,
        "exclude_known_pregnancy",
        mask,
        previous_n,
    )

    previous_n = int(
        mask.sum()
    )

    mask = (
        mask
        & iron_base[
            "_iron_ferritin"
        ].notna()
    )

    append_flow(
        flow_records,
        cycle,
        "iron",
        "observed",
        3,
        "ferritin_observed",
        mask,
        previous_n,
    )

    previous_n = int(
        mask.sum()
    )

    mask = (
        mask
        & iron_base[
            "_iron_hgb"
        ].notna()
        & iron_base[
            "_iron_anemia"
        ].notna()
    )

    append_flow(
        flow_records,
        cycle,
        "iron",
        "observed",
        4,
        "hemoglobin_and_anemia_status_available",
        mask,
        previous_n,
    )

    previous_n = int(
        mask.sum()
    )

    mask = (
        mask
        & (
            ~iron_base[
                "_iron_anemia"
            ].fillna(
                True
            )
        )
    )

    append_flow(
        flow_records,
        cycle,
        "iron",
        "observed",
        5,
        "exclude_anemia",
        mask,
        previous_n,
    )

    cohort = (
        iron_base.loc[
            mask
        ]
        .copy()
    )

    cohort = finalize_iron_cohort(
        cycle=cycle,
        cohort=cohort,
        cohort_type="observed",
        feature_blocks=
            feature_blocks,
    )

    survey_vars = (
        resolve_survey_design_vars(
            cycle,
            list(
                iron_base.columns
            ),
        )
    )

    prevalence_records = (
        build_prevalence_records(
            cohort=cohort,
            cycle=cycle,
            outcome_family="iron",
            cohort_type="observed",
            outcome_columns=[
                "outcome_iron_lt15",
                "outcome_iron_lt25",
            ],
            survey_vars=
                survey_vars,
        )
    )

    return (
        cohort,
        prevalence_records,
    )


# =============================================================================
# 19. IRON MATCHED COHORT
# =============================================================================

def construct_iron_matched_cohort(
    cycle: str,
    iron_base: pd.DataFrame,
    feature_blocks: Dict[str, List[str]],
    flow_records: List[Dict],
) -> Tuple[
    pd.DataFrame,
    List[Dict],
]:

    mask = pd.Series(
        True,
        index=iron_base.index,
    )

    previous_n = None

    append_flow(
        flow_records,
        cycle,
        "iron",
        "matched",
        0,
        "native_joined_population",
        mask,
        previous_n,
    )

    previous_n = int(
        mask.sum()
    )

    mask = (
        mask
        & (
            iron_base[
                "_iron_sex"
            ]
            == FEMALE_CODE
        )
    )

    append_flow(
        flow_records,
        cycle,
        "iron",
        "matched",
        1,
        "female",
        mask,
        previous_n,
    )

    previous_n = int(
        mask.sum()
    )

    mask = (
        mask
        & iron_base[
            "_iron_age"
        ].notna()
        & (
            iron_base[
                "_iron_age"
            ]
            >= MATCHED_IRON_MIN_AGE
        )
        & (
            iron_base[
                "_iron_age"
            ]
            <= MATCHED_IRON_MAX_AGE
        )
    )

    append_flow(
        flow_records,
        cycle,
        "iron",
        "matched",
        2,
        "age_18_49",
        mask,
        previous_n,
    )

    previous_n = int(
        mask.sum()
    )

    mask = (
        mask
        & (
            ~iron_base[
                "_iron_known_pregnant"
            ]
        )
    )

    append_flow(
        flow_records,
        cycle,
        "iron",
        "matched",
        3,
        "exclude_known_pregnancy",
        mask,
        previous_n,
    )

    previous_n = int(
        mask.sum()
    )

    mask = (
        mask
        & iron_base[
            "_iron_ferritin"
        ].notna()
    )

    append_flow(
        flow_records,
        cycle,
        "iron",
        "matched",
        4,
        "ferritin_observed",
        mask,
        previous_n,
    )

    previous_n = int(
        mask.sum()
    )

    mask = (
        mask
        & iron_base[
            "_iron_hgb"
        ].notna()
        & iron_base[
            "_iron_anemia"
        ].notna()
    )

    append_flow(
        flow_records,
        cycle,
        "iron",
        "matched",
        5,
        "hemoglobin_and_anemia_status_available",
        mask,
        previous_n,
    )

    previous_n = int(
        mask.sum()
    )

    mask = (
        mask
        & (
            ~iron_base[
                "_iron_anemia"
            ].fillna(
                True
            )
        )
    )

    append_flow(
        flow_records,
        cycle,
        "iron",
        "matched",
        6,
        "exclude_anemia",
        mask,
        previous_n,
    )

    cohort = (
        iron_base.loc[
            mask
        ]
        .copy()
    )

    cohort = finalize_iron_cohort(
        cycle=cycle,
        cohort=cohort,
        cohort_type="matched",
        feature_blocks=
            feature_blocks,
    )

    survey_vars = (
        resolve_survey_design_vars(
            cycle,
            list(
                iron_base.columns
            ),
        )
    )

    prevalence_records = (
        build_prevalence_records(
            cohort=cohort,
            cycle=cycle,
            outcome_family="iron",
            cohort_type="matched",
            outcome_columns=[
                "outcome_iron_lt15",
                "outcome_iron_lt25",
            ],
            survey_vars=
                survey_vars,
        )
    )

    return (
        cohort,
        prevalence_records,
    )


# =============================================================================
# 20. FINALIZE IRON COHORT
# =============================================================================

def finalize_iron_cohort(
    cycle: str,
    cohort: pd.DataFrame,
    cohort_type: str,
    feature_blocks: Dict[str, List[str]],
) -> pd.DataFrame:

    cohort[
        "ferritin_ug_L"
    ] = safe_numeric(
        cohort[
            "_iron_ferritin"
        ]
    )

    cohort[
        "hemoglobin_g_dL"
    ] = safe_numeric(
        cohort[
            "_iron_hgb"
        ]
    )

    cohort[
        "outcome_iron_lt15"
    ] = (
        cohort[
            "ferritin_ug_L"
        ]
        < FERRITIN_PRIMARY_THRESHOLD_UG_L
    ).astype(int)

    cohort[
        "outcome_iron_lt25"
    ] = (
        cohort[
            "ferritin_ug_L"
        ]
        < FERRITIN_SENSITIVITY_THRESHOLD_UG_L
    ).astype(int)

    cohort[
        "primary_outcome"
    ] = cohort[
        "outcome_iron_lt15"
    ]

    cohort[
        "outcome_family"
    ] = "iron"

    cohort[
        "cohort_type"
    ] = cohort_type

    cohort[
        "dataset_role"
    ] = (
        "main"
        if cycle in MAIN_CYCLES
        else "supplementary"
    )

    cohort[
        "cycle_order"
    ] = CYCLE_ORDER[
        cycle
    ]

    cohort[
        "survey_cycle"
    ] = cycle

    survey_vars = (
        resolve_survey_design_vars(
            cycle,
            list(
                cohort.columns
            ),
        )
    )

    extra_columns = [
        "ferritin_ug_L",
        "hemoglobin_g_dL",
        "outcome_iron_lt15",
        "outcome_iron_lt25",
        "primary_outcome",
        "outcome_family",
        "cohort_type",
    ]

    output_columns = (
        build_output_columns(
            cohort,
            feature_blocks,
            survey_vars,
            extra_columns,
        )
    )

    return cohort[
        output_columns
    ].copy()


# =============================================================================
# 21. PREVALENCE AUDIT
# =============================================================================

def build_prevalence_records(
    cohort: pd.DataFrame,
    cycle: str,
    outcome_family: str,
    cohort_type: str,
    outcome_columns: List[str],
    survey_vars: Dict[str, Optional[str]],
) -> List[Dict]:

    rows = []

    weight_var = (
        survey_vars[
            "weight_variable"
        ]
    )

    for outcome in outcome_columns:

        if outcome not in cohort.columns:
            continue

        y = safe_numeric(
            cohort[outcome]
        )

        n_valid = int(
            y.notna().sum()
        )

        n_cases = int(
            (y == 1).sum()
        )

        unweighted_prev = pct(
            n_cases,
            n_valid,
        )

        weighted_prev = np.nan

        if (
            weight_var is not None
            and weight_var
            in cohort.columns
        ):

            weighted_mean = (
                weighted_mean_binary(
                    y,
                    cohort[
                        weight_var
                    ],
                )
            )

            if not pd.isna(
                weighted_mean
            ):
                weighted_prev = (
                    100.0
                    * weighted_mean
                )

        rows.append(
            {
                "cycle": cycle,
                "cycle_label":
                    CYCLE_LABELS.get(
                        cycle,
                        cycle,
                    ),
                "dataset_role":
                    (
                        "main"
                        if cycle in MAIN_CYCLES
                        else "supplementary"
                    ),
                "outcome_family":
                    outcome_family,
                "cohort_type":
                    cohort_type,
                "outcome":
                    outcome,
                "n_analytic":
                    len(cohort),
                "n_valid_outcome":
                    n_valid,
                "n_cases":
                    n_cases,
                "prevalence_pct_unweighted":
                    unweighted_prev,
                "weight_variable":
                    weight_var,
                "prevalence_pct_weighted_descriptive":
                    weighted_prev,
                "weighted_variance_estimated":
                    False,
            }
        )

    return rows


# =============================================================================
# 22. SURVEY DESIGN AUDIT
# =============================================================================

def build_survey_design_audit(
    cycle: str,
    native: pd.DataFrame,
) -> Dict:

    survey_vars = (
        resolve_survey_design_vars(
            cycle,
            list(
                native.columns
            ),
        )
    )

    weight = (
        survey_vars[
            "weight_variable"
        ]
    )

    stratum = (
        survey_vars[
            "stratum_variable"
        ]
    )

    psu = (
        survey_vars[
            "psu_variable"
        ]
    )

    record = {
        "cycle": cycle,
        "cycle_label":
            CYCLE_LABELS.get(
                cycle,
                cycle,
            ),
        "dataset_role":
            (
                "main"
                if cycle in MAIN_CYCLES
                else "supplementary"
            ),
        "weight_variable":
            weight,
        "stratum_variable":
            stratum,
        "psu_variable":
            psu,
        "weight_available":
            weight is not None,
        "stratum_available":
            stratum is not None,
        "psu_available":
            psu is not None,
    }

    if (
        weight is not None
        and weight in native.columns
    ):

        w = safe_numeric(
            native[weight]
        )

        record.update(
            {
                "n_weight_nonmissing":
                    int(
                        w.notna().sum()
                    ),
                "n_weight_positive":
                    int(
                        (w > 0).sum()
                    ),
                "weight_min_positive":
                    (
                        w[
                            w > 0
                        ].min()
                        if (w > 0).any()
                        else np.nan
                    ),
                "weight_max":
                    w.max(),
            }
        )

    else:

        record.update(
            {
                "n_weight_nonmissing":
                    0,
                "n_weight_positive":
                    0,
                "weight_min_positive":
                    np.nan,
                "weight_max":
                    np.nan,
            }
        )

    return record


# =============================================================================
# 23. PREDICTOR INTEGRITY AUDIT
# =============================================================================

def build_predictor_integrity_audit(
    cycle: str,
    predictors: pd.DataFrame,
    feature_blocks: Dict[str, List[str]],
) -> List[Dict]:

    rows = []

    blocks = {
        "tier1":
            feature_blocks[
                "tier1"
            ],
        "cbc":
            feature_blocks[
                "cbc"
            ],
        "chemistry":
            feature_blocks[
                "chemistry"
            ],
    }

    for block_name, features in blocks.items():

        for feature in features:

            available = (
                feature
                in predictors.columns
            )

            n_nonmissing = (
                int(
                    predictors[
                        feature
                    ]
                    .notna()
                    .sum()
                )
                if available
                else 0
            )

            rows.append(
                {
                    "cycle":
                        cycle,
                    "feature_block":
                        block_name,
                    "feature":
                        feature,
                    "column_available":
                        available,
                    "n_nonmissing":
                        n_nonmissing,
                    "pct_nonmissing":
                        pct(
                            n_nonmissing,
                            len(
                                predictors
                            ),
                        ),
                }
            )

    return rows


# =============================================================================
# 24. IRON OBSERVED VS MATCHED AUDIT
# =============================================================================

def compare_iron_cohorts(
    cycle: str,
    observed: pd.DataFrame,
    matched: pd.DataFrame,
) -> Dict:

    def summarize(
        df: pd.DataFrame,
        outcome: str,
    ) -> Tuple[int, float]:

        if (
            df is None
            or len(df) == 0
            or outcome
            not in df.columns
        ):
            return (
                0,
                np.nan,
            )

        cases = int(
            df[
                outcome
            ].sum()
        )

        prevalence = pct(
            cases,
            len(df),
        )

        return (
            cases,
            prevalence,
        )

    (
        observed_cases,
        observed_prev,
    ) = summarize(
        observed,
        "outcome_iron_lt15",
    )

    (
        matched_cases,
        matched_prev,
    ) = summarize(
        matched,
        "outcome_iron_lt15",
    )

    return {
        "cycle":
            cycle,
        "cycle_label":
            CYCLE_LABELS.get(
                cycle,
                cycle,
            ),
        "dataset_role":
            (
                "main"
                if cycle in MAIN_CYCLES
                else "supplementary"
            ),
        "n_observed":
            len(
                observed
            )
            if observed is not None
            else 0,
        "n_matched":
            len(
                matched
            )
            if matched is not None
            else 0,
        "observed_lt15_cases":
            observed_cases,
        "matched_lt15_cases":
            matched_cases,
        "observed_lt15_prevalence_pct":
            observed_prev,
        "matched_lt15_prevalence_pct":
            matched_prev,
        "absolute_prevalence_difference_pp":
            (
                observed_prev
                - matched_prev
                if (
                    not pd.isna(
                        observed_prev
                    )
                    and not pd.isna(
                        matched_prev
                    )
                )
                else np.nan
            ),
    }


# =============================================================================
# 25. SAMPLE SIZE SUMMARY
# =============================================================================

def build_sample_size_row(
    cycle: str,
    vitd: Optional[pd.DataFrame],
    iron_observed: Optional[pd.DataFrame],
    iron_matched: Optional[pd.DataFrame],
) -> Dict:

    return {
        "cycle":
            cycle,
        "cycle_label":
            CYCLE_LABELS.get(
                cycle,
                cycle,
            ),
        "dataset_role":
            (
                "main"
                if cycle in MAIN_CYCLES
                else "supplementary"
            ),
        "vitamin_d_n":
            (
                len(vitd)
                if vitd is not None
                else 0
            ),
        "iron_observed_n":
            (
                len(
                    iron_observed
                )
                if iron_observed
                is not None
                else 0
            ),
        "iron_matched_n":
            (
                len(
                    iron_matched
                )
                if iron_matched
                is not None
                else 0
            ),
    }


# =============================================================================
# 26. DECISION LEDGER
# =============================================================================

def build_decision_ledger() -> pd.DataFrame:

    today = (
        datetime.now()
        .date()
        .isoformat()
    )

    rows = [

        {
            "decision_id": "D021",
            "stage": "03",
            "topic":
                "Primary adult age range",
            "decision":
                (
                    "Primary vitamin D and observed iron cohorts "
                    "are restricted to ages 18-59."
                ),
            "reason":
                (
                    "Maintains continuity with the original adult "
                    "screening study while avoiding pediatric populations."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D022",
            "stage": "03",
            "topic":
                "Pregnancy",
            "decision":
                (
                    "Participants explicitly coded pregnant are "
                    "excluded; missing RIDEXPRG is not automatically "
                    "treated as pregnancy."
                ),
            "reason":
                (
                    "Pregnancy status is only defined for selected "
                    "female age groups in NHANES."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D023",
            "stage": "03",
            "topic":
                "Vitamin D primary outcome",
            "decision":
                (
                    "Primary Vitamin D outcome is 25(OH)D <50 nmol/L; "
                    "<30 nmol/L is retained as sensitivity definition."
                ),
            "reason":
                (
                    "Primary study framing concerns screening for "
                    "insufficiency; severe deficiency remains sensitivity."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D024",
            "stage": "03",
            "topic":
                "Iron primary outcome",
            "decision":
                (
                    "Primary iron outcome is ferritin <15 µg/L among "
                    "participants without candidate anemia."
                ),
            "reason":
                (
                    "Defines pre-anemic iron deficiency before overt anemia."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D025",
            "stage": "03",
            "topic":
                "Iron sensitivity outcome",
            "decision":
                (
                    "Ferritin <25 µg/L among non-anemic participants "
                    "is retained as a predefined sensitivity outcome."
                ),
            "reason":
                (
                    "Allows threshold sensitivity without post-hoc "
                    "outcome selection."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D026",
            "stage": "03",
            "topic":
                "Observed iron cohort",
            "decision":
                (
                    "Construct an age-18-59 observed ferritin cohort "
                    "using each cycle's actual outcome ascertainment."
                ),
            "reason":
                (
                    "Preserves real-world NHANES eligibility drift "
                    "for the primary shift-decomposition experiment."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D027",
            "stage": "03",
            "topic":
                "Matched iron cohort",
            "decision":
                (
                    "Construct a female age-18-49 non-pregnant "
                    "common-population ferritin cohort."
                ),
            "reason":
                (
                    "Separates apparent temporal model degradation "
                    "from ferritin population-eligibility shift."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D028",
            "stage": "03",
            "topic":
                "Predictor missingness",
            "decision":
                (
                    "Participants are not excluded for missing predictors "
                    "during analytic-cohort construction."
                ),
            "reason":
                (
                    "Imputation and preprocessing must occur inside "
                    "future training pipelines to prevent leakage."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D029",
            "stage": "03",
            "topic":
                "Survey design",
            "decision":
                (
                    "MEC weights, masked strata and masked PSUs are "
                    "preserved in analytic files where available."
                ),
            "reason":
                (
                    "Supports survey-weighted descriptive and later "
                    "sensitivity analyses."
                ),
            "status":
                "FROZEN",
            "date":
                today,
        },

        {
            "decision_id": "D030",
            "stage": "03",
            "topic":
                "P pre-pandemic release",
            "decision":
                (
                    "P is excluded from primary multicycle masters and "
                    "retained only as supplementary output."
                ),
            "reason":
                (
                    "The 2017-March 2020 pre-pandemic release includes "
                    "the 2017-2018 sample and therefore overlaps J."
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
# 27. MASTER CONCATENATION
# =============================================================================

def concat_nonempty(
    frames: List[pd.DataFrame],
) -> pd.DataFrame:

    valid = [
        df
        for df in frames
        if (
            df is not None
            and len(df) > 0
        )
    ]

    if not valid:
        return pd.DataFrame()

    result = pd.concat(
        valid,
        axis=0,
        ignore_index=True,
        sort=False,
    )

    if (
        "cycle_order"
        in result.columns
    ):

        result = (
            result
            .sort_values(
                [
                    "cycle_order",
                    "SEQN",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    return result


# =============================================================================
# 28. MAIN
# =============================================================================

def main():

    logger.info("")
    logger.info(
        "=" * 82
    )
    logger.info(
        "STAGE 03 STARTED"
    )
    logger.info(
        "=" * 82
    )

    outcome_availability = (
        load_outcome_availability()
    )

    feature_manifest = (
        load_final_feature_manifest()
    )

    feature_blocks = (
        get_feature_blocks(
            feature_manifest
        )
    )

    # -------------------------------------------------------------------------
    # Audits
    # -------------------------------------------------------------------------

    cohort_flow_records = []
    prevalence_records = []
    survey_design_records = []
    predictor_integrity_records = []
    iron_comparison_records = []
    sample_size_records = []

    # -------------------------------------------------------------------------
    # Master containers
    # -------------------------------------------------------------------------

    vitd_main_frames = []
    iron_observed_main_frames = []
    iron_matched_main_frames = []

    vitd_supp_frames = []
    iron_observed_supp_frames = []
    iron_matched_supp_frames = []

    # =========================================================================
    # PROCESS CYCLES
    # =========================================================================

    for cycle in ALL_CYCLES:

        logger.info("")
        logger.info(
            "-" * 82
        )
        logger.info(
            f"PROCESSING {cycle}"
        )
        logger.info(
            "-" * 82
        )

        native_path = (
            joined_file_path(
                cycle
            )
        )

        predictor_path = (
            predictor_file_path(
                cycle
            )
        )

        if native_path is None:

            logger.warning(
                f"{cycle}: native file missing."
            )

            continue

        if predictor_path is None:

            logger.warning(
                f"{cycle}: harmonized "
                "predictor file missing."
            )

            continue

        native = pd.read_csv(
            native_path,
            low_memory=False,
        )

        predictors = pd.read_csv(
            predictor_path,
            low_memory=False,
        )

        logger.info(
            "Native shape: "
            f"{native.shape}"
        )

        logger.info(
            "Predictor shape: "
            f"{predictors.shape}"
        )

        # ---------------------------------------------------------------------
        # Survey design audit
        # ---------------------------------------------------------------------

        survey_design_records.append(
            build_survey_design_audit(
                cycle,
                native,
            )
        )

        # ---------------------------------------------------------------------
        # Predictor integrity
        # ---------------------------------------------------------------------

        predictor_integrity_records.extend(
            build_predictor_integrity_audit(
                cycle,
                predictors,
                feature_blocks,
            )
        )

        # ---------------------------------------------------------------------
        # Native + predictors
        # ---------------------------------------------------------------------

        merged = (
            merge_native_predictors(
                cycle,
                native,
                predictors,
            )
        )

        outcome_vars = (
            resolve_outcome_variables(
                cycle,
                outcome_availability,
            )
        )

        # =====================================================================
        # VITAMIN D
        # =====================================================================

        (
            vitd_cohort,
            vitd_prevalence,
            vitd_audit,
        ) = construct_vitamin_d_cohort(
            cycle=cycle,
            merged=merged,
            outcome_vars=
                outcome_vars,
            feature_blocks=
                feature_blocks,
            flow_records=
                cohort_flow_records,
        )

        prevalence_records.extend(
            vitd_prevalence
        )

        if (
            vitd_cohort is not None
            and len(
                vitd_cohort
            ) > 0
        ):

            if cycle in MAIN_CYCLES:

                vitd_main_frames.append(
                    vitd_cohort
                )

                path = (
                    VITD_CYCLE_DIR
                    / (
                        f"NHANES_{cycle}"
                        "_vitaminD_analytic.csv"
                    )
                )

            else:

                vitd_supp_frames.append(
                    vitd_cohort
                )

                path = (
                    VITD_DIR
                    / (
                        "NHANES_vitaminD"
                        "_supplementary_P.csv"
                    )
                )

            vitd_cohort.to_csv(
                path,
                index=False,
            )

        # =====================================================================
        # IRON
        # =====================================================================

        iron_base = (
            prepare_iron_base(
                cycle=cycle,
                merged=merged,
                outcome_vars=
                    outcome_vars,
            )
        )

        iron_observed = None
        iron_matched = None

        if iron_base is not None:

            (
                iron_observed,
                observed_prevalence,
            ) = construct_iron_observed_cohort(
                cycle=cycle,
                iron_base=
                    iron_base,
                feature_blocks=
                    feature_blocks,
                flow_records=
                    cohort_flow_records,
            )

            (
                iron_matched,
                matched_prevalence,
            ) = construct_iron_matched_cohort(
                cycle=cycle,
                iron_base=
                    iron_base,
                feature_blocks=
                    feature_blocks,
                flow_records=
                    cohort_flow_records,
            )

            prevalence_records.extend(
                observed_prevalence
            )

            prevalence_records.extend(
                matched_prevalence
            )

            iron_comparison_records.append(
                compare_iron_cohorts(
                    cycle=cycle,
                    observed=
                        iron_observed,
                    matched=
                        iron_matched,
                )
            )

            # -------------------------------------------------------------
            # Save cycle-specific iron files
            # -------------------------------------------------------------

            if cycle in MAIN_CYCLES:

                if (
                    iron_observed
                    is not None
                ):

                    iron_observed.to_csv(
                        (
                            IRON_OBSERVED_CYCLE_DIR
                            / (
                                f"NHANES_{cycle}"
                                "_iron_observed.csv"
                            )
                        ),
                        index=False,
                    )

                    iron_observed_main_frames.append(
                        iron_observed
                    )

                if (
                    iron_matched
                    is not None
                ):

                    iron_matched.to_csv(
                        (
                            IRON_MATCHED_CYCLE_DIR
                            / (
                                f"NHANES_{cycle}"
                                "_iron_matched.csv"
                            )
                        ),
                        index=False,
                    )

                    iron_matched_main_frames.append(
                        iron_matched
                    )

            else:

                if (
                    iron_observed
                    is not None
                ):

                    iron_observed.to_csv(
                        (
                            IRON_SUPP_DIR
                            / "NHANES_iron_observed_P.csv"
                        ),
                        index=False,
                    )

                    iron_observed_supp_frames.append(
                        iron_observed
                    )

                if (
                    iron_matched
                    is not None
                ):

                    iron_matched.to_csv(
                        (
                            IRON_SUPP_DIR
                            / "NHANES_iron_matched_P.csv"
                        ),
                        index=False,
                    )

                    iron_matched_supp_frames.append(
                        iron_matched
                    )

        # =====================================================================
        # SAMPLE SIZE SUMMARY
        # =====================================================================

        sample_size_records.append(
            build_sample_size_row(
                cycle=cycle,
                vitd=
                    vitd_cohort,
                iron_observed=
                    iron_observed,
                iron_matched=
                    iron_matched,
            )
        )

    # =========================================================================
    # BUILD PRIMARY MULTICYCLE MASTERS
    # =========================================================================

    vitamin_d_master = (
        concat_nonempty(
            vitd_main_frames
        )
    )

    iron_observed_master = (
        concat_nonempty(
            iron_observed_main_frames
        )
    )

    iron_matched_master = (
        concat_nonempty(
            iron_matched_main_frames
        )
    )

    if not vitamin_d_master.empty:

        vitamin_d_master.to_csv(
            (
                VITD_DIR
                / "NHANES_vitaminD_primary_multicycle.csv"
            ),
            index=False,
        )

    if not iron_observed_master.empty:

        iron_observed_master.to_csv(
            (
                IRON_OBSERVED_DIR
                / "NHANES_iron_observed_multicycle.csv"
            ),
            index=False,
        )

    if not iron_matched_master.empty:

        iron_matched_master.to_csv(
            (
                IRON_MATCHED_DIR
                / "NHANES_iron_matched_multicycle.csv"
            ),
            index=False,
        )

    # =========================================================================
    # BUILD AUDIT TABLES
    # =========================================================================

    cohort_flow_df = (
        pd.DataFrame(
            cohort_flow_records
        )
    )

    prevalence_df = (
        pd.DataFrame(
            prevalence_records
        )
    )

    survey_design_df = (
        pd.DataFrame(
            survey_design_records
        )
    )

    predictor_integrity_df = (
        pd.DataFrame(
            predictor_integrity_records
        )
    )

    iron_comparison_df = (
        pd.DataFrame(
            iron_comparison_records
        )
    )

    sample_size_df = (
        pd.DataFrame(
            sample_size_records
        )
    )

    decision_ledger = (
        build_decision_ledger()
    )

    # =========================================================================
    # SAVE AUDIT CSVs
    # =========================================================================

    cohort_flow_df.to_csv(
        (
            AUDIT_DIR
            / "03_cohort_flow_audit.csv"
        ),
        index=False,
    )

    prevalence_df.to_csv(
        (
            AUDIT_DIR
            / "03_outcome_prevalence_audit.csv"
        ),
        index=False,
    )

    survey_design_df.to_csv(
        (
            AUDIT_DIR
            / "03_survey_design_audit.csv"
        ),
        index=False,
    )

    predictor_integrity_df.to_csv(
        (
            AUDIT_DIR
            / "03_predictor_integrity_audit.csv"
        ),
        index=False,
    )

    iron_comparison_df.to_csv(
        (
            AUDIT_DIR
            / "03_iron_observed_vs_matched_audit.csv"
        ),
        index=False,
    )

    sample_size_df.to_csv(
        (
            AUDIT_DIR
            / "03_cycle_outcome_sample_sizes.csv"
        ),
        index=False,
    )

    decision_ledger.to_csv(
        (
            AUDIT_DIR
            / "decision_ledger_stage03.csv"
        ),
        index=False,
    )

    # =========================================================================
    # EXCEL SUMMARY
    # =========================================================================

    summary_path = (
        AUDIT_DIR
        / "03_stage03_summary.xlsx"
    )

    with pd.ExcelWriter(
        summary_path,
        engine="openpyxl",
    ) as writer:

        sample_size_df.to_excel(
            writer,
            sheet_name="Sample_Sizes",
            index=False,
        )

        cohort_flow_df.to_excel(
            writer,
            sheet_name="Cohort_Flow",
            index=False,
        )

        prevalence_df.to_excel(
            writer,
            sheet_name="Outcome_Prevalence",
            index=False,
        )

        iron_comparison_df.to_excel(
            writer,
            sheet_name="Iron_Observed_vs_Matched",
            index=False,
        )

        survey_design_df.to_excel(
            writer,
            sheet_name="Survey_Design",
            index=False,
        )

        predictor_integrity_df.to_excel(
            writer,
            sheet_name="Predictor_Integrity",
            index=False,
        )

        decision_ledger.to_excel(
            writer,
            sheet_name="Decision_Ledger",
            index=False,
        )

    # =========================================================================
    # FINAL INTEGRITY CHECKS
    # =========================================================================

    print("\n")
    print(
        "=" * 84
    )
    print(
        "STAGE 03 COMPLETE"
    )
    print(
        "=" * 84
    )

    # -------------------------------------------------------------------------
    # Sample sizes
    # -------------------------------------------------------------------------

    if not sample_size_df.empty:

        print(
            "\nANALYTIC SAMPLE SIZES"
        )

        print(
            "-" * 84
        )

        print(
            sample_size_df.to_string(
                index=False
            )
        )

    # -------------------------------------------------------------------------
    # Iron observed vs matched
    # -------------------------------------------------------------------------

    if not iron_comparison_df.empty:

        print(
            "\nIRON OBSERVED VS MATCHED"
        )

        print(
            "-" * 84
        )

        display_cols = [
            "cycle",
            "n_observed",
            "n_matched",
            "observed_lt15_prevalence_pct",
            "matched_lt15_prevalence_pct",
            "absolute_prevalence_difference_pp",
        ]

        print(
            iron_comparison_df[
                display_cols
            ]
            .round(2)
            .to_string(
                index=False
            )
        )

    # -------------------------------------------------------------------------
    # Primary masters
    # -------------------------------------------------------------------------

    print(
        "\nPRIMARY MULTICYCLE MASTER FILES"
    )

    print(
        "-" * 84
    )

    print(
        "Vitamin D:"
    )

    print(
        "  "
        + str(
            VITD_DIR
            / "NHANES_vitaminD_primary_multicycle.csv"
        )
    )

    print(
        "\nIron observed:"
    )

    print(
        "  "
        + str(
            IRON_OBSERVED_DIR
            / "NHANES_iron_observed_multicycle.csv"
        )
    )

    print(
        "\nIron matched:"
    )

    print(
        "  "
        + str(
            IRON_MATCHED_DIR
            / "NHANES_iron_matched_multicycle.csv"
        )
    )

    print(
        "\nPRIMARY AUDIT WORKBOOK:"
    )

    print(
        "  "
        + str(
            summary_path
        )
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  • No predictor imputation has been performed."
    )

    print(
        "  • No scaling or feature selection has been performed."
    )

    print(
        "  • No models have been trained."
    )

    print(
        "  • Vitamin D and ferritin remain outcome-only variables."
    )

    print(
        "  • P is supplementary and excluded from primary masters."
    )

    print(
        "  • Weighted prevalence is descriptive only; "
        "no design-based confidence intervals are computed yet."
    )

    print(
        "\nNext step after review:"
    )

    print(
        "  Stage 04 — analytic cohort QC, survey-weighted descriptive "
        "statistics, temporal-shift diagnostics, and locked modeling design."
    )

    print(
        "\n" + "=" * 84
    )


# =============================================================================
# 29. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()