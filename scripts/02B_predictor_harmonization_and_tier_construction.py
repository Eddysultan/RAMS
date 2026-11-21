"""
===============================================================================
STAGE 02B — CORRECTED FINAL VERSION
NHANES Predictor Harmonization and Resource-Tier Construction
===============================================================================


        harmonized_cycles/
            NHANES_2009_2010_predictors_harmonized.csv
            ...
            NHANES_2021_2023_predictors_harmonized.csv

===============================================================================
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAIN_JOIN_DIR = (
    PROJECT_ROOT
    / "processed"
    / "02_joined_native"
    / "main_cycles"
)

SUPP_JOIN_DIR = (
    PROJECT_ROOT
    / "processed"
    / "02_joined_native"
    / "supplementary"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed"
    / "04_predictor_harmonization"
)

HARMONIZED_DIR = OUTPUT_DIR / "harmonized_cycles"

LOG_DIR = PROJECT_ROOT / "processed" / "logs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

HARMONIZED_DIR.mkdir(
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

SUPPLEMENTARY_CYCLES = [
    "2017_2020_PrePandemic",
]

ALL_CYCLES = (
    MAIN_CYCLES
    + SUPPLEMENTARY_CYCLES
)

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
# 3. LOGGING
# =============================================================================

LOG_FILE = (
    LOG_DIR
    / "stage02B_predictor_harmonization.log"
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
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


# =============================================================================
# 4. FEATURE ARCHITECTURE
# =============================================================================
#
# inclusion_role:
#
#   PRIMARY_T1
#       Final Tier-1 predictor.
#
#   PRIMARY_T2_CBC
#       Routine CBC predictor added in Tier 2.
#
#   T2_SENSITIVITY
#       Routine chemistry predictor used only in predefined sensitivity
#       analyses.
#
#   SUPPLEMENTARY_AUDIT
#       Retained for documentation but excluded from the primary common-core
#       predictor set.
#
# transform:
#
#   numeric
#   sex
#   race5
#   education5
#   yes_no
#   yes_no_borderline
#   food_security4
#   routine_care
#
# Derived variables can list >1 input variable.
# =============================================================================


FEATURE_SPECS = {

    # =========================================================================
    # TIER 1 — DEMOGRAPHICS
    # =========================================================================

    "age_years": {
        "domain": "demographics",
        "tier": "T1",
        "inclusion_role": "PRIMARY_T1",
        "preferred": ["RIDAGEYR"],
        "transform": "numeric",
        "derived": False,
    },

    "sex": {
        "domain": "demographics",
        "tier": "T1",
        "inclusion_role": "PRIMARY_T1",
        "preferred": ["RIAGENDR"],
        "transform": "sex",
        "derived": False,
    },

    "race_ethnicity_5cat": {
        "domain": "demographics",
        "tier": "T1",
        "inclusion_role": "PRIMARY_T1",
        "preferred": ["RIDRETH1"],
        "transform": "race5",
        "derived": False,
    },

    # =========================================================================
    # TIER 1 — SOCIOECONOMIC
    # =========================================================================

    "education_adult_5cat": {
        "domain": "socioeconomic",
        "tier": "T1",
        "inclusion_role": "PRIMARY_T1",
        "preferred": ["DMDEDUC2"],
        "transform": "education5",
        "derived": False,
    },

    "poverty_income_ratio": {
        "domain": "socioeconomic",
        "tier": "T1",
        "inclusion_role": "PRIMARY_T1",
        "preferred": ["INDFMPIR"],
        "transform": "numeric",
        "derived": False,
    },

    # =========================================================================
    # TIER 1 — INSURANCE / ACCESS
    # =========================================================================

    "health_insurance": {
        "domain": "healthcare_access",
        "tier": "T1",
        "inclusion_role": "PRIMARY_T1",
        "preferred": ["HIQ011"],
        "transform": "yes_no",
        "derived": False,
    },

    "has_routine_place_for_care": {
        "domain": "healthcare_access",
        "tier": "T1",
        "inclusion_role": "PRIMARY_T1",
        "preferred": ["HUQ030"],
        "transform": "routine_care",
        "derived": False,
    },

    # =========================================================================
    # TIER 1 — FOOD SECURITY
    # =========================================================================

    "adult_food_security_4cat": {
        "domain": "food_security",
        "tier": "T1",
        "inclusion_role": "PRIMARY_T1",
        "preferred": ["FSDAD"],
        "transform": "food_security4",
        "derived": False,
    },

    # =========================================================================
    # TIER 1 — ANTHROPOMETRICS
    # =========================================================================

    "bmi": {
        "domain": "anthropometric",
        "tier": "T1",
        "inclusion_role": "PRIMARY_T1",
        "preferred": ["BMXBMI"],
        "transform": "numeric",
        "derived": False,
    },

    "waist_cm": {
        "domain": "anthropometric",
        "tier": "T1",
        "inclusion_role": "PRIMARY_T1",
        "preferred": ["BMXWAIST"],
        "transform": "numeric",
        "derived": False,
    },

    # =========================================================================
    # TIER 1 — SMOKING
    # =========================================================================
    #
    # This is intentionally DERIVED from SMQ020 and SMQ040.
    #
    # 0 = never smoker
    # 1 = former smoker
    # 2 = current smoker
    #
    # Never-smokers are not missing merely because SMQ040 was not asked.
    # =========================================================================

    "smoking_status_3cat": {
        "domain": "lifestyle",
        "tier": "T1",
        "inclusion_role": "PRIMARY_T1",
        "preferred": [
            "SMQ020",
            "SMQ040",
        ],
        "transform": "derived_smoking",
        "derived": True,
    },

    # =========================================================================
    # TIER 1 — MEDICAL HISTORY
    # =========================================================================

    "doctor_diagnosed_diabetes": {
        "domain": "medical_history",
        "tier": "T1",
        "inclusion_role": "PRIMARY_T1",
        "preferred": ["DIQ010"],
        "transform": "yes_no_borderline",
        "derived": False,
    },

    "doctor_diagnosed_hypertension": {
        "domain": "medical_history",
        "tier": "T1",
        "inclusion_role": "PRIMARY_T1",
        "preferred": ["BPQ020"],
        "transform": "yes_no",
        "derived": False,
    },

    # =========================================================================
    # TIER 1 — DIETARY
    # =========================================================================

    "day1_energy_kcal": {
        "domain": "dietary",
        "tier": "T1",
        "inclusion_role": "PRIMARY_T1",
        "preferred": ["DR1TKCAL"],
        "transform": "numeric",
        "derived": False,
    },

    "day1_dietary_iron_mg": {
        "domain": "dietary",
        "tier": "T1",
        "inclusion_role": "PRIMARY_T1",
        "preferred": ["DR1TIRON"],
        "transform": "numeric",
        "derived": False,
    },

    "day1_dietary_vitamin_d": {
        "domain": "dietary",
        "tier": "T1",
        "inclusion_role": "PRIMARY_T1",
        "preferred": ["DR1TVD"],
        "transform": "numeric",
        "derived": False,
    },

    # =========================================================================
    # TIER 2 PRIMARY — CBC
    # =========================================================================

    "cbc_wbc": {
        "domain": "cbc",
        "tier": "T2",
        "inclusion_role": "PRIMARY_T2_CBC",
        "preferred": [
            "LBXWBCSI",
            "LBXWBC",
        ],
        "transform": "numeric",
        "derived": False,
    },

    "cbc_rbc": {
        "domain": "cbc",
        "tier": "T2",
        "inclusion_role": "PRIMARY_T2_CBC",
        "preferred": [
            "LBXRBCSI",
            "LBXRBC",
        ],
        "transform": "numeric",
        "derived": False,
    },

    "cbc_hemoglobin": {
        "domain": "cbc",
        "tier": "T2",
        "inclusion_role": "PRIMARY_T2_CBC",
        "preferred": ["LBXHGB"],
        "transform": "numeric",
        "derived": False,
    },

    "cbc_hematocrit": {
        "domain": "cbc",
        "tier": "T2",
        "inclusion_role": "PRIMARY_T2_CBC",
        "preferred": ["LBXHCT"],
        "transform": "numeric",
        "derived": False,
    },

    "cbc_mcv": {
        "domain": "cbc",
        "tier": "T2",
        "inclusion_role": "PRIMARY_T2_CBC",
        "preferred": [
            "LBXMCVSI",
            "LBDMCVSI",
            "LBXMCV",
        ],
        "transform": "numeric",
        "derived": False,
    },

    "cbc_mch": {
        "domain": "cbc",
        "tier": "T2",
        "inclusion_role": "PRIMARY_T2_CBC",
        "preferred": [
            "LBXMCHSI",
            "LBDMCHSI",
            "LBXMCH",
        ],
        "transform": "numeric",
        "derived": False,
    },

    "cbc_mchc": {
        "domain": "cbc",
        "tier": "T2",
        "inclusion_role": "PRIMARY_T2_CBC",
        "preferred": [
            "LBXMC",
            "LBXMCHC",
            "LBDMCHC",
        ],
        "transform": "numeric",
        "derived": False,
    },

    "cbc_rdw": {
        "domain": "cbc",
        "tier": "T2",
        "inclusion_role": "PRIMARY_T2_CBC",
        "preferred": [
            "LBXRDW",
            "LBXRDWSD",
        ],
        "transform": "numeric",
        "derived": False,
    },

    "cbc_platelets": {
        "domain": "cbc",
        "tier": "T2",
        "inclusion_role": "PRIMARY_T2_CBC",
        "preferred": [
            "LBXPLTSI",
            "LBXPLT",
        ],
        "transform": "numeric",
        "derived": False,
    },

    # =========================================================================
    # TIER 2 SENSITIVITY EXTENSION — ROUTINE CHEMISTRY
    # =========================================================================

    "serum_albumin": {
        "domain": "routine_chemistry",
        "tier": "T2_SENSITIVITY",
        "inclusion_role": "T2_SENSITIVITY",
        "preferred": [
            "LBXSAL",
            "LBDSALSI",
        ],
        "transform": "numeric",
        "derived": False,
    },

    "serum_creatinine": {
        "domain": "routine_chemistry",
        "tier": "T2_SENSITIVITY",
        "inclusion_role": "T2_SENSITIVITY",
        "preferred": [
            "LBXSCR",
            "LBDSCRSI",
        ],
        "transform": "numeric",
        "derived": False,
    },

    "serum_glucose": {
        "domain": "routine_chemistry",
        "tier": "T2_SENSITIVITY",
        "inclusion_role": "T2_SENSITIVITY",
        "preferred": [
            "LBXSGL",
            "LBDGLUSI",
        ],
        "transform": "numeric",
        "derived": False,
    },

    # =========================================================================
    # SUPPLEMENTARY / AUDIT-ONLY VARIABLES
    # =========================================================================

    "household_food_security_candidate": {
        "domain": "food_security",
        "tier": "SUPPLEMENTARY",
        "inclusion_role": "SUPPLEMENTARY_AUDIT",
        "preferred": [
            "FSDHH",
            "FSDHHAD",
        ],
        "transform": None,
        "derived": False,
    },

    "healthcare_visits_candidate": {
        "domain": "healthcare_access",
        "tier": "SUPPLEMENTARY",
        "inclusion_role": "SUPPLEMENTARY_AUDIT",
        "preferred": [
            "HUQ051",
            "HUQ050",
        ],
        "transform": None,
        "derived": False,
    },

    "alcohol_use_candidate": {
        "domain": "lifestyle",
        "tier": "SUPPLEMENTARY",
        "inclusion_role": "SUPPLEMENTARY_AUDIT",
        "preferred": [
            "ALQ101",
            "ALQ111",
        ],
        "transform": None,
        "derived": False,
    },

    "vigorous_recreational_activity_candidate": {
        "domain": "lifestyle",
        "tier": "SUPPLEMENTARY",
        "inclusion_role": "SUPPLEMENTARY_AUDIT",
        "preferred": [
            "PAQ650",
        ],
        "transform": None,
        "derived": False,
    },

    "moderate_recreational_activity_candidate": {
        "domain": "lifestyle",
        "tier": "SUPPLEMENTARY",
        "inclusion_role": "SUPPLEMENTARY_AUDIT",
        "preferred": [
            "PAQ665",
        ],
        "transform": None,
        "derived": False,
    },
}


# =============================================================================
# 5. CONFIRMATORY / OUTCOME VARIABLES THAT MUST NEVER ENTER PREDICTORS
# =============================================================================

FORBIDDEN_EXACT_VARIABLES = {
    "LBXFER",
    "LBDFERSI",
    "LBXVIDMS",
    "LBDVIDMS",
}

FORBIDDEN_PREFIXES = [
    "LBXFER",
    "LBDFER",
    "LBXVID",
    "LBDVID",
]


# =============================================================================
# 6. BASIC HELPERS
# =============================================================================

def safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a Series to numeric without raising on nonnumeric values.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def pct(
    numerator: float,
    denominator: float,
) -> float:
    """
    Safe percentage.
    """

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


def cycle_file_path(
    cycle: str,
) -> Optional[Path]:
    """
    Resolve Stage-01 joined-native dataset.
    """

    filename = (
        f"NHANES_{cycle}_joined_native.csv"
    )

    if cycle in MAIN_CYCLES:

        path = (
            MAIN_JOIN_DIR
            / filename
        )

    else:

        path = (
            SUPP_JOIN_DIR
            / filename
        )

    if path.exists():
        return path

    return None


def resolve_single_variable(
    columns: List[str],
    preferred: List[str],
) -> Optional[str]:
    """
    Resolve one preferred raw variable.

    Priority:
        1. exact native variable;
        2. unique Stage-01 collision-renamed variant.

    This is used only for non-derived features.
    """

    for variable in preferred:

        if variable in columns:
            return variable

    for variable in preferred:

        candidates = [
            col
            for col in columns
            if col.startswith(
                variable + "__"
            )
        ]

        if len(candidates) == 1:
            return candidates[0]

    return None


def resolve_exact_or_collision(
    columns: List[str],
    variable: str,
) -> Optional[str]:
    """
    Resolve a specific required variable for a derived feature.
    """

    if variable in columns:
        return variable

    candidates = [
        col
        for col in columns
        if col.startswith(
            variable + "__"
        )
    ]

    if len(candidates) == 1:
        return candidates[0]

    return None


def find_related_candidates(
    columns: List[str],
    preferred: List[str],
) -> List[str]:
    """
    Find columns related to preferred variable names for audit purposes.

    These candidates are NOT automatically treated as equivalent.
    """

    matches = set()

    for preferred_var in preferred:

        root = re.sub(
            r"[^A-Z0-9]",
            "",
            preferred_var.upper(),
        )

        for col in columns:

            cleaned = re.sub(
                r"[^A-Z0-9]",
                "",
                col.upper(),
            )

            if root in cleaned:
                matches.add(col)

    return sorted(matches)


def is_forbidden_predictor(
    variable: str,
) -> bool:
    """
    Detect outcome-defining nutrient biomarkers.
    """

    base = (
        variable.split("__")[0]
        if "__" in variable
        else variable
    )

    base_upper = base.upper()

    if base_upper in FORBIDDEN_EXACT_VARIABLES:
        return True

    return any(
        base_upper.startswith(prefix)
        for prefix in FORBIDDEN_PREFIXES
    )


# =============================================================================
# 7. TRANSFORM FUNCTIONS
# =============================================================================

def transform_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    Preserve numeric value.
    """

    return safe_numeric(series)


def transform_yes_no(
    series: pd.Series,
) -> pd.Series:
    """
    Standard common NHANES yes/no recode:

        1 = Yes  -> 1
        2 = No   -> 0

    Refused, don't know, missing -> NaN.
    """

    x = safe_numeric(series)

    out = pd.Series(
        np.nan,
        index=series.index,
        dtype="float64",
    )

    out.loc[x == 1] = 1.0
    out.loc[x == 2] = 0.0

    return out


def transform_yes_no_borderline(
    series: pd.Series,
) -> pd.Series:
    """
    DIQ010-style recode:

        1 = Yes        -> 1
        2 = No         -> 0
        3 = Borderline -> 2

    Borderline is preserved rather than silently combined with yes/no.
    Later modeling should encode this as categorical.
    """

    x = safe_numeric(series)

    out = pd.Series(
        np.nan,
        index=series.index,
        dtype="float64",
    )

    out.loc[x == 2] = 0.0
    out.loc[x == 1] = 1.0
    out.loc[x == 3] = 2.0

    return out


def transform_sex(
    series: pd.Series,
) -> pd.Series:
    """
    RIAGENDR:

        1 = Male
        2 = Female

    Preserve native stable coding.
    """

    x = safe_numeric(series)

    return x.where(
        x.isin([1, 2]),
        np.nan,
    )


def transform_race5(
    series: pd.Series,
) -> pd.Series:
    """
    RIDRETH1 harmonized five-category representation:

        1 = Mexican American
        2 = Other Hispanic
        3 = Non-Hispanic White
        4 = Non-Hispanic Black
        5 = Other race / multiracial

    RIDRETH1 is deliberately retained because it supports the long-horizon
    study consistently.

    Later models must treat this as categorical, not continuous.
    """

    x = safe_numeric(series)

    return x.where(
        x.isin([1, 2, 3, 4, 5]),
        np.nan,
    )


def transform_education5(
    series: pd.Series,
) -> pd.Series:
    """
    DMDEDUC2:

        1 = Less than 9th grade
        2 = 9–11th grade
        3 = High-school graduate / GED
        4 = Some college / AA degree
        5 = College graduate or above

    Later models must treat this as categorical/ordinal according to the
    predefined modeling specification.

    Structural age-related nonresponse is retained as missing.
    """

    x = safe_numeric(series)

    return x.where(
        x.isin([1, 2, 3, 4, 5]),
        np.nan,
    )


def transform_food_security4(
    series: pd.Series,
) -> pd.Series:
    """
    Adult food security category (FSDAD):

        1 = Full food security
        2 = Marginal food security
        3 = Low food security
        4 = Very low food security

    Preserve the four-category construct.

    Refused/unknown/out-of-scope -> NaN.

    Later modeling should treat this as categorical or explicitly ordinal.
    """

    x = safe_numeric(series)

    return x.where(
        x.isin([1, 2, 3, 4]),
        np.nan,
    )


def transform_routine_care(
    series: pd.Series,
) -> pd.Series:
    """
    HUQ030:
        1 = Yes, there is a routine place
        2 = There is no place
        3 = There is more than one place

    Harmonized binary:
        1 = Has at least one usual place for healthcare
            (native code 1 or 3)
        0 = No usual place
            (native code 2)

    7/9/missing remain NaN.
    """

    x = safe_numeric(series)

    out = pd.Series(
        np.nan,
        index=series.index,
        dtype="float64",
    )

    out.loc[
        x.isin([1, 3])
    ] = 1.0

    out.loc[
        x == 2
    ] = 0.0

    return out


TRANSFORM_FUNCTIONS = {
    "numeric": transform_numeric,
    "yes_no": transform_yes_no,
    "yes_no_borderline":
        transform_yes_no_borderline,
    "sex": transform_sex,
    "race5": transform_race5,
    "education5": transform_education5,
    "food_security4":
        transform_food_security4,
    "routine_care":
        transform_routine_care,
}


# =============================================================================
# 8. DERIVED SMOKING STATUS
# =============================================================================

def derive_smoking_status(
    df: pd.DataFrame,
    smq020_var: Optional[str],
    smq040_var: Optional[str],
) -> Tuple[pd.Series, Dict]:
    """
    Derive mutually exclusive smoking status.

    NHANES logic:

        SMQ020:
            1 = smoked at least 100 cigarettes lifetime
            2 = has not

        SMQ040 among ever-smokers:
            1 = smokes every day
            2 = smokes some days
            3 = not at all

    Final coding:

        0 = Never smoker
            SMQ020 == 2

        1 = Former smoker
            SMQ020 == 1 AND SMQ040 == 3

        2 = Current smoker
            SMQ020 == 1 AND SMQ040 in {1, 2}

        NaN = cannot classify

    Critically, people with SMQ020 == 2 are classified as NEVER even though
    SMQ040 is legitimately skipped for them.
    """

    out = pd.Series(
        np.nan,
        index=df.index,
        dtype="float64",
    )

    audit = {
        "smq020_variable": smq020_var,
        "smq040_variable": smq040_var,
        "n_never": 0,
        "n_former": 0,
        "n_current": 0,
        "n_unclassified": len(df),
    }

    if smq020_var is None:
        return out, audit

    ever = safe_numeric(
        df[smq020_var]
    )

    # Never smoker.
    never_mask = (
        ever == 2
    )

    out.loc[
        never_mask
    ] = 0.0

    # SMQ040 required only for those with lifetime smoking history.
    if smq040_var is not None:

        current = safe_numeric(
            df[smq040_var]
        )

        former_mask = (
            (ever == 1)
            & (current == 3)
        )

        current_mask = (
            (ever == 1)
            & current.isin([1, 2])
        )

        out.loc[
            former_mask
        ] = 1.0

        out.loc[
            current_mask
        ] = 2.0

    audit.update(
        {
            "n_never":
                int((out == 0).sum()),

            "n_former":
                int((out == 1).sum()),

            "n_current":
                int((out == 2).sum()),

            "n_unclassified":
                int(out.isna().sum()),
        }
    )

    return out, audit


# =============================================================================
# 9. FEATURE RESOLUTION
# =============================================================================

def resolve_feature_variables(
    df: pd.DataFrame,
    feature_name: str,
    spec: Dict,
) -> Dict:
    """
    Resolve raw variable(s) for one feature.

    Derived smoking is treated separately because both SMQ020 and SMQ040
    are conceptually required.
    """

    columns = list(df.columns)

    if (
        spec["transform"]
        == "derived_smoking"
    ):

        smq020 = resolve_exact_or_collision(
            columns,
            "SMQ020",
        )

        smq040 = resolve_exact_or_collision(
            columns,
            "SMQ040",
        )

        return {
            "resolved_variable":
                (
                    "|".join(
                        [
                            x
                            for x in [
                                smq020,
                                smq040,
                            ]
                            if x is not None
                        ]
                    )
                    or None
                ),
            "resolved_variables_list": [
                smq020,
                smq040,
            ],
            "available":
                smq020 is not None,
            "complete_derived_inputs":
                (
                    smq020 is not None
                    and smq040 is not None
                ),
        }

    variable = resolve_single_variable(
        columns,
        spec["preferred"],
    )

    return {
        "resolved_variable": variable,
        "resolved_variables_list":
            [variable],
        "available":
            variable is not None,
        "complete_derived_inputs":
            True,
    }


# =============================================================================
# 10. AUDIT FEATURE AVAILABILITY
# =============================================================================

def audit_cycle_features(
    cycle: str,
    df: pd.DataFrame,
) -> Tuple[
    List[Dict],
    Dict[str, Dict],
]:
    """
    Audit all requested features against one cycle.
    """

    records = []
    resolved_map = {}

    columns = list(df.columns)

    for (
        feature_name,
        spec,
    ) in FEATURE_SPECS.items():

        resolution = (
            resolve_feature_variables(
                df=df,
                feature_name=feature_name,
                spec=spec,
            )
        )

        resolved_map[
            feature_name
        ] = resolution

        related = (
            find_related_candidates(
                columns,
                spec["preferred"],
            )
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
                "feature":
                    feature_name,
                "domain":
                    spec["domain"],
                "tier":
                    spec["tier"],
                "inclusion_role":
                    spec[
                        "inclusion_role"
                    ],
                "derived":
                    spec["derived"],
                "preferred_variables":
                    "|".join(
                        spec["preferred"]
                    ),
                "resolved_variable":
                    resolution[
                        "resolved_variable"
                    ],
                "available":
                    resolution[
                        "available"
                    ],
                "complete_derived_inputs":
                    resolution[
                        "complete_derived_inputs"
                    ],
                "related_candidates":
                    "|".join(related),
            }
        )

    return (
        records,
        resolved_map,
    )


# =============================================================================
# 11. HARMONIZE ONE CYCLE
# =============================================================================

def harmonize_cycle(
    cycle: str,
    df: pd.DataFrame,
    resolved_map: Dict[str, Dict],
) -> Tuple[
    pd.DataFrame,
    List[Dict],
    List[Dict],
]:
    """
    Create the standardized predictor table for one cycle.
    """

    if "SEQN" not in df.columns:

        raise RuntimeError(
            f"{cycle}: SEQN missing."
        )

    if df["SEQN"].duplicated().any():

        raise RuntimeError(
            f"{cycle}: input joined-native file "
            f"contains duplicate SEQN."
        )

    out = pd.DataFrame(
        {
            "SEQN": df["SEQN"],
            "survey_cycle": cycle,
        }
    )

    coverage_records = []
    smoking_audit_records = []

    # -------------------------------------------------------------------------
    # Process features
    # -------------------------------------------------------------------------

    for (
        feature_name,
        spec,
    ) in FEATURE_SPECS.items():

        # Audit-only fields are intentionally NOT written to the harmonized
        # predictor table.
        if (
            spec["inclusion_role"]
            == "SUPPLEMENTARY_AUDIT"
        ):
            continue

        resolution = (
            resolved_map[
                feature_name
            ]
        )

        # ---------------------------------------------------------------------
        # Derived smoking
        # ---------------------------------------------------------------------

        if (
            spec["transform"]
            == "derived_smoking"
        ):

            inputs = (
                resolution[
                    "resolved_variables_list"
                ]
            )

            smq020_var = (
                inputs[0]
                if len(inputs) >= 1
                else None
            )

            smq040_var = (
                inputs[1]
                if len(inputs) >= 2
                else None
            )

            smoking_status, smoking_audit = (
                derive_smoking_status(
                    df=df,
                    smq020_var=smq020_var,
                    smq040_var=smq040_var,
                )
            )

            out[
                feature_name
            ] = smoking_status

            n_nonmissing = int(
                smoking_status
                .notna()
                .sum()
            )

            coverage_records.append(
                {
                    "cycle": cycle,
                    "feature":
                        feature_name,
                    "tier":
                        spec["tier"],
                    "inclusion_role":
                        spec[
                            "inclusion_role"
                        ],
                    "raw_variable":
                        resolution[
                            "resolved_variable"
                        ],
                    "status":
                        (
                            "harmonized_derived"
                            if smq020_var
                            is not None
                            else "not_available"
                        ),
                    "n_nonmissing":
                        n_nonmissing,
                    "pct_nonmissing":
                        pct(
                            n_nonmissing,
                            len(df),
                        ),
                }
            )

            smoking_audit_records.append(
                {
                    "cycle": cycle,
                    **smoking_audit,
                }
            )

            continue

        # ---------------------------------------------------------------------
        # Normal harmonized fields
        # ---------------------------------------------------------------------

        raw_variable = (
            resolution[
                "resolved_variable"
            ]
        )

        if raw_variable is None:

            out[
                feature_name
            ] = np.nan

            coverage_records.append(
                {
                    "cycle": cycle,
                    "feature":
                        feature_name,
                    "tier":
                        spec["tier"],
                    "inclusion_role":
                        spec[
                            "inclusion_role"
                        ],
                    "raw_variable":
                        None,
                    "status":
                        "not_available",
                    "n_nonmissing":
                        0,
                    "pct_nonmissing":
                        0.0,
                }
            )

            continue

        if is_forbidden_predictor(
            raw_variable
        ):

            raise RuntimeError(
                "FATAL: outcome-defining "
                "biomarker attempted as "
                "predictor: "
                f"{raw_variable}"
            )

        transform_name = (
            spec["transform"]
        )

        if (
            transform_name
            not in
            TRANSFORM_FUNCTIONS
        ):

            raise RuntimeError(
                f"No transform registered "
                f"for {feature_name}: "
                f"{transform_name}"
            )

        transform_function = (
            TRANSFORM_FUNCTIONS[
                transform_name
            ]
        )

        transformed = (
            transform_function(
                df[raw_variable]
            )
        )

        out[
            feature_name
        ] = transformed

        n_nonmissing = int(
            transformed
            .notna()
            .sum()
        )

        coverage_records.append(
            {
                "cycle": cycle,
                "feature":
                    feature_name,
                "tier":
                    spec["tier"],
                "inclusion_role":
                    spec[
                        "inclusion_role"
                    ],
                "raw_variable":
                    raw_variable,
                "status":
                    "harmonized",
                "n_nonmissing":
                    n_nonmissing,
                "pct_nonmissing":
                    pct(
                        n_nonmissing,
                        len(df),
                    ),
            }
        )

    # -------------------------------------------------------------------------
    # Tier completeness summaries
    #
    # These are DESCRIPTIVE only and must NOT later be used as predictors.
    # -------------------------------------------------------------------------

    tier1_features = [
        feature
        for feature, spec
        in FEATURE_SPECS.items()
        if (
            spec[
                "inclusion_role"
            ]
            == "PRIMARY_T1"
        )
    ]

    tier2_cbc_features = [
        feature
        for feature, spec
        in FEATURE_SPECS.items()
        if (
            spec[
                "inclusion_role"
            ]
            == "PRIMARY_T2_CBC"
        )
    ]

    tier2_sensitivity_features = [
        feature
        for feature, spec
        in FEATURE_SPECS.items()
        if (
            spec[
                "inclusion_role"
            ]
            == "T2_SENSITIVITY"
        )
    ]

    out[
        "_audit_tier1_nonmissing_count"
    ] = (
        out[
            tier1_features
        ]
        .notna()
        .sum(axis=1)
    )

    out[
        "_audit_tier2_cbc_nonmissing_count"
    ] = (
        out[
            tier2_cbc_features
        ]
        .notna()
        .sum(axis=1)
    )

    out[
        "_audit_t2s_chem_nonmissing_count"
    ] = (
        out[
            tier2_sensitivity_features
        ]
        .notna()
        .sum(axis=1)
    )

    return (
        out,
        coverage_records,
        smoking_audit_records,
    )


# =============================================================================
# 12. FEATURE AVAILABILITY MATRIX
# =============================================================================

def build_availability_matrix(
    candidate_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Concept-by-cycle availability matrix.
    """

    if candidate_df.empty:
        return pd.DataFrame()

    matrix = (
        candidate_df
        .pivot_table(
            index=[
                "feature",
                "domain",
                "tier",
                "inclusion_role",
            ],
            columns="cycle",
            values="available",
            aggfunc="first",
        )
        .reset_index()
    )

    for cycle in ALL_CYCLES:

        if cycle not in matrix.columns:
            matrix[cycle] = False

    ordered_columns = [
        "feature",
        "domain",
        "tier",
        "inclusion_role",
    ] + ALL_CYCLES

    return matrix[
        ordered_columns
    ]


# =============================================================================
# 13. FINAL FEATURE MANIFEST
# =============================================================================

def build_final_feature_set() -> pd.DataFrame:
    """
    Explicit model-feature manifest.

    This file is the authoritative Stage-02B feature-set definition.
    """

    rows = []

    for (
        feature,
        spec,
    ) in FEATURE_SPECS.items():

        role = (
            spec[
                "inclusion_role"
            ]
        )

        if role == "PRIMARY_T1":

            modeling_block = "Tier1"

            use_primary = True

            use_sensitivity = True

        elif role == "PRIMARY_T2_CBC":

            modeling_block = (
                "Tier2_CBC_increment"
            )

            use_primary = True

            use_sensitivity = True

        elif role == "T2_SENSITIVITY":

            modeling_block = (
                "Tier2_chemistry_sensitivity"
            )

            use_primary = False

            use_sensitivity = True

        else:

            modeling_block = (
                "Supplementary_audit_only"
            )

            use_primary = False

            use_sensitivity = False

        rows.append(
            {
                "feature": feature,
                "domain":
                    spec["domain"],
                "tier":
                    spec["tier"],
                "inclusion_role":
                    role,
                "modeling_block":
                    modeling_block,
                "primary_model_feature":
                    use_primary,
                "sensitivity_model_feature":
                    use_sensitivity,
                "derived":
                    spec["derived"],
                "raw_source_candidates":
                    "|".join(
                        spec["preferred"]
                    ),
                "transform":
                    spec["transform"],
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 14. HARMONIZATION DICTIONARY
# =============================================================================

def build_harmonization_dictionary() -> pd.DataFrame:
    """
    Human-readable harmonization dictionary.
    """

    rows = []

    notes = {

        "age_years":
            "Age in years; retained numerically.",

        "sex":
            "RIAGENDR stable coding retained: 1 male, 2 female.",

        "race_ethnicity_5cat":
            "RIDRETH1 five-category representation retained for "
            "long-horizon consistency.",

        "education_adult_5cat":
            "DMDEDUC2 codes 1–5 retained. Structural age-related "
            "missingness is not imputed in Stage 02B.",

        "poverty_income_ratio":
            "Family poverty-income ratio retained numerically.",

        "health_insurance":
            "HIQ011 recoded yes=1, no=0.",

        "has_routine_place_for_care":
            "HUQ030 recoded to has at least one routine place: "
            "native 1/3 -> 1; native 2 -> 0.",

        "adult_food_security_4cat":
            "FSDAD retained as four-category adult food-security "
            "construct: full, marginal, low, very low.",

        "bmi":
            "Measured BMI retained numerically.",

        "waist_cm":
            "Measured waist circumference retained numerically.",

        "smoking_status_3cat":
            "Derived from SMQ020 + SMQ040: "
            "0 never, 1 former, 2 current. "
            "Corrects conditional-question missingness.",

        "doctor_diagnosed_diabetes":
            "DIQ010 recoded: no=0, yes=1, borderline=2.",

        "doctor_diagnosed_hypertension":
            "BPQ020 recoded yes=1, no=0.",

        "day1_energy_kcal":
            "Day-1 participant-level total energy.",

        "day1_dietary_iron_mg":
            "Day-1 participant-level dietary iron.",

        "day1_dietary_vitamin_d":
            "Day-1 participant-level dietary vitamin D.",

        "serum_albumin":
            "Routine chemistry; sensitivity extension only.",

        "serum_creatinine":
            "Routine chemistry; sensitivity extension only.",

        "serum_glucose":
            "Routine chemistry; sensitivity extension only.",
    }

    for (
        feature,
        spec,
    ) in FEATURE_SPECS.items():

        rows.append(
            {
                "harmonized_feature":
                    feature,
                "domain":
                    spec["domain"],
                "tier":
                    spec["tier"],
                "inclusion_role":
                    spec[
                        "inclusion_role"
                    ],
                "raw_variable_candidates":
                    "|".join(
                        spec["preferred"]
                    ),
                "derived":
                    spec["derived"],
                "transform":
                    spec["transform"],
                "stage02B_action":
                    (
                        "HARMONIZE"
                        if spec[
                            "inclusion_role"
                        ]
                        !=
                        "SUPPLEMENTARY_AUDIT"
                        else
                        "AUDIT_ONLY"
                    ),
                "notes":
                    notes.get(
                        feature,
                        (
                            "Primary CBC feature."
                            if spec[
                                "inclusion_role"
                            ]
                            ==
                            "PRIMARY_T2_CBC"
                            else
                            "Retained for availability audit only."
                        ),
                    ),
            }
        )

    # Explicit outcome leakage exclusions.
    rows.extend(
        [
            {
                "harmonized_feature":
                    "ferritin",
                "domain":
                    "confirmatory_biomarker",
                "tier":
                    "CONFIRMATORY_ONLY",
                "inclusion_role":
                    "EXCLUDE",
                "raw_variable_candidates":
                    "LBXFER|LBDFERSI",
                "derived":
                    False,
                "transform":
                    None,
                "stage02B_action":
                    "PROHIBIT_AS_PREDICTOR",
                "notes":
                    "Iron outcome-defining confirmatory biomarker.",
            },
            {
                "harmonized_feature":
                    "25oh_vitamin_d",
                "domain":
                    "confirmatory_biomarker",
                "tier":
                    "CONFIRMATORY_ONLY",
                "inclusion_role":
                    "EXCLUDE",
                "raw_variable_candidates":
                    "LBXVIDMS|LBDVIDMS",
                "derived":
                    False,
                "transform":
                    None,
                "stage02B_action":
                    "PROHIBIT_AS_PREDICTOR",
                "notes":
                    "Vitamin-D outcome-defining confirmatory biomarker.",
            },
        ]
    )

    return pd.DataFrame(rows)


# =============================================================================
# 15. TIER DEFINITION
# =============================================================================

def build_tier_definition() -> pd.DataFrame:
    """
    Explicit resource-tier structure.
    """

    rows = []

    for (
        feature,
        spec,
    ) in FEATURE_SPECS.items():

        role = (
            spec[
                "inclusion_role"
            ]
        )

        if role == "PRIMARY_T1":

            rows.append(
                {
                    "analysis_tier":
                        "Tier 1",
                    "feature":
                        feature,
                    "domain":
                        spec["domain"],
                    "incremental_block":
                        "Low-resource core",
                    "primary_analysis":
                        True,
                    "resource_interpretation":
                        (
                            "Non-specialized / "
                            "low-resource screening information"
                        ),
                }
            )

        elif role == "PRIMARY_T2_CBC":

            rows.append(
                {
                    "analysis_tier":
                        "Tier 2 primary",
                    "feature":
                        feature,
                    "domain":
                        spec["domain"],
                    "incremental_block":
                        "Routine CBC",
                    "primary_analysis":
                        True,
                    "resource_interpretation":
                        (
                            "Tier 1 plus routine "
                            "hematology information"
                        ),
                }
            )

        elif role == "T2_SENSITIVITY":

            rows.append(
                {
                    "analysis_tier":
                        "Tier 2 sensitivity",
                    "feature":
                        feature,
                    "domain":
                        spec["domain"],
                    "incremental_block":
                        "Routine chemistry",
                    "primary_analysis":
                        False,
                    "resource_interpretation":
                        (
                            "Expanded routine-laboratory "
                            "sensitivity analysis"
                        ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# 16. MISSINGNESS TABLE
# =============================================================================

def build_missingness(
    cycle: str,
    harmonized: pd.DataFrame,
) -> List[Dict]:
    """
    Feature-level missingness.

    _audit_ columns are excluded because they are not modeling features.
    """

    records = []

    protected = {
        "SEQN",
        "survey_cycle",
    }

    for column in harmonized.columns:

        if column in protected:
            continue

        if column.startswith(
            "_audit_"
        ):
            continue

        n = len(
            harmonized
        )

        n_missing = int(
            harmonized[
                column
            ]
            .isna()
            .sum()
        )

        records.append(
            {
                "cycle": cycle,
                "cycle_label":
                    CYCLE_LABELS.get(
                        cycle,
                        cycle,
                    ),
                "feature":
                    column,
                "n_participants":
                    n,
                "n_nonmissing":
                    n - n_missing,
                "n_missing":
                    n_missing,
                "missing_pct":
                    pct(
                        n_missing,
                        n,
                    ),
            }
        )

    return records


# =============================================================================
# 17. TIER COVERAGE TABLE
# =============================================================================

def build_cycle_tier_coverage(
    cycle: str,
    harmonized: pd.DataFrame,
) -> List[Dict]:
    """
    Summarize complete / partial availability for each resource tier.
    """

    tier1 = [
        feature
        for feature, spec
        in FEATURE_SPECS.items()
        if spec[
            "inclusion_role"
        ] == "PRIMARY_T1"
    ]

    cbc = [
        feature
        for feature, spec
        in FEATURE_SPECS.items()
        if spec[
            "inclusion_role"
        ] == "PRIMARY_T2_CBC"
    ]

    chemistry = [
        feature
        for feature, spec
        in FEATURE_SPECS.items()
        if spec[
            "inclusion_role"
        ] == "T2_SENSITIVITY"
    ]

    tier2_primary = (
        tier1 + cbc
    )

    tier2_extended = (
        tier1
        + cbc
        + chemistry
    )

    tier_groups = [
        (
            "Tier1",
            tier1,
        ),
        (
            "Tier2_primary",
            tier2_primary,
        ),
        (
            "Tier2_extended_sensitivity",
            tier2_extended,
        ),
    ]

    rows = []

    for (
        tier_name,
        features,
    ) in tier_groups:

        complete = (
            harmonized[
                features
            ]
            .notna()
            .all(axis=1)
        )

        observed_counts = (
            harmonized[
                features
            ]
            .notna()
            .sum(axis=1)
        )

        rows.append(
            {
                "cycle": cycle,
                "cycle_label":
                    CYCLE_LABELS.get(
                        cycle,
                        cycle,
                    ),
                "tier":
                    tier_name,
                "n_features":
                    len(features),
                "n_participants":
                    len(
                        harmonized
                    ),
                "n_complete_cases":
                    int(
                        complete.sum()
                    ),
                "complete_case_pct":
                    pct(
                        complete.sum(),
                        len(
                            harmonized
                        ),
                    ),
                "median_features_observed":
                    float(
                        observed_counts
                        .median()
                    ),
                "min_features_observed":
                    int(
                        observed_counts
                        .min()
                    ),
                "max_features_observed":
                    int(
                        observed_counts
                        .max()
                    ),
            }
        )

    return rows


# =============================================================================
# 18. STRUCTURAL MISSINGNESS NOTES
# =============================================================================

def build_structural_missingness_notes() -> pd.DataFrame:
    """
    Explicitly distinguish known questionnaire structure from generic
    missing data.

    These notes guide Stage 03 / modeling and prevent inappropriate
    interpretation of missingness.
    """

    rows = [
        {
            "feature":
                "education_adult_5cat",
            "issue_type":
                "eligibility_or_questionnaire_structure",
            "stage02B_handling":
                "Retain missing values",
            "note":
                (
                    "DMDEDUC2 is not uniformly observed across all ages. "
                    "Do not replace structural missingness during harmonization."
                ),
        },
        {
            "feature":
                "smoking_status_3cat",
            "issue_type":
                "skip_pattern_resolved",
            "stage02B_handling":
                (
                    "Derive using SMQ020 and SMQ040"
                ),
            "note":
                (
                    "SMQ040 is conditionally administered. "
                    "Never-smokers are derived from SMQ020 rather than "
                    "being mislabeled as missing."
                ),
        },
        {
            "feature":
                "day1_energy_kcal",
            "issue_type":
                "dietary_recall_availability",
            "stage02B_handling":
                "Retain observed Day-1 value; no Stage-02B imputation",
            "note":
                (
                    "Dietary recall nonresponse is preserved. "
                    "Training-pipeline handling is deferred."
                ),
        },
        {
            "feature":
                "day1_dietary_iron_mg",
            "issue_type":
                "dietary_recall_availability",
            "stage02B_handling":
                "Retain observed Day-1 value; no Stage-02B imputation",
            "note":
                (
                    "Dietary recall nonresponse is preserved."
                ),
        },
        {
            "feature":
                "day1_dietary_vitamin_d",
            "issue_type":
                "dietary_recall_availability",
            "stage02B_handling":
                "Retain observed Day-1 value; no Stage-02B imputation",
            "note":
                (
                    "Dietary recall nonresponse is preserved."
                ),
        },
        {
            "feature":
                "household_food_security_candidate",
            "issue_type":
                "cross_cycle_availability",
            "stage02B_handling":
                "Supplementary audit only",
            "note":
                (
                    "Not used in common primary Tier 1 because the "
                    "availability audit did not support a complete "
                    "long-horizon series."
                ),
        },
        {
            "feature":
                "healthcare_visits_candidate",
            "issue_type":
                "variable_change_and_availability",
            "stage02B_handling":
                "Supplementary audit only",
            "note":
                (
                    "HUQ050/HUQ051 structure changes and lack of a "
                    "complete common-cycle series preclude primary use."
                ),
        },
        {
            "feature":
                "alcohol_use_candidate",
            "issue_type":
                "questionnaire_definition_change",
            "stage02B_handling":
                "Supplementary audit only",
            "note":
                (
                    "ALQ101 and ALQ111 represent a questionnaire "
                    "transition; no forced equivalence is imposed."
                ),
        },
        {
            "feature":
                "recreational_physical_activity",
            "issue_type":
                "cross_cycle_availability",
            "stage02B_handling":
                "Supplementary audit only",
            "note":
                (
                    "PAQ650/PAQ665 do not form a complete primary "
                    "long-horizon feature series in the audit."
                ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 19. DECISION LEDGER
# =============================================================================

def build_decision_ledger() -> pd.DataFrame:
    """
    Final Stage-02B decisions.

    D008+ continue the Stage-01/02A ledger.
    """

    today = (
        datetime.now()
        .date()
        .isoformat()
    )

    decisions = [

        {
            "decision_id": "D008",
            "stage": "02B",
            "topic":
                "Predictor harmonization",
            "decision":
                (
                    "Only predictors with explicit and reproducible "
                    "cross-cycle recoding rules enter harmonized model blocks."
                ),
            "reason":
                (
                    "Prevents silent equivalence assumptions across "
                    "changing NHANES components."
                ),
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D009",
            "stage": "02B",
            "topic":
                "Confirmatory biomarkers",
            "decision":
                (
                    "Ferritin and 25-hydroxyvitamin D are prohibited "
                    "from all predictor tiers."
                ),
            "reason":
                (
                    "They define/confirm study outcomes and would "
                    "produce target leakage."
                ),
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D010",
            "stage": "02B",
            "topic":
                "Tier 1 definition",
            "decision":
                (
                    "Tier 1 contains demographics, socioeconomic status, "
                    "insurance/access, adult food security, anthropometrics, "
                    "smoking, medical history and Day-1 dietary intake."
                ),
            "reason":
                (
                    "Represents information potentially available before "
                    "nutrient-specific confirmatory testing."
                ),
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D011",
            "stage": "02B",
            "topic":
                "Tier 2 primary definition",
            "decision":
                (
                    "Primary Tier 2 equals Tier 1 plus routine CBC."
                ),
            "reason":
                (
                    "Provides a clean incremental resource tier and "
                    "tests the value of routine hematology."
                ),
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D012",
            "stage": "02B",
            "topic":
                "Routine chemistry",
            "decision":
                (
                    "Albumin, creatinine and glucose form a predefined "
                    "Tier-2 sensitivity extension, not the primary Tier 2."
                ),
            "reason":
                (
                    "Preserves a parsimonious CBC-centered primary "
                    "resource hierarchy while allowing expanded-lab analysis."
                ),
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D013",
            "stage": "02B",
            "topic":
                "Smoking",
            "decision":
                (
                    "Replace separate SMQ020/SMQ040 model features with "
                    "one derived never/former/current smoking variable."
                ),
            "reason":
                (
                    "SMQ040 has questionnaire skip logic; deriving status "
                    "avoids misclassifying legitimate skips as ordinary missingness."
                ),
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D014",
            "stage": "02B",
            "topic":
                "Food security",
            "decision":
                (
                    "Use adult food-security category FSDAD as the "
                    "primary common-core food-security predictor."
                ),
            "reason":
                (
                    "Stage-02B availability audit showed FSDAD across "
                    "the complete study cycle series."
                ),
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D015",
            "stage": "02B",
            "topic":
                "Healthcare access",
            "decision":
                (
                    "Use HUQ030-derived routine-place-for-care indicator "
                    "in primary Tier 1."
                ),
            "reason":
                (
                    "Availability audit showed HUQ030 consistently "
                    "across the complete cycle series."
                ),
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D016",
            "stage": "02B",
            "topic":
                "Dietary vitamin D",
            "decision":
                (
                    "Use DR1TVD as a Tier-1 dietary predictor."
                ),
            "reason":
                (
                    "Availability audit demonstrated complete "
                    "cross-cycle presence."
                ),
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D017",
            "stage": "02B",
            "topic":
                "Supplementary questionnaire variables",
            "decision":
                (
                    "Alcohol, recreational physical activity, healthcare "
                    "visit count and household food security are excluded "
                    "from the primary common-core model."
                ),
            "reason":
                (
                    "Cross-cycle questionnaire changes or availability "
                    "breaks prevent clean long-horizon primary use."
                ),
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D018",
            "stage": "02B",
            "topic":
                "Missing data",
            "decision":
                (
                    "No statistical imputation, standardization or "
                    "data-driven missing-value replacement occurs in Stage 02B."
                ),
            "reason":
                (
                    "These operations must be learned exclusively "
                    "inside training data to avoid leakage."
                ),
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D019",
            "stage": "02B",
            "topic":
                "Dietary grain",
            "decision":
                (
                    "Primary dietary predictors come from participant-level "
                    "Day-1 total-intake variables rather than food-item rows."
                ),
            "reason":
                (
                    "Preserves one-row-per-SEQN structure and avoids "
                    "premature DR1IFF/DR2IFF aggregation."
                ),
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D020",
            "stage": "02B",
            "topic":
                "Pre-pandemic release",
            "decision":
                (
                    "P-cycle predictors are harmonized for supplementary "
                    "analysis but are not treated as an independent period "
                    "following J."
                ),
            "reason":
                (
                    "The 2017–March 2020 pre-pandemic release overlaps "
                    "with the 2017–2018 cycle."
                ),
            "status": "FROZEN",
            "date": today,
        },
    ]

    return pd.DataFrame(
        decisions
    )


# =============================================================================
# 20. VALIDATE FINAL FEATURE ARCHITECTURE
# =============================================================================

def validate_feature_architecture():
    """
    Static checks before loading any data.
    """

    # -------------------------------------------------------------------------
    # Outcome biomarkers must not appear in any preferred predictor list.
    # -------------------------------------------------------------------------

    for (
        feature,
        spec,
    ) in FEATURE_SPECS.items():

        for raw in spec["preferred"]:

            if is_forbidden_predictor(
                raw
            ):

                raise RuntimeError(
                    "Feature specification contains "
                    "a forbidden confirmatory biomarker:\n"
                    f"feature={feature}\n"
                    f"raw_variable={raw}"
                )

    # -------------------------------------------------------------------------
    # Unique feature names automatically guaranteed by dict, but explicitly
    # check primary blocks are non-empty.
    # -------------------------------------------------------------------------

    n_t1 = sum(
        1
        for spec
        in FEATURE_SPECS.values()
        if (
            spec[
                "inclusion_role"
            ]
            == "PRIMARY_T1"
        )
    )

    n_cbc = sum(
        1
        for spec
        in FEATURE_SPECS.values()
        if (
            spec[
                "inclusion_role"
            ]
            == "PRIMARY_T2_CBC"
        )
    )

    if n_t1 == 0:
        raise RuntimeError(
            "Tier 1 contains no features."
        )

    if n_cbc == 0:
        raise RuntimeError(
            "Tier 2 CBC block contains no features."
        )


# =============================================================================
# 21. MAIN
# =============================================================================

def main():

    logger.info("")
    logger.info(
        "=" * 80
    )
    logger.info(
        "STAGE 02B — CORRECTED FINAL VERSION"
    )
    logger.info(
        "=" * 80
    )

    validate_feature_architecture()

    candidate_records = []
    coverage_records = []
    missingness_records = []
    tier_coverage_records = []
    smoking_audit_records = []

    loaded_cycles = []

    # =========================================================================
    # PROCESS EACH CYCLE INDEPENDENTLY
    # =========================================================================

    for cycle in ALL_CYCLES:

        path = cycle_file_path(
            cycle
        )

        if path is None:

            logger.warning(
                "Joined native dataset not found: "
                f"{cycle}"
            )

            continue

        logger.info("")
        logger.info(
            "-" * 80
        )
        logger.info(
            f"PROCESSING {cycle}"
        )
        logger.info(
            "-" * 80
        )

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        loaded_cycles.append(
            cycle
        )

        logger.info(
            "Native shape: "
            f"{df.shape[0]:,} rows × "
            f"{df.shape[1]:,} columns"
        )

        # ---------------------------------------------------------------------
        # Feature availability audit
        # ---------------------------------------------------------------------

        (
            cycle_candidates,
            resolved_map,
        ) = audit_cycle_features(
            cycle=cycle,
            df=df,
        )

        candidate_records.extend(
            cycle_candidates
        )

        # ---------------------------------------------------------------------
        # Harmonization
        # ---------------------------------------------------------------------

        (
            harmonized,
            cycle_coverage,
            cycle_smoking_audit,
        ) = harmonize_cycle(
            cycle=cycle,
            df=df,
            resolved_map=resolved_map,
        )

        coverage_records.extend(
            cycle_coverage
        )

        smoking_audit_records.extend(
            cycle_smoking_audit
        )

        # ---------------------------------------------------------------------
        # Missingness
        # ---------------------------------------------------------------------

        missingness_records.extend(
            build_missingness(
                cycle=cycle,
                harmonized=harmonized,
            )
        )

        # ---------------------------------------------------------------------
        # Tier coverage
        # ---------------------------------------------------------------------

        tier_coverage_records.extend(
            build_cycle_tier_coverage(
                cycle=cycle,
                harmonized=harmonized,
            )
        )

        # ---------------------------------------------------------------------
        # Integrity checks
        # ---------------------------------------------------------------------

        if (
            len(harmonized)
            != len(df)
        ):

            raise RuntimeError(
                f"{cycle}: row count changed "
                "during harmonization."
            )

        if (
            harmonized[
                "SEQN"
            ]
            .duplicated()
            .any()
        ):

            raise RuntimeError(
                f"{cycle}: duplicate SEQN "
                "generated during harmonization."
            )

        # ---------------------------------------------------------------------
        # Explicit outcome leakage scan
        # ---------------------------------------------------------------------

        for col in harmonized.columns:

            if is_forbidden_predictor(
                col
            ):

                raise RuntimeError(
                    "Outcome leakage detected "
                    "in harmonized predictors: "
                    f"{col}"
                )

        # ---------------------------------------------------------------------
        # Save cycle file
        # ---------------------------------------------------------------------

        output_path = (
            HARMONIZED_DIR
            / (
                f"NHANES_{cycle}"
                "_predictors_harmonized.csv"
            )
        )

        harmonized.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            f"Saved: {output_path.name}"
        )

        logger.info(
            "Harmonized shape: "
            f"{harmonized.shape[0]:,} × "
            f"{harmonized.shape[1]:,}"
        )

    # =========================================================================
    # CREATE AUDIT TABLES
    # =========================================================================

    candidate_df = pd.DataFrame(
        candidate_records
    )

    coverage_df = pd.DataFrame(
        coverage_records
    )

    missingness_df = pd.DataFrame(
        missingness_records
    )

    tier_coverage_df = pd.DataFrame(
        tier_coverage_records
    )

    smoking_audit_df = pd.DataFrame(
        smoking_audit_records
    )

    availability_matrix = (
        build_availability_matrix(
            candidate_df
        )
    )

    harmonization_dictionary = (
        build_harmonization_dictionary()
    )

    final_feature_set = (
        build_final_feature_set()
    )

    tier_definition = (
        build_tier_definition()
    )

    structural_missingness = (
        build_structural_missingness_notes()
    )

    decision_ledger = (
        build_decision_ledger()
    )

    # =========================================================================
    # CROSS-CYCLE PRIMARY FEATURE CHECK
    # =========================================================================

    if not candidate_df.empty:

        primary_features = [
            feature
            for feature, spec
            in FEATURE_SPECS.items()
            if spec[
                "inclusion_role"
            ]
            in {
                "PRIMARY_T1",
                "PRIMARY_T2_CBC",
            }
        ]

        main_availability = (
            candidate_df[
                (
                    candidate_df[
                        "cycle"
                    ]
                    .isin(
                        MAIN_CYCLES
                    )
                )
                &
                (
                    candidate_df[
                        "feature"
                    ]
                    .isin(
                        primary_features
                    )
                )
            ]
        )

        availability_check = (
            main_availability
            .groupby(
                "feature"
            )[
                "available"
            ]
            .agg(
                [
                    "count",
                    "sum",
                ]
            )
            .reset_index()
        )

        availability_check[
            "available_all_main_cycles"
        ] = (
            availability_check[
                "sum"
            ]
            ==
            len(
                MAIN_CYCLES
            )
        )

    else:

        availability_check = (
            pd.DataFrame()
        )

    # =========================================================================
    # SAVE CSV OUTPUTS
    # =========================================================================

    candidate_df.to_csv(
        OUTPUT_DIR
        / "02B_predictor_variable_candidates.csv",
        index=False,
    )

    availability_matrix.to_csv(
        OUTPUT_DIR
        / "02B_feature_availability_matrix.csv",
        index=False,
    )

    harmonization_dictionary.to_csv(
        OUTPUT_DIR
        / "02B_harmonization_dictionary.csv",
        index=False,
    )

    final_feature_set.to_csv(
        OUTPUT_DIR
        / "02B_final_feature_set.csv",
        index=False,
    )

    tier_definition.to_csv(
        OUTPUT_DIR
        / "02B_tier_definition.csv",
        index=False,
    )

    coverage_df.to_csv(
        OUTPUT_DIR
        / "02B_cycle_feature_coverage.csv",
        index=False,
    )

    missingness_df.to_csv(
        OUTPUT_DIR
        / "02B_predictor_missingness.csv",
        index=False,
    )

    tier_coverage_df.to_csv(
        OUTPUT_DIR
        / "02B_cycle_tier_coverage.csv",
        index=False,
    )

    smoking_audit_df.to_csv(
        OUTPUT_DIR
        / "02B_smoking_derivation_audit.csv",
        index=False,
    )

    structural_missingness.to_csv(
        OUTPUT_DIR
        / "02B_structural_missingness_notes.csv",
        index=False,
    )

    availability_check.to_csv(
        OUTPUT_DIR
        / "02B_primary_feature_crosscycle_check.csv",
        index=False,
    )

    decision_ledger.to_csv(
        OUTPUT_DIR
        / "decision_ledger_stage02B.csv",
        index=False,
    )

    # =========================================================================
    # EXCEL AUDIT WORKBOOK
    # =========================================================================

    excel_path = (
        OUTPUT_DIR
        / "02B_predictor_audit_summary.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:

        availability_matrix.to_excel(
            writer,
            sheet_name=
                "Availability_Matrix",
            index=False,
        )

        final_feature_set.to_excel(
            writer,
            sheet_name=
                "Final_Feature_Set",
            index=False,
        )

        tier_definition.to_excel(
            writer,
            sheet_name=
                "Tier_Definition",
            index=False,
        )

        candidate_df.to_excel(
            writer,
            sheet_name=
                "Variable_Candidates",
            index=False,
        )

        coverage_df.to_excel(
            writer,
            sheet_name=
                "Feature_Coverage",
            index=False,
        )

        tier_coverage_df.to_excel(
            writer,
            sheet_name=
                "Tier_Coverage",
            index=False,
        )

        missingness_df.to_excel(
            writer,
            sheet_name=
                "Missingness",
            index=False,
        )

        smoking_audit_df.to_excel(
            writer,
            sheet_name=
                "Smoking_Audit",
            index=False,
        )

        availability_check.to_excel(
            writer,
            sheet_name=
                "Primary_CrossCycle",
            index=False,
        )

        harmonization_dictionary.to_excel(
            writer,
            sheet_name=
                "Harmonization_Dict",
            index=False,
        )

        structural_missingness.to_excel(
            writer,
            sheet_name=
                "Structural_Missing",
            index=False,
        )

        decision_ledger.to_excel(
            writer,
            sheet_name=
                "Decision_Ledger",
            index=False,
        )

    # =========================================================================
    # FINAL CONSOLE REPORT
    # =========================================================================

    print("\n")
    print(
        "=" * 82
    )
    print(
        "STAGE 02B — CORRECTED FINAL VERSION COMPLETE"
    )
    print(
        "=" * 82
    )

    print(
        "\nCycles processed:"
    )

    for cycle in loaded_cycles:

        print(
            f"  ✓ {cycle}"
        )

    # -------------------------------------------------------------------------
    # Final feature counts
    # -------------------------------------------------------------------------

    t1_features = [
        feature
        for feature, spec
        in FEATURE_SPECS.items()
        if spec[
            "inclusion_role"
        ]
        == "PRIMARY_T1"
    ]

    cbc_features = [
        feature
        for feature, spec
        in FEATURE_SPECS.items()
        if spec[
            "inclusion_role"
        ]
        == "PRIMARY_T2_CBC"
    ]

    chemistry_features = [
        feature
        for feature, spec
        in FEATURE_SPECS.items()
        if spec[
            "inclusion_role"
        ]
        == "T2_SENSITIVITY"
    ]

    print(
        "\nFINAL RESOURCE-TIER ARCHITECTURE"
    )
    print(
        "-" * 82
    )

    print(
        f"\nTier 1: "
        f"{len(t1_features)} features"
    )

    for feature in t1_features:

        print(
            f"  • {feature}"
        )

    print(
        f"\nTier 2 CBC increment: "
        f"{len(cbc_features)} features"
    )

    for feature in cbc_features:

        print(
            f"  • {feature}"
        )

    print(
        "\nTier 2 sensitivity chemistry:"
    )

    for feature in chemistry_features:

        print(
            f"  • {feature}"
        )

    # -------------------------------------------------------------------------
    # Availability failures
    # -------------------------------------------------------------------------

    if not availability_check.empty:

        failures = (
            availability_check[
                ~availability_check[
                    "available_all_main_cycles"
                ]
            ]
        )

        print(
            "\nPRIMARY FEATURE CROSS-CYCLE CHECK"
        )
        print(
            "-" * 82
        )

        if failures.empty:

            print(
                "PASS: All primary Tier-1 and "
                "Tier-2 CBC concepts resolved "
                "in all six main cycles."
            )

        else:

            print(
                "WARNING: The following primary "
                "features did not resolve in all "
                "main cycles:"
            )

            print(
                failures.to_string(
                    index=False
                )
            )

    # -------------------------------------------------------------------------
    # Smoking status
    # -------------------------------------------------------------------------

    if not smoking_audit_df.empty:

        print(
            "\nSMOKING DERIVATION AUDIT"
        )
        print(
            "-" * 82
        )

        display = [
            "cycle",
            "n_never",
            "n_former",
            "n_current",
            "n_unclassified",
        ]

        print(
            smoking_audit_df[
                display
            ]
            .to_string(
                index=False
            )
        )

    # -------------------------------------------------------------------------
    # Output locations
    # -------------------------------------------------------------------------

    print(
        "\nPrimary audit workbook:"
    )

    print(
        f"  {excel_path}"
    )

    print(
        "\nFinal feature manifest:"
    )

    print(
        "  "
        + str(
            OUTPUT_DIR
            / "02B_final_feature_set.csv"
        )
    )

    print(
        "\nHarmonized cycle files:"
    )

    print(
        f"  {HARMONIZED_DIR}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  • Do not stack cycles yet."
    )

    print(
        "  • Do not impute predictors yet."
    )

    print(
        "  • Do not fit models yet."
    )

    print(
        "  • Ferritin and 25(OH)D remain prohibited predictors."
    )

    print(
        "  • P remains supplementary because it overlaps J."
    )

    print(
        "\nIf the primary cross-cycle check passes, "
        "Stage 02B can be frozen and Stage 03 can begin."
    )

    print(
        "\n" + "=" * 82
    )


# =============================================================================
# 22. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()