"""
===============================================================================
STAGE 02A
NHANES Outcome Availability, Eligibility, Definition & Comparability Audit
===============================================================================

PURPOSE
-------
Stage 02A audits the two proposed outcomes:

    1. Vitamin D status
    2. Pre-anemic iron deficiency


Expected Stage 01 outputs:
--------------------------
processed/
    01_inventory/
        01_file_inventory.csv
        01_variable_inventory.csv
        01_data_lineage.csv

    02_joined_native/
        main_cycles/
            NHANES_2009_2010_joined_native.csv
            ...
        supplementary/
            NHANES_2017_2020_PrePandemic_joined_native.csv

Outputs:
--------
processed/
    03_outcome_audit/
        02A_outcome_variable_candidates.csv
        02A_outcome_availability_matrix.csv
        02A_outcome_eligibility_audit.csv
        02A_lab_measurement_comparability.csv
        02A_candidate_outcome_definitions.csv
        02A_cycle_outcome_descriptives.csv
        02A_matched_iron_population_audit.csv
        02A_outcome_audit_summary.xlsx
        decision_ledger.csv

Author: AISC NHANES project
===============================================================================
"""

from __future__ import annotations

import re
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INVENTORY_DIR = PROJECT_ROOT / "processed" / "01_inventory"

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
    / "03_outcome_audit"
)

LOG_DIR = PROJECT_ROOT / "processed" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


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

ALL_CYCLES = MAIN_CYCLES + SUPPLEMENTARY_CYCLES


CYCLE_LABELS = {
    "2009_2010": "2009–2010",
    "2011_2012": "2011–2012",
    "2013_2014": "2013–2014",
    "2015_2016": "2015–2016",
    "2017_2018": "2017–2018",
    "2017_2020_PrePandemic": "2017–March 2020 pre-pandemic",
    "2021_2023": "2021–2023",
}


# =============================================================================
# 3. CANDIDATE VARIABLE NAMES
# =============================================================================

# Preferred variables are searched first.
#
# We intentionally also perform keyword searches later because:
#   - NHANES naming can change
#   - Stage 01 may have renamed collisions with "__SOURCEFILE"

PREFERRED_VARIABLES = {

    "age": [
        "RIDAGEYR",
    ],

    "sex": [
        "RIAGENDR",
    ],

    "pregnancy": [
        "RIDEXPRG",
    ],

    "vitamin_d": [
        "LBXVIDMS",
        "LBDVIDMS",
        "LBXVID",
    ],

    "ferritin": [
        "LBXFER",
        "LBDFERSI",
    ],

    "hemoglobin": [
        "LBXHGB",
    ],

    "crp": [
        "LBXCRP",
        "LBXHSCRP",
    ],

    "agp": [
        "LBXAGP",
    ],
}


# Keyword patterns are for AUDIT / discovery.
# They are not automatically accepted as equivalent variables.

DISCOVERY_PATTERNS = {

    "vitamin_d": [
        r"VID",
        r"25OH",
        r"VITD",
    ],

    "ferritin": [
        r"FER",
        r"FERR",
    ],

    "hemoglobin": [
        r"HGB",
        r"HEMOGLOBIN",
    ],

    "crp": [
        r"CRP",
    ],

    "agp": [
        r"AGP",
        r"GLYCOPROTEIN",
    ],

    "pregnancy": [
        r"RIDEXPRG",
        r"PREG",
    ],
}


# =============================================================================
# 4. CANDIDATE OUTCOME DEFINITIONS
# =============================================================================

# These are CANDIDATE thresholds for audit purposes.
# Stage 02A does not declare all of them final.

# Vitamin D:
#
# LBXVIDMS is typically expressed in nmol/L.
#
# 30 nmol/L = 12 ng/mL
# 50 nmol/L = 20 ng/mL
#
# We calculate both so the eventual primary definition can be frozen
# after protocol/literature review.

VITD_SEVERE_THRESHOLD_NMOL_L = 30.0
VITD_INSUFFICIENCY_THRESHOLD_NMOL_L = 50.0


# Ferritin candidate definitions.
#
# 15 µg/L is retained as the conservative primary candidate because it
# aligns with the project's previously established iron work.
#
# 25 µg/L is included only as a sensitivity candidate.

FERRITIN_PRIMARY_UG_L = 15.0
FERRITIN_SENSITIVITY_UG_L = 25.0


# Candidate adult anemia thresholds.
#
# These are used only for descriptive candidate construction in Stage 02A.
# Final thresholds must be frozen before modeling.

ANEMIA_HGB_MALE_G_DL = 13.0
ANEMIA_HGB_FEMALE_G_DL = 12.0


# Initial adult age range to audit.
# No final analytic exclusion is imposed on native files.

ADULT_MIN_AGE = 18


# Candidate population for matched ferritin transportability analysis.
MATCHED_IRON_MIN_AGE = 18
MATCHED_IRON_MAX_AGE = 49
MATCHED_IRON_SEX_CODE = 2  # NHANES RIAGENDR: female


# =============================================================================
# 5. LOGGING
# =============================================================================

LOG_FILE = LOG_DIR / "stage02A_outcome_audit.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


# =============================================================================
# 6. BASIC HELPERS
# =============================================================================

def safe_numeric(series: pd.Series) -> pd.Series:
    """
    Convert values to numeric without raising errors.
    """
    return pd.to_numeric(series, errors="coerce")


def pct(numerator: float, denominator: float) -> float:
    """
    Safe percentage.
    """
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return np.nan

    return 100.0 * numerator / denominator


def cycle_file_path(cycle: str) -> Optional[Path]:
    """
    Locate joined native dataset for requested cycle.
    """

    filename = f"NHANES_{cycle}_joined_native.csv"

    if cycle in MAIN_CYCLES:
        path = MAIN_JOIN_DIR / filename
    else:
        path = SUPP_JOIN_DIR / filename

    if path.exists():
        return path

    return None


def resolve_preferred_variable(
    columns: List[str],
    concept: str,
) -> Optional[str]:
    """
    Return preferred variable if present.

    Exact native variable is preferred.

    Collision-renamed versions such as:
        LBXHGB__CBC_L

    may be used only if the original variable is not present.
    """

    preferred = PREFERRED_VARIABLES.get(concept, [])

    # Exact native match first.
    for candidate in preferred:
        if candidate in columns:
            return candidate

    # Collision-renamed fallback.
    for candidate in preferred:
        matches = [
            c
            for c in columns
            if c.startswith(candidate + "__")
        ]

        if len(matches) == 1:
            return matches[0]

    return None


def discover_candidate_variables(
    columns: List[str],
    concept: str,
) -> List[str]:
    """
    Discover variables whose names resemble an outcome concept.

    These are NOT automatically treated as equivalent.
    """

    patterns = DISCOVERY_PATTERNS.get(concept, [])

    matches = []

    for col in columns:

        upper = col.upper()

        for pattern in patterns:

            if re.search(pattern, upper):

                matches.append(col)
                break

    return sorted(set(matches))


def measurement_summary(series: pd.Series) -> Dict:
    """
    Descriptive summary for a numeric biomarker.
    """

    x = safe_numeric(series)

    observed = x.dropna()

    if len(observed) == 0:

        return {
            "n_measured": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "n_measured": int(observed.notna().sum()),
        "mean": observed.mean(),
        "sd": observed.std(),
        "median": observed.median(),
        "q1": observed.quantile(0.25),
        "q3": observed.quantile(0.75),
        "min": observed.min(),
        "max": observed.max(),
    }


# =============================================================================
# 7. LOAD STAGE 01 LINEAGE
# =============================================================================

def load_lineage() -> pd.DataFrame:
    """
    Load Stage 01 data lineage if available.
    """

    path = INVENTORY_DIR / "01_data_lineage.csv"

    if not path.exists():

        logger.warning(
            "Stage 01 lineage not found. "
            "Source-file mapping will be incomplete."
        )

        return pd.DataFrame()

    return pd.read_csv(path, low_memory=False)


def source_for_variable(
    lineage: pd.DataFrame,
    cycle: str,
    variable: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Retrieve original source file/domain from Stage 01 lineage.
    """

    if (
        variable is None
        or lineage.empty
        or "final_variable" not in lineage.columns
    ):
        return None, None

    subset = lineage[
        (lineage["cycle"] == cycle)
        & (lineage["final_variable"] == variable)
    ]

    if len(subset) == 0:
        return None, None

    row = subset.iloc[0]

    return (
        row.get("source_domain"),
        row.get("source_file"),
    )


# =============================================================================
# 8. OUTCOME VARIABLE DISCOVERY
# =============================================================================

def build_candidate_variable_audit(
    cycle: str,
    df: pd.DataFrame,
    lineage: pd.DataFrame,
) -> List[Dict]:
    """
    List relevant candidate variables present in each cycle.
    """

    records = []

    concepts = [
        "vitamin_d",
        "ferritin",
        "hemoglobin",
        "pregnancy",
        "crp",
        "agp",
    ]

    for concept in concepts:

        preferred = resolve_preferred_variable(
            list(df.columns),
            concept,
        )

        discovered = discover_candidate_variables(
            list(df.columns),
            concept,
        )

        if preferred is not None and preferred not in discovered:
            discovered.insert(0, preferred)

        if len(discovered) == 0:

            records.append(
                {
                    "cycle": cycle,
                    "cycle_label": CYCLE_LABELS.get(cycle, cycle),
                    "concept": concept,
                    "variable": None,
                    "selected_preferred_variable": None,
                    "is_selected_preferred": False,
                    "source_domain": None,
                    "source_file": None,
                }
            )

            continue

        for variable in discovered:

            source_domain, source_file = source_for_variable(
                lineage=lineage,
                cycle=cycle,
                variable=variable,
            )

            records.append(
                {
                    "cycle": cycle,
                    "cycle_label": CYCLE_LABELS.get(cycle, cycle),
                    "concept": concept,
                    "variable": variable,
                    "selected_preferred_variable": preferred,
                    "is_selected_preferred": variable == preferred,
                    "source_domain": source_domain,
                    "source_file": source_file,
                }
            )

    return records


# =============================================================================
# 9. OUTCOME AVAILABILITY
# =============================================================================

def build_availability_record(
    cycle: str,
    df: pd.DataFrame,
) -> Dict:
    """
    Determine basic outcome-variable availability.
    """

    columns = list(df.columns)

    age_var = resolve_preferred_variable(columns, "age")
    sex_var = resolve_preferred_variable(columns, "sex")
    preg_var = resolve_preferred_variable(columns, "pregnancy")

    vitd_var = resolve_preferred_variable(columns, "vitamin_d")
    ferritin_var = resolve_preferred_variable(columns, "ferritin")
    hgb_var = resolve_preferred_variable(columns, "hemoglobin")
    crp_var = resolve_preferred_variable(columns, "crp")
    agp_var = resolve_preferred_variable(columns, "agp")

    return {
        "cycle": cycle,
        "cycle_label": CYCLE_LABELS.get(cycle, cycle),
        "dataset_role": (
            "main"
            if cycle in MAIN_CYCLES
            else "supplementary"
        ),
        "n_native_participants": len(df),

        "age_variable": age_var,
        "sex_variable": sex_var,
        "pregnancy_variable": preg_var,

        "vitamin_d_available": vitd_var is not None,
        "vitamin_d_variable": vitd_var,

        "ferritin_available": ferritin_var is not None,
        "ferritin_variable": ferritin_var,

        "hemoglobin_available": hgb_var is not None,
        "hemoglobin_variable": hgb_var,

        "crp_candidate_available": crp_var is not None,
        "crp_candidate_variable": crp_var,

        "agp_candidate_available": agp_var is not None,
        "agp_candidate_variable": agp_var,
    }


# =============================================================================
# 10. ELIGIBILITY / ASCERTAINMENT AUDIT
# =============================================================================

def biomarker_eligibility_record(
    cycle: str,
    df: pd.DataFrame,
    biomarker_name: str,
    biomarker_var: Optional[str],
) -> Dict:
    """
    Describe who was measured for an outcome biomarker.
    """

    base = {
        "cycle": cycle,
        "cycle_label": CYCLE_LABELS.get(cycle, cycle),
        "biomarker": biomarker_name,
        "variable": biomarker_var,
        "n_native_participants": len(df),
    }

    if biomarker_var is None:

        base.update(
            {
                "n_measured": 0,
                "pct_native_measured": 0.0,
                "measured_age_min": np.nan,
                "measured_age_max": np.nan,
                "measured_age_mean": np.nan,
                "measured_male_n": np.nan,
                "measured_male_pct": np.nan,
                "measured_female_n": np.nan,
                "measured_female_pct": np.nan,
                "pregnant_measured_n": np.nan,
                "eligibility_signal": "biomarker_not_available",
            }
        )

        return base

    measured = safe_numeric(df[biomarker_var]).notna()

    n_measured = int(measured.sum())

    age_var = resolve_preferred_variable(
        list(df.columns),
        "age",
    )

    sex_var = resolve_preferred_variable(
        list(df.columns),
        "sex",
    )

    preg_var = resolve_preferred_variable(
        list(df.columns),
        "pregnancy",
    )

    age_min = np.nan
    age_max = np.nan
    age_mean = np.nan

    male_n = np.nan
    female_n = np.nan

    male_pct = np.nan
    female_pct = np.nan

    pregnant_measured_n = np.nan

    if age_var is not None and n_measured > 0:

        ages = safe_numeric(
            df.loc[measured, age_var]
        ).dropna()

        if len(ages) > 0:
            age_min = ages.min()
            age_max = ages.max()
            age_mean = ages.mean()

    if sex_var is not None and n_measured > 0:

        sex = safe_numeric(
            df.loc[measured, sex_var]
        )

        male_n = int((sex == 1).sum())
        female_n = int((sex == 2).sum())

        male_pct = pct(male_n, n_measured)
        female_pct = pct(female_n, n_measured)

    if preg_var is not None and n_measured > 0:

        preg = safe_numeric(
            df.loc[measured, preg_var]
        )

        pregnant_measured_n = int(
            (preg == 1).sum()
        )

    # Purely empirical signal.
    # This does not claim the official NHANES eligibility rule.
    if n_measured == 0:

        signal = "not_measured"

    elif n_measured < 0.40 * len(df):

        signal = "strong_subsample_or_eligibility_signal"

    elif n_measured < 0.75 * len(df):

        signal = "partial_measurement_signal"

    else:

        signal = "broad_measurement"

    base.update(
        {
            "n_measured": n_measured,
            "pct_native_measured": pct(
                n_measured,
                len(df),
            ),
            "measured_age_min": age_min,
            "measured_age_max": age_max,
            "measured_age_mean": age_mean,
            "measured_male_n": male_n,
            "measured_male_pct": male_pct,
            "measured_female_n": female_n,
            "measured_female_pct": female_pct,
            "pregnant_measured_n": pregnant_measured_n,
            "eligibility_signal": signal,
        }
    )

    return base


# =============================================================================
# 11. CANDIDATE VITAMIN D DEFINITIONS
# =============================================================================

def vitamin_d_descriptives(
    cycle: str,
    df: pd.DataFrame,
    vitd_var: Optional[str],
) -> List[Dict]:
    """
    Calculate candidate vitamin D outcome distributions.
    """

    records = []

    if vitd_var is None:
        return records

    value = safe_numeric(df[vitd_var])

    measured = value.notna()

    n = int(measured.sum())

    summary = measurement_summary(value)

    candidate_definitions = [
        (
            "vitd_lt_30_nmol_L",
            VITD_SEVERE_THRESHOLD_NMOL_L,
            "Candidate severe/deficiency threshold; not yet frozen",
        ),
        (
            "vitd_lt_50_nmol_L",
            VITD_INSUFFICIENCY_THRESHOLD_NMOL_L,
            "Candidate insufficiency threshold; not yet frozen",
        ),
    ]

    for name, threshold, note in candidate_definitions:

        cases = (
            measured
            & (value < threshold)
        )

        n_cases = int(cases.sum())

        records.append(
            {
                "cycle": cycle,
                "cycle_label": CYCLE_LABELS.get(cycle, cycle),
                "outcome_family": "vitamin_d",
                "candidate_definition": name,
                "biomarker_variable": vitd_var,
                "threshold": threshold,
                "assumed_unit_for_audit": "nmol/L",
                "n_measured": n,
                "n_cases": n_cases,
                "prevalence_pct_unweighted": pct(
                    n_cases,
                    n,
                ),
                "biomarker_mean": summary["mean"],
                "biomarker_sd": summary["sd"],
                "biomarker_median": summary["median"],
                "biomarker_q1": summary["q1"],
                "biomarker_q3": summary["q3"],
                "note": note,
            }
        )

    return records


# =============================================================================
# 12. CANDIDATE IRON DEFINITIONS
# =============================================================================

def classify_candidate_anemia(
    df: pd.DataFrame,
    hgb_var: str,
    sex_var: str,
) -> pd.Series:
    """
    Candidate sex-specific adult anemia classification.

    Returns nullable boolean-like series.
    """

    hgb = safe_numeric(df[hgb_var])
    sex = safe_numeric(df[sex_var])

    anemia = pd.Series(
        pd.NA,
        index=df.index,
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

    anemia.loc[male_valid] = (
        hgb.loc[male_valid]
        < ANEMIA_HGB_MALE_G_DL
    )

    anemia.loc[female_valid] = (
        hgb.loc[female_valid]
        < ANEMIA_HGB_FEMALE_G_DL
    )

    return anemia


def iron_descriptives(
    cycle: str,
    df: pd.DataFrame,
    ferritin_var: Optional[str],
    hgb_var: Optional[str],
) -> List[Dict]:
    """
    Calculate candidate raw ferritin and pre-anemic iron-deficiency
    definitions.
    """

    records = []

    if ferritin_var is None:
        return records

    ferritin = safe_numeric(df[ferritin_var])

    ferritin_measured = ferritin.notna()

    ferritin_summary = measurement_summary(
        ferritin
    )

    sex_var = resolve_preferred_variable(
        list(df.columns),
        "sex",
    )

    anemia = None

    if (
        hgb_var is not None
        and sex_var is not None
    ):

        anemia = classify_candidate_anemia(
            df=df,
            hgb_var=hgb_var,
            sex_var=sex_var,
        )

    thresholds = [
        (
            "ferritin_lt_15",
            FERRITIN_PRIMARY_UG_L,
            "Primary conservative candidate",
        ),
        (
            "ferritin_lt_25",
            FERRITIN_SENSITIVITY_UG_L,
            "Sensitivity candidate only",
        ),
    ]

    for name, threshold, note in thresholds:

        low_ferritin = (
            ferritin_measured
            & (ferritin < threshold)
        )

        n_low = int(low_ferritin.sum())

        records.append(
            {
                "cycle": cycle,
                "cycle_label": CYCLE_LABELS.get(cycle, cycle),
                "outcome_family": "iron",
                "candidate_definition": name,
                "biomarker_variable": ferritin_var,
                "threshold": threshold,
                "assumed_unit_for_audit": "µg/L",
                "n_measured": int(
                    ferritin_measured.sum()
                ),
                "n_cases": n_low,
                "prevalence_pct_unweighted": pct(
                    n_low,
                    ferritin_measured.sum(),
                ),
                "biomarker_mean": ferritin_summary["mean"],
                "biomarker_sd": ferritin_summary["sd"],
                "biomarker_median": ferritin_summary["median"],
                "biomarker_q1": ferritin_summary["q1"],
                "biomarker_q3": ferritin_summary["q3"],
                "note": note,
            }
        )

        # Pre-anemic version.
        if anemia is not None:

            eligible = (
                ferritin_measured
                & anemia.notna()
                & (~anemia.fillna(True))
            )

            cases = (
                eligible
                & (ferritin < threshold)
            )

            records.append(
                {
                    "cycle": cycle,
                    "cycle_label": CYCLE_LABELS.get(cycle, cycle),
                    "outcome_family": "iron",
                    "candidate_definition":
                        f"preanemic_{name}",
                    "biomarker_variable": ferritin_var,
                    "threshold": threshold,
                    "assumed_unit_for_audit": "µg/L",
                    "n_measured": int(
                        eligible.sum()
                    ),
                    "n_cases": int(
                        cases.sum()
                    ),
                    "prevalence_pct_unweighted": pct(
                        cases.sum(),
                        eligible.sum(),
                    ),
                    "biomarker_mean":
                        ferritin_summary["mean"],
                    "biomarker_sd":
                        ferritin_summary["sd"],
                    "biomarker_median":
                        ferritin_summary["median"],
                    "biomarker_q1":
                        ferritin_summary["q1"],
                    "biomarker_q3":
                        ferritin_summary["q3"],
                    "note":
                        note
                        + "; excludes candidate anemia cases",
                }
            )

    return records


# =============================================================================
# 13. MATCHED IRON POPULATION AUDIT
# =============================================================================

def matched_iron_population_audit(
    cycle: str,
    df: pd.DataFrame,
    ferritin_var: Optional[str],
    hgb_var: Optional[str],
) -> Dict:
    """
    Evaluate the proposed common ferritin population:

        Female
        age 18–49
        non-pregnant where pregnancy information is available

    This does NOT yet declare this as the final cohort.
    """

    record = {
        "cycle": cycle,
        "cycle_label": CYCLE_LABELS.get(cycle, cycle),
        "ferritin_variable": ferritin_var,
        "hemoglobin_variable": hgb_var,
    }

    if ferritin_var is None:

        record.update(
            {
                "n_ferritin_measured": 0,
                "n_candidate_matched_population": 0,
                "n_candidate_matched_with_ferritin": 0,
                "pct_matched_with_ferritin": np.nan,
                "ferritin_lt15_cases": 0,
                "preanemic_ferritin_lt15_cases": np.nan,
                "preanemic_ferritin_lt15_prevalence_pct":
                    np.nan,
            }
        )

        return record

    columns = list(df.columns)

    age_var = resolve_preferred_variable(
        columns,
        "age",
    )

    sex_var = resolve_preferred_variable(
        columns,
        "sex",
    )

    preg_var = resolve_preferred_variable(
        columns,
        "pregnancy",
    )

    if age_var is None or sex_var is None:

        record["audit_error"] = (
            "Age or sex variable unavailable"
        )

        return record

    age = safe_numeric(df[age_var])
    sex = safe_numeric(df[sex_var])
    ferritin = safe_numeric(df[ferritin_var])

    matched = (
        (age >= MATCHED_IRON_MIN_AGE)
        & (age <= MATCHED_IRON_MAX_AGE)
        & (sex == MATCHED_IRON_SEX_CODE)
    )

    # Exclude known pregnancy.
    if preg_var is not None:

        pregnancy = safe_numeric(
            df[preg_var]
        )

        matched = matched & (
            pregnancy.ne(1)
            | pregnancy.isna()
        )

    matched_with_ferritin = (
        matched
        & ferritin.notna()
    )

    low15 = (
        matched_with_ferritin
        & (ferritin < FERRITIN_PRIMARY_UG_L)
    )

    preanemic_cases = np.nan
    preanemic_n = np.nan
    preanemic_prev = np.nan

    if hgb_var is not None:

        anemia = classify_candidate_anemia(
            df=df,
            hgb_var=hgb_var,
            sex_var=sex_var,
        )

        preanemic_eligible = (
            matched_with_ferritin
            & anemia.notna()
            & (~anemia.fillna(True))
        )

        preanemic_case_mask = (
            preanemic_eligible
            & (
                ferritin
                < FERRITIN_PRIMARY_UG_L
            )
        )

        preanemic_cases = int(
            preanemic_case_mask.sum()
        )

        preanemic_n = int(
            preanemic_eligible.sum()
        )

        preanemic_prev = pct(
            preanemic_cases,
            preanemic_n,
        )

    record.update(
        {
            "age_variable": age_var,
            "sex_variable": sex_var,
            "pregnancy_variable": preg_var,

            "n_ferritin_measured":
                int(ferritin.notna().sum()),

            "n_candidate_matched_population":
                int(matched.sum()),

            "n_candidate_matched_with_ferritin":
                int(matched_with_ferritin.sum()),

            "pct_matched_with_ferritin":
                pct(
                    matched_with_ferritin.sum(),
                    matched.sum(),
                ),

            "ferritin_lt15_cases":
                int(low15.sum()),

            "preanemic_ferritin_lt15_n":
                preanemic_n,

            "preanemic_ferritin_lt15_cases":
                preanemic_cases,

            "preanemic_ferritin_lt15_prevalence_pct":
                preanemic_prev,
        }
    )

    return record


# =============================================================================
# 14. MEASUREMENT COMPARABILITY AUDIT
# =============================================================================

def build_comparability_record(
    cycle: str,
    availability: Dict,
) -> List[Dict]:
    """
    Create explicit cross-cycle documentation requirements.

    IMPORTANT:
    CSV values alone cannot establish assay equivalence.
    """

    records = []

    outcome_specs = [

        (
            "vitamin_d",
            availability["vitamin_d_available"],
            availability["vitamin_d_variable"],
        ),

        (
            "ferritin",
            availability["ferritin_available"],
            availability["ferritin_variable"],
        ),

        (
            "hemoglobin",
            availability["hemoglobin_available"],
            availability["hemoglobin_variable"],
        ),
    ]

    for (
        biomarker,
        available,
        variable,
    ) in outcome_specs:

        if not available:

            status = "not_available"

        else:

            status = "documentation_review_required"

        records.append(
            {
                "cycle": cycle,
                "cycle_label": CYCLE_LABELS.get(cycle, cycle),
                "biomarker": biomarker,
                "variable": variable,
                "available": available,
                "unit_confirmed_from_csv_only": False,
                "assay_method_confirmed": False,
                "laboratory_instrument_confirmed": False,
                "eligibility_rule_confirmed_from_documentation": False,
                "cross_cycle_comparability_status": status,
                "required_action":
                    (
                        "Review NHANES cycle-specific laboratory "
                        "documentation for assay, instrument, units, "
                        "lower detection limits, and eligibility."
                        if available
                        else
                        "No biomarker variable selected in joined data."
                    ),
            }
        )

    return records


# =============================================================================
# 15. CANDIDATE DEFINITIONS TABLE
# =============================================================================

def candidate_definition_table() -> pd.DataFrame:
    """
    Record all current candidate definitions explicitly.
    """

    rows = [

        {
            "outcome_family": "vitamin_d",
            "definition_name": "vitd_lt_30_nmol_L",
            "definition":
                "25-hydroxyvitamin D < 30 nmol/L",
            "role":
                "candidate sensitivity/severity definition",
            "status":
                "NOT FROZEN",
            "requires_confirmation":
                "Confirm biomarker units and assay harmonization "
                "across cycles.",
        },

        {
            "outcome_family": "vitamin_d",
            "definition_name": "vitd_lt_50_nmol_L",
            "definition":
                "25-hydroxyvitamin D < 50 nmol/L",
            "role":
                "candidate insufficiency screening definition",
            "status":
                "NOT FROZEN",
            "requires_confirmation":
                "Confirm target clinical framing and assay "
                "harmonization across cycles.",
        },

        {
            "outcome_family": "iron",
            "definition_name": "ferritin_lt_15",
            "definition":
                "Ferritin < 15 µg/L",
            "role":
                "primary conservative candidate",
            "status":
                "CANDIDATE PRIMARY — NOT YET FROZEN",
            "requires_confirmation":
                "Confirm ferritin units, assay comparability, "
                "eligibility and inflammation strategy.",
        },

        {
            "outcome_family": "iron",
            "definition_name": "preanemic_ferritin_lt_15",
            "definition":
                "Ferritin <15 µg/L among participants not meeting "
                "candidate sex-specific anemia thresholds",
            "role":
                "primary pre-anemic iron candidate",
            "status":
                "CANDIDATE PRIMARY — NOT YET FROZEN",
            "requires_confirmation":
                "Confirm hemoglobin/anemia thresholds, ferritin "
                "eligibility and inflammation handling.",
        },

        {
            "outcome_family": "iron",
            "definition_name": "preanemic_ferritin_lt_25",
            "definition":
                "Ferritin <25 µg/L among participants not meeting "
                "candidate sex-specific anemia thresholds",
            "role":
                "sensitivity analysis candidate",
            "status":
                "NOT PRIMARY",
            "requires_confirmation":
                "Use only if scientifically justified.",
        },

        {
            "outcome_family": "iron",
            "definition_name": "matched_transport_population",
            "definition":
                "Females age 18–49, excluding known pregnancy, "
                "with ferritin measured",
            "role":
                "candidate population-matched temporal sensitivity",
            "status":
                "NOT FROZEN",
            "requires_confirmation":
                "Compare against official ferritin eligibility "
                "rules for I/J/L and earlier cycles.",
        },
    ]

    return pd.DataFrame(rows)


# =============================================================================
# 16. DECISION LEDGER
# =============================================================================

def initialize_decision_ledger() -> pd.DataFrame:
    """
    Build first permanent decision ledger entries.
    """

    today = datetime.now().date().isoformat()

    rows = [

        {
            "decision_id": "D001",
            "stage": "01",
            "topic": "Raw data integrity",
            "decision":
                "Raw NHANES CSV files remain immutable.",
            "reason":
                "All processing outputs must be reproducible "
                "from untouched source data.",
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D002",
            "stage": "01",
            "topic": "Participant joins",
            "decision":
                "Direct native joins restricted to one-row-per-SEQN "
                "files.",
            "reason":
                "Prevent Cartesian row multiplication from item-level "
                "files such as DR1IFF, DR2IFF, DSQIDS and RXQ_RX.",
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D003",
            "stage": "01",
            "topic": "Pre-pandemic release",
            "decision":
                "2017–March 2020 pre-pandemic release remains "
                "supplementary and separate from 2017–2018.",
            "reason":
                "Avoid overlap/double counting until release "
                "relationship is formally resolved.",
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D004",
            "stage": "02A",
            "topic": "Ferritin eligibility",
            "decision":
                "Do not interpret ferritin missingness as ordinary "
                "random missingness.",
            "reason":
                "Stage 01 demonstrated substantial cycle-specific "
                "ferritin ascertainment differences.",
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D005",
            "stage": "02A",
            "topic": "Iron temporal analysis",
            "decision":
                "Audit both observed eligible cohorts and a common "
                "population-matched ferritin cohort before modeling.",
            "reason":
                "Separate temporal model degradation from outcome "
                "eligibility/population shift.",
            "status": "PROVISIONAL",
            "date": today,
        },

        {
            "decision_id": "D006",
            "stage": "02A",
            "topic": "Outcome definitions",
            "decision":
                "Candidate thresholds may be calculated for audit, "
                "but final Vitamin D and iron labels are not frozen "
                "until measurement and eligibility review is complete.",
            "reason":
                "Avoid outcome-definition leakage and post-hoc "
                "selection based on model performance.",
            "status": "FROZEN",
            "date": today,
        },

        {
            "decision_id": "D007",
            "stage": "02A",
            "topic": "Laboratory comparability",
            "decision":
                "Assay or instrument changes will not be inferred "
                "from biomarker distributions alone.",
            "reason":
                "Cycle-specific NHANES documentation is required to "
                "establish measurement-method changes.",
            "status": "FROZEN",
            "date": today,
        },
    ]

    return pd.DataFrame(rows)


# =============================================================================
# 17. MAIN
# =============================================================================

def main():

    logger.info("")
    logger.info("=" * 80)
    logger.info("STAGE 02A STARTED")
    logger.info("=" * 80)

    lineage = load_lineage()

    candidate_variable_records = []
    availability_records = []
    eligibility_records = []
    comparability_records = []
    descriptive_records = []
    matched_iron_records = []

    loaded_cycles = []

    for cycle in ALL_CYCLES:

        path = cycle_file_path(cycle)

        if path is None:

            logger.warning(
                f"Joined dataset not found for {cycle}"
            )

            continue

        logger.info("")
        logger.info("-" * 70)
        logger.info(
            f"Loading {cycle}: {path.name}"
        )
        logger.info("-" * 70)

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        loaded_cycles.append(cycle)

        logger.info(
            f"Shape: {df.shape[0]:,} rows × "
            f"{df.shape[1]:,} columns"
        )

        # ---------------------------------------------------------
        # Variable candidates
        # ---------------------------------------------------------

        candidate_variable_records.extend(
            build_candidate_variable_audit(
                cycle=cycle,
                df=df,
                lineage=lineage,
            )
        )

        # ---------------------------------------------------------
        # Availability
        # ---------------------------------------------------------

        availability = build_availability_record(
            cycle=cycle,
            df=df,
        )

        availability_records.append(
            availability
        )

        # ---------------------------------------------------------
        # Eligibility / ascertainment
        # ---------------------------------------------------------

        eligibility_records.append(
            biomarker_eligibility_record(
                cycle=cycle,
                df=df,
                biomarker_name="vitamin_d",
                biomarker_var=
                    availability[
                        "vitamin_d_variable"
                    ],
            )
        )

        eligibility_records.append(
            biomarker_eligibility_record(
                cycle=cycle,
                df=df,
                biomarker_name="ferritin",
                biomarker_var=
                    availability[
                        "ferritin_variable"
                    ],
            )
        )

        # ---------------------------------------------------------
        # Candidate vitamin D definitions
        # ---------------------------------------------------------

        descriptive_records.extend(
            vitamin_d_descriptives(
                cycle=cycle,
                df=df,
                vitd_var=
                    availability[
                        "vitamin_d_variable"
                    ],
            )
        )

        # ---------------------------------------------------------
        # Candidate iron definitions
        # ---------------------------------------------------------

        descriptive_records.extend(
            iron_descriptives(
                cycle=cycle,
                df=df,
                ferritin_var=
                    availability[
                        "ferritin_variable"
                    ],
                hgb_var=
                    availability[
                        "hemoglobin_variable"
                    ],
            )
        )

        # ---------------------------------------------------------
        # Matched iron population
        # ---------------------------------------------------------

        matched_iron_records.append(
            matched_iron_population_audit(
                cycle=cycle,
                df=df,
                ferritin_var=
                    availability[
                        "ferritin_variable"
                    ],
                hgb_var=
                    availability[
                        "hemoglobin_variable"
                    ],
            )
        )

        # ---------------------------------------------------------
        # Comparability documentation flags
        # ---------------------------------------------------------

        comparability_records.extend(
            build_comparability_record(
                cycle=cycle,
                availability=availability,
            )
        )

    # =========================================================================
    # OUTPUT TABLES
    # =========================================================================

    candidate_vars = pd.DataFrame(
        candidate_variable_records
    )

    availability_df = pd.DataFrame(
        availability_records
    )

    eligibility_df = pd.DataFrame(
        eligibility_records
    )

    comparability_df = pd.DataFrame(
        comparability_records
    )

    descriptives_df = pd.DataFrame(
        descriptive_records
    )

    matched_iron_df = pd.DataFrame(
        matched_iron_records
    )

    definitions_df = candidate_definition_table()

    decision_ledger = initialize_decision_ledger()

    # -------------------------------------------------------------
    # Save CSVs
    # -------------------------------------------------------------

    candidate_vars.to_csv(
        OUTPUT_DIR
        / "02A_outcome_variable_candidates.csv",
        index=False,
    )

    availability_df.to_csv(
        OUTPUT_DIR
        / "02A_outcome_availability_matrix.csv",
        index=False,
    )

    eligibility_df.to_csv(
        OUTPUT_DIR
        / "02A_outcome_eligibility_audit.csv",
        index=False,
    )

    comparability_df.to_csv(
        OUTPUT_DIR
        / "02A_lab_measurement_comparability.csv",
        index=False,
    )

    definitions_df.to_csv(
        OUTPUT_DIR
        / "02A_candidate_outcome_definitions.csv",
        index=False,
    )

    descriptives_df.to_csv(
        OUTPUT_DIR
        / "02A_cycle_outcome_descriptives.csv",
        index=False,
    )

    matched_iron_df.to_csv(
        OUTPUT_DIR
        / "02A_matched_iron_population_audit.csv",
        index=False,
    )

    decision_ledger.to_csv(
        OUTPUT_DIR
        / "decision_ledger.csv",
        index=False,
    )

    # =========================================================================
    # EXCEL SUMMARY
    # =========================================================================

    excel_path = (
        OUTPUT_DIR
        / "02A_outcome_audit_summary.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:

        availability_df.to_excel(
            writer,
            sheet_name="Availability",
            index=False,
        )

        eligibility_df.to_excel(
            writer,
            sheet_name="Eligibility",
            index=False,
        )

        descriptives_df.to_excel(
            writer,
            sheet_name="Outcome_Descriptives",
            index=False,
        )

        matched_iron_df.to_excel(
            writer,
            sheet_name="Matched_Iron",
            index=False,
        )

        candidate_vars.to_excel(
            writer,
            sheet_name="Variable_Candidates",
            index=False,
        )

        comparability_df.to_excel(
            writer,
            sheet_name="Comparability",
            index=False,
        )

        definitions_df.to_excel(
            writer,
            sheet_name="Definitions",
            index=False,
        )

        decision_ledger.to_excel(
            writer,
            sheet_name="Decision_Ledger",
            index=False,
        )

    # =========================================================================
    # CONSOLE SUMMARY
    # =========================================================================

    print("\n")
    print("=" * 78)
    print("STAGE 02A COMPLETE")
    print("=" * 78)

    print(
        f"\nCycles processed: "
        f"{', '.join(loaded_cycles)}"
    )

    print(
        "\nOutputs saved to:\n"
        f"{OUTPUT_DIR}"
    )

    if not availability_df.empty:

        print("\nOUTCOME AVAILABILITY")
        print("-" * 78)

        display_cols = [
            "cycle",
            "vitamin_d_available",
            "vitamin_d_variable",
            "ferritin_available",
            "ferritin_variable",
            "hemoglobin_available",
            "hemoglobin_variable",
        ]

        existing = [
            c
            for c in display_cols
            if c in availability_df.columns
        ]

        print(
            availability_df[
                existing
            ].to_string(index=False)
        )

    if not eligibility_df.empty:

        print("\nBIOMARKER ASCERTAINMENT")
        print("-" * 78)

        display_cols = [
            "cycle",
            "biomarker",
            "n_measured",
            "pct_native_measured",
            "measured_age_min",
            "measured_age_max",
            "measured_male_pct",
            "measured_female_pct",
            "eligibility_signal",
        ]

        existing = [
            c
            for c in display_cols
            if c in eligibility_df.columns
        ]

        print(
            eligibility_df[
                existing
            ].round(2).to_string(
                index=False
            )
        )

    print("\n")
    print("IMPORTANT:")
    print(
        "Stage 02A has NOT frozen either outcome definition."
    )

    print(
        "Do not begin modeling until:"
    )

    print(
        "  1. Ferritin eligibility differences are reviewed."
    )

    print(
        "  2. Vitamin D assay/unit comparability is confirmed."
    )

    print(
        "  3. Ferritin assay/unit comparability is confirmed."
    )

    print(
        "  4. The matched iron cohort is inspected."
    )

    print(
        "  5. Final outcome thresholds are explicitly frozen."
    )

    print(
        "\nPrimary file to send for review:"
    )

    print(
        "02A_outcome_audit_summary.xlsx"
    )

    print("\n" + "=" * 78)


# =============================================================================
# 18. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()