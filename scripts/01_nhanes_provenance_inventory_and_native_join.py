"""
======================================================================
STAGE 01
NHANES Provenance, Inventory, Cardinality Audit, and Native Cycle Joins
======================================================================


PURPOSE
-------
This script performs the first reproducible data-engineering stage.

It:

1. Scans the raw NHANES repository.
2. Inventories every CSV file.
3. Audits SEQN availability and uniqueness.
4. Classifies files as:
      - participant_level
      - item_level
      - no_seqn
      - empty
5. Preserves all raw files unchanged.
6. Creates cycle-specific "native joined" datasets using DEMO as the
   participant anchor.
7. Joins ONLY one-row-per-SEQN files.
8. Leaves multi-row files such as DR1IFF, DR2IFF, DSQIDS, and RXQ_RX
   outside the participant-level join.
9. Detects variable-name collisions.
10. Creates complete data-lineage and join-audit outputs.


Author: Project pipeline
======================================================================
"""

from __future__ import annotations

import hashlib
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ======================================================================
# 1. CONFIGURATION
# ======================================================================

# If this script is stored in:
#
# Nutrient_NHANES_EXP/Scripts/
#
# the following automatically identifies Nutrient_NHANES_EXP as root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


RAW_DOMAINS = [
    "Demographics",
    "Dietary",
    "Examination",
    "Laboratory",
    "Questionnaire",
]


# Principal AISC temporal cohorts.
MAIN_CYCLES = [
    "2009_2010",
    "2011_2012",
    "2013_2014",
    "2015_2016",
    "2017_2018",
    "2021_2023",
]


# Keep the overlapping pre-pandemic release separate for now.
SUPPLEMENTARY_CYCLES = [
    "2017_2020_PrePandemic",
]


ALL_JOIN_CYCLES = MAIN_CYCLES + SUPPLEMENTARY_CYCLES


# Known multi-row / relational NHANES file families.
#
# Even if a particular downloaded file happens not to contain duplicate
# participants, these should NOT be blindly merged into a one-row-per-person
# analytic table.
KNOWN_ITEM_LEVEL_PREFIXES = {
    "DR1IFF",
    "DR2IFF",
    "DSQIDS",
    "RXQ_RX",
}


# CSV chunk size for large files.
CHUNK_SIZE = 100_000


# Hashing gives us reproducible raw-file provenance.
COMPUTE_SHA256 = True


# ======================================================================
# 2. OUTPUT DIRECTORIES
# ======================================================================

PROCESSED_ROOT = PROJECT_ROOT / "processed"

INVENTORY_DIR = PROCESSED_ROOT / "01_inventory"

JOINED_DIR = PROCESSED_ROOT / "02_joined_native"

MAIN_JOINED_DIR = JOINED_DIR / "main_cycles"

SUPPLEMENTARY_JOINED_DIR = JOINED_DIR / "supplementary"

LOG_DIR = PROCESSED_ROOT / "logs"


for directory in [
    INVENTORY_DIR,
    MAIN_JOINED_DIR,
    SUPPLEMENTARY_JOINED_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 3. LOGGING
# ======================================================================

LOG_FILE = LOG_DIR / "stage01_native_join.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)


logger = logging.getLogger(__name__)


# ======================================================================
# 4. HELPER FUNCTIONS
# ======================================================================

def sha256_file(path: Path) -> str:
    """
    Calculate SHA256 hash for file provenance.
    """
    hash_obj = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hash_obj.update(block)

    return hash_obj.hexdigest()


def safe_read_header(path: Path) -> List[str]:
    """
    Read only CSV column names.
    """
    try:
        return pd.read_csv(
            path,
            nrows=0,
            low_memory=False,
        ).columns.tolist()

    except Exception as exc:
        logger.error(f"HEADER READ FAILED: {path} | {exc}")
        return []


def normalize_seqn(series: pd.Series) -> pd.Series:
    """
    Standardize SEQN for safe joining.

    NHANES SEQN should represent integer participant identifiers.

    Numeric coercion is used because CSV conversion can sometimes
    represent SEQN as float (e.g., 12345.0).
    """
    numeric = pd.to_numeric(series, errors="coerce")

    return numeric.astype("Int64")


def known_item_level(filename_stem: str) -> bool:
    """
    Determine whether a file belongs to a known multi-row file family.
    """
    clean_stem = filename_stem.upper()

    for prefix in KNOWN_ITEM_LEVEL_PREFIXES:
        if clean_stem.startswith(prefix):
            return True

    return False


def inspect_seqn(path: Path) -> Dict:
    """
    Inspect SEQN cardinality without loading all columns.

    Returns:
        row_count
        nonmissing_seqn
        unique_seqn
        duplicate_seqn_rows
        missing_seqn_rows
    """
    row_count = 0
    nonmissing_seqn = 0
    seqn_values = []

    try:
        for chunk in pd.read_csv(
            path,
            usecols=["SEQN"],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):

            row_count += len(chunk)

            seqn = normalize_seqn(chunk["SEQN"])

            nonmissing_seqn += seqn.notna().sum()

            seqn_values.append(
                seqn.dropna().astype("int64")
            )

        if not seqn_values:
            return {
                "row_count": row_count,
                "nonmissing_seqn": 0,
                "unique_seqn": 0,
                "duplicate_seqn_rows": 0,
                "missing_seqn_rows": row_count,
            }

        all_seqn = pd.concat(seqn_values, ignore_index=True)

        unique_seqn = all_seqn.nunique()

        duplicate_seqn_rows = len(all_seqn) - unique_seqn

        missing_seqn_rows = row_count - nonmissing_seqn

        return {
            "row_count": row_count,
            "nonmissing_seqn": int(nonmissing_seqn),
            "unique_seqn": int(unique_seqn),
            "duplicate_seqn_rows": int(duplicate_seqn_rows),
            "missing_seqn_rows": int(missing_seqn_rows),
        }

    except Exception as exc:
        logger.error(f"SEQN INSPECTION FAILED: {path} | {exc}")

        return {
            "row_count": np.nan,
            "nonmissing_seqn": np.nan,
            "unique_seqn": np.nan,
            "duplicate_seqn_rows": np.nan,
            "missing_seqn_rows": np.nan,
        }


def classify_file(
    filename_stem: str,
    row_count,
    has_seqn: bool,
    duplicate_seqn_rows,
) -> str:
    """
    Classify file join cardinality.
    """

    if row_count == 0:
        return "empty"

    if not has_seqn:
        return "no_seqn"

    if known_item_level(filename_stem):
        return "item_level_known"

    if pd.notna(duplicate_seqn_rows) and duplicate_seqn_rows > 0:
        return "item_level_detected"

    return "participant_level"


def get_relative_path(path: Path) -> str:
    """
    Return repository-relative path when possible.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))

    except ValueError:
        return str(path)


# ======================================================================
# 5. BUILD COMPLETE RAW FILE INVENTORY
# ======================================================================

def build_file_inventory() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan all raw-domain cycle folders.

    Returns:
        file_inventory
        variable_inventory
    """

    logger.info("=" * 80)
    logger.info("BUILDING RAW FILE INVENTORY")
    logger.info("=" * 80)

    file_records = []
    variable_records = []

    for domain in RAW_DOMAINS:

        domain_dir = PROJECT_ROOT / domain

        if not domain_dir.exists():
            logger.warning(f"DOMAIN NOT FOUND: {domain_dir}")
            continue

        logger.info(f"Scanning domain: {domain}")

        for cycle_dir in sorted(domain_dir.iterdir()):

            if not cycle_dir.is_dir():
                continue

            cycle = cycle_dir.name

            csv_files = sorted(cycle_dir.glob("*.csv"))

            if not csv_files:
                logger.warning(
                    f"No CSV files found: {domain}/{cycle}"
                )
                continue

            for csv_path in csv_files:

                logger.info(
                    f"Inspecting: {domain}/{cycle}/{csv_path.name}"
                )

                columns = safe_read_header(csv_path)

                has_seqn = "SEQN" in columns

                file_size_mb = (
                    csv_path.stat().st_size / (1024 ** 2)
                )

                if has_seqn:
                    seqn_info = inspect_seqn(csv_path)

                else:
                    # Count rows cheaply.
                    try:
                        row_count = sum(
                            len(chunk)
                            for chunk in pd.read_csv(
                                csv_path,
                                chunksize=CHUNK_SIZE,
                                low_memory=False,
                            )
                        )

                    except Exception as exc:
                        logger.error(
                            f"ROW COUNT FAILED: {csv_path} | {exc}"
                        )
                        row_count = np.nan

                    seqn_info = {
                        "row_count": row_count,
                        "nonmissing_seqn": np.nan,
                        "unique_seqn": np.nan,
                        "duplicate_seqn_rows": np.nan,
                        "missing_seqn_rows": np.nan,
                    }

                classification = classify_file(
                    filename_stem=csv_path.stem,
                    row_count=seqn_info["row_count"],
                    has_seqn=has_seqn,
                    duplicate_seqn_rows=seqn_info[
                        "duplicate_seqn_rows"
                    ],
                )

                if COMPUTE_SHA256:
                    file_hash = sha256_file(csv_path)
                else:
                    file_hash = None

                file_record = {
                    "domain": domain,
                    "cycle": cycle,
                    "filename": csv_path.name,
                    "file_stem": csv_path.stem,
                    "relative_path": get_relative_path(csv_path),
                    "file_size_mb": round(file_size_mb, 3),
                    "n_rows": seqn_info["row_count"],
                    "n_columns": len(columns),
                    "has_seqn": has_seqn,
                    "nonmissing_seqn": seqn_info[
                        "nonmissing_seqn"
                    ],
                    "unique_seqn": seqn_info[
                        "unique_seqn"
                    ],
                    "duplicate_seqn_rows": seqn_info[
                        "duplicate_seqn_rows"
                    ],
                    "missing_seqn_rows": seqn_info[
                        "missing_seqn_rows"
                    ],
                    "known_item_level": known_item_level(
                        csv_path.stem
                    ),
                    "classification": classification,
                    "sha256": file_hash,
                }

                file_records.append(file_record)

                for col in columns:

                    variable_records.append(
                        {
                            "domain": domain,
                            "cycle": cycle,
                            "filename": csv_path.name,
                            "file_stem": csv_path.stem,
                            "variable": col,
                            "is_seqn": col == "SEQN",
                            "file_classification": classification,
                        }
                    )

    file_inventory = pd.DataFrame(file_records)

    variable_inventory = pd.DataFrame(variable_records)

    return file_inventory, variable_inventory


# ======================================================================
# 6. FIND DEMOGRAPHIC ANCHOR
# ======================================================================

def find_demo_file(
    file_inventory: pd.DataFrame,
    cycle: str,
) -> Path:
    """
    Identify the demographic participant-level anchor file.
    """

    subset = file_inventory[
        (file_inventory["cycle"] == cycle)
        & (file_inventory["domain"] == "Demographics")
        & (
            file_inventory["classification"]
            == "participant_level"
        )
    ].copy()

    # DEMO files only.
    subset = subset[
        subset["file_stem"]
        .str.upper()
        .str.contains("DEMO")
    ]

    if len(subset) == 0:
        raise RuntimeError(
            f"No participant-level DEMO file found for {cycle}"
        )

    if len(subset) > 1:
        logger.warning(
            f"Multiple DEMO candidates for {cycle}: "
            f"{subset['filename'].tolist()}"
        )

    chosen = subset.iloc[0]

    return PROJECT_ROOT / chosen["relative_path"]


# ======================================================================
# 7. HANDLE COLUMN COLLISIONS
# ======================================================================

def rename_collisions(
    incoming: pd.DataFrame,
    existing_columns: set,
    source_stem: str,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Preserve native NHANES names whenever possible.

    If an incoming variable already exists in the joined dataset,
    rename ONLY the incoming version:

        VARIABLE
            ->
        VARIABLE__SOURCEFILE

    This prevents silent overwriting while preserving provenance.
    """

    collision_records = []

    rename_map = {}

    for col in incoming.columns:

        if col == "SEQN":
            continue

        if col in existing_columns:

            new_name = f"{col}__{source_stem}"

            rename_map[col] = new_name

            collision_records.append(
                {
                    "source_file": source_stem,
                    "original_variable": col,
                    "renamed_variable": new_name,
                    "reason": "column_name_collision",
                }
            )

    if rename_map:
        incoming = incoming.rename(columns=rename_map)

    return incoming, collision_records


# ======================================================================
# 8. LOAD PARTICIPANT-LEVEL FILE
# ======================================================================

def load_participant_file(path: Path) -> pd.DataFrame:
    """
    Load one-row-per-SEQN file and enforce SEQN type.
    """

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if "SEQN" not in df.columns:
        raise RuntimeError(
            f"Participant file unexpectedly lacks SEQN: {path}"
        )

    df["SEQN"] = normalize_seqn(df["SEQN"])

    # Remove rows without an identifiable participant.
    missing_seqn = df["SEQN"].isna().sum()

    if missing_seqn:
        logger.warning(
            f"{path.name}: dropping {missing_seqn} rows "
            f"with missing SEQN."
        )

        df = df[df["SEQN"].notna()].copy()

    if df["SEQN"].duplicated().any():
        raise RuntimeError(
            f"Unsafe join stopped: duplicate SEQN found in "
            f"participant-level file {path}"
        )

    return df


# ======================================================================
# 9. JOIN ONE CYCLE
# ======================================================================

def join_cycle(
    cycle: str,
    file_inventory: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Safely build one-row-per-participant native cycle table.

    DEMO is the anchor.

    All other one-row-per-SEQN files are left-joined.
    """

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"JOINING CYCLE: {cycle}")
    logger.info("=" * 80)

    demo_path = find_demo_file(
        file_inventory=file_inventory,
        cycle=cycle,
    )

    logger.info(f"DEMO anchor: {demo_path.name}")

    joined = load_participant_file(demo_path)

    starting_n = len(joined)

    logger.info(
        f"DEMO participants: {starting_n:,}"
    )

    cycle_files = file_inventory[
        (file_inventory["cycle"] == cycle)
        & (
            file_inventory["classification"]
            == "participant_level"
        )
    ].copy()

    # Do not rejoin DEMO anchor.
    cycle_files = cycle_files[
        cycle_files["relative_path"]
        != get_relative_path(demo_path)
    ]

    # Stable order:
    # Demographics -> Examination -> Questionnaire ->
    # Dietary -> Laboratory
    domain_order = {
        "Demographics": 0,
        "Examination": 1,
        "Questionnaire": 2,
        "Dietary": 3,
        "Laboratory": 4,
    }

    cycle_files["domain_order"] = (
        cycle_files["domain"]
        .map(domain_order)
        .fillna(99)
    )

    cycle_files = cycle_files.sort_values(
        ["domain_order", "filename"]
    )

    join_records = []

    collision_records = []

    lineage_records = []

    # Record demographic anchor lineage.
    for col in joined.columns:

        if col == "SEQN":
            continue

        lineage_records.append(
            {
                "cycle": cycle,
                "final_variable": col,
                "original_variable": col,
                "source_domain": "Demographics",
                "source_file": demo_path.name,
                "source_path": get_relative_path(demo_path),
                "join_role": "DEMO_anchor",
            }
        )

    for _, row in cycle_files.iterrows():

        file_path = PROJECT_ROOT / row["relative_path"]

        source_stem = row["file_stem"]

        logger.info(
            f"Joining: {row['domain']} | {file_path.name}"
        )

        incoming = load_participant_file(file_path)

        incoming_n = len(incoming)

        incoming_columns_original = [
            c for c in incoming.columns if c != "SEQN"
        ]

        incoming, collisions = rename_collisions(
            incoming=incoming,
            existing_columns=set(joined.columns),
            source_stem=source_stem,
        )

        for collision in collisions:
            collision["cycle"] = cycle
            collision["domain"] = row["domain"]
            collision["filename"] = row["filename"]

        collision_records.extend(collisions)

        # Map original -> final column name for lineage.
        collision_lookup = {
            x["original_variable"]: x["renamed_variable"]
            for x in collisions
        }

        matched_before = joined["SEQN"].isin(
            incoming["SEQN"]
        ).sum()

        columns_before = joined.shape[1]

        joined = joined.merge(
            incoming,
            on="SEQN",
            how="left",
            validate="one_to_one",
        )

        columns_after = joined.shape[1]

        # A left join on DEMO must NEVER change participant count.
        if len(joined) != starting_n:

            raise RuntimeError(
                f"Participant count changed during {cycle} join.\n"
                f"Expected: {starting_n}\n"
                f"Observed: {len(joined)}\n"
                f"Problem file: {file_path}"
            )

        match_rate = (
            matched_before / starting_n
            if starting_n > 0
            else np.nan
        )

        join_records.append(
            {
                "cycle": cycle,
                "domain": row["domain"],
                "filename": row["filename"],
                "participants_in_source": incoming_n,
                "demo_anchor_participants": starting_n,
                "matched_demo_participants": int(
                    matched_before
                ),
                "match_rate": match_rate,
                "columns_added": (
                    columns_after - columns_before
                ),
                "participant_count_after_join": len(joined),
            }
        )

        for original_col in incoming_columns_original:

            final_col = collision_lookup.get(
                original_col,
                original_col,
            )

            lineage_records.append(
                {
                    "cycle": cycle,
                    "final_variable": final_col,
                    "original_variable": original_col,
                    "source_domain": row["domain"],
                    "source_file": row["filename"],
                    "source_path": row["relative_path"],
                    "join_role": "left_join_participant_level",
                }
            )

    # Final integrity test.
    if joined["SEQN"].duplicated().any():

        raise RuntimeError(
            f"Final joined dataset contains duplicate "
            f"SEQN values for {cycle}"
        )

    if len(joined) != starting_n:

        raise RuntimeError(
            f"Final participant count differs from DEMO "
            f"for {cycle}"
        )

    join_audit = pd.DataFrame(join_records)

    collision_audit = pd.DataFrame(collision_records)

    lineage = pd.DataFrame(lineage_records)

    return joined, join_audit, collision_audit, lineage


# ======================================================================
# 10. CREATE SKIPPED-FILE AUDIT
# ======================================================================

def build_skipped_file_audit(
    file_inventory: pd.DataFrame,
) -> pd.DataFrame:
    """
    Record files intentionally excluded from native participant joins.
    """

    subset = file_inventory[
        file_inventory["cycle"].isin(
            ALL_JOIN_CYCLES
        )
        & (
            file_inventory["classification"]
            != "participant_level"
        )
    ].copy()

    reasons = {
        "item_level_known":
            "Known multi-row/item-level table; requires aggregation "
            "or relational handling before participant-level use.",

        "item_level_detected":
            "Duplicate SEQN values detected; unsafe for direct "
            "one-to-one participant join.",

        "no_seqn":
            "SEQN not present; cannot join directly to participant "
            "table.",

        "empty":
            "File contains no observations.",
    }

    subset["skip_reason"] = (
        subset["classification"].map(reasons)
    )

    return subset


# ======================================================================
# 11. RUN COMPLETE STAGE
# ======================================================================

def main():

    logger.info("")
    logger.info("#" * 80)
    logger.info("NHANES STAGE 01 STARTED")
    logger.info("#" * 80)

    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(
        f"Started: {datetime.now().isoformat(timespec='seconds')}"
    )

    # --------------------------------------------------------------
    # Verify expected raw domains.
    # --------------------------------------------------------------

    missing_domains = [
        domain
        for domain in RAW_DOMAINS
        if not (PROJECT_ROOT / domain).exists()
    ]

    if missing_domains:

        raise FileNotFoundError(
            "Missing expected raw-data folders:\n"
            + "\n".join(missing_domains)
        )

    # --------------------------------------------------------------
    # Inventory
    # --------------------------------------------------------------

    file_inventory, variable_inventory = (
        build_file_inventory()
    )

    file_inventory.to_csv(
        INVENTORY_DIR / "01_file_inventory.csv",
        index=False,
    )

    variable_inventory.to_csv(
        INVENTORY_DIR / "01_variable_inventory.csv",
        index=False,
    )

    logger.info(
        f"Files inventoried: {len(file_inventory):,}"
    )

    logger.info(
        f"Variable-file records: "
        f"{len(variable_inventory):,}"
    )

    # --------------------------------------------------------------
    # Cardinality summary
    # --------------------------------------------------------------

    cardinality_summary = (
        file_inventory
        .groupby(
            ["domain", "cycle", "classification"],
            dropna=False,
        )
        .size()
        .reset_index(name="n_files")
    )

    cardinality_summary.to_csv(
        INVENTORY_DIR
        / "01_cardinality_summary.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Skipped files
    # --------------------------------------------------------------

    skipped = build_skipped_file_audit(
        file_inventory
    )

    skipped.to_csv(
        INVENTORY_DIR
        / "01_files_excluded_from_native_join.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Join each selected cycle independently.
    # --------------------------------------------------------------

    all_join_audits = []

    all_collision_audits = []

    all_lineage = []

    cycle_summary_records = []

    for cycle in ALL_JOIN_CYCLES:

        cycle_exists = (
            file_inventory["cycle"] == cycle
        ).any()

        if not cycle_exists:

            logger.warning(
                f"Requested cycle not found: {cycle}"
            )
            continue

        try:

            (
                joined,
                join_audit,
                collision_audit,
                lineage,
            ) = join_cycle(
                cycle=cycle,
                file_inventory=file_inventory,
            )

            if cycle in MAIN_CYCLES:

                output_dir = MAIN_JOINED_DIR

            else:

                output_dir = SUPPLEMENTARY_JOINED_DIR

            output_path = (
                output_dir
                / f"NHANES_{cycle}_joined_native.csv"
            )

            joined.to_csv(
                output_path,
                index=False,
            )

            logger.info(
                f"SAVED: {output_path}"
            )

            logger.info(
                f"Final shape: "
                f"{joined.shape[0]:,} rows × "
                f"{joined.shape[1]:,} columns"
            )

            if not join_audit.empty:
                all_join_audits.append(join_audit)

            if not collision_audit.empty:
                all_collision_audits.append(
                    collision_audit
                )

            if not lineage.empty:
                all_lineage.append(lineage)

            cycle_summary_records.append(
                {
                    "cycle": cycle,
                    "dataset_type": (
                        "main"
                        if cycle in MAIN_CYCLES
                        else "supplementary"
                    ),
                    "output_filename": output_path.name,
                    "n_participants": joined.shape[0],
                    "n_columns": joined.shape[1],
                    "unique_seqn": joined[
                        "SEQN"
                    ].nunique(),
                    "duplicate_seqn": joined[
                        "SEQN"
                    ].duplicated().sum(),
                }
            )

        except Exception as exc:

            logger.exception(
                f"CYCLE JOIN FAILED: {cycle} | {exc}"
            )

            cycle_summary_records.append(
                {
                    "cycle": cycle,
                    "dataset_type": (
                        "main"
                        if cycle in MAIN_CYCLES
                        else "supplementary"
                    ),
                    "output_filename": None,
                    "n_participants": np.nan,
                    "n_columns": np.nan,
                    "unique_seqn": np.nan,
                    "duplicate_seqn": np.nan,
                    "error": str(exc),
                }
            )

    # --------------------------------------------------------------
    # Combine audit outputs.
    # --------------------------------------------------------------

    if all_join_audits:

        join_audit_final = pd.concat(
            all_join_audits,
            ignore_index=True,
        )

    else:

        join_audit_final = pd.DataFrame()

    join_audit_final.to_csv(
        INVENTORY_DIR / "01_join_audit.csv",
        index=False,
    )

    if all_collision_audits:

        collision_audit_final = pd.concat(
            all_collision_audits,
            ignore_index=True,
        )

    else:

        collision_audit_final = pd.DataFrame(
            columns=[
                "cycle",
                "domain",
                "filename",
                "source_file",
                "original_variable",
                "renamed_variable",
                "reason",
            ]
        )

    collision_audit_final.to_csv(
        INVENTORY_DIR
        / "01_column_collision_audit.csv",
        index=False,
    )

    if all_lineage:

        lineage_final = pd.concat(
            all_lineage,
            ignore_index=True,
        )

    else:

        lineage_final = pd.DataFrame()

    lineage_final.to_csv(
        INVENTORY_DIR / "01_data_lineage.csv",
        index=False,
    )

    cycle_summary = pd.DataFrame(
        cycle_summary_records
    )

    cycle_summary.to_csv(
        INVENTORY_DIR
        / "01_native_join_cycle_summary.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Completion report
    # --------------------------------------------------------------

    logger.info("")
    logger.info("#" * 80)
    logger.info("STAGE 01 COMPLETE")
    logger.info("#" * 80)

    logger.info(
        f"Inventory folder: {INVENTORY_DIR}"
    )

    logger.info(
        f"Joined datasets: {JOINED_DIR}"
    )

    logger.info(
        f"Log: {LOG_FILE}"
    )

    logger.info(
        f"Completed: "
        f"{datetime.now().isoformat(timespec='seconds')}"
    )

    print("\n")
    print("=" * 72)
    print("STAGE 01 COMPLETE")
    print("=" * 72)

    print(
        "\nReview these files BEFORE moving to Stage 02:\n"
    )

    print(
        "1. processed/01_inventory/"
        "01_file_inventory.csv"
    )

    print(
        "2. processed/01_inventory/"
        "01_cardinality_summary.csv"
    )

    print(
        "3. processed/01_inventory/"
        "01_files_excluded_from_native_join.csv"
    )

    print(
        "4. processed/01_inventory/"
        "01_join_audit.csv"
    )

    print(
        "5. processed/01_inventory/"
        "01_column_collision_audit.csv"
    )

    print(
        "6. processed/01_inventory/"
        "01_native_join_cycle_summary.csv"
    )

    print(
        "7. processed/01_inventory/"
        "01_data_lineage.csv"
    )

    print(
        "\nDO NOT begin harmonization until these audits "
        "have been reviewed.\n"
    )


# ======================================================================
# 12. ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()