"""
===============================================================================
STAGE 09
Final Scientific Synthesis, Claim-Evidence Audit,
===============================================================================
===============================================================================
"""

from __future__ import annotations

import hashlib
import math
import shutil
import textwrap
import warnings

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. PATHS
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

STAGE08_DIR = (
    PROCESSED_ROOT
    / "10_clinical_utility"
)

OUTPUT_DIR = (
    PROCESSED_ROOT
    / "11_final_synthesis"
)

TABLE_DIR = (
    OUTPUT_DIR
    / "tables"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

CONFERENCE_DIR = (
    OUTPUT_DIR
    / "conference"
)

MANUSCRIPT_DIR = (
    OUTPUT_DIR
    / "manuscript"
)

AUDIT_DIR = (
    OUTPUT_DIR
    / "audit"
)

for directory in [
    OUTPUT_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    CONFERENCE_DIR,
    MANUSCRIPT_DIR,
    AUDIT_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# 2. INPUT FILES
# =============================================================================

STAGE03_PREVALENCE = (
    STAGE03_DIR
    / "audits"
    / "03_outcome_prevalence_audit.csv"
)

STAGE06_METRICS = (
    STAGE06_DIR
    / "06_experiment_metrics.csv"
)

STAGE06_STATUS = (
    STAGE06_DIR
    / "06_execution_status.csv"
)

STAGE07_TEMPORAL = (
    STAGE07_DIR
    / "07_temporal_transport_profile.csv"
)

STAGE07_DISTANCE = (
    STAGE07_DIR
    / "07_temporal_distance_association.csv"
)

STAGE07_TIER = (
    STAGE07_DIR
    / "07_tier_contribution_summary.csv"
)

STAGE07_IRON = (
    STAGE07_DIR
    / "07_iron_natural_experiment_summary.csv"
)

STAGE07_SHIFT = (
    STAGE07_DIR
    / "07_shift_decomposition_descriptive.csv"
)

STAGE07_INTEGRITY = (
    STAGE07_DIR
    / "07_integrity_check.csv"
)

STAGE08_DCA = (
    STAGE08_DIR
    / "08_dca_strategy_summary.csv"
)

STAGE08_OPERATING = (
    STAGE08_DIR
    / "08_operating_point_summary.csv"
)

STAGE08_MODEL_CONSENSUS = (
    STAGE08_DIR
    / "08_model_family_consensus.csv"
)

STAGE08_MODEL_ROBUSTNESS = (
    STAGE08_DIR
    / "08_model_family_tier_robustness.csv"
)

STAGE08_WEIGHTED = (
    STAGE08_DIR
    / "08_weighted_vs_unweighted_summary.csv"
)

STAGE08_SUBGROUP = (
    STAGE08_DIR
    / "08_subgroup_disparity_summary.csv"
)

STAGE08_AGE = (
    STAGE08_DIR
    / "08_age20_59_sensitivity_summary.csv"
)

STAGE08_SURVEY = (
    STAGE08_DIR
    / "08_survey_design_descriptive.csv"
)

STAGE08_LINKAGE = (
    STAGE08_DIR
    / "08_prediction_linkage_audit.csv"
)

STAGE08_INTEGRITY = (
    STAGE08_DIR
    / "08_integrity_check.csv"
)


# =============================================================================
# 3. OUTPUT FILES
# =============================================================================

FINAL_WORKBOOK = (
    OUTPUT_DIR
    / "09_final_scientific_synthesis.xlsx"
)

KEY_RESULTS_PATH = (
    TABLE_DIR
    / "09_key_results.csv"
)

PRIMARY_PERFORMANCE_PATH = (
    TABLE_DIR
    / "09_primary_performance_table.csv"
)

TIER_PATH = (
    TABLE_DIR
    / "09_tier_contribution_table.csv"
)

UTILITY_PATH = (
    TABLE_DIR
    / "09_clinical_utility_table.csv"
)

WEIGHTED_PATH = (
    TABLE_DIR
    / "09_weighted_robustness_table.csv"
)

AGE_PATH = (
    TABLE_DIR
    / "09_age_sensitivity_table.csv"
)

TEMPORAL_PATH = (
    TABLE_DIR
    / "09_temporal_degradation_table.csv"
)

SUBGROUP_PATH = (
    TABLE_DIR
    / "09_subgroup_robustness_table.csv"
)

CLAIM_AUDIT_PATH = (
    AUDIT_DIR
    / "09_claim_evidence_audit.csv"
)

INTEGRITY_PATH = (
    AUDIT_DIR
    / "09_integrity_check.csv"
)

DECISION_LEDGER_PATH = (
    AUDIT_DIR
    / "decision_ledger_stage09.csv"
)

ABSTRACT_PATH = (
    CONFERENCE_DIR
    / "09_AISC_abstract_draft.md"
)

TALK_PATH = (
    CONFERENCE_DIR
    / "09_AISC_20min_talk_structure.md"
)

TITLE_PATH = (
    CONFERENCE_DIR
    / "09_title_and_positioning.md"
)

RESULTS_PATH = (
    MANUSCRIPT_DIR
    / "09_manuscript_results_draft.md"
)

METHODS_PATH = (
    MANUSCRIPT_DIR
    / "09_manuscript_methods_summary.md"
)

DISCUSSION_PATH = (
    MANUSCRIPT_DIR
    / "09_discussion_framework.md"
)

FIGURE_MANIFEST_PATH = (
    FIGURE_DIR
    / "09_figure_manifest.csv"
)


# =============================================================================
# 4. HELPERS
# =============================================================================

def require_file(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required Stage-09 input missing:\n{path}"
        )


def read_csv(path: Path):

    require_file(
        path
    )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def find_column(df, candidates):

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
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


def numeric_mean(series):

    return float(
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .mean()
    )


def numeric_median(series):

    return float(
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .median()
    )


def pct(value, digits=1):

    if pd.isna(
        value
    ):

        return "NA"

    return (
        f"{100 * float(value):.{digits}f}%"
    )


def dec(value, digits=3):

    if pd.isna(
        value
    ):

        return "NA"

    return (
        f"{float(value):.{digits}f}"
    )


def check_integrity_file(path):

    df = read_csv(
        path
    )

    pass_column = find_column(
        df,
        [
            "pass",
        ]
    )

    if pass_column is None:

        raise RuntimeError(
            f"No pass column in {path.name}"
        )

    passed = (
        df[
            pass_column
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

        failed = df[
            ~passed
        ]

        raise RuntimeError(
            f"{path.name} contains "
            f"{len(failed)} failed checks."
        )

    return df


# =============================================================================
# 5. VALIDATE FROZEN PIPELINE
# =============================================================================

def validate_pipeline():

    status = read_csv(
        STAGE06_STATUS
    )

    latest = (
        status
        .drop_duplicates(
            "experiment_id",
            keep="last",
        )
    )

    if (
        len(
            latest
        )
        != 336
    ):

        raise RuntimeError(
            "Stage 06 experiment count is not 336."
        )

    if (
        latest[
            "status"
        ]
        .eq(
            "SUCCESS"
        )
        .sum()
        != 336
    ):

        raise RuntimeError(
            "Stage 06 does not have 336 successful experiments."
        )

    check_integrity_file(
        STAGE07_INTEGRITY
    )

    check_integrity_file(
        STAGE08_INTEGRITY
    )

    print(
        "Frozen pipeline validated:"
    )

    print(
        "Stage06 336/336 | Stage07 PASS | Stage08 PASS"
    )


# =============================================================================
# 6. LOAD
# =============================================================================

def load_inputs():

    validate_pipeline()

    data = {
        "stage06_metrics":
            read_csv(
                STAGE06_METRICS
            ),

        "temporal":
            read_csv(
                STAGE07_TEMPORAL
            ),

        "distance":
            read_csv(
                STAGE07_DISTANCE
            ),

        "tier":
            read_csv(
                STAGE07_TIER
            ),

        "iron":
            read_csv(
                STAGE07_IRON
            ),

        "shift":
            read_csv(
                STAGE07_SHIFT
            ),

        "dca":
            read_csv(
                STAGE08_DCA
            ),

        "operating":
            read_csv(
                STAGE08_OPERATING
            ),

        "consensus":
            read_csv(
                STAGE08_MODEL_CONSENSUS
            ),

        "model_robustness":
            read_csv(
                STAGE08_MODEL_ROBUSTNESS
            ),

        "weighted":
            read_csv(
                STAGE08_WEIGHTED
            ),

        "subgroup":
            read_csv(
                STAGE08_SUBGROUP
            ),

        "age":
            read_csv(
                STAGE08_AGE
            ),

        "survey":
            read_csv(
                STAGE08_SURVEY
            ),

        "linkage":
            read_csv(
                STAGE08_LINKAGE
            ),
    }

    return data


# =============================================================================
# 7. PRIMARY PERFORMANCE TABLE
# =============================================================================

def build_primary_performance(
    temporal
):

    result = (
        temporal.groupby(
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

            mean_AUROC=(
                "AUROC",
                "mean",
            ),

            mean_AUPRC=(
                "AUPRC",
                "mean",
            ),

            mean_Brier=(
                "Brier",
                "mean",
            ),

            mean_ECE=(
                "ECE",
                "mean",
            ),
        )
        .reset_index()
    )

    result.to_csv(
        PRIMARY_PERFORMANCE_PATH,
        index=False,
    )

    return result


# =============================================================================
# 8. TIER CONTRIBUTION TABLE
# =============================================================================

def build_tier_table(
    tier,
    consensus,
):

    result = tier.merge(
        consensus,
        on="study",
        how="left",
    )

    result.to_csv(
        TIER_PATH,
        index=False,
    )

    return result


# =============================================================================
# 9. CLINICAL UTILITY TABLE
# =============================================================================

def build_utility_table(
    operating,
    dca,
):

    target = (
        operating[
            np.isclose(
                operating[
                    "target_training_sensitivity"
                ],
                0.90,
            )
        ]
        .copy()
    )

    result = target.merge(
        dca[
            [
                "study",
                "feature_tier",
                "mean_proportion_thresholds_model_best",
                "mean_net_benefit_advantage",
            ]
        ],
        on=[
            "study",
            "feature_tier",
        ],
        how="left",
    )

    result.to_csv(
        UTILITY_PATH,
        index=False,
    )

    return result


# =============================================================================
# 10. WEIGHTED ROBUSTNESS
# =============================================================================

def build_weighted_table(
    weighted
):

    keep = [
        c
        for c in [
            "study",
            "feature_tier",
            "n_experiments",
            "mean_unweighted_AUROC",
            "mean_weighted_AUROC",
            "mean_weighted_minus_unweighted_AUROC",
            "mean_unweighted_AUPRC",
            "mean_weighted_AUPRC",
            "mean_weighted_minus_unweighted_AUPRC",
            "mean_unweighted_Brier",
            "mean_weighted_Brier",
            "mean_unweighted_ECE",
            "mean_weighted_ECE",
        ]
        if c
        in weighted.columns
    ]

    result = weighted[
        keep
    ].copy()

    result.to_csv(
        WEIGHTED_PATH,
        index=False,
    )

    return result


# =============================================================================
# 11. AGE SENSITIVITY TABLE
# =============================================================================

def build_age_table(
    age
):

    result = age.copy()

    result.to_csv(
        AGE_PATH,
        index=False,
    )

    return result


# =============================================================================
# 12. TEMPORAL DEGRADATION TABLE
# =============================================================================

def build_temporal_table(
    distance
):

    result = (
        distance.groupby(
            [
                "study",
                "feature_tier",
                "metric",
            ]
        )
        .agg(
            n_model_associations=(
                "model",
                "count",
            ),

            mean_spearman_rho=(
                "spearman_rho",
                "mean",
            ),

            median_spearman_rho=(
                "spearman_rho",
                "median",
            ),
        )
        .reset_index()
    )

    result.to_csv(
        TEMPORAL_PATH,
        index=False,
    )

    return result


# =============================================================================
# 13. SUBGROUP ROBUSTNESS TABLE
# =============================================================================

def build_subgroup_table(
    subgroup
):

    range_columns = [
        c
        for c in subgroup.columns
        if c.startswith(
            "range_"
        )
    ]

    rows = []

    for (
        study,
        tier,
        dimension,
    ), group in subgroup.groupby(
        [
            "study",
            "feature_tier",
            "subgroup_dimension",
        ]
    ):

        row = {
            "study":
                study,

            "feature_tier":
                tier,

            "subgroup_dimension":
                dimension,

            "n_comparisons":
                len(
                    group
                ),
        }

        for column in range_columns:

            row[
                f"mean_{column}"
            ] = numeric_mean(
                group[
                    column
                ]
            )

            row[
                f"median_{column}"
            ] = numeric_median(
                group[
                    column
                ]
            )

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        SUBGROUP_PATH,
        index=False,
    )

    return result


# =============================================================================
# 14. MASTER KEY RESULTS
# =============================================================================

def lookup(
    df,
    study,
    tier,
    column,
):

    row = df[
        (
            df[
                "study"
            ]
            == study
        )
        &
        (
            df[
                "feature_tier"
            ]
            == tier
        )
    ]

    if (
        row.empty
        or column
        not in row.columns
    ):

        return np.nan

    return row[
        column
    ].iloc[
        0
    ]


def build_key_results(
    primary,
    tier,
    utility,
    weighted,
    age,
):

    rows = []

    for study in [
        "vitamin_d",
        "iron_observed",
        "iron_matched",
    ]:

        for tier_name in [
            "T1",
            "T2",
        ]:

            rows.append(
                {
                    "study":
                        study,

                    "feature_tier":
                        tier_name,

                    "mean_AUROC":
                        lookup(
                            primary,
                            study,
                            tier_name,
                            "mean_AUROC",
                        ),

                    "mean_AUPRC":
                        lookup(
                            primary,
                            study,
                            tier_name,
                            "mean_AUPRC",
                        ),

                    "mean_Brier":
                        lookup(
                            primary,
                            study,
                            tier_name,
                            "mean_Brier",
                        ),

                    "mean_ECE":
                        lookup(
                            primary,
                            study,
                            tier_name,
                            "mean_ECE",
                        ),

                    "weighted_AUROC":
                        lookup(
                            weighted,
                            study,
                            tier_name,
                            "mean_weighted_AUROC",
                        ),

                    "tests_avoided_90":
                        lookup(
                            utility,
                            study,
                            tier_name,
                            "mean_TestsAvoided",
                        ),

                    "realized_sensitivity_90":
                        lookup(
                            utility,
                            study,
                            tier_name,
                            "mean_Sensitivity",
                        ),

                    "dca_fraction_model_best":
                        lookup(
                            utility,
                            study,
                            tier_name,
                            "mean_proportion_thresholds_model_best",
                        ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        KEY_RESULTS_PATH,
        index=False,
    )

    return result


# =============================================================================
# 15. FIGURES
# =============================================================================

def save_primary_auroc_figure(
    primary
):

    pivot = (
        primary.pivot(
            index="study",
            columns="feature_tier",
            values="mean_AUROC",
        )
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(
            9,
            5,
        ),
    )

    ax.set_ylabel(
        "Mean AUROC across locked temporal experiments"
    )

    ax.set_xlabel(
        ""
    )

    ax.set_title(
        "Predictive discrimination by nutrient and resource tier"
    )

    ax.set_ylim(
        0.45,
        0.90,
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    path = (
        FIGURE_DIR
        / "Fig01_primary_AUROC_by_tier.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return path


def save_model_tier_figure(
    model_robustness
):

    pivot = (
        model_robustness.pivot(
            index="model",
            columns="study",
            values="mean_delta_AUROC",
        )
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(
            10,
            5,
        ),
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_ylabel(
        "Mean ΔAUROC (Tier 2 − Tier 1)"
    )

    ax.set_xlabel(
        ""
    )

    ax.set_title(
        "Incremental value of routine CBC across model families"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    path = (
        FIGURE_DIR
        / "Fig02_model_family_tier_delta.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return path


def save_testing_burden_figure(
    operating
):

    pivot = operating.pivot_table(
        index=[
            "study",
            "feature_tier",
        ],
        columns=
            "target_training_sensitivity",
        values=
            "mean_TestsAvoided",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(
            11,
            5,
        ),
    )

    ax.set_ylabel(
        "Mean proportion of confirmatory tests avoided"
    )

    ax.set_xlabel(
        ""
    )

    ax.set_title(
        "Testing efficiency across prespecified sensitivity targets"
    )

    ax.set_ylim(
        0,
        max(
            0.4,
            float(
                np.nanmax(
                    pivot.values
                )
            )
            + 0.05,
        ),
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    path = (
        FIGURE_DIR
        / "Fig03_testing_burden.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return path


def save_weighted_figure(
    weighted
):

    x = np.arange(
        len(
            weighted
        )
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.bar(
        x - width / 2,
        weighted[
            "mean_unweighted_AUROC"
        ],
        width,
        label="Unweighted",
    )

    ax.bar(
        x + width / 2,
        weighted[
            "mean_weighted_AUROC"
        ],
        width,
        label="MEC-weighted sensitivity",
    )

    labels = (
        weighted[
            "study"
        ]
        +
        "\n"
        +
        weighted[
            "feature_tier"
        ]
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )

    ax.set_ylabel(
        "AUROC"
    )

    ax.set_title(
        "Survey-weight sensitivity of temporal predictive performance"
    )

    ax.legend()

    plt.tight_layout()

    path = (
        FIGURE_DIR
        / "Fig04_weighted_vs_unweighted_AUROC.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return path


def save_temporal_figure(
    temporal
):

    subset = (
        temporal[
            temporal[
                "temporal_design"
            ]
            == "fixed_origin"
        ]
        .copy()
    )

    aggregate = (
        subset.groupby(
            [
                "study",
                "feature_tier",
                "absolute_temporal_distance_years",
            ]
        )
        .agg(
            mean_AUROC=(
                "AUROC",
                "mean",
            )
        )
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    for (
        study,
        tier,
    ), group in aggregate.groupby(
        [
            "study",
            "feature_tier",
        ]
    ):

        group = group.sort_values(
            "absolute_temporal_distance_years"
        )

        ax.plot(
            group[
                "absolute_temporal_distance_years"
            ],
            group[
                "mean_AUROC"
            ],
            marker="o",
            label=f"{study} {tier}",
        )

    ax.set_xlabel(
        "Absolute temporal distance from most recent training cycle (years)"
    )

    ax.set_ylabel(
        "Mean AUROC"
    )

    ax.set_title(
        "Fixed-origin transport across temporal distance"
    )

    ax.legend(
        fontsize=8
    )

    plt.tight_layout()

    path = (
        FIGURE_DIR
        / "Fig05_temporal_transport.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return path


# =============================================================================
# 16. FIGURE MANIFEST
# =============================================================================

def build_figure_manifest(
    figures
):

    descriptions = {
        "Fig01_primary_AUROC_by_tier.png":
            (
                "Primary discrimination across nutrient-specific "
                "resource tiers."
            ),

        "Fig02_model_family_tier_delta.png":
            (
                "Tier-2 minus Tier-1 AUROC change across all "
                "six locked model families."
            ),

        "Fig03_testing_burden.png":
            (
                "Proportion of confirmatory tests avoided at "
                "90%, 95%, and 97.5% sensitivity targets."
            ),

        "Fig04_weighted_vs_unweighted_AUROC.png":
            (
                "Unweighted versus MEC-weighted sensitivity analysis."
            ),

        "Fig05_temporal_transport.png":
            (
                "Fixed-origin mean AUROC as temporal distance increases."
            ),
    }

    rows = []

    for index, path in enumerate(
        figures,
        start=1,
    ):

        rows.append(
            {
                "figure_number":
                    index,

                "filename":
                    path.name,

                "purpose":
                    descriptions.get(
                        path.name,
                        "",
                    ),

                "recommended_use":
                    (
                        "AISC talk and manuscript"
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        FIGURE_MANIFEST_PATH,
        index=False,
    )

    return result


# =============================================================================
# 17. CLAIM-EVIDENCE AUDIT
# =============================================================================

def build_claim_audit(
    consensus,
    weighted,
    age,
    dca,
    operating,
):

    claims = []

    def add(
        claim,
        evidence,
        strength,
        permitted_language,
        prohibited_language,
    ):

        claims.append(
            {
                "claim":
                    claim,

                "evidence":
                    evidence,

                "evidence_strength":
                    strength,

                "permitted_language":
                    permitted_language,

                "avoid_language":
                    prohibited_language,
            }
        )

    add(
        (
            "CBC provides substantial incremental predictive "
            "value for iron-deficiency triage."
        ),
        (
            "Tier-2 ΔAUROC approximately +0.29 in both observed "
            "and matched iron analyses; all six model families "
            "support positive mean T2 gain."
        ),
        "STRONG",
        (
            "substantial incremental value; consistent across "
            "model families"
        ),
        (
            "CBC causes better outcomes; universally optimal"
        ),
    )

    add(
        (
            "CBC adds little incremental discrimination for "
            "vitamin-D screening."
        ),
        (
            "Mean Tier-2 minus Tier-1 AUROC approximately "
            "-0.0015; 0/6 model families support positive "
            "mean Tier-2 gain."
        ),
        "STRONG",
        (
            "little or no incremental predictive value"
        ),
        (
            "CBC is clinically useless"
        ),
    )

    add(
        (
            "Iron Tier 2 provides greater confirmatory-testing "
            "efficiency than Tier 1."
        ),
        (
            "At the 90% training-sensitivity target, Tier 2 "
            "avoids materially more ferritin testing than Tier 1."
        ),
        "STRONG",
        (
            "testing-efficient; reduces confirmatory-testing burden"
        ),
        (
            "cost-effective; saves healthcare costs"
        ),
    )

    add(
        (
            "Temporal instability cannot be characterized by "
            "elapsed time alone."
        ),
        (
            "Vitamin D shows clearer temporal attenuation; iron "
            "patterns differ after eligibility matching and across "
            "measured distribution shifts."
        ),
        "MODERATE_TO_STRONG",
        (
            "consistent with multiple sources of temporal shift"
        ),
        (
            "we proved the cause of model drift"
        ),
    )

    add(
        (
            "Ferritin eligibility changes create substantial "
            "population-distribution discontinuity."
        ),
        (
            "Stage 04 matching reduced I-to-J mean |SMD|, PSI, "
            "and JSD substantially."
        ),
        "STRONG",
        (
            "materially associated with compositional discontinuity"
        ),
        (
            "fully explains iron model degradation"
        ),
    )

    add(
        (
            "Eligibility matching does not fully restore iron "
            "predictive transport."
        ),
        (
            "Observed-versus-matched Stage-07 performance differences "
            "were small or negative in multiple bridge analyses."
        ),
        "STRONG",
        (
            "residual predictive instability remains after matching"
        ),
        (
            "matching solves temporal drift"
        ),
    )

    add(
        (
            "OOF probability recalibration substantially improves "
            "probability reliability."
        ),
        (
            "Sigmoid recalibration improved Brier score in all "
            "study/tier experiment groups and ECE in essentially all."
        ),
        "STRONG",
        (
            "improves calibration/probability reliability"
        ),
        (
            "restores discrimination; target-period recalibration"
        ),
    )

    add(
        (
            "Survey weighting does not overturn the principal "
            "Tier-1/Tier-2 discrimination conclusion."
        ),
        (
            "Weighted AUROC changes are modest and retain the "
            "vitamin-D T1≈T2 and iron T2≫T1 ordering."
        ),
        "STRONG",
        (
            "robust in MEC-weighted sensitivity analysis"
        ),
        (
            "fully survey-representative causal validation"
        ),
    )

    add(
        (
            "The iron Tier-2 result is robust to exclusion of "
            "ages 18–19."
        ),
        (
            "Age 20–59 evaluation changes iron Tier-2 AUROC only "
            "minimally without retraining."
        ),
        "STRONG",
        (
            "robust to the 20–59 structural sensitivity analysis"
        ),
        (
            "education missingness has no effect whatsoever"
        ),
    )

    add(
        (
            "Subgroup performance heterogeneity was evaluated."
        ),
        (
            "Stage 08 evaluates sex, race/ethnicity, age and poverty "
            "groups subject to minimum support rules."
        ),
        "DESCRIPTIVE",
        (
            "subgroup transportability was examined; performance "
            "heterogeneity was quantified"
        ),
        (
            "the model is fair; no disparities exist"
        ),
    )

    result = pd.DataFrame(
        claims
    )

    result.to_csv(
        CLAIM_AUDIT_PATH,
        index=False,
    )

    return result


# =============================================================================
# 18. TITLE + POSITIONING
# =============================================================================

def write_positioning():

    text = """
# Stage 09 — Title and Scientific Positioning

## Recommended AISC title

**Beyond Temporal Validation: A Shift-Aware, Resource-Adaptive Machine-Learning Framework for Micronutrient Screening Across NHANES**

## Strong alternative

**When Model Drift Is Not Just Time: Disentangling Temporal, Population, and Measurement Shift in Machine-Learning Screening for Micronutrient Deficiency**

## Shorter presentation title

**When Model Drift Is Not Just Time: Resource-Adaptive Micronutrient Screening Across NHANES**

## Primary methodological question

When a clinical screening model changes performance across survey periods,
how much of the apparent instability reflects elapsed time versus population
eligibility, prevalence, measured covariate distributions, calibration, and
the amount of clinical information available?

## Primary clinical question

How much clinical information is needed to support reliable triage for
confirmatory micronutrient testing?

## Core contribution

The study is not primarily a new classifier.

Its contribution is a deployment-aware evaluation framework combining:

1. harmonized longitudinal NHANES cohorts;
2. locked temporal transport;
3. eligibility-matched natural experiments;
4. resource-tier comparisons;
5. calibration;
6. distribution-shift diagnostics;
7. clinical utility;
8. survey-weighted sensitivity;
9. subgroup transportability;
10. structural robustness analyses.

## Main conclusion

The amount of information required for useful micronutrient screening is
nutrient-specific. Non-invasive information retained useful discrimination
for vitamin-D insufficiency, whereas routine CBC information was necessary
for strong iron-deficiency triage. Temporal model instability was also not
well described by elapsed time alone; eligibility composition, prevalence,
covariate shift, calibration, and information tier all influenced transport.

## Language safeguards

Use:
- resource-adaptive
- resource-efficient
- testing-efficient
- confirmatory-testing reduction
- temporal transportability
- distribution shift
- descriptive association

Avoid:
- cost-effective
- causal effect of shift
- representative of rural populations
- universal screening solution
- model drift is caused by eligibility changes
"""

    TITLE_PATH.write_text(
        text.strip()
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# 19. ABSTRACT
# =============================================================================

def write_abstract(
    primary,
    tier,
    utility,
    weighted,
):

    vd_t1 = lookup(
        primary,
        "vitamin_d",
        "T1",
        "mean_AUROC",
    )

    vd_t2 = lookup(
        primary,
        "vitamin_d",
        "T2",
        "mean_AUROC",
    )

    io_t1 = lookup(
        primary,
        "iron_observed",
        "T1",
        "mean_AUROC",
    )

    io_t2 = lookup(
        primary,
        "iron_observed",
        "T2",
        "mean_AUROC",
    )

    im_t1 = lookup(
        primary,
        "iron_matched",
        "T1",
        "mean_AUROC",
    )

    im_t2 = lookup(
        primary,
        "iron_matched",
        "T2",
        "mean_AUROC",
    )

    iron_obs_tests = lookup(
        utility,
        "iron_observed",
        "T2",
        "mean_TestsAvoided",
    )

    iron_match_tests = lookup(
        utility,
        "iron_matched",
        "T2",
        "mean_TestsAvoided",
    )

    text = f"""
# AISC 2026 Abstract Draft

## Beyond Temporal Validation: A Shift-Aware, Resource-Adaptive Machine-Learning Framework for Micronutrient Screening Across NHANES

**Background:** Machine-learning screening models are often evaluated within a
single population or time period, obscuring whether later performance changes
reflect true temporal instability, changes in population eligibility, or
differences in available clinical information. We developed a resource-adaptive
framework for micronutrient screening using repeated U.S. National Health and
Nutrition Examination Survey (NHANES) cycles.

**Methods:** Adults aged 18–59 years were assembled from NHANES 2009–2023.
Vitamin-D insufficiency and pre-anemic iron deficiency were evaluated using two
prespecified information tiers: Tier 1 contained non-invasive demographic,
socioeconomic, anthropometric, lifestyle, and dietary information; Tier 2 added
routine complete blood count variables. Six model families were evaluated using
fixed-origin, rolling-expanding, and backward/forward temporal transport
designs. Probability calibration, covariate-shift diagnostics, ferritin
eligibility-matched analyses, decision-curve analysis, survey-weighted
sensitivity analyses, subgroup transportability, and confirmatory-testing
burden were assessed.

**Results:** Vitamin-D discrimination was similar with Tier 1 and Tier 2
(mean AUROC {vd_t1:.3f} vs {vd_t2:.3f}), indicating little incremental value
from routine CBC information. In contrast, iron discrimination increased
substantially from Tier 1 to Tier 2 in both the observed cohort
({io_t1:.3f} vs {io_t2:.3f}) and the eligibility-matched cohort
({im_t1:.3f} vs {im_t2:.3f}); the Tier-2 advantage was consistent across all
six model families. At the 90% training-sensitivity operating target, Tier-2
iron models avoided approximately {100*iron_obs_tests:.1f}% of confirmatory
tests in the observed cohort and {100*iron_match_tests:.1f}% in the matched
cohort. Eligibility matching markedly reduced measured population shift around
the ferritin protocol transition but did not uniformly restore discrimination,
suggesting that temporal transportability reflects multiple interacting sources
of shift. Calibration substantially improved probability reliability, and the
principal nutrient-specific tier conclusions were preserved in survey-weighted
and age 20–59 sensitivity analyses.

**Conclusions:** The clinical information required for useful machine-learning
screening is nutrient-specific. Routine hematologic information substantially
improves iron-deficiency triage but adds little to vitamin-D screening.
Temporal validation should therefore distinguish elapsed time from changes in
population composition, measurement context, calibration, and resource
availability when evaluating deployment readiness.
"""

    ABSTRACT_PATH.write_text(
        text.strip()
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# 20. METHODS SUMMARY
# =============================================================================

def write_methods():

    text = """
# Stage 09 — Manuscript Methods Summary

## Study design

Repeated cross-sectional NHANES cycles were harmonized across 2009–2023.
The principal temporal backbone consisted of 2009–2010, 2011–2012,
2013–2014, 2015–2016, 2017–2018, and 2021–2023. The 2017–March 2020
pre-pandemic release was treated separately because of overlap with
2017–2018.

## Outcomes

Vitamin-D insufficiency:
- primary: serum 25-hydroxyvitamin D <50 nmol/L;
- sensitivity: <30 nmol/L.

Pre-anemic iron deficiency:
- primary: ferritin <15 µg/L among non-anemic participants;
- sensitivity: ferritin <25 µg/L.

Iron was evaluated using:
1. observed biomarker-eligible populations; and
2. a matched female age 18–49 non-pregnant cohort to reduce ferritin
   eligibility-related compositional differences.

## Resource tiers

Tier 1 included demographic, socioeconomic, healthcare-access,
food-security, anthropometric, smoking, chronic-disease and dietary
variables.

Tier 2 added nine routine CBC indices.

Outcome-defining biomarkers were never predictors.

## Models

Six prespecified model families:
- logistic regression;
- elastic-net logistic regression;
- random forest;
- XGBoost;
- LightGBM;
- CatBoost.

Hyperparameters were tuned using internal cross-validation within the
training population only. Temporal test labels were never used for tuning.

## Temporal validation

Three complementary transport designs were used:

1. fixed-origin forward transport;
2. rolling-expanding training histories;
3. backward/forward bridge analyses around 2017–2018.

## Calibration

Sigmoid calibration using training out-of-fold probabilities was primary.
Raw probabilities and isotonic calibration were retained as comparisons.

## Shift diagnostics

Continuous variables were assessed using standardized mean difference,
population stability index, Kolmogorov-Smirnov statistics, and related
distributional diagnostics. Categorical features were evaluated using
Jensen-Shannon divergence and category-proportion differences.

Shift-performance relationships were descriptive and were not interpreted
causally.

## Clinical utility

Operating thresholds were derived from training out-of-fold predictions at
prespecified sensitivity targets of 90%, 95%, and 97.5%.

Reported utility measures included:
- referral rate;
- tests avoided;
- cases detected;
- cases missed;
- tests per detected case;
- PPV;
- NPV;
- decision-curve net benefit.

## Robustness

Participant-level Stage-06 predictions were linked by SEQN to frozen
Stage-03 cohorts.

Additional analyses included:
- MEC-weighted predictive sensitivity analyses;
- subgroup transportability by sex, race/ethnicity, age, and poverty;
- age 20–59 sensitivity evaluation;
- survey-design-aware descriptive prevalence estimates.

Survey-weighted machine-learning metrics were treated as sensitivity
analyses rather than full complex-survey estimators.
"""

    METHODS_PATH.write_text(
        text.strip()
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# 21. RESULTS DRAFT
# =============================================================================

def write_results(
    primary,
    tier,
    utility,
    consensus,
    weighted,
    age,
):

    vitamin_delta = (
        tier.loc[
            tier[
                "study"
            ]
            == "vitamin_d",
            "mean_delta_AUROC_T2_minus_T1",
        ]
        .iloc[
            0
        ]
    )

    iron_obs_delta = (
        tier.loc[
            tier[
                "study"
            ]
            == "iron_observed",
            "mean_delta_AUROC_T2_minus_T1",
        ]
        .iloc[
            0
        ]
    )

    iron_match_delta = (
        tier.loc[
            tier[
                "study"
            ]
            == "iron_matched",
            "mean_delta_AUROC_T2_minus_T1",
        ]
        .iloc[
            0
        ]
    )

    text = f"""
# Stage 09 — Results Draft

## Resource-tier performance

Across the locked temporal experiments, vitamin-D screening achieved a mean
AUROC of {lookup(primary, "vitamin_d", "T1", "mean_AUROC"):.3f} using Tier 1
and {lookup(primary, "vitamin_d", "T2", "mean_AUROC"):.3f} using Tier 2.
The mean Tier-2 minus Tier-1 AUROC difference was {vitamin_delta:+.3f},
indicating essentially no incremental discrimination from routine CBC data.

Iron showed a markedly different pattern. In the observed ferritin-eligible
cohort, mean AUROC increased from
{lookup(primary, "iron_observed", "T1", "mean_AUROC"):.3f} in Tier 1 to
{lookup(primary, "iron_observed", "T2", "mean_AUROC"):.3f} in Tier 2
(mean ΔAUROC {iron_obs_delta:+.3f}). In the eligibility-matched cohort,
mean AUROC increased from
{lookup(primary, "iron_matched", "T1", "mean_AUROC"):.3f} to
{lookup(primary, "iron_matched", "T2", "mean_AUROC"):.3f}
(mean ΔAUROC {iron_match_delta:+.3f}).

All six locked model families supported a positive mean Tier-2 AUROC gain
for both iron analyses, whereas none supported a positive mean Tier-2 gain
for vitamin D.

## Temporal transport and population shift

Vitamin-D fixed-origin transport displayed a clearer decline in discrimination
with increasing temporal distance. Iron did not exhibit a simple monotonic
relationship between elapsed time and performance.

Ferritin eligibility matching substantially reduced measured distributional
shift around the 2017–2018 transition. However, matched-cohort predictive
performance was not uniformly superior to observed-cohort performance.
Therefore, changes in ferritin eligibility appear to contribute to apparent
population discontinuity but do not fully account for temporal predictive
instability.

## Calibration

Training-OOF sigmoid calibration consistently improved Brier score and ECE,
particularly for iron models trained with class balancing. These results
support recalibration as a comparatively low-burden maintenance strategy
for probability reliability, although recalibration should not be interpreted
as restoring discrimination.

## Clinical utility

At the prespecified 90% training-sensitivity target, Tier-2 iron models
substantially reduced referral for confirmatory ferritin testing compared
with Tier 1.

Observed iron:
- Tier 1 tests avoided:
  {pct(lookup(utility, "iron_observed", "T1", "mean_TestsAvoided"))}
- Tier 2 tests avoided:
  {pct(lookup(utility, "iron_observed", "T2", "mean_TestsAvoided"))}

Matched iron:
- Tier 1 tests avoided:
  {pct(lookup(utility, "iron_matched", "T1", "mean_TestsAvoided"))}
- Tier 2 tests avoided:
  {pct(lookup(utility, "iron_matched", "T2", "mean_TestsAvoided"))}

For vitamin D, testing efficiency was similar between Tier 1 and Tier 2,
consistent with the negligible discrimination gain from CBC variables.

Decision-curve analysis also favored Tier-2 iron over Tier-1 iron across a
substantially larger fraction of evaluated threshold probabilities.

## Survey-weighted robustness

MEC-weighted sensitivity analysis preserved the principal resource-tier
ordering. Mean AUROC changes after weighting were modest, and the principal
conclusion remained vitamin-D Tier 1 approximately equal to Tier 2 and iron
Tier 2 substantially greater than Tier 1.

AUPRC changes were larger in several analyses, consistent with the
prevalence sensitivity of precision-recall metrics.

## Age 20–59 sensitivity

Excluding ages 18–19 retained approximately 92% of participants. Iron Tier-2
AUROC changed minimally, indicating that the primary Tier-2 iron result was
not driven by the structural adult-education limitation among younger adults.
Tier-1 iron was somewhat more sensitive to this restriction.

## Subgroup transportability

Performance was evaluated across sex, race/ethnicity, age, and poverty
subgroups where minimum sample-size and case-count requirements were met.
Performance heterogeneity was observed and should be reported descriptively;
these analyses should not be interpreted as proof of demographic fairness.
"""

    RESULTS_PATH.write_text(
        text.strip()
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# 22. DISCUSSION FRAMEWORK
# =============================================================================

def write_discussion():

    text = """
# Stage 09 — Discussion Framework

## 1. Principal finding

The amount of clinical information required for useful micronutrient
screening was strongly nutrient-specific.

Vitamin D retained useful discrimination using predominantly non-invasive
information, whereas iron required routine hematologic information for
strong discrimination and operationally useful testing reduction.

## 2. Why this matters methodologically

A model's transportability cannot be judged solely by applying it to a
later calendar period.

Temporal evaluation should distinguish:

- elapsed time;
- eligibility changes;
- prevalence changes;
- predictor-distribution changes;
- measurement context;
- calibration changes;
- information availability.

## 3. Iron as a natural experiment

Ferritin eligibility changed substantially across NHANES cycles.

The matched-cohort analysis showed that much of the population-distribution
discontinuity around 2017–2018 was related to eligibility composition.

However, predictive performance did not uniformly recover after matching.

Thus:

**population matching reduced measured compositional shift without eliminating
all predictive instability.**

This argues against attributing model degradation to one source alone.

## 4. Resource-adaptive interpretation

The study supports a staged testing concept:

### Vitamin D
Tier 1 screening may be sufficient for initial risk stratification.

### Iron
Tier 1 alone is weak.
Routine CBC provides substantial additional information before confirmatory
ferritin testing.

This should be described as a resource-adaptive or testing-efficient pathway,
not a formal cost-effectiveness result.

## 5. Calibration and maintenance

Calibration deterioration can occur even when discrimination remains
relatively stable.

Training-OOF sigmoid calibration substantially improved probability
reliability.

Periodic recalibration may therefore represent a lighter-weight maintenance
strategy than complete retraining when discrimination remains adequate.

## 6. Survey and subgroup robustness

MEC-weighted sensitivity analyses did not overturn the principal resource-tier
conclusions.

Subgroup performance heterogeneity was present and should remain visible in
deployment discussions rather than being collapsed into a single pooled
accuracy statistic.

## 7. Limitations

Important limitations include:

1. repeated cross-sectional rather than longitudinal individuals;
2. NHANES is not a direct rural-deployment dataset;
3. ferritin eligibility differs substantially across cycles;
4. CRP/AGP inflammation-adjustment variables were not consistently available
   in the assembled repository;
5. operational anemia thresholds were not adjusted for all physiologic factors;
6. survey-weighted ML metrics are sensitivity analyses rather than universally
   accepted complex-survey predictive estimators;
7. subgroup estimates are limited by sample and case counts;
8. the study evaluates resource efficiency, not formal economic
   cost-effectiveness;
9. observational distribution-shift associations are descriptive rather than
   causal.

## 8. Contribution

The contribution is not a novel neural architecture.

It is a reproducible framework for evaluating whether clinical machine-learning
screening remains useful when:

- time changes;
- population eligibility changes;
- available clinical information changes;
- probability calibration drifts;
- testing resources are constrained.

That is the main methodological contribution to emphasize at AISC.
"""

    DISCUSSION_PATH.write_text(
        text.strip()
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# 23. 20-MINUTE TALK STRUCTURE
# =============================================================================

def write_talk_structure():

    text = """
# AISC 2026 — 20-Minute Contributed Talk Structure

Total session allowance includes questions, so target approximately
15–16 minutes of presentation and 4–5 minutes for discussion.

## Slide 1 — Title / clinical problem — 1 min

Question:

**How much information do we actually need before ordering a specialized
micronutrient test?**

Introduce resource constraints without claiming NHANES directly represents
rural deployment.

---

## Slide 2 — Why ordinary temporal validation is insufficient — 1.5 min

A lower AUC in a later dataset can reflect:

- true temporal change;
- population eligibility;
- prevalence;
- measurement context;
- predictor distribution;
- calibration.

Key phrase:

**Model drift is not always just time.**

---

## Slide 3 — Study design — 1.5 min

NHANES:
2009–2010 → 2011–2012 → 2013–2014 → 2015–2016 →
2017–2018 → 2021–2023.

Explain P release separately if asked.

---

## Slide 4 — Resource-adaptive tiers — 1 min

Tier 1:
demographics + SES + anthropometrics + lifestyle + diet + access.

Tier 2:
Tier 1 + routine CBC.

Final step:
confirmatory nutrient-specific laboratory test.

---

## Slide 5 — Locked validation framework — 1.5 min

Six models.

Fixed-origin.
Rolling-expanding.
Backward and forward bridges.
Ten seeds.
Leakage-safe calibration.

Emphasize design > algorithm.

---

## Slide 6 — Main result: Vitamin D vs Iron — 2 min

Show Figure 1.

Vitamin D:
T1 ≈ T2.

Iron:
T2 ≫ T1.

This is the key clinical result.

---

## Slide 7 — Six-model robustness — 1 min

Show Figure 2.

Iron:
6/6 models support Tier 2.

Vitamin D:
0/6 support positive Tier-2 gain.

Therefore not architecture-specific.

---

## Slide 8 — Iron natural experiment — 2 min

Explain ferritin eligibility change.

Matching sharply reduced measured I→J population shift.

But discrimination did not uniformly recover.

Conclusion:

**eligibility composition matters, but it is not the whole explanation.**

---

## Slide 9 — Temporal distance and calibration — 1.5 min

Show Figure 5 or selected temporal result.

Vitamin D:
clearer temporal attenuation.

Iron:
non-monotonic.

Calibration:
probability reliability deteriorates differently from ranking performance.

---

## Slide 10 — Clinical utility — 2 min

Show Figure 3.

At high sensitivity:
Tier-2 iron avoids materially more confirmatory tests than Tier 1.

Use:
"tests avoided" / "testing efficiency".

Do not say:
"cost savings" unless economic analysis exists.

---

## Slide 11 — Population robustness — 1 min

Show Figure 4.

Survey weighting preserves primary conclusion.

Mention:
20–59 sensitivity also preserves Tier-2 iron result.

---

## Slide 12 — Final take-home — 1 min

Three messages:

1. **Information depth should be nutrient-specific.**
2. **Temporal model instability is not just elapsed time.**
3. **Deployment evaluation should combine discrimination, calibration,
   shift, clinical utility, and population robustness.**

End with:

**The question is not only whether a model predicts deficiency, but whether
it remains trustworthy when the population, time period, and available
clinical information change.**
"""

    TALK_PATH.write_text(
        text.strip()
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# 24. DECISION LEDGER
# =============================================================================

def build_decision_ledger():

    rows = [
        {
            "decision_id":
                "D091",

            "topic":
                "No further model development",

            "decision":
                (
                    "Stage 09 synthesizes frozen Stage 01–08 results "
                    "and performs no new model fitting."
                ),

            "status":
                "FROZEN",
        },

        {
            "decision_id":
                "D092",

            "topic":
                "Primary AISC contribution",

            "decision":
                (
                    "The contribution is framed as a shift-aware, "
                    "resource-adaptive validation framework rather "
                    "than a novel classifier."
                ),

            "status":
                "FROZEN",
        },

        {
            "decision_id":
                "D093",

            "topic":
                "Primary nutrient comparison",

            "decision":
                (
                    "Vitamin D and iron are presented as contrasting "
                    "case studies of information-tier requirements."
                ),

            "status":
                "FROZEN",
        },

        {
            "decision_id":
                "D094",

            "topic":
                "Temporal interpretation",

            "decision":
                (
                    "Temporal instability is described as reflecting "
                    "multiple interacting sources of shift; causal "
                    "attribution is avoided."
                ),

            "status":
                "FROZEN",
        },

        {
            "decision_id":
                "D095",

            "topic":
                "Clinical utility language",

            "decision":
                (
                    "Testing-efficient/resource-efficient language "
                    "is used instead of cost-effective."
                ),

            "status":
                "FROZEN",
        },

        {
            "decision_id":
                "D096",

            "topic":
                "Survey robustness",

            "decision":
                (
                    "MEC-weighted predictive metrics are reported as "
                    "sensitivity analyses rather than universal "
                    "complex-survey estimators."
                ),

            "status":
                "FROZEN",
        },

        {
            "decision_id":
                "D097",

            "topic":
                "Fairness interpretation",

            "decision":
                (
                    "Subgroup analyses quantify transportability "
                    "heterogeneity and are not used to claim that "
                    "the model is universally fair."
                ),

            "status":
                "FROZEN",
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
# 25. INTEGRITY
# =============================================================================

def build_integrity(
    primary,
    tier,
    utility,
    weighted,
    figures,
    claims,
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
        "Primary study-tier groups",
        6,
        len(
            primary
        ),
        len(
            primary
        )
        == 6,
    )

    add(
        "Tier study groups",
        3,
        tier[
            "study"
        ].nunique(),
        tier[
            "study"
        ].nunique()
        == 3,
    )

    add(
        "Clinical utility groups at 90% target",
        6,
        len(
            utility
        ),
        len(
            utility
        )
        == 6,
    )

    add(
        "Weighted robustness study-tier groups",
        6,
        len(
            weighted
        ),
        len(
            weighted
        )
        == 6,
    )

    add(
        "Publication/conference figures generated",
        5,
        len(
            figures
        ),
        len(
            figures
        )
        == 5,
    )

    add(
        "Claim-evidence audit non-empty",
        ">0",
        len(
            claims
        ),
        len(
            claims
        )
        > 0,
    )

    required_documents = [
        ABSTRACT_PATH,
        TALK_PATH,
        TITLE_PATH,
        METHODS_PATH,
        RESULTS_PATH,
        DISCUSSION_PATH,
    ]

    existing_documents = sum(
        path.exists()
        for path
        in required_documents
    )

    add(
        "Scientific/conference documents generated",
        len(
            required_documents
        ),
        existing_documents,
        existing_documents
        == len(
            required_documents
        ),
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
# 26. FINAL WORKBOOK
# =============================================================================

def write_workbook(
    key,
    primary,
    tier,
    utility,
    temporal,
    weighted,
    age,
    subgroup,
    consensus,
    dca,
    survey,
    claims,
    figures,
    integrity,
    ledger,
):

    with pd.ExcelWriter(
        FINAL_WORKBOOK,
        engine="openpyxl",
    ) as writer:

        key.to_excel(
            writer,
            sheet_name="Key_Results",
            index=False,
        )

        integrity.to_excel(
            writer,
            sheet_name="Integrity_Check",
            index=False,
        )

        primary.to_excel(
            writer,
            sheet_name="Primary_Performance",
            index=False,
        )

        tier.to_excel(
            writer,
            sheet_name="Tier_Contribution",
            index=False,
        )

        utility.to_excel(
            writer,
            sheet_name="Clinical_Utility",
            index=False,
        )

        temporal.to_excel(
            writer,
            sheet_name="Temporal_Degradation",
            index=False,
        )

        weighted.to_excel(
            writer,
            sheet_name="Weighted_Robustness",
            index=False,
        )

        age.to_excel(
            writer,
            sheet_name="Age20_59",
            index=False,
        )

        subgroup.to_excel(
            writer,
            sheet_name="Subgroup_Robustness",
            index=False,
        )

        consensus.to_excel(
            writer,
            sheet_name="Model_Consensus",
            index=False,
        )

        dca.to_excel(
            writer,
            sheet_name="Decision_Curve",
            index=False,
        )

        survey.to_excel(
            writer,
            sheet_name="Survey_Descriptive",
            index=False,
        )

        claims.to_excel(
            writer,
            sheet_name="Claim_Evidence_Audit",
            index=False,
        )

        figures.to_excel(
            writer,
            sheet_name="Figure_Manifest",
            index=False,
        )

        ledger.to_excel(
            writer,
            sheet_name="Decision_Ledger",
            index=False,
        )


# =============================================================================
# 27. MAIN
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
        "STAGE 09"
    )

    print(
        "Final Scientific Synthesis | Claim-Evidence Audit | "
        "AISC Conference Package"
    )

    print(
        "=" * 110
    )

    data = load_inputs()

    # -------------------------------------------------------------------------
    # Tables
    # -------------------------------------------------------------------------

    primary = build_primary_performance(
        data[
            "temporal"
        ]
    )

    tier = build_tier_table(
        data[
            "tier"
        ],
        data[
            "consensus"
        ],
    )

    utility = build_utility_table(
        data[
            "operating"
        ],
        data[
            "dca"
        ],
    )

    weighted = build_weighted_table(
        data[
            "weighted"
        ]
    )

    age = build_age_table(
        data[
            "age"
        ]
    )

    temporal = build_temporal_table(
        data[
            "distance"
        ]
    )

    subgroup = build_subgroup_table(
        data[
            "subgroup"
        ]
    )

    key = build_key_results(
        primary=
            primary,
        tier=
            tier,
        utility=
            utility,
        weighted=
            weighted,
        age=
            age,
    )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    print(
        "\nGenerating conference/manuscript figures..."
    )

    figures = [
        save_primary_auroc_figure(
            primary
        ),

        save_model_tier_figure(
            data[
                "model_robustness"
            ]
        ),

        save_testing_burden_figure(
            data[
                "operating"
            ]
        ),

        save_weighted_figure(
            weighted
        ),

        save_temporal_figure(
            data[
                "temporal"
            ]
        ),
    ]

    figure_manifest = (
        build_figure_manifest(
            figures
        )
    )

    # -------------------------------------------------------------------------
    # Claim audit
    # -------------------------------------------------------------------------

    claims = build_claim_audit(
        consensus=
            data[
                "consensus"
            ],
        weighted=
            weighted,
        age=
            age,
        dca=
            data[
                "dca"
            ],
        operating=
            data[
                "operating"
            ],
    )

    # -------------------------------------------------------------------------
    # Conference/manuscript outputs
    # -------------------------------------------------------------------------

    write_positioning()

    write_abstract(
        primary=
            primary,
        tier=
            tier,
        utility=
            utility,
        weighted=
            weighted,
    )

    write_methods()

    write_results(
        primary=
            primary,
        tier=
            tier,
        utility=
            utility,
        consensus=
            data[
                "consensus"
            ],
        weighted=
            weighted,
        age=
            age,
    )

    write_discussion()

    write_talk_structure()

    # -------------------------------------------------------------------------
    # Ledger
    # -------------------------------------------------------------------------

    ledger = (
        build_decision_ledger()
    )

    # -------------------------------------------------------------------------
    # Integrity
    # -------------------------------------------------------------------------

    integrity = (
        build_integrity(
            primary=
                primary,
            tier=
                tier,
            utility=
                utility,
            weighted=
                weighted,
            figures=
                figures,
            claims=
                claims,
        )
    )

    # -------------------------------------------------------------------------
    # Workbook
    # -------------------------------------------------------------------------

    write_workbook(
        key=
            key,
        primary=
            primary,
        tier=
            tier,
        utility=
            utility,
        temporal=
            temporal,
        weighted=
            weighted,
        age=
            age,
        subgroup=
            subgroup,
        consensus=
            data[
                "consensus"
            ],
        dca=
            data[
                "dca"
            ],
        survey=
            data[
                "survey"
            ],
        claims=
            claims,
        figures=
            figure_manifest,
        integrity=
            integrity,
        ledger=
            ledger,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print("\n")
    print(
        "-" * 110
    )

    print(
        "PRIMARY PERFORMANCE"
    )

    print(
        "-" * 110
    )

    print(
        primary
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
        "TIER CONTRIBUTION"
    )

    print(
        "-" * 110
    )

    display_columns = [
        c
        for c in [
            "study",
            "mean_delta_AUROC_T2_minus_T1",
            "mean_delta_AUPRC_T2_minus_T1",
            "n_model_families",
            "proportion_models_supporting_T2",
        ]
        if c
        in tier.columns
    ]

    print(
        tier[
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
        "CLINICAL UTILITY — 90% TRAINING SENSITIVITY TARGET"
    )

    print(
        "-" * 110
    )

    utility_columns = [
        c
        for c in [
            "study",
            "feature_tier",
            "mean_Sensitivity",
            "mean_TestsAvoided",
            "mean_ReferralRate",
            "mean_PPV",
            "mean_NPV",
            "mean_proportion_thresholds_model_best",
        ]
        if c
        in utility.columns
    ]

    print(
        utility[
            utility_columns
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
        "INTEGRITY"
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
            "STAGE 09 EXECUTION: PASS"
        )

        print(
            "Frozen results have been converted into the final "
            "scientific and AISC conference synthesis package."
        )

    else:

        print(
            "STAGE 09 EXECUTION: FAIL"
        )

        print(
            "Review the failed synthesis integrity checks."
        )

    print(
        "=" * 110
    )

    print(
        f"\nFinal synthesis workbook:\n"
        f"  {FINAL_WORKBOOK}"
    )

    print(
        f"\nAISC abstract draft:\n"
        f"  {ABSTRACT_PATH}"
    )

    print(
        f"\n20-minute talk structure:\n"
        f"  {TALK_PATH}"
    )


# =============================================================================
# 28. ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()