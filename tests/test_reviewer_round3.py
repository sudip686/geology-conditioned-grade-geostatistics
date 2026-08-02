from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nrr_study.reviewer_revision import (
    PairScoreDesign,
    vectorized_pair_universe_score_sensitivities,
)
from nrr_study.reviewer_round3 import (
    public_geology_group_summary,
    residual_distribution_diagnostics,
    version_flag_exclusion_sensitivities,
)


STUDY = Path(__file__).resolve().parents[1]
EMPIRICAL_STUDY_AVAILABLE = (
    STUDY / "derived" / "analysis" / "primary_analysis_cohort.csv"
).is_file()
PRIVATE_CONFIG_AVAILABLE = (STUDY / "config" / "study_config.json").is_file()


@pytest.mark.skipif(
    not EMPIRICAL_STUDY_AVAILABLE,
    reason="requires the restricted empirical study workspace",
)
def test_public_geology_group_summary_covers_primary_cohort() -> None:
    primary = pd.read_csv(
        STUDY / "derived" / "analysis" / "primary_analysis_cohort.csv"
    )
    hierarchy = pd.read_csv(
        STUDY
        / "derived"
        / "post_analysis"
        / "geology_public_hierarchy_support.csv"
    )
    summary = public_geology_group_summary(primary, hierarchy)
    assert len(summary) == 7
    assert int(summary["rows"].sum()) == 3542
    assert summary["public_grouping"].tolist() == hierarchy[
        "public_grouping"
    ].tolist()


def _flag_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(20):
        hole = f"H{index:02d}"
        lithology = "graphitic_schist" if index % 2 == 0 else "qfr"
        subtype = "grsc1" if lithology == "graphitic_schist" else "not_graphitic"
        for offset in (0, 1):
            easting = float(index * 10)
            northing = float(index * 2 + offset)
            rows.append(
                {
                    "BHID": hole,
                    "mid_easting": easting,
                    "mid_northing": northing,
                    "mid_rl": float(100 - offset),
                    "tgc_pct": (
                        2.0
                        + 0.01 * easting
                        + (1.0 if lithology == "graphitic_schist" else 0.0)
                    ),
                    "support_m": 2.0,
                    "canonical_lithology": lithology,
                    "grsc_subtype": subtype,
                    "weathering": "fresh",
                    "depth_within_hole_m": float(offset - 0.5),
                    "hole_mean_depth_m": 1.0,
                    "northing_block": int(index // 4 + 1),
                    "spatial_buffer_m": 0.1,
                    "source_lithology_difference": (
                        index == 0 and offset == 0
                    ),
                    "source_weathering_difference": (
                        index == 1
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_pair_universe_variants_use_declared_denominators() -> None:
    design = PairScoreDesign(
        first=np.asarray([0, 0, 3, 3, 0]),
        second=np.asarray([1, 2, 4, 5, 6]),
        inverse=np.asarray([0, 0, 1, 1, 2]),
        overall_counts=np.asarray([2.0, 2.0, 1.0]),
        short_mask=np.asarray([True, False, True, False, False]),
        short_counts=np.asarray([1.0, 1.0, 0.0]),
        pair_count_requested=5,
        pair_count_used=5,
        eligible_hole_pairs=3,
        short_lag_quantile=0.2,
        random_state=17,
    )
    values = np.asarray(
        [0.0, 1.0, 2.0, 0.0, np.sqrt(8.0), np.sqrt(12.0), 10.0]
    )[:, None]
    scores = vectorized_pair_universe_score_sensitivities(values, design)
    np.testing.assert_allclose(
        scores["legacy_all_pair_denominator"], [1.0 - 2.25 / 18.75]
    )
    np.testing.assert_allclose(
        scores["same_supported_hole_pairs"], [1.0 - 2.25 / 3.125]
    )
    np.testing.assert_allclose(
        scores["within_hole_pair_ratio"], [1.0 - (0.4 + 0.8) / 2.0]
    )


def test_version_flag_exclusions_refit_paired_grouped_models() -> None:
    data = _flag_fixture()
    cohorts, predictions, metrics, evidence, tuning = (
        version_flag_exclusion_sensitivities(
            data,
            bootstraps=50,
            alphas=(1.0,),
            inner_grouped_folds=3,
        )
    )
    assert cohorts["removed_rows"].tolist() == [1, 2]
    assert cohorts["retained_holes"].tolist() == [20, 19]
    assert len(evidence) == 8
    assert evidence["joint_success_rate"].eq(1.0).all()
    assert set(evidence["comparison"]) == {
        "lithology_minus_global",
        "lithology_spatial_minus_coordinate",
    }
    assert set(metrics["exclusion"]) == set(cohorts["exclusion"])
    assert predictions["success"].astype(bool).all()
    assert tuning["inner_group_count"].eq(3).all()


@pytest.mark.skipif(
    not EMPIRICAL_STUDY_AVAILABLE,
    reason="requires the restricted empirical study workspace",
)
def test_residual_transform_is_diagnostic_not_automatic() -> None:
    primary = pd.read_csv(
        STUDY / "derived" / "analysis" / "primary_analysis_cohort.csv"
    )
    table, summary = residual_distribution_diagnostics(primary)
    assert int(table.loc[table["canonical_lithology"].eq("all"), "rows"].iloc[0]) == 3542
    assert summary["transformation_variogram_sensitivity_implemented"] is False
    assert summary["normal_score_sensitivity_implemented"] is False
    assert summary["changes_frozen_gate"] is False


@pytest.mark.skipif(
    not PRIVATE_CONFIG_AVAILABLE,
    reason="requires the restricted empirical study configuration",
)
def test_round3_config_does_not_replace_frozen_gate() -> None:
    config = json.loads(
        (STUDY / "config" / "study_config.json").read_text(encoding="utf-8")
    )
    round3 = config["reviewer_motivated_post_analysis"]["reviewer_round3"]
    assert round3["changes_frozen_covariance_gate"] is False
    assert round3["pair_universe_score"]["pair_count"] == 20_000
    assert round3["pair_universe_score"]["residual_standardization"] is False
    assert round3["source_version_flag_exclusions"]["inner_grouped_folds"] == 5
