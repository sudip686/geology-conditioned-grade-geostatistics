import numpy as np
import pandas as pd
import pytest

from nrr_study.geology_stats import (
    assign_signed_contact_distances,
    build_adjacent_contact_registry,
    build_rq_evidence_table,
    decompose_depth_by_hole,
    grouped_hole_bootstrap_stats,
    summarize_contact_profiles,
    summarize_support_sensitivities,
)


def test_depth_decomposition_separates_within_and_between_hole_terms():
    data = pd.DataFrame(
        {
            "BHID": ["A", "A", "B", "B", "B"],
            "MIDPOINT": [1.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    result = decompose_depth_by_hole(data)
    assert np.allclose(
        result.groupby("BHID")["depth_within_hole"].sum().to_numpy(),
        0.0,
    )
    assert result.loc[result["BHID"] == "A", "depth_between_hole"].eq(2.0).all()
    assert result.loc[result["BHID"] == "B", "depth_between_hole"].eq(20.0).all()


def test_support_censor_cap_and_declustering_summary_is_deterministic():
    data = pd.DataFrame(
        {
            "BHID": ["A", "A", "B", "C"],
            "tgc_pct": [0.025, 2.0, 4.0, 20.0],
            "support_m": [1.0, 1.0, 2.0, 2.0],
            "censored_support_fraction": [1.0, 0.0, 0.0, 0.0],
            "mid_easting": [0.0, 1.0, 1000.0, 1001.0],
            "mid_northing": [0.0, 1.0, 1000.0, 1001.0],
        }
    )
    first = summarize_support_sensitivities(
        {"2m": data},
        cell_sizes_m=(100.0,),
        grid_origin_fractions=((0.0, 0.0),),
    )
    second = summarize_support_sensitivities(
        {"2m": data},
        cell_sizes_m=(100.0,),
        grid_origin_fractions=((0.0, 0.0),),
    )
    pd.testing.assert_frame_equal(first, second)
    base = first[
        (first["scenario"] == "base")
        & (first["estimator"] == "length_weighted")
    ].iloc[0]
    censor_zero = first[
        (first["scenario"] == "censor_0")
        & (first["estimator"] == "length_weighted")
    ].iloc[0]
    capped = first[
        (first["scenario"] == "cap_p99")
        & (first["estimator"] == "length_weighted")
    ].iloc[0]
    assert censor_zero["mean_tgc_pct"] < base["mean_tgc_pct"]
    assert capped["mean_tgc_pct"] < base["mean_tgc_pct"]
    assert set(first["estimator"]) == {
        "length_weighted",
        "equal_hole",
        "equal_cell",
    }


def test_grouped_stats_bootstrap_holes_not_intervals():
    data = pd.DataFrame(
        {
            "BHID": ["A", "A", "B", "C", "C"],
            "canonical_lithology": [
                "qfr",
                "qfr",
                "qfr",
                "graphitic_schist",
                "graphitic_schist",
            ],
            "weathering": ["HW", "HW", "SW", "HW", "HW"],
            "tgc_pct": [1.0, 3.0, 5.0, 10.0, 14.0],
            "support_m": [1.0, 1.0, 2.0, 1.0, 3.0],
        }
    )
    result = grouped_hole_bootstrap_stats(
        data,
        group_cols=["canonical_lithology", "weathering"],
        n_boot=100,
        seed=7,
    )
    qfr_hw = result[
        (result["canonical_lithology"] == "qfr")
        & (result["weathering"] == "HW")
    ].iloc[0]
    assert qfr_hw["n_records"] == 2
    assert qfr_hw["n_holes"] == 1
    assert qfr_hw["equal_hole_mean_tgc_pct"] == pytest.approx(2.0)
    repeat = grouped_hole_bootstrap_stats(
        data,
        group_cols=["canonical_lithology", "weathering"],
        n_boot=100,
        seed=7,
    )
    pd.testing.assert_frame_equal(result, repeat)


def test_adjacent_contacts_signed_distances_and_profiles_preserve_geology():
    geology = pd.DataFrame(
        {
            "BHID": ["A", "A", "A", "A", "B", "B"],
            "FROM": [0.0, 5.0, 10.0, 15.5, 0.0, 5.2],
            "TO": [5.0, 10.0, 15.0, 20.0, 5.0, 10.0],
            "canonical_lithology": [
                "qfr",
                "graphitic_schist",
                "other",
                "qfr",
                "graphitic_schist",
                "qfr",
            ],
            "geology_interval_id": ["g1", "g2", "g3", "g4", "g5", "g6"],
            "interval_key_difference": [False] * 6,
            "lithology_difference": [False, True, False, False, False, False],
        }
    )
    contacts = build_adjacent_contact_registry(geology)
    assert len(contacts) == 1
    contact = contacts.iloc[0]
    assert contact["contact_depth"] == pytest.approx(5.0)
    assert contact["graphitic_position"] == "deep"
    assert not bool(contact["source_version_stable"])
    assert contact["boundary_policy"] == "untested"

    composites = pd.DataFrame(
        {
            "composite_id": ["c1", "c2", "c3", "c4"],
            "BHID": ["A", "A", "A", "B"],
            "MIDPOINT": [4.0, 7.0, 17.0, 2.0],
            "canonical_lithology": [
                "qfr",
                "graphitic_schist",
                "qfr",
                "graphitic_schist",
            ],
            "tgc_pct": [1.0, 8.0, 2.0, 9.0],
            "support_m": [2.0, 2.0, 2.0, 2.0],
        }
    )
    assigned = assign_signed_contact_distances(composites, contacts)
    distances = assigned.set_index("composite_id")[
        "signed_alonghole_contact_distance_m"
    ]
    assert distances["c1"] == pytest.approx(-1.0)
    assert distances["c2"] == pytest.approx(2.0)
    assert np.isnan(distances["c3"])
    assert np.isnan(distances["c4"])

    profiles = summarize_contact_profiles(
        assigned,
        windows_m=(1.0, 2.0, 5.0, 10.0),
        n_boot=20,
        min_independent_holes=8,
    )
    assert len(profiles) == 8
    qfr_one = profiles[
        (profiles["window_m"] == 1.0) & (profiles["side"] == "qfr")
    ].iloc[0]
    graphitic_one = profiles[
        (profiles["window_m"] == 1.0)
        & (profiles["side"] == "graphitic_schist")
    ].iloc[0]
    assert qfr_one["n_composites"] == 1
    assert graphitic_one["n_composites"] == 0
    assert qfr_one["evidence_status"] == "insufficient evidence/abstain"
    assert "not perpendicular distance" in qfr_one["distance_limitation"]


def test_rq_evidence_table_restricts_conclusions():
    table = build_rq_evidence_table(
        [
            {
                "rq": "RQ1",
                "question": "Does support change the result?",
                "evidence": "Native and 2 m sensitivity",
                "result": "supported",
                "limitation": "Descriptive association",
            },
            {
                "rq": "RQ2",
                "question": "Is the boundary hard?",
                "evidence": "Too few independent holes",
                "result": "abstain",
                "limitation": "Along-hole distance only",
            },
        ]
    )
    assert table["result"].tolist() == [
        "supported",
        "insufficient evidence/abstain",
    ]
    with pytest.raises(ValueError):
        build_rq_evidence_table(
            [
                {
                    "rq": "RQX",
                    "question": "Unsupported inference?",
                    "evidence": "None",
                    "result": "proven genesis",
                    "limitation": "None",
                }
            ]
        )
