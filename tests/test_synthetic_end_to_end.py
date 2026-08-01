from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _workflow_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "synthetic_end_to_end.py"
    spec = importlib.util.spec_from_file_location("synthetic_end_to_end", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_common_support_conserves_support_and_grade_mass():
    workflow = _workflow_module()
    parents = workflow.make_parent_intervals()
    children = workflow.composite_to_common_support(parents)
    audit = workflow.support_conservation(parents, children)
    assert audit.support_conserved.all()
    assert audit.grade_mass_conserved.all()


def test_parent_groups_never_cross_roles_within_a_validation_group():
    workflow = _workflow_module()
    parents = workflow.make_parent_intervals()
    children = workflow.composite_to_common_support(parents)
    registry = workflow.grouped_buffered_registry(children)
    assert registry.groupby(["validation_group", "parent_group"]).role.nunique().max() == 1
    assert registry.groupby(["validation_group", "synthetic_hole"]).role.nunique().max() == 1


def test_training_rows_respect_grade_blind_spatial_buffer():
    workflow = _workflow_module()
    children = workflow.composite_to_common_support(workflow.make_parent_intervals())
    registry = workflow.grouped_buffered_registry(children, buffer_m=225.0)
    training = registry.loc[registry.role == "training"]
    excluded = registry.loc[registry.role == "excluded_buffer"]
    assert not training.empty
    assert not excluded.empty
    assert (training.nearest_validation_distance_m > training.buffer_m).all()
    assert (excluded.nearest_validation_distance_m <= excluded.buffer_m).all()


def test_pair_weighting_gives_each_independent_hole_pair_equal_weight():
    workflow = _workflow_module()
    toy = pd.DataFrame(
        {
            "synthetic_hole": ["SYN_A", "SYN_B", "SYN_C", "SYN_C"],
            "grade_pct": [0.0, 2.0, 10.0, 10.0],
        }
    )
    audit = workflow.equal_hole_pair_semivariance(toy)
    assert audit["independent_hole_pairs"] == 3
    assert np.isclose(audit["equal_hole_pair_semivariance"], 28.0)
    assert np.isclose(audit["raw_row_pair_semivariance"], 33.2)


def test_end_to_end_workflow_writes_declared_outputs(tmp_path):
    workflow = _workflow_module()
    summary = workflow.run(tmp_path)
    assert summary["scope"] == "synthetic_only"
    assert summary["support_conservation_passed"] is True
    assert summary["parent_grouping_passed"] is True
    assert summary["buffering_passed"] is True
    assert {row["model"] for row in summary["model_comparison"]} == {
        "domain_mean", "global_mean"
    }
    assert (tmp_path / "synthetic_gate.json").is_file()
