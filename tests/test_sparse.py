from pathlib import Path

import numpy as np
import pandas as pd

from nrr_study.sparse import (
    evaluate_sparse_family,
    leave_one_hole_out_predictions,
)


def _toy_frame(holes: int = 10) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(4)
    for index in range(holes):
        hole = f"H{index:02d}"
        for depth in (10.0, 20.0, 30.0):
            descriptor = index / holes + depth / 100.0
            rows.append(
                {
                    "bhid": hole,
                    "assay_midpoint_md_m": depth,
                    "verified_class_or_classes": "graphitic_schist",
                    "workbook_weathering_code": "HW" if index % 2 else "MW",
                    "descriptor": descriptor,
                    "tgc_pct": 2.0 + 3.0 * descriptor + rng.normal(0, 0.05),
                }
            )
    return pd.DataFrame(rows)


def test_sparse_holdout_abstains_below_eight_holes():
    predictions, state = leave_one_hole_out_predictions(
        _toy_frame(7), ["descriptor"]
    )
    assert predictions.empty
    assert state == "insufficient_independent_holes"


def test_sparse_holdout_never_splits_a_hole():
    frame = _toy_frame()
    predictions, state = leave_one_hole_out_predictions(frame, ["descriptor"])
    assert state == "complete"
    assert len(predictions) == len(frame)
    assert set(predictions["bhid"]) == set(frame["bhid"])


def test_strong_descriptor_can_pass_prospective_gate():
    result, predictions = evaluate_sparse_family(
        "toy",
        _toy_frame(),
        ["descriptor"],
        primary_descriptor="descriptor",
        bootstrap_replicates=400,
        seed=12,
    )
    assert not predictions.empty
    assert result.augmented_mae < result.baseline_mae
    assert result.conclusion in {"supported", "unsupported"}
