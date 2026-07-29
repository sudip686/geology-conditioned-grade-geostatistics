from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nrr_study.information_sensitivity import (
    both_schemes_support,
    farthest_point_order,
    parent_sample_set,
    reconcile_selected_master,
    refresh_analysis_freeze,
)


class InformationSensitivityTests(unittest.TestCase):
    def test_selected_composite_requires_all_nonempty_parents(self) -> None:
        master = pd.DataFrame(
            {
                "sample_entity_key": ["S1", "S2", "S3"],
                "sample_id": ["P1", "P2", "P3"],
                "bhid": ["H1", "H1", "H2"],
                "assay_sample_id": ["A", "B", None],
                "from_m": [1.0, 2.0, 3.0],
                "to_m": [1.1, 2.1, 3.1],
            }
        )
        composites = pd.DataFrame(
            {
                "BHID": ["H1"] * 4,
                "parent_sample_ids": ["A", "A|B", "A|C", ""],
            }
        )
        summary, duplicates, flagged = reconcile_selected_master(master, composites)
        self.assertEqual(summary.at[0, "unique_exact_assay_ids"], 2)
        self.assertTrue(duplicates.empty)
        self.assertEqual(
            flagged["selected_endpoint"].tolist(), [True, True, False, False]
        )

    def test_parent_parser_drops_empty_tokens(self) -> None:
        self.assertEqual(parent_sample_set("A|| B |"), frozenset({"A", "B"}))
        self.assertFalse(parent_sample_set(None))

    def test_farthest_order_is_deterministic_and_complete(self) -> None:
        holes = pd.DataFrame(
            {
                "BHID": ["H3", "H1", "H4", "H2"],
                "collar_easting": [0.0, 0.0, 10.0, 10.0],
                "collar_northing": [0.0, 10.0, 10.0, 0.0],
            }
        )
        first = farthest_point_order(holes)
        second = farthest_point_order(holes.sample(frac=1, random_state=2))
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(holes["BHID"]))

    def test_both_primary_schemes_are_required(self) -> None:
        one = pd.DataFrame(
            {
                "scheme": ["leave_one_hole_out"],
                "upper": [-0.1],
            }
        )
        both = pd.DataFrame(
            {
                "scheme": [
                    "leave_one_hole_out",
                    "northing_block_buffered",
                ],
                "upper": [-0.1, -0.01],
            }
        )
        self.assertFalse(both_schemes_support(one))
        self.assertTrue(both_schemes_support(both))

    def test_freeze_refresh_rejects_undeclared_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "data.csv"
            target.write_text("a\n1\n", encoding="utf-8")
            freeze = root / "freeze.csv"
            pd.DataFrame(
                [
                    {
                        "path": "data.csv",
                        "sha256": "0" * 64,
                        "size_bytes": 0,
                        "frozen_before_analysis": True,
                    }
                ]
            ).to_csv(freeze, index=False)
            with self.assertRaisesRegex(ValueError, "unexpected changed"):
                refresh_analysis_freeze(
                    freeze, root, metadata_only_paths=()
                )


if __name__ == "__main__":
    unittest.main()
