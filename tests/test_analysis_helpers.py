from __future__ import annotations

import unittest

import pandas as pd

from nrr_study.analysis_helpers import (
    add_signed_contact_distance,
    directional_structure_support,
    prepare_primary_composites,
)


class AnalysisHelperTests(unittest.TestCase):
    def test_directional_structure_support_separates_s1_from_other_beta(self) -> None:
        geology = pd.DataFrame(
            {
                "BHID": ["H1", "H1", "H2", "H3", "H4"],
                "STRUCT-BETA (°)": [10, 20, 30, 40, None],
                "TYPE OF STRUCTURE\n(S0/S1/S2/L1/L2)": [
                    "S1",
                    "FR",
                    "S1 ",
                    "",
                    "S1",
                ],
            }
        )
        support = directional_structure_support(geology)
        self.assertEqual(support["beta_measurements"], 4)
        self.assertEqual(support["beta_holes"], 3)
        self.assertEqual(support["s1_beta_measurements"], 2)
        self.assertEqual(support["s1_beta_holes"], 2)

    def test_contact_distance_sign_is_lithology_defined(self) -> None:
        geology = pd.DataFrame(
            {
                "BHID": ["H1", "H1"],
                "FROM": [0.0, 10.0],
                "TO": [10.0, 20.0],
                "canonical_lithology": ["qfr", "graphitic_schist"],
            }
        )
        composites = pd.DataFrame(
            {
                "BHID": ["H1", "H1"],
                "MIDPOINT": [8.0, 12.0],
                "canonical_lithology": ["qfr", "graphitic_schist"],
            }
        )
        linked, contacts = add_signed_contact_distance(composites, geology)
        self.assertEqual(len(contacts), 1)
        self.assertEqual(linked["signed_contact_distance_m"].tolist(), [-2.0, 2.0])

    def test_fold_registry_must_be_grade_blind(self) -> None:
        composites = pd.DataFrame(
            {
                "composite_id": ["C1"],
                "BHID": ["H1"],
                "FROM": [0.0],
                "TO": [2.0],
                "MIDPOINT": [1.0],
                "support_m": [2.0],
                "support_complete": [True],
                "tgc_pct": [5.0],
                "parent_assay_ids": ["A1"],
                "canonical_lithology": ["qfr"],
                "grsc_subtype": ["not_graphitic_schist"],
                "weathering": ["fresh"],
                "BATCH_NUMBER": ["B1"],
                "primary_spatial_eligible": [True],
                "mid_easting": [1.0],
                "mid_northing": [2.0],
                "mid_rl": [3.0],
                "mid_tvd": [1.0],
            }
        )
        folds = pd.DataFrame(
            {
                "BHID": ["H1"],
                "northing_block_label": ["N1"],
                "spatial_buffer_m": [10.0],
                "loho_fold": ["H1"],
                "grade_used": [True],
            }
        )
        with self.assertRaisesRegex(ValueError, "not grade blind"):
            prepare_primary_composites(composites, folds)


if __name__ == "__main__":
    unittest.main()
