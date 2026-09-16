"""Regression tests for this patient's airway geometry path.

These fixtures are synthetic. They lock the failure modes the audit reproduced
on empty morphology, short stenoses, thin lumens, and fine-grid recovery.
They are not a clinical accuracy score.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from airway import profile  # noqa: E402
from mesh import mask_to_mesh  # noqa: E402
from segment import (  # noqa: E402
    dilate_mm,
    open_mm,
    refine_conducting_airway,
)


class EmptyMorphologyTests(unittest.TestCase):
    def test_dilate_and_open_of_empty_stay_empty(self):
        empty = np.zeros((9, 9, 9), dtype=bool)
        sp = (0.5, 0.5, 0.5)
        self.assertFalse(dilate_mm(empty, 1.0, sp).any())
        self.assertFalse(open_mm(empty, 1.6, sp).any())


class ProfileTests(unittest.TestCase):
    def test_one_millimetre_stenosis_is_not_smoothed_away(self):
        mask = np.zeros((60, 14, 14), dtype=bool)
        mask[:, 2:12, 2:12] = True
        mask[30] = False
        mask[30, 2:4, 2:12] = True
        distance = np.broadcast_to(np.arange(mask.shape[0])[:, None, None], mask.shape)
        measured = profile(mask, distance, (1.0, 1.0, 1.0), np.zeros(3), mask.shape[0])
        self.assertLess(min(measured["area_mm2"]), 30.0)
        self.assertAlmostEqual(min(measured["area_mm2"]), 20.0, delta=2.0)


class MeshTests(unittest.TestCase):
    def test_sdf_airway_mesh_keeps_a_one_voxel_lumen(self):
        mask = np.zeros((30, 35, 35), dtype=bool)
        mask[:, 6:15, 6:15] = True
        mask[:, 25, 25] = True
        mesh = mask_to_mesh(mask, (0.625, 0.2774, 0.2774), mask.shape,
                            np.zeros(3), (0.35, 0.45, 0.45), 1_000_000, field="sdf")
        self.assertGreaterEqual(len(mesh.split()), 2)


class FineRecoveryTests(unittest.TestCase):
    def test_keeps_both_passages_and_excludes_a_sinus_bulb(self):
        # Fine grid 0.5 mm; coarse is 1 mm in-plane (factor 2). z matches.
        gray = np.full((8, 24, 40), 80, np.float32)
        # Two conducting tubes (left/right). The right tube is smaller.
        gray[:, 8:16, 6:12] = 8
        gray[:, 8:14, 22:26] = 10
        # Partial-volume wall voxels along the small tube (gray 25).
        gray[:, 8:14, 26] = 25
        # A wide sinus bulb, linked by a 1-voxel ostium already labeled sinus.
        gray[:, 8:18, 30:38] = 6
        gray[:, 11:13, 26:31] = 7

        coarse_airway = np.zeros((8, 12, 20), dtype=bool)
        coarse_airway[:, 4:8, 3:6] = True
        coarse_airway[:, 4:7, 11:13] = True
        coarse_sinus = np.zeros_like(coarse_airway)
        coarse_sinus[:, 4:9, 15:19] = True
        coarse_sinus[:, 5:7, 13:16] = True  # ostium

        sp_f, sp_c = (1.0, 0.5, 0.5), (1.0, 1.0, 1.0)
        out = refine_conducting_airway(gray, coarse_airway, coarse_sinus, sp_f, sp_c)
        self.assertTrue(out[:, 8:16, 6:12].any(), "large passage kept")
        self.assertTrue(out[:, 8:14, 22:26].any(), "small passage kept")
        self.assertTrue(out[:, 8:14, 26].any(), "partial-volume wall recovered")
        self.assertFalse(out[:, 8:18, 32:38].any(), "sinus bulb stays out")
        self.assertEqual(int(ndi.label(out)[1]), 2)


if __name__ == "__main__":
    unittest.main()
