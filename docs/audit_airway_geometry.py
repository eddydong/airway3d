"""Reproduce synthetic airway geometry audit probes; print JSON only.

Run from the repository root:
    .venv/bin/python docs/audit_airway_geometry.py

These are implementation probes, not clinical accuracy estimates. They use no
patient inputs, do not regenerate datasets, and do not write output files.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import numpy as np
from scipy import ndimage as ndi

from airway import profile
from mesh import mask_to_mesh


def cylinder_probe():
    """Straight circular tube; supplied distances equal axial inlet distance."""
    spacing = np.array([0.5, 0.5, 0.5])
    shape = (100, 45, 45)
    y, x = np.mgrid[:shape[1], :shape[2]]
    mask = np.broadcast_to((x - 22) ** 2 + (y - 22) ** 2 <= 10 ** 2, shape).copy()
    distance = np.broadcast_to(np.arange(shape[0])[:, None, None] * spacing[0], shape)
    measured = profile(mask, distance, spacing, np.zeros(3), shape[0])
    interior = slice(8, -8)

    def median(name):
        return float(np.median(np.array(measured[name])[interior]))

    perimeter = median("perimeter_mm")
    return {
        "scope": "Synthetic radius-5-mm cylinder, 0.5-mm isotropic mask; interior stations exclude end caps.",
        "analytic_area_mm2": float(np.pi * 25),
        "measured_area_mm2": median("area_mm2"),
        "analytic_perimeter_mm": float(2 * np.pi * 5),
        "measured_perimeter_mm": perimeter,
        "perimeter_relative_error": float(perimeter / (2 * np.pi * 5) - 1),
        "analytic_hydraulic_diameter_mm": 10.0,
        "measured_hydraulic_diameter_mm": median("hyd_diam_mm"),
    }


def stenosis_probe():
    """Isolate profile smoothing with known station areas and prescribed distances.

    The distance field is axial station distance, not a computed geodesic through
    this constriction. This intentionally isolates the production profile's
    smoothing stage from upstream distance-estimation error.
    """
    mask = np.zeros((60, 14, 14), dtype=bool)
    mask[:, 2:12, 2:12] = True
    mask[30] = False
    mask[30, 2:4, 2:12] = True
    distance = np.broadcast_to(np.arange(mask.shape[0])[:, None, None], mask.shape)
    measured = profile(mask, distance, (1.0, 1.0, 1.0), np.zeros(3), mask.shape[0])
    return {
        "scope": "Synthetic one-mm station constriction; prescribed axial distances isolate production profile smoothing.",
        "baseline_area_mm2": 100.0,
        "unsmoothed_minimum_area_mm2": float(mask.sum(axis=(1, 2)).min()),
        "smoothed_minimum_area_mm2": float(min(measured["area_mm2"])),
        "constriction_length_mm": 1.0,
    }


def lumen_probe():
    """Two source lumens, one a single voxel wide; apply production display settings."""
    mask = np.zeros((30, 35, 35), dtype=bool)
    mask[:, 6:15, 6:15] = True
    mask[:, 25, 25] = True
    sigma = (0.5, 0.9, 0.9)
    field = ndi.gaussian_filter(mask.astype(np.float32), sigma)
    # A high face target disables decimation, isolating Gaussian isosurfacing.
    mesh = mask_to_mesh(mask, (0.625, 0.2774, 0.2774), mask.shape,
                        np.zeros(3), (0.35, 0.45, 0.45), 1_000_000, field="sdf")
    return {
        "scope": "Synthetic parallel lumens; one source lumen is one voxel wide. Display SDF is production airway meshing; no decimation.",
        "source_components": int(ndi.label(mask)[1]),
        "smoothed_field_components_at_half": int(ndi.label(field >= 0.5)[1]),
        "small_lumen_peak_scalar": float(field[:, 25, 25].max()),
        "mesh_components": len(mesh.split()) if mesh is not None else 0,
        "mesh_watertight": bool(mesh.is_watertight),
    }


def main():
    # Preserve a JSON-only stdout if a production function emits diagnostic logs.
    diagnostics = io.StringIO()
    with contextlib.redirect_stdout(diagnostics):
        results = {
            "evidence_type": "synthetic implementation probes; not clinical accuracy or patient validation",
            "production_functions": ["pipeline.airway.profile", "pipeline.mesh.mask_to_mesh"],
            "cylinder": cylinder_probe(),
            "short_stenosis": stenosis_probe(),
            "parallel_lumens": lumen_probe(),
        }
    if diagnostics.getvalue():
        results["production_diagnostics"] = diagnostics.getvalue().splitlines()
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
