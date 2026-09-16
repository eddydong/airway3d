"""Run small, patient-free probes of the current airway separation assumptions.

Usage from the repository root:
    .venv/bin/python docs/audit_airway_separation.py

This audit writes JSON to stdout only. It does not load patient volumes or modify
pipeline output. The examples demonstrate algorithmic failure modes; they do not
measure the accuracy of an actual patient segmentation.

Production morphology and component helpers are imported directly. The two
inline stages below replay the current logic in pipeline/segment.py because that
logic is embedded in main(). A source hash identifies the production version
against which the replay was run. These probes are not production segmentation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.segment import dilate_mm, largest_cc, open_mm  # noqa: E402


def replay_sinus_split(airway_all, sp=(1.0, 1.0, 1.0)):
    """Replay segment.py's opening, bottom-root and radius classification stage.

    Corresponds to the stage beginning ``opened = open_mm(airway_all, 1.6,
    sp)`` and ending with assignment of ``sinus_from_airway``. Preserve its
    thresholds and Euclidean nearest-core assignment exactly for this audit.
    """
    opened = open_mm(airway_all, 1.6, sp)
    lab_core, k = ndi.label(opened)
    airway = airway_all.copy()
    bottom_cores = set()
    if k > 1:
        wide = open_mm(airway_all, 1.3, sp)
        lab_wide, _ = ndi.label(wide)
        edt_all = ndi.distance_transform_edt(airway_all, sampling=sp)
        bottom_cores = set(np.unique(lab_core[-1])) - {0}
        wide_ids_airway = set(
            np.unique(lab_wide[np.isin(lab_core, list(bottom_cores))])
        ) - {0}
        core_ids = np.arange(1, k + 1)
        max_r = ndi.maximum(edt_all, lab_core, core_ids)
        wide_of_core = ndi.maximum(lab_wide, lab_core, core_ids)
        is_airway = np.ones(k + 1, bool)
        for cid, r, w in zip(core_ids, max_r, wide_of_core):
            if cid in bottom_cores or w in wide_ids_airway:
                continue
            if r > 4.2:
                is_airway[cid] = False
        if not is_airway.all():
            idx = ndi.distance_transform_edt(
                lab_core == 0,
                sampling=sp,
                return_distances=False,
                return_indices=True,
            )
            assigned = lab_core[idx[0], idx[1], idx[2]]
            airway = airway_all & is_airway[assigned]
    return {
        "input_voxels": int(airway_all.sum()),
        "opened_core_count": int(k),
        "bottom_root_ids": sorted(int(i) for i in bottom_cores),
        "airway_voxels": int(airway.sum()),
    }


def replay_exterior_seed(air, head_sealed, depth):
    """Replay segment.py from ``ext_air`` through ``airway_all`` assignment."""
    ext_air = air & ~head_sealed
    lab, _ = ndi.label(air)
    outside_ids = np.unique(lab[ext_air])
    connected_air = np.isin(lab, outside_ids[outside_ids > 0]) & head_sealed
    core = connected_air & (depth > 12.0)
    core, _ = largest_cc(core)
    grow_region = connected_air & (depth > 2.5)
    lab, _ = ndi.label(grow_region)
    ids = np.unique(lab[core])
    return np.isin(lab, ids[ids > 0])


def main():
    # A wide tube reaches the final z face. A second bulb is linked to the tube
    # by a narrow passage. Reversing slice order preserves physical geometry.
    z, y, x = np.indices((50, 42, 60))
    tube = ((x - 15) ** 2 + (y - 20) ** 2 <= 25) & (z >= 10)
    bulb = (x - 38) ** 2 + (y - 20) ** 2 + (z - 15) ** 2 <= 36
    joined = tube | bulb
    joined[15, 20, 15:39] = True

    # A scan cropped within tissue can contain a lumen reaching both z faces,
    # with no external-to-head air visible in the image field of view.
    cropped_lumen = np.zeros((24, 24, 24), bool)
    cropped_lumen[:, 10:14, 10:14] = True
    head_sealed = np.ones_like(cropped_lumen)
    depth = np.full(cropped_lumen.shape, 15.0)
    cropped_result = replay_exterior_seed(cropped_lumen, head_sealed, depth)

    # Two observed passages need not connect inside the acquired scan coverage.
    bilateral = np.zeros((12, 12, 24), bool)
    bilateral[:, 2:8, 2:8] = True
    bilateral[:, 2:7, 15:20] = True
    largest, _ = largest_cc(bilateral)

    # EDT has no in-array background site for ~empty; unchecked dilation can
    # consequently manufacture foreground near an array corner.
    empty = np.zeros((9, 9, 9), bool)
    empty_dilated = dilate_mm(empty, 1.0, (0.5, 0.5, 0.5))
    empty_opened = open_mm(empty, 1.6, (0.5, 0.5, 0.5))

    results = {
        "scope": "Synthetic failure-mode probes; no patient accuracy estimate",
        "segment_source_sha256": hashlib.sha256(
            (ROOT / "pipeline" / "segment.py").read_bytes()
        ).hexdigest(),
        "sinus_split_slice_order": {
            "original": replay_sinus_split(joined),
            "z_reversed": replay_sinus_split(joined[::-1]),
        },
        "no_visible_exterior_air": {
            "input_lumen_voxels": int(cropped_lumen.sum()),
            "lumen_touches_both_z_faces": bool(
                cropped_lumen[0].any() and cropped_lumen[-1].any()
            ),
            "airway_voxels": int(cropped_result.sum()),
        },
        "largest_component_only": {
            "input_voxels": int(bilateral.sum()),
            "kept_voxels": int(largest.sum()),
            "discarded_voxels": int((bilateral & ~largest).sum()),
        },
        "empty_morphology": {
            "input_voxels": int(empty.sum()),
            "dilated_voxels": int(empty_dilated.sum()),
            "opened_voxels": int(empty_opened.sum()),
        },
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
