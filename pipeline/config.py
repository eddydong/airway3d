"""Shared configuration for the airway3d reconstruction pipeline.

Geometry
--------
The source images are 1235x1498 screenshots of the hospital viewer at 100% zoom
(bone window C 400 / W 1500, series "Sinus Bone 0.625mm").  The in-plane scale
is calibrated from the patient's measured inter-pupillary distance (68 mm): the
fitted centres of the two globes are 490.4 source px apart in-plane (1.5 mm
apart in z), so PX_MM = sqrt(68^2 - 1.5^2) / 490.4 = 0.1387 mm/px.  With that
the circular reconstruction FOV (1531 px) is 212 mm and the head is 164 mm wide
at the supraorbital level.  Override with AIRWAY3D_PX_MM when a ruler
measurement or a DICOM header becomes available.

HU mapping
----------
For a linear window with centre C and width W the gray value g (0..255) maps to
HU = (g / 255) * W + (C - W / 2)  ->  HU = g * 5.882 - 350.  Values below
-350 HU (all air) saturate to 0 and above 1150 HU (dense cortical bone, enamel)
saturate to 255.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLICES_DIR = ROOT / "slices" / "skull"
WORK_DIR = ROOT / "work"
DATA_DIR = ROOT / "viewer" / "data"

IPD_MM = 68.0  # patient's inter-pupillary distance (given)
PX_MM = float(os.environ.get("AIRWAY3D_PX_MM", 0.1387))  # in-plane mm per screenshot pixel
SLICE_MM = 0.625  # slice spacing (from viewer header "切片厚度 0.63 毫米")

WINDOW_C = 400.0   # bone window of slices/skull
WINDOW_W = 1500.0

# Soft-tissue series (slices/head): C 40 / W 350, smooth kernel.  It is a separate
# reconstruction displayed 0.37 % smaller and shifted; fitted on saturated-bone masks:
#   head_px = HEAD_S * skull_px + (HEAD_TX, HEAD_TY)
SOFT_SLICES_DIR = ROOT / "slices" / "head"
SOFT_WINDOW_C = 40.0
SOFT_WINDOW_W = 350.0
HEAD_S, HEAD_TX, HEAD_TY = 0.9963, 6.23, -3.05


def soft_gray_to_hu(g):
    return g * (SOFT_WINDOW_W / 255.0) + (SOFT_WINDOW_C - SOFT_WINDOW_W / 2.0)


def hu_to_soft_gray(hu):
    return (hu - (SOFT_WINDOW_C - SOFT_WINDOW_W / 2.0)) * (255.0 / SOFT_WINDOW_W)

# Working grids: in-plane downsample factors relative to the screenshots.
FINE_DS = 2    # 0.277 mm in-plane, used for bone / airway
COARSE_DS = 4  # 0.555 mm in-plane, used for tissues / web volume
SRC_W, SRC_H = 1235, 1498  # source screenshot size (px)


def gray_to_hu(g):
    return g * (WINDOW_W / 255.0) + (WINDOW_C - WINDOW_W / 2.0)


def hu_to_gray(hu):
    return (hu - (WINDOW_C - WINDOW_W / 2.0)) * (255.0 / WINDOW_W)


# Tissue label table (values stored in the label volumes).  Keep in sync with
# viewer/app.js (it reads the table from meta.json, so only this file matters).
LABELS = {
    0: dict(key="background", name="Background", color="#000000", opacity=0.0),
    1: dict(key="skin", name="Skin", color="#e8b89a", opacity=0.35),
    2: dict(key="fat", name="Fat", color="#f2d16b", opacity=0.6),
    3: dict(key="muscle", name="Muscle", color="#b5493f", opacity=0.7),
    10: dict(key="soft", name="Soft tissue (glands, mucosa, connective)", color="#d9907c", opacity=0.6),
    4: dict(key="bone", name="Bone", color="#f3ecd8", opacity=1.0),
    5: dict(key="brain", name="Brain / neural", color="#e2a3b3", opacity=0.9),
    6: dict(key="eye", name="Eyeballs", color="#eef4ff", opacity=1.0),
    7: dict(key="airway", name="Airway (nose → pharynx)", color="#33c5ff", opacity=0.9),
    8: dict(key="sinus", name="Sinuses / air cells", color="#7fe0c2", opacity=0.5),
    9: dict(key="teeth", name="Teeth", color="#ffffff", opacity=1.0),
}
