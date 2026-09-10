"""Stack the de-labelled slice PNGs into 3D gray volumes at two resolutions.

Outputs (work/):
  vol_fine.npy    uint8 (nz, ny, nx)  in-plane FINE_DS   downsample (box filter)
  vol_coarse.npy  uint8 (nz, ny, nx)  in-plane COARSE_DS downsample
  volume_meta.json  spacing / origin / positions
"""
from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C  # noqa: E402


def read_index():
    rows = []
    with open(C.SLICES_DIR / "index.csv") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["index"]), r["file"], float(r["position_mm"])))
    rows.sort()
    return rows


def fit_fov_circle(vol, ds):
    """The reconstruction FOV is a disc; the occiput lies outside it, so the union of
    all slices ends on a circular arc at the bottom.  Fit that arc (least squares)
    and return the circle in source-pixel units."""
    ever = (vol > 30).any(axis=0)
    ny, nx = ever.shape
    pts = []
    for x in range(nx):
        rows = np.nonzero(ever[:, x])[0]
        if len(rows):
            pts.append((x, rows.max()))
    pts = np.array(pts, float)
    sel = pts[pts[:, 1] > 0.85 * ny]
    if len(sel) < 20:
        return dict(cx=nx * ds / 2, cy=ny * ds / 2, r=1e9)
    x, y = sel[:, 0] + 0.5, sel[:, 1] + 1.0
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    r = float(np.sqrt(c + cx ** 2 + cy ** 2))
    return dict(cx=float(cx * ds), cy=float(cy * ds), r=float(r * ds))


def main():
    t0 = time.time()
    C.WORK_DIR.mkdir(exist_ok=True)
    rows = read_index()
    first = Image.open(C.SLICES_DIR / rows[0][1])
    W, H = first.size
    grids = {}
    for name, ds in (("fine", C.FINE_DS), ("coarse", C.COARSE_DS)):
        w, h = W // ds, H // ds
        grids[name] = dict(ds=ds, w=w, h=h, vol=np.zeros((len(rows), h, w), np.uint8))
    for zi, (_, fn, _) in enumerate(rows):
        im = Image.open(C.SLICES_DIR / fn).convert("L")
        for g in grids.values():
            ds, w, h = g["ds"], g["w"], g["h"]
            small = im.crop((0, 0, w * ds, h * ds)).resize((w, h), Image.BOX)
            g["vol"][zi] = np.asarray(small)
        if zi % 40 == 0:
            print(f"  slice {zi + 1}/{len(rows)}")
    # ---- soft-tissue series, resampled into the skull frame ----
    soft = {name: np.zeros_like(g["vol"]) for name, g in grids.items()}
    if C.SOFT_SLICES_DIR.exists():
        # skull pixel (x, y) samples the head image at HEAD_S * (x, y) + (HEAD_TX, HEAD_TY)
        from scipy import ndimage as ndi
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        sy = C.HEAD_S * yy + C.HEAD_TY
        sx = C.HEAD_S * xx + C.HEAD_TX
        for zi, (_, fn, _) in enumerate(rows):
            p = C.SOFT_SLICES_DIR / fn
            if not p.exists():
                continue
            im = np.asarray(Image.open(p).convert("L")).astype(np.float32)
            reg = ndi.map_coordinates(im, [sy, sx], order=1, cval=0).clip(0, 255).astype(np.uint8)
            reg_im = Image.fromarray(reg)
            for name, g in grids.items():
                ds, w, h = g["ds"], g["w"], g["h"]
                soft[name][zi] = np.asarray(reg_im.crop((0, 0, w * ds, h * ds)).resize((w, h), Image.BOX))
            if zi % 40 == 0:
                print(f"  soft slice {zi + 1}/{len(rows)}")
        for name in grids:
            np.save(C.WORK_DIR / f"vol_soft_{name}.npy", soft[name])
        print("soft-tissue volumes written")

    positions = [r[2] for r in rows]
    fov = fit_fov_circle(grids["fine"]["vol"], C.FINE_DS)
    print("FOV circle (source px): cx=%.1f cy=%.1f r=%.1f -> diameter %.1f mm" % (
        fov["cx"], fov["cy"], fov["r"], 2 * fov["r"] * C.PX_MM))
    meta = dict(
        fov_circle_px=fov,
        px_mm=C.PX_MM,
        slice_mm=C.SLICE_MM,
        window_center=C.WINDOW_C,
        window_width=C.WINDOW_W,
        source_size=[W, H],
        n_slices=len(rows),
        positions_mm=positions,
        grids={},
    )
    for name, g in grids.items():
        np.save(C.WORK_DIR / f"vol_{name}.npy", g["vol"])
        meta["grids"][name] = dict(
            shape=list(g["vol"].shape),  # (nz, ny, nx)
            spacing_mm=[C.SLICE_MM, C.PX_MM * g["ds"], C.PX_MM * g["ds"]],  # (z, y, x)
            downsample=g["ds"],
        )
        print(name, g["vol"].shape, "spacing", meta["grids"][name]["spacing_mm"])
    with open(C.WORK_DIR / "volume_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    # quick histogram report for threshold sanity
    v = grids["coarse"]["vol"]
    hist = np.bincount(v.ravel(), minlength=256)
    print("gray histogram (coarse), 8-bin groups:")
    print(" ".join(f"{i * 8}:{hist[i * 8:(i + 1) * 8].sum()}" for i in range(32)))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
