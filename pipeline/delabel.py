"""Remove the viewer's light-blue text overlays (header text, "P" marker) from a slice series.

Two modes:

  same-window   (used for slices/skull, reference slices/skull_90)
      The reference is the same window at 90% zoom; label pixels are filled from the
      reference after a global similarity transform (scale ~1.104, shift) fitted once.

  cross-window  (used for slices/head, reference slices/skull)
      The reference is the *de-labelled* 100% bone-window series.  Label pixels are
      filled from the reference after (a) the fitted similarity transform between the
      two series and (b) a per-slice intensity look-up table (binned mean of target
      gray per reference gray, over the unlabelled pixels), with the reference
      smoothed a little because the bone kernel is noisier than the soft kernel.

Pixels labelled in both images fall back to nearest-neighbour fill (only the topmost
header rows, which are background anyway).

Usage:
  python pipeline/delabel.py same-window  slices/skull  slices/skull_90  [--out DIR]
  python pipeline/delabel.py cross-window slices/head   slices/skull     [--out DIR]
With no --out the series is overwritten in place.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi, optimize


def label_mask(rgb, dil):
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    m = (b - r > 12) & (b > g) & (g >= r)
    return ndi.binary_dilation(m, iterations=dil)


def gray(rgb):
    return np.asarray(Image.fromarray(rgb[..., :3]).convert("L")).astype(np.float32)


def fit_similarity(tgt, ref, mask_t, mask_r, x0, mode):
    """Find s, tx, ty with  tgt(x) ~ ref((x - t) / s)  by minimising a masked error."""
    H, W = tgt.shape
    step = 2
    yy, xx = np.mgrid[0:H:step, 0:W:step].astype(np.float32)
    if mode == "same-window":
        A, B = tgt, ref
        wa = ~mask_t[::step, ::step]
    else:  # compare saturated-bone masks: soft window 255 <=> HU >= 215 <=> bone gray >= 96
        A = ndi.gaussian_filter((tgt >= 250).astype(np.float32), 1.5)
        B = ndi.gaussian_filter((ref >= 96).astype(np.float32), 1.5)
        wa = ~mask_t[::step, ::step]
    Bm = mask_r.astype(np.float32)

    def err(p):
        s, tx, ty = p
        sy = (yy - ty) / s
        sx = (xx - tx) / s
        Bs = ndi.map_coordinates(B, [sy, sx], order=1, cval=0)
        Ms = ndi.map_coordinates(Bm, [sy, sx], order=0, cval=1) > 0
        valid = wa & ~Ms & (sy >= 0) & (sy < H - 1) & (sx >= 0) & (sx < W - 1)
        d = A[::step, ::step] - Bs
        return float(np.mean(np.abs(d[valid]) if mode == "same-window" else (d[valid] ** 2)))

    s0, tx0, ty0 = x0
    simplex = [[s0, tx0, ty0], [s0 + 0.003, tx0, ty0], [s0, tx0 + 3, ty0], [s0, tx0, ty0 + 3]]
    r = optimize.minimize(err, x0, method="Nelder-Mead",
                          options=dict(xatol=1e-3, fatol=1e-7, initial_simplex=simplex, maxiter=600))
    return r.x, r.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["same-window", "cross-window"])
    ap.add_argument("target")
    ap.add_argument("reference")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fit-slices", default="60,100,160", help="slice indices (1-based) used to fit the transform")
    args = ap.parse_args()
    out = args.out or args.target
    os.makedirs(out, exist_ok=True)
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(args.target, "*.png")))

    # ---- fit the similarity transform on a few slices, use the median ----
    fits = []
    for k in [int(v) - 1 for v in args.fit_slices.split(",")]:
        t = np.array(Image.open(os.path.join(args.target, files[k])).convert("RGBA"))
        r = np.array(Image.open(os.path.join(args.reference, files[k])).convert("RGBA"))
        mt, mr = label_mask(t, 3), label_mask(r, 3)
        x0 = [1 / 0.9, 0, 0] if args.mode == "same-window" else [1.0, 0.0, 0.0]
        if args.mode == "same-window":
            H, W = t.shape[:2]
            x0 = [1 / 0.9, W / 2 - W / 2 / 0.9, H / 2 - H / 2 / 0.9]
        x, f = fit_similarity(gray(t), gray(r), mt, mr, x0, args.mode)
        fits.append(x)
        print(f"fit on {files[k]}: scale {x[0]:.4f} tx {x[1]:.2f} ty {x[2]:.2f} (err {f:.4f})")
    S, TX, TY = np.median(np.array(fits), axis=0)
    print(f"using scale {S:.4f} tx {TX:.2f} ty {TY:.2f}")

    first = np.array(Image.open(os.path.join(args.target, files[0])))
    H, W = first.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sy = (yy - TY) / S
    sx = (xx - TX) / S
    inb = (sy >= 0) & (sy <= H - 1) & (sx >= 0) & (sx <= W - 1)

    tot_fb = 0
    for name in files:
        t = np.array(Image.open(os.path.join(args.target, name)).convert("RGBA")).astype(np.float32)
        r = np.array(Image.open(os.path.join(args.reference, name)).convert("RGBA")).astype(np.float32)
        mt = label_mask(t.astype(np.uint8), 2)
        mr = label_mask(r.astype(np.uint8), 3)
        if args.mode == "same-window":
            src = np.stack([ndi.map_coordinates(r[..., c], [sy, sx], order=1, cval=0) for c in range(3)], -1)
        else:
            rg = gray(r.astype(np.uint8))
            rg = ndi.gaussian_filter(rg, 1.5)
            rg_w = ndi.map_coordinates(rg, [sy, sx], order=1, cval=0)
            tg = gray(t.astype(np.uint8))
            # per-slice LUT: mean target gray per reference gray over unlabelled pixels
            valid = ~mt & ~(ndi.map_coordinates(mr.astype(np.float32), [sy, sx], order=0, cval=1) > 0) & inb
            rq = np.clip(np.round(rg_w[valid]).astype(int), 0, 255)
            sums = np.bincount(rq, tg[valid], minlength=256)
            cnts = np.bincount(rq, minlength=256)
            lut = np.where(cnts > 20, sums / np.maximum(cnts, 1), np.nan)
            # fill gaps by interpolation, enforce monotonic-ish smoothness
            idx = np.arange(256)
            ok = ~np.isnan(lut)
            lut = np.interp(idx, idx[ok], lut[ok]) if ok.sum() > 2 else idx.astype(float)
            lut = ndi.gaussian_filter1d(lut, 2.0)
            v = np.interp(rg_w, idx, lut)
            src = np.stack([v, v, v], -1)
        wmask = ndi.map_coordinates(mr.astype(np.float32), [sy, sx], order=0, cval=1) > 0
        good = mt & ~wmask & inb
        o = t.copy()
        o[good, :3] = src[good]
        bad = mt & ~good
        if bad.any():
            tot_fb += int(bad.sum())
            idx3 = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
            o[bad, :3] = o[idx3[0][bad], idx3[1][bad], :3]
        o[..., 3] = 255
        Image.fromarray(o.clip(0, 255).astype(np.uint8), "RGBA").save(os.path.join(out, name), optimize=True)
    print("done", len(files), "files; fallback px total", tot_fb)


if __name__ == "__main__":
    main()
