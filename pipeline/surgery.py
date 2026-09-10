"""Virtual surgery: parametric inferior-turbinate reduction.

The inferior turbinate is the tissue mass the nasal airway wraps around in the lower part
of each nasal passage.  It is found geometrically: closing the airway of one side with a
large radius fills the concavities between the meatuses; the filled tissue that lies
lateral to the passage and in the lower part of the cavity is the turbinate body.
Reduction removes the outer `depth` mm of that body (mucosa and thin conchal bone), i.e. a
volumetric reduction such as a submucous / microdebrider / radiofrequency turbinoplasty.
Thick bone (lateral nasal wall, maxilla, palate) and the septum are never touched.

Produces work/airway_fine_<name>.npy, work/labels_coarse_<name>.npy,
work/segment_stats_<name>.json.  Then run airway.py <name> and mesh.py <name>.

Usage: python pipeline/surgery.py [--side both|L|R] [--depth 2.0] [--lower 0.45] [--name post]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from segment import close_mm, dilate_mm  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="both", choices=["both", "L", "R"])
    ap.add_argument("--depth", type=float, default=2.0, help="reduction depth in mm")
    ap.add_argument("--lower", type=float, default=0.45, help="fraction of cavity height above which tissue is left alone (0.45 = lower 55 %%)")
    ap.add_argument("--name", default="post")
    args = ap.parse_args()

    meta = json.load(open(C.WORK_DIR / "volume_meta.json"))
    spf = tuple(meta["grids"]["fine"]["spacing_mm"])
    spc = tuple(meta["grids"]["coarse"]["spacing_mm"])
    airway = np.load(C.WORK_DIR / "airway_fine.npy")
    side = np.load(C.WORK_DIR / "airway_side_fine.npy")
    labels = np.load(C.WORK_DIR / "labels_coarse.npy")
    vf = np.load(C.WORK_DIR / "vol_fine.npy")
    vfs = ndi.gaussian_filter(vf.astype(np.float32), sigma=(0.3, 0.7, 0.7))
    bone = vfs >= 96
    air_any = vfs < 22
    del vf
    nz, ny, nx = airway.shape
    f = C.COARSE_DS // C.FINE_DS
    # tissue that may be removed: not air, and not thick bone (conchal bone is < ~1.5 mm)
    bone_thick = ndi.distance_transform_edt(bone, sampling=spf) > 0.9
    removable = ~air_any & ~bone_thick
    # soft-tissue-only voxels on the fine grid (upsampled coarse labels: fat/soft/muscle/skin)
    lab_up = np.repeat(np.repeat(labels, f, axis=1), f, axis=2)[:, :ny, :nx]
    pad_y, pad_x = ny - lab_up.shape[1], nx - lab_up.shape[2]
    if pad_y or pad_x:
        lab_up = np.pad(lab_up, ((0, 0), (0, pad_y), (0, pad_x)), mode="edge")
    removable &= np.isin(lab_up, [1, 2, 3, 10, 4]) | bone  # never brain / eyes

    # nasal region per side: all voxels of that side's passage
    xs_all = np.nonzero(side > 0)[2]
    new_air = np.zeros_like(airway)
    removed_cc = {}
    for s_name, sid in (("L", 1), ("R", 2)):
        if args.side not in ("both", s_name):
            continue
        passage = side == sid
        if not passage.any():
            continue
        # septal plane ~ the medial-most extent of the passage per coronal slice; use the
        # x-position of the other side's passage as the medial limit (with a 2 mm margin)
        other = side == (3 - sid)
        filled = close_mm(passage, 7.0, spf) & ~airway & removable
        # keep only tissue lateral to the passage core: per coronal column (y), tissue must be
        # lateral of the passage's medial-most x by > 2 mm  (left = +x, right = -x)
        zz, yy, xx = np.nonzero(passage)
        med = np.full(ny, np.nan)
        for y in np.unique(yy):
            sel = yy == y
            med[y] = xx[sel].min() if sid == 1 else xx[sel].max()
        lateral = np.zeros_like(filled)
        for y in np.nonzero(~np.isnan(med))[0]:
            if sid == 1:
                lateral[:, y, int(med[y] + 2.0 / spf[2]):] = True
            else:
                lateral[:, y, : max(int(med[y] - 2.0 / spf[2]), 0)] = True
        filled &= lateral & ~dilate_mm(other, 3.0, spf)
        # lower part of the cavity: per coronal slice, z below top + lower*(height)
        zone = np.zeros_like(filled)
        for y in np.unique(yy):
            zs = zz[yy == y]
            z0, z1 = zs.min(), zs.max()
            zc = int(z0 + args.lower * (z1 - z0))
            zone[zc:, y, :] = True
        turb = filled & zone
        # the turbinate body is the largest connected piece(s) - drop specks
        lab_t, k = ndi.label(turb)
        if k:
            sizes = np.bincount(lab_t.ravel()); sizes[0] = 0
            keep = sizes >= 0.05 * sizes.max()
            turb = keep[lab_t]
        # reduction: outer `depth` mm of the turbinate (distance from the current airway)
        d_air = ndi.distance_transform_edt(~airway, sampling=spf)
        cut = turb & (d_air <= args.depth)
        new_air |= cut
        vox = float(np.prod(spf))
        removed_cc[s_name] = float(cut.sum() * vox / 1000.0)
        log("%s: turbinate body %.2f cc, removed %.2f cc (depth %.1f mm)" % (s_name, turb.sum() * vox / 1000, removed_cc[s_name], args.depth))

    airway_post = airway | new_air
    # keep it one connected piece with the original airway
    lab_a, _ = ndi.label(airway_post)
    ids = np.unique(lab_a[airway])
    airway_post = np.isin(lab_a, ids[ids > 0])
    np.save(C.WORK_DIR / f"airway_fine_{args.name}.npy", airway_post)

    # coarse labels: a coarse voxel becomes airway when >= half of its fine voxels are airway
    added = airway_post & ~airway
    nzc, nyc, nxc = labels.shape
    a = added[:, : nyc * f, : nxc * f]
    if a.shape[1] < nyc * f or a.shape[2] < nxc * f:
        a = np.pad(a, ((0, 0), (0, nyc * f - a.shape[1]), (0, nxc * f - a.shape[2])))
    frac = a.reshape(nz, nyc, f, nxc, f).mean(axis=(2, 4))
    labels_post = labels.copy()
    conv = (frac >= 0.5) & np.isin(labels_post, [1, 2, 3, 10, 4])
    labels_post[conv] = 7
    np.save(C.WORK_DIR / f"labels_coarse_{args.name}.npy", labels_post)
    vox_c = float(np.prod(spc))
    stats = json.load(open(C.WORK_DIR / "segment_stats.json"))
    for k, v in stats["classes"].items():
        v["volume_cc"] = float((labels_post == int(k)).sum() * vox_c / 1000.0)
    stats["surgery"] = dict(side=args.side, depth_mm=args.depth, lower_fraction=args.lower, removed_cc=removed_cc)
    json.dump(stats, open(C.WORK_DIR / f"segment_stats_{args.name}.json", "w"), indent=1)
    log("post airway %.1f cc (pre %.1f cc); %d coarse voxels converted" % (
        airway_post.sum() * np.prod(spf) / 1000, airway.sum() * np.prod(spf) / 1000, int(conv.sum())))


if __name__ == "__main__":
    main()
