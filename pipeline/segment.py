"""Tissue segmentation of the bone-window CT volume.

Classes (see config.LABELS): skin, fat, soft tissue, bone, brain/neural,
eyeballs, airway, sinuses/air cells.  Intensity thresholds only separate
air / fat / soft / bone reliably in a C400/W1500 window; the remaining classes
are derived from anatomy (enclosure by bone, sphericity, depth under the skin).

Outputs (work/):
  labels_coarse.npy   uint8 label volume on the coarse grid
  airway_fine.npy     bool  airway on the fine grid (for metrics / meshing)
  sinus_fine.npy      bool  sinuses & air cells on the fine grid
  segment_stats.json  volumes per class etc.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

T_AIR, T_FAT, T_BONE = 22, 53, 96  # gray thresholds (see build_volume histogram)
# Weak air on the fine grid: partial-volume lumen darker than fat, still connected
# to the conducting airway. Gray 28 ≈ −185 display-equivalent HU on C400/W1500.
T_AIR_WEAK = 28
AIRWAY_ENVELOPE_MM = 2.5  # coarse airway → fine ROI; was 1 mm and missed wall voxels
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ---------- anisotropic morphology via distance transforms ----------
def dilate_mm(mask, r, sp):
    if r <= 0:
        return np.asarray(mask, dtype=bool).copy()
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return ndi.distance_transform_edt(~mask, sampling=sp) <= r


def erode_mm(mask, r, sp):
    if r <= 0:
        return np.asarray(mask, dtype=bool).copy()
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return ndi.distance_transform_edt(mask, sampling=sp) > r


def open_mm(mask, r, sp):
    return dilate_mm(erode_mm(mask, r, sp), r, sp)


def close_mm(mask, r, sp):
    # pad so the closing does not treat the volume border as background
    p = int(np.ceil(r / min(sp))) + 1
    m = np.pad(mask, p, constant_values=False)
    m = erode_mm(dilate_mm(m, r, sp), r, sp)
    return m[p:-p, p:-p, p:-p]


def largest_cc(mask, n=1, conn=None):
    lab, k = ndi.label(mask, structure=conn)
    if k == 0:
        return np.zeros_like(mask), []
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1][:n]
    return np.isin(lab, order), sizes[order]


def fill_2d_slices(mask):
    out = np.empty_like(mask)
    for z in range(mask.shape[0]):
        out[z] = ndi.binary_fill_holes(mask[z])
    return out


def fill_3d_capped(mask):
    """3D hole fill with the top/bottom z faces treated as solid walls."""
    m = np.pad(mask, ((1, 1), (0, 0), (0, 0)), constant_values=True)
    return ndi.binary_fill_holes(m)[1:-1]


def upsample_mask(mask, out_shape, factor):
    """Repeat a coarse (z,y,x) mask onto a finer in-plane grid."""
    u = np.repeat(np.repeat(np.asarray(mask, dtype=bool), factor, axis=1), factor, axis=2)
    out = np.zeros(out_shape, dtype=bool)
    h, w = min(u.shape[1], out_shape[1]), min(u.shape[2], out_shape[2])
    out[:, :h, :w] = u[:, :h, :w]
    return out


def downsample_any(mask, out_shape, factor):
    """Coarse voxel is True if any child fine voxel is True."""
    out = np.zeros(out_shape, dtype=bool)
    h, w = (mask.shape[1] // factor) * factor, (mask.shape[2] // factor) * factor
    if h == 0 or w == 0:
        return out
    pooled = np.asarray(mask, dtype=bool)[:, :h, :w].reshape(
        mask.shape[0], h // factor, factor, w // factor, factor
    ).any(axis=(2, 4))
    out[:, : pooled.shape[1], : pooled.shape[2]] = pooled
    return out


def downsample_majority(mask, out_shape, factor, frac=0.5):
    """Coarse voxel follows the fine mask when at least `frac` of children agree."""
    out = np.zeros(out_shape, dtype=bool)
    h, w = (mask.shape[1] // factor) * factor, (mask.shape[2] // factor) * factor
    if h == 0 or w == 0:
        return out
    pooled = np.asarray(mask, dtype=bool)[:, :h, :w].reshape(
        mask.shape[0], h // factor, factor, w // factor, factor
    ).mean(axis=(2, 4)) >= frac
    out[:, : pooled.shape[1], : pooled.shape[2]] = pooled
    return out


def refine_conducting_airway(gray, coarse_airway, coarse_sinus, spacing_fine, spacing_coarse,
                             t_strong=T_AIR, t_weak=T_AIR_WEAK, envelope_mm=AIRWAY_ENVELOPE_MM):
    """Fine conducting lumen from native air, using coarse labels only as ROI/identity.

    Coarse airway/sinus say which cavity is which. Fine intensities place the wall.
    A wider envelope recovers partial-volume lumen the coarse grid dropped. Large
    sinus cavities stay excluded. Every fragment that touches the coarse airway is
    kept, so a narrower side is not deleted just because it is not the largest.
    """
    factor = int(round(spacing_coarse[1] / spacing_fine[1]))
    region = upsample_mask(dilate_mm(coarse_airway, envelope_mm, spacing_coarse), gray.shape, factor)
    sinus_keepout = upsample_mask(coarse_sinus, gray.shape, factor)
    strong = (gray < t_strong) & region & ~sinus_keepout
    weak = (gray < t_weak) & region & ~sinus_keepout
    grown = ndi.binary_propagation(strong, mask=weak) if strong.any() else weak
    seed = upsample_mask(coarse_airway, gray.shape, factor) & grown
    if not seed.any():
        grown, _ = largest_cc(grown)
        return grown
    lab, _ = ndi.label(grown)
    keep = np.unique(lab[seed])
    keep = keep[keep > 0]
    return np.isin(lab, keep) if keep.size else grown


def remove_small(mask, min_vox):
    lab, k = ndi.label(mask)
    if k == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_vox
    keep[0] = False
    return keep[lab]


def main():
    meta = json.load(open(C.WORK_DIR / "volume_meta.json"))
    sp = tuple(meta["grids"]["coarse"]["spacing_mm"])  # (z, y, x)
    vox_mm3 = float(np.prod(sp))
    vol = np.load(C.WORK_DIR / "vol_coarse.npy")
    log("coarse volume", vol.shape, "spacing", sp)
    vs = ndi.gaussian_filter(vol.astype(np.float32), sigma=(0.4, 0.7, 0.7))

    # The reconstruction FOV is a disc and the occiput lies outside it; pixels outside
    # the disc are "no data", never air.  Treat them as a wall ("outside the frame").
    fov = meta["fov_circle_px"]
    ds = meta["grids"]["coarse"]["downsample"]
    yy2, xx2 = np.mgrid[0:vol.shape[1], 0:vol.shape[2]]
    disc = np.hypot((xx2 + 0.5) * ds - fov["cx"], (yy2 + 0.5) * ds - fov["cy"]) < fov["r"] - 1.5 * ds
    frame = np.broadcast_to(disc, vol.shape).copy()
    log("frame: FOV disc radius %.1f mm, %.0f%% of the image area" % (fov["r"] * C.PX_MM, 100 * disc.mean()))

    # bone: thin plates (orbital roofs, lamina papyracea, septum) are ~0.3 mm and vanish
    # in the box-averaged coarse grid, so detect bone on the fine grid and max-pool it.
    vf = np.load(C.WORK_DIR / "vol_fine.npy")
    spf = tuple(meta["grids"]["fine"]["spacing_mm"])
    vfs = ndi.gaussian_filter(vf.astype(np.float32), sigma=(0.3, 0.7, 0.7))
    f = C.COARSE_DS // C.FINE_DS
    nzf, nyf, nxf = vf.shape
    bone_f = vfs >= T_BONE
    bone = np.zeros(vol.shape, bool)
    bp = bone_f[:, : (nyf // f) * f, : (nxf // f) * f].reshape(nzf, nyf // f, f, nxf // f, f).any(axis=(2, 4))
    bone[:, : bp.shape[1], : bp.shape[2]] = bp
    bone &= frame
    bone = remove_small(bone, 12)
    air = (vs < T_AIR) & frame & ~bone

    # Soft classes come from the soft-tissue series (C40/W350, ~1.4 HU per gray level,
    # registered onto this grid): fat < -12 HU, muscle > 47 HU, everything in between is
    # "soft tissue" (glands, mucosa, connective tissue, CSF, brain parenchyma).
    soft_path = C.WORK_DIR / "vol_soft_coarse.npy"
    have_soft = soft_path.exists()
    if have_soft:
        ss = ndi.gaussian_filter(np.load(soft_path).astype(np.float32), sigma=(0.4, 0.7, 0.7))
        S_AIR, S_FAT, S_MUSCLE = 12, 90, 133  # soft-window gray: -118 / -12 / +47 HU
        tissue = frame & ~bone & ~air & (ss >= S_AIR)
        fat = tissue & (ss < S_FAT)
        soft = tissue & (ss >= S_FAT) & (ss < S_MUSCLE)
        muscle = tissue & (ss >= S_MUSCLE)
        log("soft-tissue series used for fat / soft / muscle")
    else:
        fat = (vs >= T_AIR) & (vs < T_FAT) & frame & ~bone
        soft = (vs >= T_FAT) & (vs < T_BONE) & frame & ~bone
        muscle = np.zeros_like(soft)
        log("soft-tissue series not found: fat / soft from the bone window")
    log("thresholds done: air %.0f cc, fat %.0f cc, soft %.0f cc, muscle %.0f cc, bone %.0f cc" % tuple(
        m.sum() * vox_mm3 / 1000 for m in (air, fat, soft, muscle, bone)))
    soft_any = soft | muscle

    # ---------- head, external vs internal air ----------
    head, _ = largest_cc(~air & frame)
    head_filled = fill_2d_slices(head)
    head_sealed = fill_3d_capped(close_mm(head_filled, 9.0, sp)) & frame
    solid = np.pad(head_sealed | ~frame, 1, constant_values=True)  # frame cut is not a surface
    depth = ndi.distance_transform_edt(solid, sampling=sp)[1:-1, 1:-1, 1:-1]
    depth[~head_sealed] = 0  # mm inside the sealed hull
    internal_air = air & head_sealed
    log("head %.0f cc, internal air %.0f cc" % (head_filled.sum() * vox_mm3 / 1000, internal_air.sum() * vox_mm3 / 1000))

    # airway: internal air connected to the outside (through nostrils / mouth), grown
    # from a deep core so shallow surface concavities (under the nose tip, ear folds) drop out
    ext_air = air & ~head_sealed
    lab, _ = ndi.label(air)
    outside_ids = np.unique(lab[ext_air])
    connected_air = np.isin(lab, outside_ids[outside_ids > 0]) & head_sealed
    core = connected_air & (depth > 12.0)
    core, sizes = largest_cc(core)
    log("airway core %.0f cc" % (core.sum() * vox_mm3 / 1000))
    grow_region = connected_air & (depth > 2.5)
    lab, _ = ndi.label(grow_region)
    ids = np.unique(lab[core])
    airway_all = np.isin(lab, ids[ids > 0])
    # split off sinuses hanging on the airway through narrow ostia.  Cores = the airway
    # opened at 1.6 mm.  A core is airway if it exits through the bottom of the volume
    # (pharynx), or is connected to such a core through a passage >= 2.6 mm wide, or is
    # a small fragment (narrow nasal passages).  Big bulbous cores (inscribed radius
    # > 5.5 mm) that are only reachable through narrow ostia are sinuses.
    opened = open_mm(airway_all, 1.6, sp)
    lab_core, k = ndi.label(opened)
    airway = airway_all.copy()
    sinus_from_airway = np.zeros_like(airway)
    if k > 1:
        wide = open_mm(airway_all, 1.3, sp)
        lab_wide, _ = ndi.label(wide)
        edt_all = ndi.distance_transform_edt(airway_all, sampling=sp)
        bottom_cores = set(np.unique(lab_core[-1])) - {0}
        wide_ids_airway = set(np.unique(lab_wide[np.isin(lab_core, list(bottom_cores))])) - {0}
        core_ids = np.arange(1, k + 1)
        max_r = ndi.maximum(edt_all, lab_core, core_ids)
        wide_of_core = ndi.maximum(lab_wide, lab_core, core_ids)  # each core lies in exactly one wide component
        is_airway = np.ones(k + 1, bool)
        for cid, r, w in zip(core_ids, max_r, wide_of_core):
            if cid in bottom_cores or w in wide_ids_airway:
                continue
            if r > 4.2:
                is_airway[cid] = False
        if not is_airway.all():
            idx = ndi.distance_transform_edt(lab_core == 0, sampling=sp, return_distances=False, return_indices=True)
            assigned = lab_core[idx[0], idx[1], idx[2]]
            airway = airway_all & is_airway[assigned]
            sinus_from_airway = airway_all & ~airway
            log("  cores: %d total, %d classified as sinus" % (k, int((~is_airway[1:]).sum())))
    sinus = (internal_air & ~airway_all & (depth > 1.5)) | sinus_from_airway
    sinus = remove_small(sinus, int(30 / vox_mm3))  # drop < 30 mm3 specks
    log("airway %.0f cc, sinuses/air cells %.0f cc" % (airway.sum() * vox_mm3 / 1000, sinus.sum() * vox_mm3 / 1000))

    # ---------- brain / neural: largest soft-tissue cavity enclosed in 3D ----------
    # Barriers: bone (fine-grid, so thin plates survive), internal air (sinuses, airway)
    # and fat (the orbit is fat-filled, so partial-volume gaps in the orbital roof do not
    # leak), closed by 5 mm to seal the skull-base foramina/fissures.  The spinal canal
    # opens to the neck through soft-tissue gaps, so an artificial floor is inserted
    # ~6 mm below the hard palate level (~ Chamberlain's line / foramen magnum).  The
    # cranial cavity is then the largest enclosed soft-tissue component (top/bottom faces
    # capped, FOV cut treated as a wall).
    ant = np.zeros_like(airway)
    ant[:, : int(airway.shape[1] * 0.45), :] = True
    zs = np.nonzero((airway & ant).any(axis=(1, 2)))[0]
    z_palate = int(zs.max()) if len(zs) else airway.shape[0] // 2
    z_floor = min(z_palate + 10, bone.shape[0] - 1)
    log("  hard palate ~ slice %d, cranial floor at slice %d" % (z_palate, z_floor))
    walls = close_mm(bone | internal_air | fat | ~frame, 5.0, sp)
    walls[z_floor] = True
    interior = fill_3d_capped(walls) & ~walls & frame
    interior[z_floor + 1:] = False
    cand = open_mm(interior & soft_any, 1.0, sp)  # break thin leaks through cracks
    brain, _ = largest_cc(cand)
    brain = dilate_mm(brain, 1.2, sp) & interior & soft_any
    log("brain/neural (cranial) %.0f cc" % (brain.sum() * vox_mm3 / 1000))

    # spinal canal below the floor: closed vertebral rings near the midline, posterior half
    ext_or_scalp = ~head_sealed & frame
    def enclosed_2d(z, iterations):
        bc = ndi.binary_closing(bone[z], structure=np.ones((3, 3), bool), iterations=iterations)
        w = np.pad(bc | ~frame[z], 1, constant_values=True)
        regions, _ = ndi.label(~w)
        regions = regions[1:-1, 1:-1]
        outside_ids = np.unique(regions[ext_or_scalp[z] & (regions > 0)])
        return (regions > 0) & ~np.isin(regions, outside_ids) & ~bc
    xs_head = np.nonzero(head_filled[z_floor].any(axis=0))[0]
    xmid = xs_head.mean() if len(xs_head) else bone.shape[2] / 2
    cord = np.zeros_like(bone)
    for z in range(z_floor + 1, bone.shape[0]):
        enc = enclosed_2d(z, 4) & soft_any[z]
        lab2, n2 = ndi.label(enc)
        for i in range(1, n2 + 1):
            m = lab2 == i
            a = m.sum() * sp[1] * sp[2]
            if 40 < a < 600:
                ys_, xs_ = np.nonzero(m)
                if abs(xs_.mean() - xmid) * sp[2] < 12 and ys_.mean() > 0.45 * bone.shape[1]:
                    cord[z] |= m
    brain |= cord
    log("brain/neural incl. spinal canal %.0f cc" % (brain.sum() * vox_mm3 / 1000))
    interior_air = interior & air & ~airway & ~sinus
    sinus |= remove_small(interior_air, int(30 / vox_mm3))

    # ---------- eyeballs ----------
    # The globe is the only ~12 mm inscribed sphere of soft tissue that is surrounded
    # mostly by fat/air/bone (orbital fat behind, air in front).  Find inscribed-sphere
    # peaks of the soft mask and test the shell around each candidate sphere.
    if have_soft:
        # globe (vitreous ~15 HU, lens ~100 HU) vs orbital fat (< -12 HU): clean in the soft window
        ss2 = ndi.gaussian_filter(np.load(soft_path).astype(np.float32), sigma=(0.6, 1.2, 1.2))
        eye_cand = (ss2 >= 88) & (ss2 < 235) & frame & ~brain & ~bone
    else:
        vs2 = ndi.gaussian_filter(vol.astype(np.float32), sigma=(0.8, 1.6, 1.6))  # stronger smoothing: no speckle holes in the globe
        eye_cand = (vs2 >= 50) & (vs2 < T_BONE) & frame & ~brain & ~bone
    edt = ndi.distance_transform_edt(eye_cand, sampling=sp)
    peaks = edt > 9.5
    lab, k = ndi.label(peaks)
    eyes = np.zeros_like(soft)
    found = []
    zz, yy, xx = np.mgrid[0:soft.shape[0], 0:soft.shape[1], 0:soft.shape[2]]
    nz, ny, nx = soft.shape
    debug_cands = []
    for i in range(1, k + 1):
        m = lab == i
        r = float(edt[m].max())
        cz, cy, cx = np.unravel_index(int(np.argmax(np.where(m, edt, 0))), m.shape)
        if not (9.5 < r < 14.5):
            debug_cands.append((round(r, 1), int(cz), int(cy), int(cx), "radius"))
            continue
        if cz > nz * 0.6 or cy > ny * 0.55:  # eyes are in the upper, anterior part of the scan
            debug_cands.append((round(r, 1), int(cz), int(cy), int(cx), "position"))
            continue
        dist = np.sqrt(((zz - cz) * sp[0]) ** 2 + ((yy - cy) * sp[1]) ** 2 + ((xx - cx) * sp[2]) ** 2)
        inner = eye_cand[dist <= r].mean()
        shell = eye_cand[(dist > r + 1.5) & (dist <= r + 4.5)].mean()
        debug_cands.append((round(r, 1), int(cz), int(cy), int(cx), round(float(inner), 2), round(float(shell), 2)))
        if inner > 0.85 and shell < 0.62:
            found.append(dict(center_vox=[int(cz), int(cy), int(cx)], radius_mm=r, inner=float(inner), shell=float(shell)))
    log("eye candidates:", debug_cands)
    found.sort(key=lambda d: d["shell"])
    found = found[:2]
    for d in found:
        cz, cy, cx = d["center_vox"]
        dist = np.sqrt(((zz - cz) * sp[0]) ** 2 + ((yy - cy) * sp[1]) ** 2 + ((xx - cx) * sp[2]) ** 2)
        eyes |= (dist <= d["radius_mm"]) & ~air & ~bone
    log("eyeballs:", found)

    # ---------- skin ----------
    hf_solid = np.pad(head_filled | ~frame, 1, constant_values=True)  # frame cut is not skin
    skin_depth = ndi.distance_transform_edt(hf_solid, sampling=sp)[1:-1, 1:-1, 1:-1]
    skin = head_filled & (skin_depth <= 2.0) & ~air & ~bone & ~eyes

    # ---------- assemble labels ----------
    labels = np.zeros(vol.shape, np.uint8)
    labels[fat] = 2
    labels[soft] = 10
    labels[muscle] = 3
    labels[skin] = 1
    labels[brain] = 5
    labels[eyes] = 6
    labels[bone] = 4
    labels[sinus] = 8
    labels[airway] = 7
    labels[air & ~airway & ~sinus] = 0
    labels[~head_filled & ~airway] = 0  # anything outside the head is background

    # ---------- fine-grid airway & sinuses ----------
    # Native-resolution air inside a generous conducting ROI. Coarse labels keep
    # maxillary/ethmoid air out; hysteresis recovers partial-volume walls.
    airway_f = refine_conducting_airway(vfs, airway, sinus, spf, sp)
    sinus_f = (vfs < T_AIR) & upsample_mask(dilate_mm(sinus, 0.8, sp), vf.shape, f) & ~airway_f
    airway_c = downsample_majority(airway_f, labels.shape, f)
    protected = np.isin(labels, [4, 5, 6, 9])  # bone / brain / eyes / teeth
    labels[labels == 7] = 0
    labels[airway_c & ~protected] = 7
    sinus_c = downsample_majority(sinus_f, labels.shape, f) & ~airway_c
    labels[labels == 8] = 0
    labels[sinus_c & ~protected] = 8
    np.save(C.WORK_DIR / "labels_coarse.npy", labels)
    np.save(C.WORK_DIR / "airway_fine.npy", airway_f)
    np.save(C.WORK_DIR / "sinus_fine.npy", sinus_f)

    stats = {int(k): dict(name=v["name"], volume_cc=float((labels == k).sum() * vox_mm3 / 1000)) for k, v in C.LABELS.items()}
    stats_out = dict(classes=stats, eyes=found, spacing_mm=list(sp))
    stats_out["airway_fine_cc"] = float(airway_f.sum() * np.prod(spf) / 1000)
    log("label volumes (cc):", {v["name"]: round(v["volume_cc"], 1) for v in stats.values()})
    log("fine airway %.1f cc" % stats_out["airway_fine_cc"])
    json.dump(stats_out, open(C.WORK_DIR / "segment_stats.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
