"""Airway analysis: left / right nostril split, geodesic cross-section profiles, 1-D
hydraulic metrics and a 3-D potential-flow velocity field for the particle visualisation.

Inputs (work/):  airway_fine[_<set>].npy, labels_coarse[_<set>].npy, vol_fine.npy, vol_coarse.npy
Outputs:
  work/airway_side_fine[_<set>].npy   uint8: 1 = left nostril passage, 2 = right, 3 = common (pharynx)
  viewer/data/<set>/airway.json        profiles + summary metrics
  viewer/data/<set>/flow.json, flow.bin  velocity field (int16 x3, world axes) for particles

Usage: python pipeline/airway.py [pre|post]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.sparse.linalg import cg
from skimage.graph import MCP_Geometric

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from mesh import world_centre  # noqa: E402

T0 = time.time()
RHO, MU = 1.2, 1.8e-5  # air


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def suffix(dataset):
    return "" if dataset == "pre" else f"_{dataset}"


def external_air(vol_u8, airway, disc, sigma):
    """Air outside the head = air not in the airway, connected to the image corner."""
    vs = ndi.gaussian_filter(vol_u8.astype(np.float32), sigma=sigma)
    air = (vs < 22) & disc[None] & ~airway
    lab, _ = ndi.label(air)
    # seeds: any air voxel on the first row inside the disc, mid-x, top slice
    z0 = 0
    ys = np.nonzero(disc.any(axis=1))[0]
    seed_ids = set()
    for y in (ys.min() + 2, ys.max() - 2):
        row = lab[z0, y]
        seed_ids |= set(np.unique(row[row > 0]).tolist())
    # take the largest external component among candidates (robust to stray specks)
    if not seed_ids:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        seed_ids = {int(np.argmax(sizes))}
    sizes = np.bincount(lab.ravel())
    best = max(seed_ids, key=lambda i: sizes[i])
    return lab == best


def idx_to_world_pts(idx_zyx, sp, nz, centre):
    z, y, x = idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]
    return np.c_[(x + 0.5) * sp[2] - centre[0], (nz - 1 - z) * sp[0] - centre[1], centre[2] - (y + 0.5) * sp[1]]


def geodesic(mask, seeds_zyx, sp):
    costs = np.where(mask, 1.0, np.inf)
    mcp = MCP_Geometric(costs, sampling=sp, fully_connected=True)
    d, _ = mcp.find_costs([tuple(s) for s in seeds_zyx])
    d[~mask] = np.inf
    return d


def profile(mask, dist, sp, centre, nz, step=1.0):
    """Cross-section profile along the geodesic parameter: area = shell volume / step."""
    vox = float(np.prod(sp))
    d = dist[mask]
    if d.size == 0:
        return None
    smax = float(np.nanmax(d[np.isfinite(d)]))
    nst = int(np.floor(smax / step)) - 2  # the last shells are truncated by the domain end
    if nst < 3:
        return None
    # wall faces per voxel (6-neighbourhood), weighted by face area
    face = np.zeros(mask.shape, np.float32)
    for ax, area in ((0, sp[1] * sp[2]), (1, sp[0] * sp[2]), (2, sp[0] * sp[1])):
        for sh in (1, -1):
            nb = np.roll(mask, sh, axis=ax)
            if sh == 1:
                nb[(slice(None),) * ax + (0,)] = False
            else:
                nb[(slice(None),) * ax + (-1,)] = False
            face += (mask & ~nb) * area
    zz, yy, xx = np.nonzero(mask)
    dd = dist[zz, yy, xx]
    k = np.floor(dd / step).astype(int)
    ok = (k >= 0) & (k < nst)
    zz, yy, xx, k = zz[ok], yy[ok], xx[ok], k[ok]
    cnt = np.bincount(k, minlength=nst).astype(float)
    area = cnt * vox / step
    per = np.bincount(k, face[zz, yy, xx], minlength=nst) / step
    pts = idx_to_world_pts(np.c_[zz, yy, xx], sp, nz, centre)
    cx = np.bincount(k, pts[:, 0], minlength=nst) / np.maximum(cnt, 1)
    cy = np.bincount(k, pts[:, 1], minlength=nst) / np.maximum(cnt, 1)
    cz = np.bincount(k, pts[:, 2], minlength=nst) / np.maximum(cnt, 1)
    # Display curve only. Median+Gaussian smoothing erased real 1 mm stenoses
    # (a 20 mm² dip in a 100 mm² tube reported as 100 mm²). Hydraulics and MCA
    # use the raw shell areas; keep a lightly smoothed copy for the plot.
    area_s = ndi.gaussian_filter1d(area, 0.35, mode="nearest")
    per_s = ndi.gaussian_filter1d(per, 0.35, mode="nearest")
    hyd = 4.0 * area / np.maximum(per, 1e-6)
    s = (np.arange(nst) + 0.5) * step
    centers = np.c_[cx, cy, cz]
    # Partial inlet/outlet shells are not anatomical sections. They used to be
    # hidden by median smoothing; crop them so MCA and hydraulics stay consistent.
    keep = (s >= 3.0) & (s <= s[-1] - 4.0)
    if int(keep.sum()) >= 3:
        s, area, area_s, per, hyd, centers = s[keep], area[keep], area_s[keep], per[keep], hyd[keep], centers[keep]
    jumps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    bad = np.nonzero(jumps > 5.0)[0]
    n_ok = int(bad[0] + 1) if len(bad) else len(s)
    return dict(s_mm=s.round(2).tolist(), area_mm2=area.round(2).tolist(), area_mm2_smooth=area_s.round(2).tolist(),
                perimeter_mm=per.round(2).tolist(), hyd_diam_mm=hyd.round(3).tolist(),
                centers=centers.round(2).tolist(), n_centerline=n_ok)


def bbox(mask, pad=2):
    sl = []
    for ax in range(3):
        nz_ = np.nonzero(mask.any(axis=tuple(a for a in range(3) if a != ax)))[0]
        sl.append(slice(max(int(nz_.min()) - pad, 0), min(int(nz_.max()) + pad + 1, mask.shape[ax])))
    return tuple(sl)


def min_cost_path(cost, starts, end, sp):
    """Voxel path (N x 3) from any of `starts` to `end` minimising the integrated cost."""
    cost = cost.copy()
    for s in list(starts) + [end]:
        if not np.isfinite(cost[tuple(s)]):
            cost[tuple(s)] = 1.0
    mcp = MCP_Geometric(cost, sampling=sp, fully_connected=True)
    mcp.find_costs([tuple(s) for s in starts], ends=[tuple(end)])
    try:
        return np.array(mcp.traceback(tuple(end)), dtype=np.int64)
    except Exception:
        return None


def resample_path(pts, step=1.0, sigma=1.5):
    """Resample a polyline at `step` mm and smooth it; end points are kept."""
    if pts is None or len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 2 * step:
        return pts
    si = np.arange(0.0, s[-1], step)
    si = np.r_[si, s[-1]]
    out = np.c_[[np.interp(si, s, pts[:, i]) for i in range(3)]].T
    sm = ndi.gaussian_filter1d(out, sigma, axis=0, mode="nearest")
    sm[0], sm[-1] = out[0], out[-1]
    return sm


def pressure_drop(prof, Q_mL_s):
    Q = Q_mL_s * 1e-6
    A = np.array(prof["area_mm2"]) * 1e-6
    Dh = np.maximum(np.array(prof["hyd_diam_mm"]) * 1e-3, 1e-4)
    s = np.array(prof["s_mm"]) * 1e-3
    v = Q / np.maximum(A, 1e-7)
    Re = RHO * v * Dh / MU
    f = np.where(Re < 2300, 64.0 / np.maximum(Re, 1.0), 0.316 / np.maximum(Re, 1.0) ** 0.25)
    ds = np.gradient(s)
    dP = float(np.sum(f * ds / Dh * 0.5 * RHO * v ** 2))
    a0, a1 = A[:-1], A[1:]
    v0, v1 = v[:-1], v[1:]
    K = np.where(a1 > a0, (1 - a0 / a1) ** 2, 0.5 * (1 - a1 / np.maximum(a0, 1e-7)))
    dP += float(np.sum(K * 0.5 * RHO * np.where(a1 > a0, v0 ** 2, v1 ** 2)))
    dP += 0.5 * RHO * v[0] ** 2  # entry loss
    tau = f * RHO * v ** 2 / 8
    return dict(dP=dP, vmax=float(v.max()), vmean=float(v.mean()), tau_max=float(tau.max()), Re_max=float(Re.max()),
                speed_m_s=v.round(3).tolist())


def side_summary(mask, dist, sp, prof, sp_vol):
    vox = float(np.prod(sp))
    out = dict(volume_cc=float(mask.sum() * vox / 1000.0))
    if prof is None:
        return out
    s = np.array(prof["s_mm"]); a = np.array(prof["area_mm2"])
    out["length_mm"] = float(s[-1] + 0.5)
    i = int(np.argmin(a))
    out.update(min_area_mm2=float(a[i]), min_area_at_mm=float(s[i]), min_area_index=i, min_area_pos=prof["centers"][i],
               mean_area_mm2=float(a.mean()))
    return out


def main(dataset="pre"):
    sfx = suffix(dataset)
    meta = json.load(open(C.WORK_DIR / "volume_meta.json"))
    spf = tuple(meta["grids"]["fine"]["spacing_mm"])
    spc = tuple(meta["grids"]["coarse"]["spacing_mm"])
    airway_f = np.load(C.WORK_DIR / f"airway_fine{sfx}.npy")
    labels_c = np.load(C.WORK_DIR / f"labels_coarse{sfx}.npy")
    nz, nyf, nxf = airway_f.shape
    centre = world_centre(nz)
    fov = meta["fov_circle_px"]
    out_dir = C.DATA_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    def disc_for(shape, ds):
        yy2, xx2 = np.mgrid[0:shape[1], 0:shape[2]]
        return np.hypot((xx2 + 0.5) * ds - fov["cx"], (yy2 + 0.5) * ds - fov["cy"]) < fov["r"] - 1.5 * ds

    # ---------------- inlets on the fine grid ----------------
    vf = np.load(C.WORK_DIR / "vol_fine.npy")
    disc_f = disc_for(vf.shape, C.FINE_DS)
    ext_f = external_air(vf, airway_f, disc_f, (0.3, 0.7, 0.7))
    del vf
    inlet_f = airway_f & ndi.binary_dilation(ext_f, structure=ndi.generate_binary_structure(3, 1))
    lab_in, n_in = ndi.label(ndi.binary_dilation(inlet_f, iterations=2) & airway_f)
    sizes = np.bincount(lab_in.ravel()); sizes[0] = 0
    keep = [i for i in np.argsort(sizes)[::-1] if sizes[i] > 0.02 * sizes.max()][:3]
    clusters = []
    for i in keep:
        m = (lab_in == i) & inlet_f
        if not m.any():
            continue
        pts = idx_to_world_pts(np.argwhere(m), spf, nz, centre)
        clusters.append(dict(id=int(i), n=int(m.sum()), centre=pts.mean(axis=0).round(1).tolist(), mask=m))
    log("inlet clusters:", [(c["n"], c["centre"]) for c in clusters])
    # nostrils: the two clusters highest up (largest world Y); left = +X (patient's left)
    nostrils = sorted(clusters, key=lambda c: -c["centre"][1])[:2]
    nostrils.sort(key=lambda c: c["centre"][0])
    right, left = nostrils[0], nostrils[1]
    other = [c for c in clusters if c is not left and c is not right]
    if other:
        log("  additional external opening(s) (mouth?) at", [c["centre"] for c in other], "- treated as inlets too")

    # ---------------- geodesic distances and side split ----------------
    dL = geodesic(airway_f, np.argwhere(left["mask"]), spf)
    dR = geodesic(airway_f, np.argwhere(right["mask"]), spf)
    log("geodesics done; reach L %.0f mm, R %.0f mm" % (np.nanmax(dL[np.isfinite(dL)]), np.nanmax(dR[np.isfinite(dR)])))
    isL = airway_f & (dL <= dR)
    isR = airway_f & (dR < dL)
    # choanae: most anterior coronal index (image y) where L and R passages touch
    touch = isL & ndi.binary_dilation(isR, structure=ndi.generate_binary_structure(3, 1))
    ys = np.nonzero(touch.any(axis=(0, 2)))[0]
    if len(ys):
        # ignore tiny anterior contacts (septal perforation / partial volume): need >= 40 voxels in the column
        col = touch.sum(axis=(0, 2))
        good = np.nonzero(col >= 40)[0]
        y_ch = int(good.min()) if len(good) else int(ys.min())
    else:
        y_ch = nyf // 2
    log("choanal plane at image y = %d (world Z = %.1f mm)" % (y_ch, centre[2] - (y_ch + 0.5) * spf[1]))
    side = np.zeros(airway_f.shape, np.uint8)
    nasal = airway_f.copy()
    nasal[:, y_ch:, :] = False
    common = airway_f & ~nasal
    # anything anterior to the choanae but only reachable through the pharynx (e.g. oral cavity) is common too
    lab_n, k = ndi.label(nasal)
    keep_ids = set(np.unique(lab_n[left["mask"]])) | set(np.unique(lab_n[right["mask"]]))
    keep_ids.discard(0)
    stray = nasal & ~np.isin(lab_n, list(keep_ids))
    common |= stray
    nasal &= ~stray
    side[nasal & isL] = 1
    side[nasal & isR] = 2
    side[common] = 3
    np.save(C.WORK_DIR / f"airway_side_fine{sfx}.npy", side)
    vox_f = float(np.prod(spf))
    log("volumes: L %.1f cc, R %.1f cc, common %.1f cc" % tuple((side == i).sum() * vox_f / 1000 for i in (1, 2, 3)))

    # ---------------- profiles ----------------
    sides = {}
    profL = profile(side == 1, dL, spf, centre, nz)
    profR = profile(side == 2, dR, spf, centre, nz)
    seeds_c = np.argwhere(common & ndi.binary_dilation(nasal, structure=ndi.generate_binary_structure(3, 1)))
    dC = geodesic(common, seeds_c, spf) if len(seeds_c) else None
    profC = profile(common, dC, spf, centre, nz) if dC is not None else None
    for key, m, d, p in (("L", side == 1, dL, profL), ("R", side == 2, dR, profR), ("common", common, dC, profC)):
        sides[key] = side_summary(m, d, spf, p, vox_f)
        if p is not None:
            sides[key]["profile"] = p

    # ---------------- centrelines ----------------
    # Minimal-cost paths through the passages, weighted to stay away from the walls:
    # nostril -> choana (per side), then both choanae -> a meeting point in the nasopharynx,
    # then meeting point -> outlet.  Unlike the shell centroids (which fall between the meatuses
    # when a shell is split into several pockets) these lie inside the airway and join up.
    bb = bbox(airway_f)
    off = np.array([s.start for s in bb])
    aw_c = airway_f[bb]
    side_c = side[bb]
    common_c = side_c == 3
    st6 = ndi.generate_binary_structure(3, 1)
    edt_c = ndi.distance_transform_edt(aw_c, sampling=spf)
    cost_c = np.where(aw_c, 1.0 + 3.0 / np.maximum(edt_c, 0.25), np.inf)

    def most_central(mask_c):
        pts = np.argwhere(mask_c)
        return pts[np.argmax(edt_c[tuple(pts.T)])] if len(pts) else None

    def choana_point(sid, dist_full):
        m = (side_c == sid) & ndi.binary_dilation(common_c, structure=st6)
        if not m.any():  # no contact with the pharynx: use the deepest point instead
            m = side_c == sid
            d = dist_full[bb]
            pts = np.argwhere(m)
            return pts[np.argmax(np.nan_to_num(d[tuple(pts.T)], nan=-1, posinf=-1))]
        return most_central(m)

    def region_cost(mask_c):
        return np.where(mask_c, cost_c, np.inf)

    paths = {}
    chL = choana_point(1, dL)
    chR = choana_point(2, dR)
    if common_c.any():
        mL = common_c.copy(); mL[tuple(chL)] = True
        mR = common_c.copy(); mR[tuple(chR)] = True
        gL = geodesic(mL, [chL], spf)
        gR = geodesic(mR, [chR], spf)
        tot = gL + gR
        ok = common_c & np.isfinite(tot)
        if ok.any():
            # meeting point: on the geodesic joining the two choanae (tot minimal), at the spot
            # equidistant from both, i.e. the midline just behind the posterior edge of the
            # septum (taking the widest spot instead would pull it into the roomier side)
            tmin = tot[ok].min()
            cand = ok & (tot <= tmin + 1.0)
            pts = np.argwhere(cand)
            score = np.abs(gL[cand] - gR[cand]) - 0.2 * edt_c[cand]
            meet = pts[np.argmin(score)]
        else:
            meet = most_central(common_c)
        zl = int(np.nonzero(common_c.any(axis=(1, 2)))[0].max())
        bottom = np.zeros_like(common_c); bottom[zl] = common_c[zl]
        outlet_pt = most_central(bottom)
    else:
        meet = outlet_pt = None
    for key, sid, ch, inlet_mask in (("L", 1, chL, left["mask"]), ("R", 2, chR, right["mask"])):
        starts = np.argwhere(inlet_mask[bb] & (side_c == sid))
        if not len(starts):
            starts = np.argwhere(inlet_mask[bb])
        p = min_cost_path(region_cost(side_c == sid), starts, ch, spf) if len(starts) else None
        if p is not None and meet is not None:
            pb = min_cost_path(region_cost(common_c), [ch], meet, spf)
            if pb is not None and len(pb) > 1:
                p = np.r_[p, pb[1:]]
        paths[key] = p
    if meet is not None and outlet_pt is not None:
        paths["common"] = min_cost_path(region_cost(common_c), [meet], outlet_pt, spf)
    else:
        paths["common"] = None
    for key, dist_full in (("L", dL), ("R", dR), ("common", dC)):
        p = paths.get(key)
        if p is None or len(p) < 2:
            continue
        full = p + off
        pts = idx_to_world_pts(full, spf, nz, centre)
        sides[key]["path"] = resample_path(pts).round(2).tolist()
        # put the minimum-area marker on the path at the same geodesic station
        s_min = sides[key].get("min_area_at_mm")
        if s_min is not None and dist_full is not None:
            dpath = dist_full[tuple(full.T)]
            hit = np.nonzero(np.isfinite(dpath) & (dpath >= s_min))[0]
            if len(hit):
                sides[key]["min_area_pos"] = pts[int(hit[0])].round(2).tolist()
    log("centrelines: " + ", ".join(f"{k} {len(v)} pts" for k, v in paths.items() if v is not None))
    del edt_c, cost_c
    # reference hydraulics at 250 mL/s (the viewer re-solves live for other flow rates)
    Q = 250.0
    if profL and profR:
        lo, hi = 0.01 * Q, 0.99 * Q
        for _ in range(50):
            m = 0.5 * (lo + hi)
            if pressure_drop(profL, m)["dP"] > pressure_drop(profR, Q - m)["dP"]:
                hi = m
            else:
                lo = m
        QL = 0.5 * (lo + hi)
        for key, p, q in (("L", profL, QL), ("R", profR, Q - QL)):
            h = pressure_drop(p, q)
            sides[key].update(Q_mL_s=q, dP_Pa=h["dP"], resistance_Pa_s_per_mL=h["dP"] / q, peak_speed_m_s=h["vmax"],
                              mean_speed_m_s=h["vmean"], wall_shear_max_Pa=h["tau_max"], Re_max=h["Re_max"])
            sides[key]["profile"]["speed_m_s"] = h["speed_m_s"]
        if profC:
            h = pressure_drop(profC, Q)
            sides["common"].update(Q_mL_s=Q, dP_Pa=h["dP"], resistance_Pa_s_per_mL=h["dP"] / Q, peak_speed_m_s=h["vmax"],
                                   mean_speed_m_s=h["vmean"], wall_shear_max_Pa=h["tau_max"], Re_max=h["Re_max"])
            sides["common"]["profile"]["speed_m_s"] = h["speed_m_s"]
        log("1-D model @250 mL/s: L %.0f mL/s dP %.1f Pa, R %.0f mL/s dP %.1f Pa" % (
            QL, sides["L"]["dP_Pa"], Q - QL, sides["R"]["dP_Pa"]))

    # ---------------- potential flow on the coarse grid ----------------
    airway_c = labels_c == 7
    vc = np.load(C.WORK_DIR / "vol_coarse.npy")
    disc_c = disc_for(vc.shape, C.COARSE_DS)
    ext_c = external_air(vc, airway_c, disc_c, (0.4, 0.7, 0.7))
    del vc
    inlet_c = airway_c & ndi.binary_dilation(ext_c, structure=ndi.generate_binary_structure(3, 1))
    outlet_c = np.zeros_like(airway_c)
    outlet_c[-1] = airway_c[-1]
    if not outlet_c.any():  # fall back: lowest slice that has airway
        zl = np.nonzero(airway_c.any(axis=(1, 2)))[0].max()
        outlet_c[zl] = airway_c[zl]
    # keep only the component of the airway that connects inlets and outlet
    lab_c, _ = ndi.label(airway_c)
    ids = set(np.unique(lab_c[inlet_c])) & set(np.unique(lab_c[outlet_c])) - {0}
    dom = np.isin(lab_c, list(ids)) if ids else airway_c
    idx = -np.ones(dom.shape, np.int64)
    n = int(dom.sum())
    idx[dom] = np.arange(n)
    log("potential flow: %d unknowns, %d inlet, %d outlet voxels" % (n, (inlet_c & dom).sum(), (outlet_c & dom).sum()))
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    cond = {0: spc[1] * spc[2] / spc[0], 1: spc[0] * spc[2] / spc[1], 2: spc[0] * spc[1] / spc[2]}
    for ax in range(3):
        a = dom.copy()
        sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3
        sl_a[ax] = slice(0, -1); sl_b[ax] = slice(1, None)
        pair = dom[tuple(sl_a)] & dom[tuple(sl_b)]
        ia = idx[tuple(sl_a)][pair]; ib = idx[tuple(sl_b)][pair]
        w = cond[ax]
        rows += [ia, ib]; cols += [ib, ia]; vals += [np.full(ia.size, -w), np.full(ia.size, -w)]
        np.add.at(diag, ia, w); np.add.at(diag, ib, w)
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    Lap = sparse.coo_matrix((np.concatenate([vals, diag]), (np.concatenate([rows, np.arange(n)]), np.concatenate([cols, np.arange(n)]))), shape=(n, n)).tocsr()
    phi = np.zeros(n)
    fixed = np.zeros(n, bool)
    fixed[idx[inlet_c & dom]] = True; phi[idx[inlet_c & dom]] = 1.0
    fixed[idx[outlet_c & dom]] = True; phi[idx[outlet_c & dom]] = 0.0
    free = ~fixed
    A = Lap[free][:, free]
    b = -Lap[free][:, fixed] @ phi[fixed]
    Minv = sparse.diags(1.0 / A.diagonal())
    x, info = cg(A, b, M=Minv, rtol=1e-8, maxiter=20000)
    phi[free] = x
    log("  CG info %d, phi range %.3f..%.3f" % (info, phi.min(), phi.max()))
    PHI = np.zeros(dom.shape, np.float32); PHI[dom] = phi
    # face fluxes -> voxel velocities (units: conductance * dphi), then scale to Q_ref
    vel = np.zeros(dom.shape + (3,), np.float32)
    flux_total = 0.0
    for ax in range(3):
        sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3
        sl_a[ax] = slice(0, -1); sl_b[ax] = slice(1, None)
        pair = dom[tuple(sl_a)] & dom[tuple(sl_b)]
        fl = np.zeros(pair.shape, np.float32)
        fl[pair] = cond[ax] * (PHI[tuple(sl_a)][pair] - PHI[tuple(sl_b)][pair])  # flow from a to b (+axis direction)
        face_area = cond[ax] * spc[ax]  # = product of the other two spacings
        v_face = fl / face_area  # mm/s per unit
        # cell-centre velocity = mean of the two faces along this axis; a wall face carries no
        # flux, so boundary voxels get half the interior face value (otherwise particles are
        # pushed straight into the wall)
        acc = np.zeros(dom.shape, np.float32)
        acc[tuple(sl_a)] += v_face
        acc[tuple(sl_b)] += v_face
        vel[..., ax] = np.where(dom, acc / 2.0, 0)
        if ax == 0:  # the outlet layer has no outer face: its flux equals the inner face flux
            vel[-1, ..., 0] = np.where(outlet_c[-1] & dom[-1], acc[-1], vel[-1, ..., 0])
    # total flux out of the inlet voxels (sum over faces leaving inlet voxels into the domain)
    for ax in range(3):
        sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3
        sl_a[ax] = slice(0, -1); sl_b[ax] = slice(1, None)
        pair = dom[tuple(sl_a)] & dom[tuple(sl_b)]
        fl = cond[ax] * (PHI[tuple(sl_a)] - PHI[tuple(sl_b)]) * pair
        ina = (inlet_c & dom)[tuple(sl_a)] & ~(inlet_c & dom)[tuple(sl_b)]
        inb = (inlet_c & dom)[tuple(sl_b)] & ~(inlet_c & dom)[tuple(sl_a)]
        flux_total += float(fl[ina].sum()) - float(fl[inb].sum())
    Q_ref = 250.0 * 1000.0  # mm^3/s
    scale = Q_ref / max(flux_total, 1e-9)
    vel *= scale / 1000.0  # -> m/s
    speed = np.linalg.norm(vel, axis=-1)
    log("  flux scale %.3g; speed max %.2f m/s, 99%% %.2f m/s" % (scale, speed.max(), np.percentile(speed[dom], 99)))
    # per-nostril split from the potential solution: inflow through each inlet cluster (coarse)
    lab_ic, _ = ndi.label(ndi.binary_dilation(inlet_c & dom, iterations=2) & dom)
    split = {}
    side_inlet = {}
    for name, c in (("L", left), ("R", right)):
        # map the fine cluster centre to the nearest coarse inlet cluster
        cz, cy, cx = np.argwhere(c["mask"]).mean(axis=0)
        f = C.COARSE_DS // C.FINE_DS
        ci = lab_ic[int(cz), int(cy // f), int(cx // f)]
        if ci == 0:
            near = np.argwhere(lab_ic > 0)
            d2 = ((near - np.array([cz, cy / f, cx / f])) ** 2).sum(axis=1)
            ci = lab_ic[tuple(near[np.argmin(d2)])]
        m_in = (lab_ic == ci) & inlet_c & dom
        side_inlet[name] = m_in
        fl_sum = 0.0
        for ax in range(3):
            sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3
            sl_a[ax] = slice(0, -1); sl_b[ax] = slice(1, None)
            pair = dom[tuple(sl_a)] & dom[tuple(sl_b)]
            fl = cond[ax] * (PHI[tuple(sl_a)] - PHI[tuple(sl_b)]) * pair
            ina = m_in[tuple(sl_a)] & ~(inlet_c & dom)[tuple(sl_b)]
            inb = m_in[tuple(sl_b)] & ~(inlet_c & dom)[tuple(sl_a)]
            fl_sum += float(fl[ina].sum()) - float(fl[inb].sum())
        split[name] = fl_sum / max(flux_total, 1e-9)
    log("  potential-flow split L %.0f%% / R %.0f%%" % (100 * split.get("L", 0), 100 * split.get("R", 0)))

    # ---------------- export flow field (world axes, cropped bbox) ----------------
    zz, yy, xx = np.nonzero(dom)
    z0, z1 = max(zz.min() - 1, 0), min(zz.max() + 2, dom.shape[0])
    y0, y1 = max(yy.min() - 1, 0), min(yy.max() + 2, dom.shape[1])
    x0, x1 = max(xx.min() - 1, 0), min(xx.max() + 2, dom.shape[2])
    sub = vel[z0:z1, y0:y1, x0:x1]
    # index-space (z, y, x) velocity -> world (X, Y, Z) = (vx, -vz, -vy)
    wv = np.stack([sub[..., 2], -sub[..., 0], -sub[..., 1]], axis=-1)
    vmax = float(np.percentile(speed[dom], 99.9)) if dom.any() else 1.0
    v99 = float(np.percentile(speed[dom], 99.0)) if dom.any() else 1.0  # colour-scale top: voxel hot spots are numerical
    q = np.clip(wv / max(vmax, 1e-6) * 32000.0, -32767, 32767).astype(np.int16)
    # reorder to [Zw][Yw][Xw]: world Z = -y (reverse), world Y = -z (reverse)
    q = np.transpose(q[::-1, ::-1, :, :], (1, 0, 2, 3))
    q = np.ascontiguousarray(q)
    q.tofile(out_dir / "flow.bin")
    nzc, nyc, nxc = dom.shape
    box_min = [(x0) * spc[2] - centre[0], (nzc - 1 - (z1 - 1)) * spc[0] - spc[0] / 2 - centre[1], centre[2] - (y1) * spc[1]]
    box_size = [(x1 - x0) * spc[2], (z1 - z0) * spc[0], (y1 - y0) * spc[1]]
    inl = np.argwhere(inlet_c & dom)
    inl_w = idx_to_world_pts(inl, spc, nzc, centre)
    inl_speed = speed[inlet_c & dom]
    inlets = np.c_[inl_w, np.maximum(inl_speed, 1e-3)].round(3).tolist()
    # outlet voxels (tracheal end) - particles enter here during exhalation
    outm = outlet_c & dom
    outlets = np.c_[idx_to_world_pts(np.argwhere(outm), spc, nzc, centre), np.maximum(speed[outm], 1e-3)].round(3).tolist()
    # nostril descriptors: centre, outward direction (against the inhaled flow), share of the
    # flow, equivalent radius - the viewer seeds room air around them and lets the exhaled
    # plume continue along `out`
    nostrils = []
    for name in ("L", "R"):
        m_in = side_inlet.get(name)
        if m_in is None or not m_in.any():
            continue
        pw = idx_to_world_pts(np.argwhere(m_in), spc, nzc, centre)
        vm = vel[m_in].mean(axis=0)  # index-space (z, y, x)
        vw = np.array([vm[2], -vm[0], -vm[1]])
        out = -vw / max(float(np.linalg.norm(vw)), 1e-9)
        nostrils.append(dict(name=name, centre=pw.mean(axis=0).round(2).tolist(), out=out.round(3).tolist(),
                             frac=float(split.get(name, 0.5)), radius_mm=round(float(np.sqrt(m_in.sum() * spc[1] * spc[2] / np.pi)), 2)))
    flow_meta = dict(dims=[int(x1 - x0), int(z1 - z0), int(y1 - y0)], box_min=box_min, box_size=box_size,
                     scale=vmax / 32000.0, q_ref_mL_s=250.0, max_speed_m_s=vmax, color_speed_m_s=v99, inlets=inlets, outlets=outlets,
                     nostrils=nostrils, split=split,
                     summary_html=f"potential flow (inviscid, irrotational) at 250 mL/s: 99th-percentile speed {v99:.1f} m/s · split L {100 * split.get('L', 0):.0f}% / R {100 * split.get('R', 0):.0f}%")
    json.dump(flow_meta, open(out_dir / "flow.json", "w"))
    log("flow field %s -> %.1f MB" % (flow_meta["dims"], q.nbytes / 1e6))

    total = dict(volume_cc=float(airway_f.sum() * vox_f / 1000.0))
    aw = dict(dataset=dataset, flow_rate_mL_s=250.0, sides=sides, total=total, flow_split=split,
              inlets=[dict(name=n, centre=c["centre"], n_vox=c["n"]) for n, c in (("L", left), ("R", right))] +
                     [dict(name="other", centre=c["centre"], n_vox=c["n"]) for c in other],
              choana_world_z=float(centre[2] - (y_ch + 0.5) * spf[1]))
    json.dump(aw, open(out_dir / "airway.json", "w"))
    log("wrote", out_dir / "airway.json")


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
