"""Surface meshes (GLB) for every tissue + volume/label textures + meta.json for the viewer.

World frame used everywhere in the viewer (right-handed, mm, centred on the volume):
  X = patient's left (image right),  Y = superior (up),  Z = anterior (towards the face).
Volume index (z, y, x) -> world:  X = x*sx,  Y = (nz-1-z)*sz,  Z = (ny-1-y)*sy  (minus centre).

Outputs (viewer/data/<dataset>/):
  <tissue>.glb        one mesh per tissue (decimated, smoothed)
  volume_u8.bin       coarse gray volume, uint8, order [Z][Y][X] of the world frame
  labels_u8.bin       coarse label volume, same order
  meta.json           dims, spacing, label table, mesh list, stats
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage as ndi
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

try:
    import fast_simplification
except ImportError:  # pragma: no cover
    fast_simplification = None

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def world_centre(nz):
    """World frame: X right (image x), Y up (against slice index), Z anterior (image -y).
    Origin at the centre of the source image and mid-height of the stack, so every grid
    (fine/coarse) maps consistently through its own spacing."""
    return np.array([C.SRC_W * C.PX_MM / 2.0, (nz - 1) * C.SLICE_MM / 2.0, C.SRC_H * C.PX_MM / 2.0])


def idx_to_world(verts_zyx, spacing, shape, centre):
    nz, ny, nx = shape
    sz, sy, sx = spacing
    z, y, x = verts_zyx[:, 0], verts_zyx[:, 1], verts_zyx[:, 2]
    X = (x + 0.5) * sx - centre[0]
    Y = (nz - 1 - z) * sz - centre[1]
    Z = centre[2] - (y + 0.5) * sy
    return np.c_[X, Y, Z]


def fill_solid(mask):
    """The body as one solid: every enclosed air pocket (air specks under 30 mm3 that no class
    claims, unclassified cells, partial-volume gaps) is filled, so the skin mesh has only the
    outer surface and no shells floating inside the head.  Air that reaches the outside
    (nostrils, ear canals) stays open; the scan's top/bottom cuts count as walls."""
    m = np.pad(mask, ((1, 1), (0, 0), (0, 0)), constant_values=True)
    return ndi.binary_fill_holes(m)[1:-1]


def open_nostrils(body, airway, depth_mm, spacing):
    """Carve the nasal vestibules out of the solid body so the skin has real nostril openings.
    The airway label reaches the nares, so with the airway inside the body mask the skin surface
    would be a flat cap across each nostril, coincident with the airway mesh's inlet cap - the
    two then z-fight and one nostril shows the airway while the other does not.  Removing the
    airway within depth_mm (along the passage) of where it meets the outside air makes the skin
    follow the vestibule walls instead; deeper in, the airway stays filled so the skin keeps
    no inner walls."""
    outside = ~body
    reach = airway & ndi.binary_dilation(outside)  # the inlet layer (nares)
    if not reach.any():
        return body
    # geodesic growth inside the airway, ~one voxel per step, limited to the vestibules' box
    steps = int(round(depth_mm / min(spacing)))
    zz, yy, xx = np.nonzero(reach)
    pad = steps + 2
    sl = tuple(slice(max(0, a.min() - pad), a.max() + pad + 1) for a in (zz, yy, xx))
    r, aw = reach[sl], airway[sl]
    for _ in range(steps):
        r = ndi.binary_dilation(r) & aw
    out = body.copy()
    out[sl] &= ~r
    return out


def mask_to_mesh(mask, spacing, shape, centre, sigma, target_faces, min_component_vox=0, step=1, field="gaussian"):
    """Smooth a binary mask into a scalar field, extract the 0.5 iso-surface, decimate.

    field='gaussian' is the display path for bulky tissues. field='sdf' follows the
    signed voxel distance so a one-voxel-wide lumen still crosses the iso-level.
    """
    if min_component_vox:
        lab, n = ndi.label(mask)
        if n:
            sizes = np.bincount(lab.ravel())
            keep = sizes >= min_component_vox
            keep[0] = False
            mask = keep[lab]
    if not mask.any():
        return None
    if field == "sdf":
        scalar = ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)
        if np.any(np.asarray(sigma) > 0):
            scalar = ndi.gaussian_filter(scalar.astype(np.float32), sigma)
        pad_value = float(min(scalar.min(), -1.0))
        field_vol = np.pad(scalar, 1, mode="constant", constant_values=pad_value)
        level = 0.0
    else:
        field_vol = np.pad(ndi.gaussian_filter(mask.astype(np.float32), sigma=sigma), 1)
        level = 0.5
    # pad so surfaces touching the volume border are closed
    verts, faces, _, _ = measure.marching_cubes(field_vol, level=level, spacing=(1.0, 1.0, 1.0), step_size=step)
    verts -= 1.0  # undo padding
    original_verts, original_faces = verts, faces
    m = None
    if fast_simplification is not None and len(faces) > target_faces:
        full_red = 1.0 - target_faces / len(faces)
        for red in (full_red, 0.5 * full_red, 0.25 * full_red):
            if red <= 0.02:
                break
            v2, f2 = fast_simplification.simplify(verts.astype(np.float32), faces.astype(np.int64), red)
            world = idx_to_world(v2, spacing, shape, centre)
            trial = trimesh.Trimesh(vertices=world, faces=f2, process=True)
            if field != "sdf" or trial.is_watertight:
                verts, faces, m = v2, f2, trial
                break
        if m is None:
            log('decimation broke surface topology; retaining original extraction')
            verts, faces = original_verts, original_faces
    if m is None:
        world = idx_to_world(verts, spacing, shape, centre)
        m = trimesh.Trimesh(vertices=world, faces=faces, process=True)
    # Consistent winding, then each watertight body oriented outwards by the sign of its volume.
    # The winding pass propagates from an arbitrary face per body, so a body can come out
    # consistently inside-out when decimation flipped that face: its front faces then point
    # into the tissue, and when the tissue is translucent (front faces only) its near surface
    # is culled and the far inner walls show instead - the left airway did this, the right not.
    m.fix_normals(multibody=True)
    if m.volume < 0:  # dominant body not watertight after decimation: orient by the total
        m.invert()
    return m


def export_mesh(m, path):
    m.export(path)  # glb (binary glTF)
    return dict(faces=int(len(m.faces)), vertices=int(len(m.vertices)), bytes=int(Path(path).stat().st_size))


def main(dataset="pre"):
    meta_v = json.load(open(C.WORK_DIR / "volume_meta.json"))
    out = C.DATA_DIR / dataset
    out.mkdir(parents=True, exist_ok=True)
    sfx = "" if dataset == "pre" else f"_{dataset}"
    labels = np.load(C.WORK_DIR / f"labels_coarse{sfx}.npy")
    vol = np.load(C.WORK_DIR / "vol_coarse.npy")
    spc = tuple(meta_v["grids"]["coarse"]["spacing_mm"])
    spf = tuple(meta_v["grids"]["fine"]["spacing_mm"])
    shape_c = labels.shape
    airway_f = np.load(C.WORK_DIR / f"airway_fine{sfx}.npy")
    sinus_f = np.load(C.WORK_DIR / "sinus_fine.npy")
    shape_f = airway_f.shape
    side_path = C.WORK_DIR / f"airway_side_fine{sfx}.npy"
    side_f = np.load(side_path) if side_path.exists() else None
    # for a derived dataset (virtual surgery) only the airway-related tissues change; other
    # meshes/volumes are referenced from the pre-op folder instead of being duplicated
    changed = {"airway", "airway_L", "airway_R", "airway_common", "soft", "muscle", "fat"} if dataset != "pre" else None
    centre = world_centre(shape_c[0])
    centre_f = centre
    # axis-aligned box of the coarse volume texture in world mm (texel (0,0,0) at box_min)
    nx, nz, ny = shape_c[2], shape_c[0], shape_c[1]
    box_size = np.array([nx * spc[2], nz * spc[0], ny * spc[1]])  # X, Y, Z extents
    box_min = np.array([-centre[0], -centre[1] - spc[0] / 2.0, centre[2] - ny * spc[1]])

    meshes = {}
    # --- coarse-grid tissues ---
    jobs = [
        ("skin", open_nostrils(fill_solid(labels != 0), labels == 7, 12.0, spc), 1.0, 400_000, 200),
        ("fat", labels == 2, 1.2, 300_000, 60),
        ("muscle", labels == 3, 1.2, 400_000, 60),
        ("soft", labels == 10, 1.2, 400_000, 60),
        ("brain", labels == 5, 1.2, 250_000, 200),
        ("eye", labels == 6, 0.8, 30_000, 0),
    ]
    pre_meta = json.load(open(C.DATA_DIR / "pre" / "meta.json")) if changed else None

    def reuse(key):
        """Reference the pre-op mesh for tissues unchanged by the virtual surgery."""
        if changed is not None and key not in changed and pre_meta and key in pre_meta["meshes"]:
            meshes[key] = dict(pre_meta["meshes"][key], file=f"../pre/{key}.glb")
            log(key, "-> reuse pre-op mesh")
            return True
        return False

    for key, mask, sigma, target, minc in jobs:
        if reuse(key):
            continue
        m = mask_to_mesh(mask, spc, shape_c, centre, sigma, target, minc)
        if m is None:
            log(key, "empty")
            continue
        meshes[key] = export_mesh(m, out / f"{key}.glb")
        log(key, meshes[key])
    # --- fine-grid tissues (bone with thin plates, airway, sinuses) ---
    fov = meta_v["fov_circle_px"]
    ds = meta_v["grids"]["fine"]["downsample"]
    yy2, xx2 = np.mgrid[0:shape_f[1], 0:shape_f[2]]
    disc = np.hypot((xx2 + 0.5) * ds - fov["cx"], (yy2 + 0.5) * ds - fov["cy"]) < fov["r"] - 1.5 * ds
    if not reuse("bone"):
        vf = np.load(C.WORK_DIR / "vol_fine.npy")
        vfs = ndi.gaussian_filter(vf.astype(np.float32), sigma=(0.3, 0.7, 0.7))
        bone_f = (vfs >= 96) & disc[None]  # limit to the FOV disc
        del vf, vfs
        m = mask_to_mesh(bone_f, spf, shape_f, centre_f, (0.5, 0.9, 0.9), 900_000, 40)
        meshes["bone"] = export_mesh(m, out / "bone.glb")
        log("bone", meshes["bone"])
        del bone_f
    m = mask_to_mesh(airway_f, spf, shape_f, centre_f, (0.35, 0.45, 0.45), 300_000, 0, field="sdf")
    meshes["airway"] = export_mesh(m, out / "airway.glb")
    log("airway", meshes["airway"])
    if side_f is not None:
        for key, sid in (("airway_L", 1), ("airway_R", 2), ("airway_common", 3)):
            m = mask_to_mesh(side_f == sid, spf, shape_f, centre_f, (0.35, 0.45, 0.45), 200_000, 0, field="sdf")
            if m is not None:
                meshes[key] = export_mesh(m, out / f"{key}.glb")
                log(key, meshes[key])
    if not reuse("sinus"):
        m = mask_to_mesh(sinus_f, spf, shape_f, centre_f, (0.5, 0.9, 0.9), 250_000, 30)
        if m is not None:
            meshes["sinus"] = export_mesh(m, out / "sinus.glb")
            log("sinus", meshes["sinus"])

    # --- volumes for the viewer: order [Z(anterior)][Y(up)][X] ---
    def to_world_order(a):
        return np.ascontiguousarray(np.transpose(a[::-1, ::-1, :], (1, 0, 2)))
    files = {}
    if changed:
        files["volume"] = "../pre/volume_u8.bin"
        files["volume_soft"] = "../pre/volume_soft_u8.bin"
    else:
        to_world_order(vol).tofile(out / "volume_u8.bin")
    to_world_order(labels).tofile(out / "labels_u8.bin")
    soft_path = C.WORK_DIR / "vol_soft_coarse.npy"
    have_soft = soft_path.exists()
    if have_soft and not changed:
        to_world_order(np.load(soft_path)).tofile(out / "volume_soft_u8.bin")

    stats_path = C.WORK_DIR / f"segment_stats{sfx}.json"
    stats = json.load(open(stats_path if stats_path.exists() else C.WORK_DIR / "segment_stats.json"))
    meta = dict(
        dataset=dataset,
        files=files,
        px_mm=C.PX_MM,
        slice_mm=C.SLICE_MM,
        window=dict(center=C.WINDOW_C, width=C.WINDOW_W),
        soft_window=dict(center=C.SOFT_WINDOW_C, width=C.SOFT_WINDOW_W) if have_soft else None,
        volume=dict(
            dims=[int(nx), int(nz), int(ny)],  # X, Y, Z texel counts
            spacing=[spc[2], spc[0], spc[1]],  # mm per voxel along X, Y, Z
            box_min=box_min.tolist(),  # world position of the box corner (mm)
            box_size=box_size.tolist(),  # world extents (mm)
        ),
        labels={str(k): v for k, v in C.LABELS.items()},
        meshes=meshes,
        stats=stats,
    )
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    log("wrote", out)


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
