"""Project the patient's photos onto the CT skin surface (multi-view projective texture mapping).

Inputs : work/photo/<stem>.json + .jpg (from face_landmarks.py; one frontal photo, optionally
         more from other angles), viewer/data/pre/skin.glb, eye.glb
Outputs: viewer/data/pre/skin_photo.jpg, skin_photo_1.jpg, ...  - each photo cropped to the head
         viewer/data/pre/skin_uv.bin     - float32 [view][vertex][u, v, weight] (GLB vertex order)
         viewer/data/pre/skin_photo.json - per view: camera pose, fit residuals, colour match;
                                           skin tone, vertex count, weight scale
         work/photo/fit_<stem>.jpg       - diagnostics: photo + projected silhouette + landmarks

Method
------
Every photo is a pinhole camera with the focal length from EXIF (35 mm-equivalent).

The most frontal photo is the *reference*.  Its pose in the CT world frame is solved from
anatomical landmarks that exist on both sides: the two eye centres (cornea apex on the closed
eyelid), the glabella, the sellion, the nose tip and the upper-lip point, found automatically on
the skin mesh (globe centroids + the midline profile) and on the photo (MediaPipe Face Mesh).

The other photos are registered *to the reference*, not to the CT: each of the 478 Face-Mesh
landmarks of the reference photo is lifted onto the skin surface through the fitted reference
camera (the mesh vertex under that pixel), which gives a dense set of 3-D points with a known
Face-Mesh index.  The same indices in a side photo are then 2-D observations of those points,
and the side camera is a robust PnP fit over the ~300 landmarks on the visible half of the
face.  Registering photo-to-photo this way keeps the textures mutually aligned even where the
CT-to-photo fit has a millimetre or two of slack, which is what matters for blending them.

Every skin vertex is projected into every photo; per view a weight combines visibility
(z-buffer), incidence (the photo is smeared where the surface turns away from the camera) and a
prior favouring the reference where several photos are good.  The viewer normalises the
weights per pixel, so the sides come from the side photos, the face from the frontal one, and
the flat skin colour fills what no photo saw.  Side photos are colour-matched to the reference
(per-channel gain/offset fitted on the surface both see) before being written.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

PHOTO_DIR = C.WORK_DIR / "photo"
DIAG_35MM = 43.2666  # mm, full-frame diagonal: focal35 / DIAG_35MM * image diagonal = focal in px
MAX_TEX = 2048
WEIGHT_SCALE = 0.15   # sum of view weights at which a vertex counts as fully textured
PRIOR_SIDE = 0.5      # weight of a non-reference photo relative to the reference where both are good
FACING_POW = 3        # weight ~ facing^k: the better-facing photo takes over quickly

# MediaPipe Face Mesh indices of the landmarks matched to the mesh (subject's right = image left)
MP = dict(iris_R=468, iris_L=473, glabella=9, sellion=168, pronasale=1, labrale_sup=0)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
# landmarks not lifted onto the mesh: the face oval is a silhouette (depends on the viewpoint),
# the irises are hidden behind the closed lids
NO_LIFT = set(FACE_OVAL) | set(range(468, 478))


def log(*a):
    print(*a, flush=True)


def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------ mesh landmarks
def glb_vertex_count(path):
    """POSITION accessor count of the first primitive, straight from the GLB header."""
    with open(path, "rb") as f:
        _, _, _ = struct.unpack("<III", f.read(12))
        n, _ = struct.unpack("<II", f.read(8))
        js = json.loads(f.read(n))
    prim = js["meshes"][0]["primitives"][0]
    return js["accessors"][prim["attributes"]["POSITION"]]["count"]


def load_mesh(path):
    sc = trimesh.load(path, process=False)
    m = list(sc.geometry.values())[0] if isinstance(sc, trimesh.Scene) else sc
    n = glb_vertex_count(path)
    if len(m.vertices) != n:
        sys.exit(f"{path}: trimesh reordered vertices ({len(m.vertices)} vs {n} in the GLB)")
    return m


def eye_centres(eye_mesh):
    comps = sorted(eye_mesh.split(only_watertight=False), key=lambda c: -len(c.vertices))[:2]
    cs = sorted((c.centroid for c in comps), key=lambda c: c[0])  # by X: right (-X) first
    return dict(iris_R=cs[0], iris_L=cs[1])


def surface_in_front(V, centre, radius=2.0):
    """Point on the skin straight in front (+Z) of the globe centre: the (closed) eyelid over the
    cornea.  X/Y are the globe's, only the depth comes from the surface."""
    sel = (np.hypot(V[:, 0] - centre[0], V[:, 1] - centre[1]) < radius) & (V[:, 2] > centre[2])
    return np.array([centre[0], centre[1], V[sel, 2].max()])


def midline_profile(V, x_mid, half=2.5, step=0.5):
    """Most anterior skin point per Y bin along the sagittal midline (face side, Z > 0) ->
    (Y, Z, X) arrays.  Bins the decimated mesh leaves empty are interpolated."""
    sel = (np.abs(V[:, 0] - x_mid) < half) & (V[:, 2] > 0)
    P = V[sel]
    ys = np.arange(P[:, 1].min(), P[:, 1].max() + step, step)
    idx = np.clip(((P[:, 1] - ys[0]) / step).astype(int), 0, len(ys) - 1)
    z = np.full(len(ys), -np.inf)
    np.maximum.at(z, idx, P[:, 2])
    xs = np.zeros(len(ys))
    order = np.argsort(P[:, 2])  # last write wins -> the max-Z vertex of each bin
    xs[idx[order]] = P[order, 0]
    ok = np.isfinite(z)
    z = np.interp(ys, ys[ok], z[ok])
    xs = np.interp(ys, ys[ok], xs[ok])
    # a bin whose only vertex is a nostril wall dips 10+ mm: running maximum over +-1 mm
    z = ndi.maximum_filter1d(z, size=5)
    return ys, z, xs


def profile_landmarks(V, eyes):
    x_mid = 0.5 * (eyes["iris_L"][0] + eyes["iris_R"][0])
    y_eye = 0.5 * (eyes["iris_L"][1] + eyes["iris_R"][1])
    ys, zs, xs = midline_profile(V, x_mid)
    zs_s = ndi.gaussian_filter1d(zs, 1.5)

    def pick(lo, hi, fn):
        w = (ys >= lo) & (ys <= hi)
        i = np.flatnonzero(w)[fn(zs_s[w])]
        return np.array([xs[i], ys[i], zs[i]])

    # nose tip: most anterior point below the eyes
    pron = pick(y_eye - 60, y_eye - 5, np.argmax)
    # glabella: brow bump above the eyes; sellion: deepest point between glabella and nose tip
    glab = pick(y_eye + 8, y_eye + 35, np.argmax)
    sell = pick(y_eye - 5, glab[1], np.argmin)
    # subnasale: deepest point under the nose; labrale superius: most anterior lip point below it
    sub = pick(pron[1] - 25, pron[1] - 5, np.argmin)
    lab = pick(sub[1] - 30, sub[1] - 5, np.argmax)
    return dict(glabella=glab, sellion=sell, pronasale=pron, subnasale=sub, labrale_sup=lab)


# ------------------------------------------------------------------ cameras
class View:
    """One photo: landmarks, intrinsics, and (after fitting) the pose."""

    def __init__(self, path):
        d = json.load(open(path))
        self.name = path.stem
        self.source = d.get("source", path.stem)
        self.W, self.H = d["image"]
        self.L = np.array(d["landmarks"])[:, :2]
        self.focal35 = d.get("focal35") or 26.0
        self.f = self.focal35 / DIAG_35MM * np.hypot(self.W, self.H)
        self.cxy = np.array([self.W / 2.0, self.H / 2.0])
        self.yaw = float(d.get("yaw", 0.0))
        self.rvec = self.t = None
        self.fit = {}

    @property
    def R(self):
        return Rotation.from_rotvec(self.rvec).as_matrix()

    @property
    def position(self):
        return -self.R.T @ self.t

    @property
    def view_dir(self):
        return self.R.T @ np.array([0, 0, 1.0])

    def project(self, P, rvec=None, t=None):
        R = Rotation.from_rotvec(self.rvec if rvec is None else rvec).as_matrix()
        Xc = P @ R.T + (self.t if t is None else t)
        z = np.maximum(Xc[:, 2], 1e-3)
        return np.c_[self.f * Xc[:, 0] / z + self.cxy[0], self.f * Xc[:, 1] / z + self.cxy[1]], Xc[:, 2]

    def image(self):
        return Image.open(PHOTO_DIR / f"{self.name}.jpg").convert("RGB")


def fit_reference(view, P3, P2, weights):
    """Pose of the frontal camera from the six anatomical landmark pairs.
    weights: (n, 2) per-coordinate weights (lets a landmark constrain only x or only y)."""
    d_px = np.linalg.norm(P2[0] - P2[1])
    d_mm = np.linalg.norm(P3[0] - P3[1])
    dist = view.f * d_mm / max(d_px, 1.0)  # distance from the inter-ocular pixel distance

    def resid(p):
        uv, _ = view.project(P3, p[:3], p[3:6])
        return ((uv - P2) * weights).ravel()

    best = None
    for pitch in np.linspace(-40, 40, 9):  # several starts: camera in front, tilted up/down
        # camera x = +X, y = -Y (image down = inferior), z = -Z (looking at the face)
        R0 = Rotation.from_euler("x", pitch, degrees=True).as_matrix() @ np.diag([1.0, -1.0, -1.0])
        t0 = -R0 @ P3.mean(0) + np.array([0, 0, dist])
        sol = least_squares(resid, np.r_[Rotation.from_matrix(R0).as_rotvec(), t0], method="lm", xtol=1e-12, ftol=1e-12)
        if best is None or sol.cost < best.cost:
            best = sol
    view.rvec, view.t = best.x[:3], best.x[3:6]
    uv, _ = view.project(P3)
    return uv - P2


def fit_to_points(view, ref, P3, P2, centre, f_scale=12.0, x0=None):
    """Robust PnP of a side camera from lifted 3-D landmarks and their pixels in this photo.
    Starts from x0 or, failing that, from the reference camera orbited about the head's
    vertical axis (several angles, both sides)."""
    def resid(p):
        uv, _ = view.project(P3, p[:3], p[3:6])
        return (uv - P2).ravel()

    starts = []
    if x0 is not None:
        starts.append(x0)
    else:
        for a in (-100, -75, -50, -25, 25, 50, 75, 100):
            Ry = Rotation.from_euler("y", a, degrees=True).as_matrix()
            R0 = ref.R @ Ry.T
            cam0 = centre + Ry @ (ref.position - centre)
            starts.append(np.r_[Rotation.from_matrix(R0).as_rotvec(), -R0 @ cam0])
    best = None
    for s in starts:
        sol = least_squares(resid, s, method="trf", loss="soft_l1", f_scale=f_scale, xtol=1e-10, ftol=1e-10)
        if best is None or sol.cost < best.cost:
            best = sol
    view.rvec, view.t = best.x[:3], best.x[3:6]
    uv, _ = view.project(P3)
    return uv - P2


def fit_side(view, ref, LP3, LN, lvalid, centre):
    """Side camera: PnP on the lifted landmarks, keeping only those on the half of the face that
    faces this camera, then trimming the worst residuals (Face-Mesh points that slid on the
    silhouette or the occluded cheek) in two rounds."""
    use = lvalid.copy()
    fit_to_points(view, ref, LP3[use], view.L[use], centre)
    for _ in range(3):
        cam = view.position
        to_cam = cam[None, :] - LP3
        fac = np.einsum("ij,ij->i", LN, to_cam) / np.maximum(np.linalg.norm(to_cam, axis=1), 1e-6)
        res = view.project(LP3)[0] - view.L
        err = np.hypot(res[:, 0], res[:, 1])
        keep = lvalid & (fac > 0.25)
        thr = max(2.5 * np.median(err[keep]), 20.0)
        use = keep & (err < thr)
        fit_to_points(view, ref, LP3[use], view.L[use], centre, x0=np.r_[view.rvec, view.t])
    res = view.project(LP3[use])[0] - view.L[use]
    return use, res


# ------------------------------------------------------------------ per-view projection
def project_vertices(view, V, N, F):
    """Texture coordinates + visibility + incidence of every vertex in one photo."""
    uv, depth = view.project(V)
    cam = view.position
    to_cam = cam[None, :] - V
    facing = np.einsum("ij,ij->i", N, to_cam) / np.maximum(np.linalg.norm(to_cam, axis=1), 1e-6)
    # z-buffer at 1/8 photo resolution from vertices + face centroids, min-filtered to close gaps
    ds = 8
    gw, gh = view.W // ds + 1, view.H // ds + 1
    uvc, depc = view.project(V[F].mean(1))
    pts = np.r_[uv, uvc]
    dep = np.r_[depth, depc]
    ok = (depth > 1.0)
    gx = np.clip((pts[:, 0] / ds).astype(int), 0, gw - 1)
    gy = np.clip((pts[:, 1] / ds).astype(int), 0, gh - 1)
    zbuf = np.full((gh, gw), np.inf)
    np.minimum.at(zbuf, (gy, gx), dep)
    zbuf = ndi.minimum_filter(zbuf, size=3)
    vx = np.clip((uv[:, 0] / ds).astype(int), 0, gw - 1)
    vy = np.clip((uv[:, 1] / ds).astype(int), 0, gh - 1)
    visible = ok & (depth <= zbuf[vy, vx] + 3.0)  # mm tolerance
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < view.W) & (uv[:, 1] >= 0) & (uv[:, 1] < view.H)
    return uv, depth, visible, inside, facing


def neighbour_smooth(x, edges, n_iter=2):
    """Average each vertex's value with its neighbours' mean (softens occlusion edges)."""
    i, j = edges[:, 0], edges[:, 1]
    deg = np.bincount(np.r_[i, j], minlength=len(x)).astype(float)
    for _ in range(n_iter):
        s = np.bincount(np.r_[i, j], weights=np.r_[x[j], x[i]], minlength=len(x))
        x = np.where(deg > 0, 0.5 * x + 0.5 * s / np.maximum(deg, 1), x)
    return x


def lift_landmarks(ref, V, N, uv, visible, facing, px_mm):
    """3-D surface point under each Face-Mesh landmark of the reference photo (nearest visible,
    camera-facing vertices within ~1.5 mm of the pixel).  Landmarks over surface turned more
    than ~55 deg from the camera (the sides of the cheeks) are not lifted: a pixel there covers
    a long stretch of skin.  Returns (P3, normals, valid)."""
    cand = np.flatnonzero(visible & (facing > 0.55))
    tree = cKDTree(uv[cand])
    r = 1.5 * px_mm
    P3 = np.zeros((len(ref.L), 3))
    NM = np.zeros((len(ref.L), 3))
    valid = np.zeros(len(ref.L), bool)
    for i, p in enumerate(ref.L):
        if i in NO_LIFT:
            continue
        d, k = tree.query(p, k=4, distance_upper_bound=r)
        ok = np.isfinite(d)
        if not ok.any():
            continue
        w = 1.0 / (d[ok] + 0.2 * px_mm)
        idx = cand[k[ok]]
        P3[i] = (V[idx] * w[:, None]).sum(0) / w.sum()
        NM[i] = N[idx].mean(0)
        valid[i] = True
    return P3, NM, valid


def skin_median(view):
    """Median colour of the skin inside the Face-Mesh face oval (hair / nostrils / background
    excluded by brightness)."""
    im = view.image()
    small = im.resize((view.W // 8, view.H // 8))
    mk = Image.new("L", small.size, 0)
    ImageDraw.Draw(mk).polygon([tuple((view.L[i] / 8).tolist()) for i in FACE_OVAL], fill=255)
    px = np.asarray(small)[np.asarray(mk) > 0].astype(float)
    px = px[(px.mean(1) > 60) & (px.mean(1) < 235)]
    return np.median(px, axis=0)


def colour_match(view, ref):
    """Per-channel gain taking this photo's median skin colour to the reference's.  Matching the
    whole face's median (rather than the strip both photos see) keeps a side photo from being
    darkened to the shaded, grazing cheek of the frontal one: exposure and white balance are
    matched, the photos' own lighting is left alone."""
    gain = np.clip(skin_median(ref) / np.maximum(skin_median(view), 1.0), 0.5, 2.0)
    log(f"  colour match: gain {gain.round(3)} (median skin {skin_median(view).round(0)} -> {skin_median(ref).round(0)})")
    return gain, np.zeros(3)


def crop_and_save(view, uv, weight, path, gain=None, offset=None):
    """Crop the photo to the textured region (<= MAX_TEX px), apply the colour match, write it;
    returns (u, v) in the crop and the crop box."""
    im = view.image()
    sel = weight > 0.02
    if sel.sum() < 10:
        sel = np.ones(len(uv), bool)
    x0, x1 = np.percentile(uv[sel, 0], [0.2, 99.8])
    y0, y1 = np.percentile(uv[sel, 1], [0.2, 99.8])
    pad = 0.04 * max(x1 - x0, y1 - y0)
    x0, y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    x1, y1 = int(min(view.W, x1 + pad)), int(min(view.H, y1 + pad))
    crop = im.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    s = min(1.0, MAX_TEX / max(cw, ch))
    if s < 1.0:
        crop = crop.resize((round(cw * s), round(ch * s)), Image.LANCZOS)
    if gain is not None:
        a = np.asarray(crop, dtype=np.float32) * gain[None, None, :].astype(np.float32) + offset[None, None, :].astype(np.float32)
        crop = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    crop.save(path, quality=90)
    u = (uv[:, 0] - x0) / (x1 - x0)
    v = (uv[:, 1] - y0) / (y1 - y0)  # glTF convention: v down from the top-left corner
    return u, v, [x0, y0, x1, y1], crop.size


def overlay(view, uv, visible, inside, facing, box, pairs, path):
    """Diagnostic: photo (25 %) + projected silhouette (cyan) + landmark pairs (red = from the
    mesh, green = in the photo) + the crop box."""
    sc = 0.25
    im = view.image()
    vis = im.resize((int(view.W * sc), int(view.H * sc)))
    d = ImageDraw.Draw(vis)
    rim = visible & inside & (np.abs(facing) < 0.12)
    for x, y in uv[rim][::3] * sc:
        d.point((x, y), fill=(0, 255, 255))
    for m2d, p2d, label in pairs:
        m2d, p2d = np.asarray(m2d) * sc, np.asarray(p2d) * sc
        r = 5 if label else 2
        d.ellipse([m2d[0] - r, m2d[1] - r, m2d[0] + r, m2d[1] + r], outline=(255, 60, 60), width=2 if label else 1)
        d.ellipse([p2d[0] - 2, p2d[1] - 2, p2d[0] + 2, p2d[1] + 2], fill=(60, 255, 60))
        d.line([tuple(m2d), tuple(p2d)], fill=(255, 255, 255), width=1)
        if label:
            d.text((m2d[0] + 8, m2d[1] - 6), label, fill=(255, 255, 255))
    d.rectangle([box[0] * sc, box[1] * sc, box[2] * sc, box[3] * sc], outline=(255, 128, 0), width=2)
    vis.save(path, quality=85)


# ------------------------------------------------------------------ main
def main(dataset="pre"):
    out = C.DATA_DIR / dataset
    views = [View(p) for p in sorted(PHOTO_DIR.glob("*.json")) if p.name != "landmarks.json"]
    if not views:
        sys.exit(f"no landmark files in {PHOTO_DIR} - run face_landmarks.py first")
    views.sort(key=lambda v: abs(v.yaw))
    ref, sides = views[0], views[1:]
    log(f"{len(views)} photo(s); reference (most frontal): {ref.source} (yaw {ref.yaw:+.2f}); "
        + (", ".join(f"{v.source} (yaw {v.yaw:+.2f})" for v in sides) if sides else "no side views"))
    for v in views:
        log(f"  {v.source}: {v.W}x{v.H}, focal {v.focal35:.0f} mm (35 mm eq.) -> {v.f:.0f} px")

    skin = load_mesh(out / "skin.glb")
    V = np.asarray(skin.vertices, dtype=np.float64)
    N = np.asarray(skin.vertex_normals, dtype=np.float64)
    F = np.asarray(skin.faces)
    edges = np.asarray(skin.edges_unique)
    eyes = eye_centres(load_mesh(out / "eye.glb"))
    marks = {k: surface_in_front(V, c) for k, c in eyes.items()}
    marks.update(profile_landmarks(V, eyes))
    for k, v in marks.items():
        log(f"  mesh {k:12s} X {v[0]:6.1f} Y {v[1]:6.1f} Z {v[2]:6.1f}")

    # ---- reference camera from the anatomical landmarks
    keys = ["iris_R", "iris_L", "glabella", "sellion", "pronasale", "labrale_sup"]
    P3 = np.array([marks[k] for k in keys])
    P2 = np.array([ref.L[MP[k]] for k in keys])
    # Per-coordinate weights.  With the eyes closed the "iris" of the face model sits on the lid
    # line, ~5 mm below the pupil, so the eyes constrain only x (inter-ocular scale, roll, yaw);
    # the vertical alignment comes from the midline points.  The sellion is a shallow feature.
    weights = np.array([[1, 0.1], [1, 0.1], [1, 0.6], [1, 0.3], [1, 1], [1, 0.8]], float)
    d = fit_reference(ref, P3, P2, weights)
    face_centre = P3.mean(0)
    dist = np.linalg.norm(face_centre - ref.position)
    px_mm = ref.f / dist
    rms_px = float(np.sqrt(np.sum((d * weights) ** 2) / np.sum(weights ** 2)))
    log(f"reference fit ({ref.source}), mesh landmark projected minus photo landmark (px; mm at the face):")
    for k, (dx, dy) in zip(keys, d):
        log(f"  {k:12s} dx {dx:+6.1f} dy {dy:+6.1f} px   {np.hypot(dx, dy) / px_mm:4.1f} mm")
    log(f"  weighted RMS {rms_px:.1f} px = {rms_px / px_mm:.1f} mm; camera at {ref.position.round(0)} mm, "
        f"{dist:.0f} mm from the face, {px_mm:.1f} px/mm")
    ref.fit = dict(method="anatomical landmarks", n_points=len(keys), rms_px=rms_px, rms_mm=rms_px / px_mm,
                   landmarks={k: dict(mesh=marks[k].round(2).tolist(), photo=ref.L[MP[k]].round(1).tolist(),
                                      residual_px=[float(dx), float(dy)]) for k, (dx, dy) in zip(keys, d)})
    ref.pairs = [(ref.project(marks[k][None])[0][0], ref.L[MP[k]], k) for k in keys]

    # ---- project through the reference, lift its Face-Mesh landmarks onto the surface
    proj = {ref.name: project_vertices(ref, V, N, F)}
    uv_r, _, vis_r, ins_r, fac_r = proj[ref.name]
    LP3, LN, lvalid = lift_landmarks(ref, V, N, uv_r, vis_r, fac_r, px_mm)
    log(f"lifted {lvalid.sum()} of {len(lvalid)} Face-Mesh landmarks onto the skin surface")

    # ---- side cameras: robust PnP against the lifted landmarks
    for v in sides:
        use, res = fit_side(v, ref, LP3, LN, lvalid, face_centre)
        dist_v = np.linalg.norm(face_centre - v.position)
        px_mm_v = v.f / dist_v
        err = np.hypot(res[:, 0], res[:, 1])
        rms = float(np.sqrt(np.mean(err ** 2)))
        med = float(np.median(err))
        log(f"side fit ({v.source}): {use.sum()} landmarks, RMS {rms:.1f} px = {rms / px_mm_v:.1f} mm, "
            f"median {med:.1f} px = {med / px_mm_v:.1f} mm; camera at {v.position.round(0)} mm, {dist_v:.0f} mm from the face")
        v.fit = dict(method="PnP on Face-Mesh landmarks lifted through the reference photo",
                     n_points=int(use.sum()), rms_px=rms, rms_mm=rms / px_mm_v, median_px=med, median_mm=med / px_mm_v)
        v.pairs = [(m, p, None) for m, p in zip(v.project(LP3[use])[0], v.L[use])]
        proj[v.name] = project_vertices(v, V, N, F)

    # ---- per-view weights
    weights_v = {}
    base_v = {}
    for v in views:
        uv, depth, visible, inside, facing = proj[v.name]
        base = visible * inside * smoothstep(facing, 0.15, 0.5)
        base = neighbour_smooth(base, edges)
        prior = 1.0 if v is ref else PRIOR_SIDE
        weights_v[v.name] = base * np.clip(facing, 0, 1) ** FACING_POW * prior
        base_v[v.name] = base
    wsum = sum(weights_v.values())
    cover = np.clip(wsum / WEIGHT_SCALE, 0, 1)
    log(f"textured vertices: {np.mean(cover > 0.5) * 100:.0f} % of {len(V)} "
        f"(reference alone: {np.mean(np.clip(weights_v[ref.name] / WEIGHT_SCALE, 0, 1) > 0.5) * 100:.0f} %)")
    for v in views:
        share = weights_v[v.name] / np.maximum(wsum, 1e-9)
        log(f"  {v.source}: main photo on {np.mean((share > 0.5) & (cover > 0.5)) * 100:.0f} % of the textured surface")

    # ---- colour-match the side photos to the reference, write textures + UVs
    blocks = []
    meta_views = []
    for i, v in enumerate(views):
        uv, depth, visible, inside, facing = proj[v.name]
        gain = offset = None
        if v is not ref:
            gain, offset = colour_match(v, ref)
        fname = "skin_photo.jpg" if i == 0 else f"skin_photo_{i}.jpg"
        u, vv, box, size = crop_and_save(v, uv, weights_v[v.name], out / fname, gain, offset)
        blocks.append(np.c_[u, vv, weights_v[v.name]].astype(np.float32))
        overlay(v, uv, visible, inside, facing, box, v.pairs, PHOTO_DIR / f"fit_{v.name}.jpg")
        dist_v = float(np.linalg.norm(face_centre - v.position))
        meta_views.append(dict(
            name=v.name, source=v.source, file=fname, image=list(size), crop_px=box, photo_px=[v.W, v.H],
            focal_px=float(v.f), focal35=float(v.focal35), px_per_mm=float(v.f / dist_v), yaw=v.yaw,
            camera=dict(position=v.position.round(2).tolist(), view_dir=v.view_dir.round(4).tolist(), distance_mm=dist_v),
            fit=v.fit, prior=1.0 if v is ref else PRIOR_SIDE,
            colour_match=None if gain is None else dict(gain=gain.round(4).tolist(), offset=offset.round(2).tolist()),
            textured_fraction=float(np.mean(np.clip(weights_v[v.name] / WEIGHT_SCALE, 0, 1) > 0.5)),
        ))
        log(f"  wrote {fname} ({size[0]}x{size[1]}), overlay -> fit_{v.name}.jpg")
    np.concatenate(blocks).tofile(out / "skin_uv.bin")

    # ---- mean skin tone (reference photo's face oval, without the dark hair / nostrils)
    im = ref.image()
    small = im.resize((ref.W // 8, ref.H // 8))
    mk = Image.new("L", small.size, 0)
    ImageDraw.Draw(mk).polygon([tuple((ref.L[i] / 8).tolist()) for i in FACE_OVAL], fill=255)
    px = np.asarray(small)[np.asarray(mk) > 0].astype(float)
    px = px[px.mean(1) > 60]
    tone = np.median(px, axis=0)
    tone_hex = "#%02x%02x%02x" % tuple(int(c) for c in tone)
    log("skin tone", tone_hex)

    meta = dict(
        vertices=int(len(V)), uv_file="skin_uv.bin", uv_layout="[view][vertex][u, v, weight] float32",
        weight_scale=WEIGHT_SCALE, views=meta_views, skin_tone=tone_hex,
        textured_fraction=float(np.mean(cover > 0.5)),
        # the reference view's summary, for readers of the single-photo format
        source=ref.source, file="skin_photo.jpg", camera=meta_views[0]["camera"],
        rms_px=ref.fit["rms_px"], rms_mm=ref.fit["rms_mm"],
    )
    json.dump(meta, open(out / "skin_photo.json", "w"), indent=1)
    log(f"wrote skin_uv.bin ({len(views)} view(s) x {len(V)} vertices), skin_photo.json")


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
