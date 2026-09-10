"""Face landmarks for the reference photos in skin/ (runs in the MediaPipe venv, .venv-face).

Input : skin/*.(heic|jpg|png)        - photos of the patient: one frontal, optionally more from
                                       other angles (the sides, below...)
Output: work/photo/<stem>.jpg         - each photo as sRGB JPEG, EXIF orientation applied
        work/photo/<stem>.json        - 478 MediaPipe Face-Mesh landmarks (px) + focal length +
                                       a yaw estimate (0 = frontal, negative = the patient's
                                       right side of the face is turned towards the camera)

MediaPipe 0.10.x (TensorFlow Lite) has no Python 3.13 wheels, so this step lives in its own
virtual environment (make env-face); everything downstream (skin_photo.py) uses the main one.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
PHOTO_DIR = ROOT / "skin"
OUT_DIR = ROOT / "work" / "photo"
MODEL = ROOT / "work" / "models" / "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
EXTS = ("heic", "jpg", "jpeg", "png")


def find_photos():
    ps = [p for p in sorted(PHOTO_DIR.iterdir()) if p.suffix.lower().lstrip(".") in EXTS and not p.name.startswith(".")]
    if not ps:
        sys.exit(f"no photo found in {PHOTO_DIR}")
    return ps


def load_photo(path):
    """Open (HEIC via pillow-heif), apply the EXIF orientation, return (PIL RGB image, focal35)."""
    if path.suffix.lower() == ".heic":
        import pillow_heif
        pillow_heif.register_heif_opener()
    im = Image.open(path)
    exif = im.getexif()
    focal35 = None
    try:
        ex = exif.get_ifd(0x8769)  # Exif IFD
        focal35 = ex.get(0xA405) or None  # FocalLengthIn35mmFilm
        if focal35 is None and ex.get(0x920A):  # FocalLength alone is useless without the sensor size
            print("photo has FocalLength but no 35 mm equivalent; assuming a phone main camera (26 mm)")
            focal35 = 26
    except Exception:
        pass
    im = ImageOps.exif_transpose(im).convert("RGB")
    return im, (float(focal35) if focal35 else None)


def yaw_estimate(pts):
    """Head yaw from the landmark layout: where the nose tip sits between the two cheek-side
    face-oval points (234 = patient's right, 454 = patient's left), in units of that width.
    ~0 frontal, -1 / +1 towards a full right / left profile (patient's side facing the camera)."""
    r, l, nose = pts[234, :2], pts[454, :2], pts[1, :2]
    axis = l - r
    w = np.linalg.norm(axis)
    if w < 1e-6:
        return 0.0
    t = np.dot(nose - r, axis) / (w * w)  # 0 at the right oval point, 1 at the left
    return float(np.clip(2.0 * t - 1.0, -1.5, 1.5)) * -1.0  # nose towards the right point (t < 0.5) -> right side visible


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL.exists():
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", MODEL_URL)
        urllib.request.urlretrieve(MODEL_URL, MODEL)

    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(MODEL)), num_faces=1,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
    photos = find_photos()
    # outputs of photos that are no longer in skin/ (replaced or removed) must not linger:
    # skin_photo.py uses every landmark file it finds
    stems = {p.stem for p in photos}
    for old in list(OUT_DIR.glob("*.json")) + list(OUT_DIR.glob("*.jpg")):
        stem = old.stem[4:] if old.stem.startswith("fit_") else old.stem
        if stem not in stems:
            old.unlink()
            print(f"removed stale {old.name}")
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for src in photos:
            im, focal35 = load_photo(src)
            w, h = im.size
            im.save(OUT_DIR / f"{src.stem}.jpg", quality=95)
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(im)))
            if not res.face_landmarks:
                print(f"{src.name}: {w}x{h}, focal {focal35} mm - NO FACE DETECTED, skipped")
                continue
            pts = np.array([[p.x * w, p.y * h, p.z * w] for p in res.face_landmarks[0]], dtype=np.float64)
            yaw = yaw_estimate(pts)
            out = dict(source=src.name, image=[w, h], focal35=focal35, yaw=yaw, landmarks=pts.round(2).tolist())
            json.dump(out, open(OUT_DIR / f"{src.stem}.json", "w"))
            print(f"{src.name}: {w}x{h}, focal {focal35} mm (35 mm eq.), yaw {yaw:+.2f}, {len(pts)} landmarks -> {src.stem}.json")


if __name__ == "__main__":
    main()
