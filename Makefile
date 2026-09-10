# airway3d — one-command pipeline.  `make` builds everything, `make serve` opens the viewer.
PY := .venv/bin/python
PY_FACE := .venv-face/bin/python
PYTHON_FACE ?= python3.12   # MediaPipe (face landmarks) has no Python 3.13 wheels yet
PORT ?= 8765

.PHONY: all env env-face volume segment mesh airway surgery post photo serve clean

all: airway post

env: .venv/.ok
.venv/.ok:
	python3 -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q numpy scipy scikit-image pillow trimesh fast_simplification
	touch $@

# 1. stack + register the two screenshot series into calibrated volumes (work/)
volume: env
	$(PY) pipeline/build_volume.py

# 2. tissue segmentation (labels, airway, sinuses)
segment: volume
	$(PY) pipeline/segment.py

# 3. surface meshes + web volumes for the pre-op dataset
mesh: segment
	$(PY) pipeline/mesh.py pre

# 4. nostril split, cross-sections, 1-D hydraulics, potential-flow field
airway: mesh
	$(PY) pipeline/airway.py pre

# 5. virtual inferior turbinate reduction -> "post" dataset (edit the flags to plan a different cut)
surgery: segment
	$(PY) pipeline/surgery.py --side both --depth 2.0 --lower 0.45 --name post

post: mesh surgery
	$(PY) pipeline/mesh.py post
	$(PY) pipeline/airway.py post

# 6. the patient's photos (skin/*.heic|jpg|png: one frontal + any others) projected onto the skin
# mesh: face landmarks with MediaPipe, the frontal camera fitted to the mesh landmarks, the other
# cameras to the frontal photo, per-vertex UVs + blend weights per photo -> viewer/data/pre.
# Needs the pre-op meshes (`make mesh`); re-run after them (the UVs are tied to the skin mesh's vertices).
env-face: .venv-face/.ok
.venv-face/.ok:
	$(PYTHON_FACE) -m venv .venv-face
	.venv-face/bin/pip install -q --upgrade pip
	.venv-face/bin/pip install -q "mediapipe==0.10.21" pillow-heif
	touch $@

photo: env-face
	$(PY_FACE) pipeline/face_landmarks.py
	$(PY) pipeline/skin_photo.py pre

serve:
	python3 serve.py $(PORT)

clean:
	rm -rf work viewer/data
