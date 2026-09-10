# airway3d — head reconstruction & nasal airway lab

<video src="https://github.com/eddydong/airway3d/raw/main/docs/airway3d_720.mp4" controls muted playsinline width="720"></video>

[Preview video (720p)](docs/airway3d_720.mp4)

A 3-D reconstruction of one patient's head from 209 CT viewer screenshots, with a
browser viewer that separates skin, fat, muscle, soft tissue, bone, brain, eyeballs,
sinuses and the airway (nostrils → pharynx); measures each nostril's passage
(volume, cross-section profile, minimum cross-section, resistance, speed, wall
shear); shows a particle airflow field; and previews a virtual inferior
turbinate reduction as a second, comparable dataset.

Everything is open source and runs locally: a Python pipeline builds the data,
a static Three.js page renders it. **CT screenshot stacks, patient photographs,
and generated meshes/volumes are not in this repository** — keep them on the
machine that runs the pipeline.

## Layout

```
slices/           local CT screenshot stacks (PNG, gitignored) + windowing notes / slice index CSVs
skin/             local patient photos (HEIC/JPEG/PNG, gitignored) for texturing the skin mesh
pipeline/         Python: volume → segmentation → meshes → airway metrics → virtual surgery → photo skin
viewer/           Three.js viewer (vendored, no build step); viewer/data/{pre,post} is generated (gitignored)
work/             intermediate NumPy volumes (generated, gitignored)
docs/airway3d_720.mp4   viewer preview
```

## Offline playback and wall temperature

The viewer is now read-only: it loads precomputed CFD and thermal recordings
from a static catalog and cannot submit jobs. Missing settings show **Not
precomputed** immediately. The four baseline body positions and the extra 2 s
face-up recording have offline airflow results. **Airway wall display** plays
computed surface temperature (°C) and sensible mucosal heat flux (W/m²) from
3-D energy transport coupled to a finite-capacity warm tissue layer. The new
**CFD airway · solver surface** tissue row controls visualization only.
See [offline thermal workflow and limits](docs/THERMAL_PLAYBACK.md).

## Run

```bash
make            # .venv + full pipeline (pre-op and post-op datasets), several minutes
make serve      # http://127.0.0.1:8765/
```

Steps individually: `make volume`, `make segment`, `make mesh`, `make airway`,
`make surgery`, `make post`. Every step is a plain script in `pipeline/` that
reads/writes `work/` and `viewer/data/`. `make photo` maps the photos in `skin/`
onto the skin mesh (needs `python3.12` for MediaPipe; it builds its own
`.venv-face`); re-run it after `make mesh`. Photos are picked up by name-agnostic
glob: drop in a frontal one plus any others (the sides, below) and re-run.

The viewer needs WebGL 2 (Chrome, Safari 16+, Firefox). Data is ~230 MB per
dataset pair and is fetched once per page load.

## Scenario lab (interactive what-if workspace)

The CT reference is **face up (supine)**, as confirmed by the patient. Sources are
PNG screenshots only: skull **C400/W1500**, head **C40/W350**, slice spacing
**0.625 mm**, reported IPD **68 mm**. Left-side blockage and better breathing on the
right are patient observations, not conditions captured by this CT.

Open **Scenario lab** at the top left. The same button returns to tissue controls.
The head stays visible throughout scenario changes:

- Saved CT / 2 mm reduction swaps stage changed assets first, then replace only
  changed geometry and label/flow resources. Shared skin, photos, grey volumes,
  camera, clipping and tissue materials persist. Previously loaded assets are
  cached. Loading failures retain the current scene; newer requests supersede
  older ones. The full-screen loading overlay is only used on initial startup.
- Independent left/right sliders explore inferior turbinate head/body clearance
  and anterior valve widening. These are assumed equivalent-radius gains, not
  measured resection depths. **Show region guides** and **Focus region** expose
  approximate envelopes with labels. They are not anatomical segmentations or
  recommendations about safe tissue to remove.
- Swelling suppression is an independent hypothesis, not an assumed automatic
  consequence of bone removal. Save up to three intervention plans locally;
  recall preserves current physiology assumptions so comparisons remain matched.
- **Body position** supports face up (the confirmed CT reference), left side,
  right side and upright. Adjust assumed dependent swelling, immediate radial
  displacement, cycle bias and settling time; play the response over time.
  The reported left-side blockage preset is an illustrative sensitivity test.
  These tissue responses cannot be measured from this one PNG series.
- A **Congestion proxy** stays pinned above the lab controls. It shows left/right
  estimated resistance at 150 Pa (Pa·s/mL), plus each side's percentage change
  from the observed face-up CT. Higher means less airflow at the same pressure;
  closed branches display **Closed**. It updates with intervention, posture,
  swelling and playback changes, including saved-dataset switches. The fixed
  pressure makes it independent of particle speed, breathing phase and peak flow.
  This is not a patient symptom score. **About** explains patient-rated 0–10
  congestion and the role of mucosal cooling. The adjacent panel and the particle
  **mucosal heat flux · inspiration** colour mode use the same transparent
  reduced-order thermal estimate.
- The adjacent **Mucosal heat flux** readout estimates inspiratory cooling from
  local 1-D convection and bulk-air warming. It highlights cooled mucosal area
  above 50 W/m², plus peak flux and total heat loss, for each side. This is a
  transparent reduced-order thermal proxy in estimated mode. The separate CFD
  wall display now uses offline 3-D sensible energy transport and a tissue slab;
  humidity and evaporation are not solved. The comparison report and PDF include
  these values and the thermal limitations.
- The airway surfaces and nearby soft envelope deform in place from immutable
  reference geometry, with a short visual transition. Bone and other tissue
  classes stay at the scan. The deformation is schematic and may intersect
  tissue; CT cuts/volume rendering remain the selected saved anatomy.
- Flow split, viscous friction, local losses, pressure, resistance, speed and
  shear estimates are available from the live **1-D** model. A fully closed
  branch carries zero flow; bilateral closure makes prescribed flow infeasible.
  The Scenario lab also has a geometry-backed **Viscous CFD** mode. It builds a
  watertight voxel volume from the screenshot reconstruction, runs pinned
  OpenFOAM 2412 laminar Navier–Stokes with no-slip walls, and only imports a
  result when mesh, residual, mass-balance, wall-leak and pressure-stability
  gates pass. The particle system then samples that native solved velocity
  field; it does not warp a stored field or reverse it to pretend to be a new
  solve. A matching result is required for every posture/intervention/direction
  request. CFD withholds particles and readings for unrecorded settings;
  turn off **CFD playback** to return to the estimated breathing cycle.
  Geometry and flow calculations run only in offline scripts.
- **Compare plans × body positions** opens a sortable, filterable report. Select
  a row for per-side losses and area profiles, then preview its particle flow.
  Every arm shares flow and tissue-response assumptions. Identical intervention
  arms are deduplicated.
- **Download full PDF** creates a complete, paginated report locally, including
  the complete intervention/posture comparison, branch metrics, charts,
  equations, source windows, assumptions and validation gaps. **Download data**
  includes full unrounded profiles and settings. Neither requires an account,
  network service, PDF dependency or upload of the patient's scans.

This is an exploratory sensitivity tool. The current 1-D equations are tested
against analytic pipe friction and flow-conservation cases, but the patient's
segmentation, posture response and surgical outcome are not clinically validated.
For the viscous 3-D solver and doctor-facing validation workflow, see
[docs/CFD_VALIDATION.md](docs/CFD_VALIDATION.md).

Run the numerical checks after generating `viewer/data/pre/airway.json`:

```bash
node --test tests/*.test.mjs
```

## CFD geometry and recorded breathing

Offline scripts rebuild exact solver geometry for each intervention or posture.
**Recorded breathing cycle** plays saved velocity frames with synchronized
readings. **Compare plans × body positions with CFD** reads separate
geometry-matched recordings and withholds unavailable results. No calculations
are submitted from the viewer. Walls remain rigid during each breath.

See [CFD breathing and comparison workflow](docs/CFD_BREATHING.md) for usage,
validation results and remaining limits. The estimated breathing mode and its
saved post-op example remain separate.

The **GPU lattice** solver (FluidX3D lattice Boltzmann with an LES closure,
pressure-driven breathing) records the same kind of trajectory in tens of
minutes instead of days, which makes the comparison grid a precomputed
**scenario library** that replays instantly. Build it once with
`.venv/bin/python pipeline/gpu_cfd_build.py`; fill the library offline
with `pipeline/cfd_library.py`. See
[GPU lattice recordings and the scenario library](docs/CFD_GPU.md) for the
boundary treatment, validation and limits.

## Tool choices

| need | choice | why |
| --- | --- | --- |
| reconstruction / segmentation | Python, NumPy, SciPy, scikit-image | free, fast enough on 300³ grids, easy to audit and tune; no black-box |
| meshing | Gaussian-smoothed masks → scikit-image marching cubes → `fast_simplification` quadric decimation → GLB (trimesh) | standard, quality control per tissue (smoothing radius, triangle budget) |
| renderer | Three.js (WebGL 2), vendored | PBR materials, environment lighting, clipping planes, custom GLSL for volume ray-marching and particles; zero install, runs on any Mac/PC, screenshots via the page; a native Mac app (SceneKit/Metal) gives no visual advantage for this data and would cost portability |
| volume rendering | own GLSL ray-marcher over three 3-D textures (bone window, soft window, labels) | per-tissue transfer function, gradient shading, honours the same cut planes as the meshes, composited behind the surfaces with the depth buffer |
| airflow | local OpenFOAM 2412 viscous CFD + 1-D fallback | solved fields are geometry-hashed, checked and imported for matching lab requests; the explicit estimated mode provides a breathing cycle; steady CFD waits for a matching result |

## What the pipeline does

1. **Volume** (`build_volume.py`). Stacks the bone-window PNGs (0.625 mm apart); the soft-tissue series is registered onto the same grid (scale 0.9963, shift 6.2/−3.1 px, fitted on saturated bone) and resampled. Two grids: fine 0.277 mm in-plane (bone, airway) and coarse 0.555 mm (tissues, web volume). Grey → HU is the linear window mapping (bone window: HU = 5.88 g − 350).
2. **Calibration.** No DICOM header, so the pixel size comes from anatomy: the fitted centres of the two globes are 490.4 px apart and the patient's inter-pupillary distance is 68 mm → 0.1387 mm/px (`AIRWAY3D_PX_MM` overrides it). Sanity checks: reconstruction FOV 212 mm, head 164 mm wide at the supraorbital level.
3. **Segmentation** (`segment.py`). Bone and air from the bone window; fat / soft tissue / muscle from the soft window (−12 HU and 47 HU cut-offs); skin as a 2 mm shell of the body; brain as the large soft component enclosed by bone; eyeballs as the two spherical soft blobs in the orbits; airway as air connected to the nostrils and pharynx (down to the nares on both sides), split from sinuses/air cells by connectivity through the ostia. Teeth stay inside the bone class for now.
4. **Meshes** (`mesh.py`). One GLB per tissue, plus `airway_L`, `airway_R`, `airway_common` (pharynx), plus the three 8-bit volumes for the GPU renderer, plus `meta.json` (label table, per-class volumes, world box). The skin mesh is the body with every enclosed air pocket filled (thousands of sub-30 mm³ specks that no class claims), so it is a single outer surface; air that reaches the outside (nostrils, ear canals) stays open. The airway is part of that solid (so the skin has no inner cavity walls) except the first 12 mm inside each nostril, which are carved out: the skin follows the vestibule walls and the airway mesh is seen inside the opening (with the vestibules filled, the skin would have a flat cap across each nostril coincident with the airway's inlet cap, and the two would z-fight). Every mesh is oriented outwards body by body (`fix_normals(multibody=True)` plus a total-volume check): the winding pass starts from an arbitrary face, and the decimated left airway had come out inside-out — translucent tissues are drawn front-faces-only, so its inlet cap was culled and only its far walls showed through the left nostril while the right nostril showed its cap.
5. **Airway metrics** (`airway.py`, `pipeline/cfd_geometry.py`). Nostril inlets are detected where external air meets the airway; each nostril's passage is the region geodesically closer to that nostril, cut at the choanae; cross-sections are geodesic-distance shells sampled every 1 mm (area, perimeter, hydraulic diameter, centroid). The 1-D model remains the instant preview. The CFD path crops that same screenshot-derived segmentation, labels exactly two nostril contacts plus the inferior outlet, regularizes diagonal contacts without bridging the septum, requires a closed oriented manifold, and records a geometry hash. Its OpenFOAM field is the authoritative particle source when a matching solve is available.
6. **Virtual surgery** (`surgery.py`). Parametric inferior turbinate reduction: on each side, the tissue lateral to the passage in the lower part of the nasal cavity, within `--depth` mm of the airway wall, is converted to airway (mucosa and turbinate bone alike — a submucous resection removes bone but the mucosa collapses onto the new bed, so the airway gain is what the depth models). Writes a `post` dataset; unchanged meshes are shared with `pre`.
7. **Photo skin** (`face_landmarks.py` + `skin_photo.py`). Every photo in `skin/` is a pinhole camera (focal length from EXIF, 52 mm-equivalent here). The most frontal one (by the layout of its Face-Mesh landmarks) is the *reference*: its pose in the CT frame is solved from landmarks present on both sides — the two eye centres (globe centroids, projected to the eyelid surface), glabella, sellion, nose tip and upper-lip point — found on the mesh from the eyeball segmentation and the midline skin profile, and on the photo by MediaPipe Face Mesh (478 points). Because the eyes are shut, the face model's "iris" sits on the lid line ~6 mm below the pupil, so the eyes only constrain the horizontal (inter-ocular scale, roll, yaw) and the midline points set the vertical. The other photos (here the two sides, nearly in profile) are registered *to the reference photo* rather than to the CT: each Face-Mesh landmark of the reference is lifted onto the skin surface through the fitted camera, which yields ~330 3-D points with a known landmark index; the same indices in a side photo are 2-D observations of those points, and the side camera is a robust PnP fit over the ~200 on the visible half of the face (worst residuals trimmed — Face-Mesh points slide on a profile). Registering photo-to-photo keeps the textures mutually aligned where they blend even if the CT-to-photo fit has a millimetre of slack. Every skin vertex is projected into every photo; a per-view weight combines z-buffer visibility, incidence (∝ facing³ — a photo is smeared where the surface turns away from it) and a prior favouring the reference where several photos are good. Side photos are colour-matched to the reference by the median skin colour inside the face oval (exposure / white balance only; each photo keeps its own lighting). Outputs `skin_photo.jpg`, `skin_photo_1.jpg`, … (crops), `skin_uv.bin` (per view, per vertex: u, v, weight — tied to `skin.glb`'s vertex order, the viewer checks the count) and `skin_photo.json` (cameras, residuals, colour gains, mean skin tone); `work/photo/fit_<photo>.jpg` shows the projected silhouette and landmark pairs on each photo for checking. On these photos the reference fits to 1.1 mm weighted RMS (nose tip 0.7 mm, lip 1.2 mm; camera 47 cm from the face, 12 px/mm) and the sides to a 3–3.5 mm median over the lifted landmarks — part of which is the landmarks' own drift on a profile view, not texture misalignment.

## Viewer

- **Presets** Portrait, Dissect, Muscles, Skeleton, Airway, Brain (keys 1–6). Every tissue has visibility, colour, opacity, and a material style (matte/satin/soft/wet/glossy/glass).
- **Photo skin** when `make photo` has run, the skin row gains *photo* (blend between the flat skin colour and the projected photos) and *photo light* (exposure of the photos). The material samples every photo and blends them by the per-vertex weights, normalised per pixel — the face comes from the frontal photo, the cheeks, temples, ears and hair from the side ones, with a continuous hand-over between — and fades to the flat colour where their sum is small. The skin's default colour becomes the frontal photo's mean skin tone and its material *soft*, so the parts no camera saw (top of the head, under the chin, the cut faces) continue the face without a seam or plastic glare; Portrait sets the photo to 100 %. The photos carry their own baked lighting, so the skin looks most natural with the scene lights roughly frontal.
- **Render modes** Surfaces (PBR meshes), Volume (GPU ray-marching with per-tissue transfer function; the opacity sliders set how opaque ~3 mm of tissue is), Hybrid (surfaces in front, volume behind them).
- **Translucency** the tissues are nested shells, so they are drawn inside → outside (airway first, skin last) and each translucent shell blends over what it encloses: a 96 % skin shows 4 % of the inside, and 100 % none — the transition is continuous. The skin, a single closed surface, gets a depth pre-pass while translucent, so only its nearest layer is drawn (its far side never ghosts through the face, and what is inside is attenuated by exactly one layer). Multi-piece tissues (bone, muscles, glands) keep all their layers as a depth cue. Particles inside the airway are drawn just after the airway meshes, so the surrounding tissues dim them like everything else inside; the exhaled plume and room air are drawn after the skin.
- **Cut planes** sagittal / axial / coronal with flip; the CT slice is drawn at the cut — bone or soft-tissue window, or a label overlay. Outside air is transparent so the section reads as a real dissection.
- **Airway panel** per-side volume, length, minimum cross-section and where it is (a marker in 3-D), mean cross-section, flow share, pressure drop, resistance, peak/mean speed, wall shear, Reynolds number, flow at 150 Pa; area-vs-distance chart; centrelines in 3-D. The flow rate slider (default 250 mL/s ≈ quiet breathing) re-evaluates the 1-D model live.
- **Airflow** shows speed-colored particles and trails; `F` toggles them. Scenario lab also provides **View airflow** and **Pause/Resume particles**. Breathing uses 42% of the animation period for inspiration, with longer, slower expiration of equal integrated volume. Room-air seeding and the exhaled plume remain active in scenarios. CFD mode samples the native solved velocity field for the selected direction and geometry; no field reversal, local scaling or particle-count measurement is used. While a matching result is pending, CFD readings and particles are withheld. Turn off **Steady CFD mode** to return to the estimated breathing cycle. *Peak flow* drives the fallback 1-D metrics; *Speed* controls animation time. *Look* offers Glow, Heat (speed ramp), Ice, Smoke, Vapour, Streaklines and Tracer dots. *Colour by* selects speed, source nostril, or live reduced-order mucosal heat flux during inspiration; the thermal mode can color both particles and airway walls blue → cyan → green → yellow → red. Exhaled and room particles are neutral in thermal mode. Breathing and appearance controls work across posture changes and interventions.
- **Saved datasets** in Scenario lab switch between the CT reference and the saved geometric reduction without a full scene reload. Configurable what-if deltas compare against no intervention in the same posture.
- `R` resets the camera; the camera button saves a PNG.

## Results for this patient (pre-op, 250 mL/s)

| | left | right | pharynx |
| --- | --- | --- | --- |
| passage volume | 10.5 cc | 10.1 cc | 18.0 cc |
| minimum cross-section | 23.7 mm² at 37 mm | 52.2 mm² at 40 mm | 175 mm² |
| flow share | 31 % | 69 % | — |
| resistance | 0.31 Pa·s/mL | 0.14 Pa·s/mL | 0.015 Pa·s/mL |
| peak speed | 5.9 m/s | 3.3 m/s | 3.8 m/s |

Total resistance nostrils → pharynx 0.11 Pa·s/mL (the two nasal sides in
parallel, plus the pharynx), i.e. about 27 Pa at 250 mL/s in this approximation, with an asymmetric left-sided restriction: the left minimum
cross-section is 45 % of the right's and it sits ~15 mm behind the nostril
(anterior cavity / valve region), which the 2 mm bilateral inferior turbinate
reduction barely changes (24.1 mm² after). The modelled surgery adds
3.3 cc / 2.0 cc of passage volume, lowers the total resistance to 0.08 Pa·s/mL
(21 Pa), and moves the flow split to 34 / 66 %. Use the sagittal/coronal cut at
the marked minimum to judge whether the left narrowing is septal, valve, or
turbinate head.

## Limitations — read before drawing conclusions

- **Source data are screenshots, not DICOM.** Grey levels are 8-bit windowed values, so HU is approximate and everything outside the window saturates (air below −350 HU, dense bone above 1150 HU in the bone series). The in-plane scale is calibrated from the IPD (±1–2 %); a DICOM header or a ruler in one slice would remove that uncertainty.
- **Segmentation is heuristic** (thresholds + anatomy rules). Thin bone (lamina papyracea, septum plates ~0.3 mm) partly vanishes at 0.28 mm voxels; fat / muscle / gland boundaries are threshold-based; skin is a fixed 2 mm shell. Volumes per class are indicative, not clinical.
- **Airway split** is geometric (geodesic distance from each nostril, cut at the choanal plane). The turbinate zone used by the virtual surgery is also geometric (lateral, lower part of the cavity); it does not identify the turbinate bone itself.
- **The stored airflow is potential flow**: no viscosity, turbulence, separation or heating/humidification. Scenario particles adapt those trajectories using 1-D flow/area ratios. Their direction and speed have not been validated against this patient's flow, and new recirculation or rerouting is not resolved. Resistance numbers come from a 1-D approximation. Analytic pipe and particle-regression tests verify implementation behavior; they do not validate patient-specific estimates or a surgical decision.
- **Unmeasured mucosa dynamics.** The scan is one instant; posture-dependent
  tissue response is intentionally not modeled in this release. Breathing
  animation changes the stored flow field over time; it does not animate tissue
  swelling.
- **The photo skin is three projected views.** Six landmarks fix the frontal camera to ~1 mm; the side cameras are fitted to the frontal photo through Face-Mesh landmarks, which drift by a few mm on a near-profile view, so features in the hand-over band (cheek, jaw) can be doubled by that much. The eyes are shut in all photos; anything under the chin and the top of the head are untextured; each photo's own lighting is baked in (the sides were shot in the same room as the front, so the exposure match is close). A photo from below would fill in under the nose and chin the same way.

## Roadmap

1. Import DICOM directly (pydicom) → true HU, exact spacing, no de-labelling.
2. Learning-based segmentation (e.g. TotalSegmentator / nnU-Net head models) as a second opinion for soft tissue, then a cleaner turbinate bone / mucosa split so a *submucous* resection can be modelled as bone removal + mucosal re-draping.
3. Extend the local OpenFOAM path to several flow rates and both directions, then add patient-specific mesh-independence and pressure-flow validation; the current lab already runs one pinned viscous case and imports its native fields.
4. Rhinomanometry-style outputs (pressure–flow curves, both sides, both directions) and the standard clinical indices (MCA, volume 0–5 cm, NAR).
5. More operations: septoplasty (parametric septum straightening), valve widening, adenoid removal; batch comparison of variants.
6. Optional native wrapper (Tauri/Electron) if a desktop app is wanted; the WebGL viewer is the core either way.
