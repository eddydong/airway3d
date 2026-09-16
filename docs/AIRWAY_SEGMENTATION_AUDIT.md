# Airway separation audit

Audit date: 2026-09-16. Scope: the nasal and upper airway represented in this
repository, from nostrils through the nasal cavities to the pharyngeal outlet.
This is not an audit of the intrathoracic bronchial tree or of other organ labels.

**The current method is a one-patient reconstruction prototype. It has no basis
for an all-patient accuracy claim.** The appropriate target is a common workflow
that adapts to each patient's anatomy and acquisition, quantifies uncertainty,
and requires correction or rejects cases when the anatomy cannot be resolved.
This audit identifies failure mechanisms; it does not establish the anatomical
accuracy of the current patient's segmentation.

The existing sequence is already volume → voxel segmentation → surface meshes.
That order is correct. Improve airway separation in the original voxel volume;
separating objects after producing a smoothed 3D surface cannot restore lost
image information. Detailed brain, eye, fat and muscle classifications need not
be prerequisites for an airway research pipeline. The surrounding tissue, bone,
sinus boundaries and external air still provide useful context.

## Findings, in priority order

### P1 — Screenshot intensities cannot support calibrated airway boundaries

`pipeline/config.py:14–19,31–57`, `pipeline/build_volume.py:61–69`, and
`pipeline/segment.py:27,99,114,123,323` show that the pipeline box-averages
8-bit bone-window screenshots, smooths them, and uses gray < 22 for air. Under
its approximate inverse mapping, 22 corresponds to −220.6 display-equivalent
HU. This is not a calibrated native CT threshold after clipping and averaging.

The C400/W1500 display clips values below approximately −350 HU. For example,
−1000 and −600 HU map to the same black value before screenshot resampling.
The original air–tissue mixture is unrecoverable there. DICOM explicitly defines
windowing as a display operation with clipping; this limitation follows from
that transformation. [DICOM VOI LUT specification](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html)

**Consequence:** narrow lumen width, wall position and apparent patency can
depend on display settings, zoom, interpolation and smoothing. A prettier mesh
cannot resolve this ambiguity. The existing screenshot case should remain an
approximate reconstruction.

**Required change:** use original CT voxel data with verified rescale units and
geometry. Preserve native calibrated intensities for segmentation and QC.
Retain screenshots as an explicitly lower-confidence import path. DICOM CT
defines rescale slope/intercept and when output units are HU.
[DICOM CT module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.2.html)

### P1 — Geometry and orientation are specific to the present patient

`pipeline/config.py:31–57` fixes an IPD-derived 0.1387 mm/pixel scale, 0.625 mm
slice interval, screenshot dimensions and inter-series alignment.
`pipeline/build_volume.py:22–28,96–115` sorts by index and stores positions, but
uses the constant interval rather than verifying/deriving geometry from those
positions. `pipeline/mesh.py:37–55` assumes fixed world-axis directions and
source dimensions. `pipeline/cfd_geometry.py:66–73` repeats the inferior-slice
and source-size assumptions.

**Consequence:** a different patient, screenshot zoom, slice order, crop or head
orientation can yield wrong physical dimensions, laterality or outlet location.
The supplied stack happens to span 130 mm in 208 intervals; that does not
validate these defaults for the next scan. Slice thickness and center-to-center
slice spacing are separate concepts.

**Required change:** reconstruct from ImagePositionPatient,
ImageOrientationPatient and PixelSpacing; order slices by position projected
onto the slice normal. Validate series membership, repeated/missing planes,
spacing and orientation consistency, coverage, and rescale units. Handle tilted
or irregular stacks explicitly. Store one patient-space affine and use it
throughout segmentation, measurements and meshing.
[DICOM image-plane module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html)

### P1 — Airway/sinus identity is inferred from fixed morphology and scan edges

`pipeline/segment.py:148–200` constructs a head hull with a 9 mm closing,
requires connection to external air, chooses the largest core deeper than
12 mm, and uses 1.6/1.3 mm openings to split cavities. Cores touching the final
z slice are privileged as airway; otherwise cores with inscribed radius above
4.2 mm may become sinus. The explanatory comment still says 5.5 mm.
Voxels are assigned by nearest core in Euclidean space, which is not constrained
to travel through the lumen (`:195–198`).

**Consequence:** these operations encode a particular scan extent and airway
shape rather than anatomical labels. Narrow nasal passages, severe obstruction,
widened postoperative sinus ostia, large nasal cavities and incomplete pharyngeal
coverage can defeat the classification. Actual sinus connections mean that
connectivity alone is insufficient.

**Reproduced:** replaying the core classifier on an identical synthetic mask
with reversed z order changes retained airway from 3,246 voxels to zero. A
synthetic internal lumen reaching the z scan faces, but not the outside of the
head hull, loses all 384 voxels in the exterior/core selection. These are
counterexamples to generality, not measurements of patient error.

**Required change:** define nasal cavity, sinus, oral cavity and external air
semantically. Use image-supported landmarks/seeds and explicit, reviewable ostium
cuts and outlet planes. Preserve real anatomical communications in the original
labels; apply research-domain cuts to a derived copy. A nasal CBCT pilot also
explicitly blocked sinus connections before region growing and identified
variable boundaries as a source of examiner disagreement.
[Zhang et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6354662/)

### P1 — Coarse labels and largest-component selection can discard airway

`pipeline/segment.py:330–333` restricts the fine airway to a 1 mm dilation of
the coarse airway, then keeps only the largest component. Fine-scale lumen
missed outside that envelope cannot be recovered. A real disconnected segment
or separately captured side can be silently discarded. In a synthetic pair
of disjoint lumen regions, `largest_cc` retains 432 of 732 voxels.

**Required change:** use a coarse pass only to locate a generous anatomical ROI;
derive final boundaries from native/high-resolution evidence. Retain candidate
components and explain exclusions by anatomy. Represent obstruction, uncertain
connection and inadequate coverage separately. Never widen a gap merely to make
the airway connected. A two-inlet connected CFD domain is an eligibility
requirement for that simulation, not a universal segmentation requirement.

### P1 — Surface validity and airway fidelity are different checks

`pipeline/mesh.py:91–119,204–209` smooths the mask before marching cubes.
Its useful fallback tests whether decimation breaks watertightness, but does not
detect lumens lost during smoothing or changes to narrow passages.

**Reproduced:** with the production airway smoothing, two synthetic parallel
lumens become one surface component when the smaller lumen is one voxel wide;
the resulting mesh is still watertight. This demonstrates a possible failure,
not that this specific defect exists in the saved patient mesh.

The CFD geometry is generated separately from the voxel mask, not from the
display GLB. However, `pipeline/cfd_geometry.py:220–247` resamples and regularizes
the lumen, removes voxels and retains the largest remaining component. It logs
these changes (`:258–265`); logging alone does not establish that clinically or
experimentally important narrow passages survive. Solver subdivisions repeat
the same reconstructed boundary (`:269–275`) and do not recover lost anatomy.

**Required change:** keep an immutable research mask and distinguish display
surfaces from measurement/solver geometry. At every transformation, compare
local surface distance, patency, false connections, ostium cuts, volume and
cross-sectional area. Review edits near the nasal valve, meatuses and septum.
Set acceptance tolerances against image resolution, expert variability and the
research endpoint; do not invent a universal tolerance.

### P1 — Derived minimum-area and hydraulic measurements can hide narrowing

`pipeline/airway.py:74–119` estimates area from geodesic shell volume and
perimeter from exposed voxel faces, then median/Gaussian smooths both profiles.
`pipeline/airway.py:185–190` reports minimum area from the smoothed profile.
Shells need not be equivalent to a reproducible anatomical cross-section in
branching, curved or dead-end regions.

**Reproduced with production functions:** a synthetic radius-5 mm cylinder at
0.5 mm isotropic spacing yields perimeter 42.0 mm instead of the analytic
31.416 mm (+33.7%), and hydraulic diameter 7.548 mm instead of 10 mm (−24.5%).
Its area is close: 79.25 versus 78.54 mm². A one-station 20 mm² narrowing in an
otherwise 100 mm² profile is removed completely by the current smoothing.
These fixtures isolate measurement behavior; they do not model a patient's nose.

`pipeline/airway.py:216–251` also chooses two high inlet clusters and infers
the choana from where left/right geodesic assignments meet, using a 40-voxel
cutoff and a midpoint fallback. These are acquisition-dependent heuristics;
an anterior septal communication is not necessarily the choana.

**Required change:** define reviewed anatomical boundaries and a reproducible
cross-section protocol. Preserve raw profiles and minima alongside any display
smoothing. Validate area and subvoxel contour perimeter on known shapes at
multiple spacings and orientations. Confirm local minima in multiplanar CT
views. Flow accuracy requires separate validation after geometry validation.

### P2 — Empty-mask morphology can invent foreground

`pipeline/segment.py:36–49` calls distance transforms without an empty-mask
guard. At 0.5 mm spacing, dilation of an empty fixture by 1 mm produces five
corner voxels; opening the empty fixture at 1.6 mm produces 22. These are
library edge-case effects, not anatomical evidence.

**Required change:** define empty/full-mask and image-border behavior explicitly,
return empty for dilation/opening of empty input, and test the helpers across
anisotropic spacing. Then test failure propagation rather than allowing an
empty/uncertain segmentation to continue as a valid case.

### P2 — The valid image field is inferred from one head's contour

`pipeline/build_volume.py:31–50` fits a circle to the posterior outline, assuming
that it represents a cropped reconstruction boundary. It does not check fit
residuals. `pipeline/segment.py:103–107` then uses the inferred disc as a hard
exclusion mask. A different crop or head shape can make tissue contour stand in
for acquisition coverage. This is a code-level assumption with a plausible
failure mechanism; an actual exclusion error in the local patient was not
demonstrated. Obtain valid-image coverage from acquisition metadata/padding and
reviewed import geometry instead of fitting it from anatomy.

## What the local evidence does and does not show

The saved source metadata describes 209 slices and a fine grid of
209 × 749 × 617, with spacing (z,y,x) = (0.625,0.2774,0.2774) mm.
The saved fine airway contains 804,685 voxels, 38.7007 cm³, and one
6-connected component. Connectivity and volume are descriptive, not accuracy
scores. No expert reference masks or multi-patient validation set were available
in the audited pipeline.

A read-only threshold sensitivity probe within a fixed 1 mm neighborhood of the
saved airway produced candidate volumes of 37.6009, 39.7309 and 41.5902 cm³ at
gray thresholds 16, 22 and 28. This reapplied the fine-volume Gaussian filter to
an airway crop with an eight-voxel margin and did not rerun anatomical separation.
These are local candidate-mask sensitivities, not alternative validated airways,
uncertainty bounds or an optimal threshold. In particular, the threshold-22
candidate is not the saved segmentation: its support region is different.

Published evidence independently demonstrates why this matters: a three-patient
nasal CT study found a 52% resistance decrease between segmentations at −800
and −300 HU. It did not establish an optimal universal threshold. Those native
CT thresholds must not be substituted into this clipped screenshot pipeline.
[Cherobin et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6239298/)

Existing safeguards worth retaining include spacing-aware distance transforms,
fine-grid air masks, topology checks, source/geometry hashes and rejection of
unexpected exterior contacts before CFD (`pipeline/cfd_geometry.py:36–65`).
The existing world-coordinate consistency unit test passes. It verifies internal
coordinate agreement, not correct patient orientation or airway anatomy.

## Recommended airway-focused workflow

1. **Import and qualify the scan.** Original CT, calibrated intensities, verified
   patient-space geometry, acquisition/protocol provenance and coverage checks.
   Do not treat CBCT gray values as HU without suitable calibration/validation.
2. **Define the research domain.** Nasal lumen including vestibules, nasal valve,
   meatuses and choanae, with a stated pharyngeal endpoint. Label sinus cavities,
   oral cavity and external air separately. Document treatment of ostia,
   septal perforations, tubes, secretions and incomplete coverage.
3. **Propose anatomy-aware labels.** Start with expert-guided seeds and
   constrained region growing/local boundary refinement, with native-resolution
   image evidence. Compare this baseline against a trained 3D semantic model
   if representative expert-labelled data become available. Neither fixed
   thresholds nor a generic organ/chest-airway model establish nasal accuracy.
4. **Review and correct in 3D and multiplanar slices.** Review the valve,
   inferior/middle meatuses, ostia, septum and choanae. Preserve uncertain or
   unresolved channels. Store manual edits, confidence/failure reasons, and
   source/model/parameter versions. An automated score cannot replace this
   validation while the system is unvalidated.
5. **Freeze the accepted mask.** Generate display and solver derivatives with
   an auditable record of every geometric change and local comparison to the
   accepted mask. Preserve small real constrictions and real obstruction.
6. **Measure and propagate uncertainty.** Use a reviewed cross-section protocol,
   raw area profiles, surface-error maps and segmentation sensitivity runs.
   Only eligible, accepted geometries proceed to CFD; verify mesh convergence
   and sensitivity separately from anatomical accuracy.

Automation is a candidate for reducing work, not an established winning method
for this dataset. A published framework evaluated 30 head CTs and reported
90.9% Dice and 0.3 mm average distance error; its CFD validation was preliminary.
That supports feasibility, not an all-patient guarantee or acceptable nasal-valve
error for this project. [Huang et al., 2019](https://pubmed.ncbi.nlm.nih.gov/31704374/)

## Validation required before claiming generality

Use two qualified independent annotators with adjudication and a written
boundary protocol. Measure inter-reader variation and flag anatomy below the
scan's resolving ability rather than treating every drawn voxel as ground truth.
Keep all slices, repeat scans and variants of a patient in the same partition.
Tune on development patients and freeze the method before testing at held-out
institutions/scanners.

The test cohort must represent the claimed population: scanner vendors, kernels,
spacing, noise/dose, orientation, dental artifacts, age and anatomy; severe
septal deviation, small passages, obstruction, polyps and postoperative sinus
openings. Pediatric, craniofacial and other uncommon groups need evidence if
included in the claim. Explicitly test missing/cropped coverage and unsupported
inputs. Publish subgroup and failure-case results, not only pooled means.

Use complementary endpoints:

- Volumetric overlap (Dice), plus surface Dice at a justified tolerance and
  average/95th-percentile surface distance in millimeters.
- Local wall error and area error at the valve and narrow meatuses; minimum
  cross-sectional area and its location under the declared measurement protocol.
- Wrong sinus/oral/external-air inclusion, false septal connections, missed
  passages and false occlusions; port/outlet placement and laterality errors.
- Correction time, automatic failure detection, abstention rate and performance
  on accepted cases. Include rejected cases in overall reporting.
- If airflow is the endpoint: sensitivity of resistance/flow to segmentation
  and mesh resolution, plus matched physiological/phantom validation where
  appropriate. Physiological agreement alone cannot prove segmentation accuracy.

Metric selection should reflect the research task rather than a single Dice
score. [Metrics Reloaded, 2024](https://www.nature.com/articles/s41592-023-02151-z)
Define sample size, endpoint tolerances and confidence intervals before evaluation;
they cannot be justified from the one local patient. A defensible eventual claim
is performance within a stated, externally tested population and protocol,
with a review/rejection path for other cases.

## Reproducing the software counterexamples

From the repository root, using the existing environment:

```sh
.venv/bin/python docs/audit_airway_separation.py
.venv/bin/python docs/audit_airway_geometry.py
.venv/bin/python -m unittest discover -s tests -p 'test_cfd_geometry.py'
```

The audit scripts use synthetic inputs and print JSON without changing patient
data. They expose current limitations; a successful script run means the probe
ran, not that the implementation passed an anatomical accuracy requirement.
The separation script identifies any replayed inline logic explicitly.

This audit adds documentation and reproducible probes. Follow-up work on
**this patient** (not a multi-patient method) changed the production path:

- Fine conducting lumen is recovered from native air inside a 2.5 mm envelope,
  with hysteresis for partial-volume walls, without deleting a smaller side
  (`pipeline/segment.py`).
- Airway display meshes use a signed-distance isosurface so one-voxel lumens
  survive (`pipeline/mesh.py`).
- Cross-section MCA and hydraulics use unsmoothed shell area
  (`pipeline/airway.py`). Empty-mask dilation/opening stays empty.

Run `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_airway_model`.
Original CT and expert labels are still required before any all-patient claim.
