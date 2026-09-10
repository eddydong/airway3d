# From interactive estimates to a doctor-facing flow model

## Current status

A newer **Recorded breathing cycle** path and CFD-backed comparison panel are
implemented alongside the steady solver described below. See
[CFD_BREATHING.md](CFD_BREATHING.md) for the transient method, queue/playback
behavior, completed duct checks and patient-validation limits.


The source consists of PNG screenshots, not DICOM. The patient confirmed that
the CT was acquired **face up (supine)**, with skull C400/W1500 and head C40/W350.
Slice spacing is 0.625 mm; in-plane spacing is inferred from a reported 68 mm
inter-pupillary distance. Display-window clipping cannot be reversed. The
reconstruction is therefore a best-effort geometry with explicit screenshot,
scale, segmentation and voxel-wall uncertainty, not a clinical segmentation.

The Scenario lab keeps a fast quasi-steady **1-D** friction/local-loss model for
interactive edits, but it now has a separate geometry-backed CFD path. The CFD
worker uses the same screenshot-derived airway, crops one connected domain,
labels two nostril contacts and an inferior outlet, rejects unexpected exterior
contacts, and refuses to solve unless the voxel surface is closed and oriented.
The pinned OpenFOAM 2412 case is steady, incompressible, laminar viscous
Navier–Stokes with rigid no-slip walls, ambient-pressure nostrils and prescribed
outlet flow. A result is accepted as converged only after mesh, residual, mass-balance,
wall-leak, requested-flow and pressure-stability gates pass. The viewer imports
the native velocity, pressure and occupancy fields only for the matching
geometry, posture assumptions, flow rate and breathing direction.

Posture and intervention controls prescribe a recorded wall-offset hypothesis;
they are not a tissue-mechanics or fluid-structure solve. Each change creates a
new request identity and must be solved again. While it is queued, CFD readings and particles are withheld. The explicit mode
switch returns to the estimated breathing cycle; there is no automatic fallback.

The lab also reports a reduced-order mucosal heat-flux estimate during
inspiration. It uses each transformed profile's local hydraulic diameter,
perimeter and 1-D branch flow with assumed air properties, 32.6 C wall
temperature and 20 C inspired air. It exposes peak heat flux, heat loss and
mucosal area above 50 W/m2. This remains an estimate: no 3-D energy equation,
humidity, wall-temperature field or near-wall thermal boundary layer is solved.

The particle viewer's **mucosal heat flux · inspiration** colour mode uses the
same per-shell flux values. In CFD mode particles sample the native solved
velocity field; during a pending or unmatched request it withholds CFD
particles and readings. Steady CFD replaces the old breathing engine and does
not animate a respiratory cycle. Breath period, cycle, plume, thermal walls and
legacy trail/style controls are hidden in this mode. Rate changes require a new
solve; particle count, size, colour, pause and playback speed use the shared UI.

The reported left-down obstruction and better right-down breathing are retained
as patient observations. A labeled sensitivity preset explores them using
assumed asymmetric swelling. Face-up, left-down, right-down and upright scenarios
share adjustable response amplitudes, a settling time and cycle bias. These
parameters are not recovered from CT. No intervention is ranked as a clinical
recommendation.

## Recommended free solver route

Use OpenFOAM for a background viscous solve, separate from the interactive viewer.
Start with rigid-wall, steady incompressible flow and test whether laminar or
transitional/unsteady treatment is appropriate for the anatomy and flow range.
Do not select turbulence treatment solely from the current 1-D Reynolds proxy.

The repository now pins and runs an OpenFOAM 2412 ARM64 container locally
(`microfluidica/openfoam` by digest). The worker keeps the case and diagnostics
under `work/cfd`, runs `checkMesh`, partitions larger voxel domains, solves with
`simpleFoam`, reconstructs the latest fields, and writes a geometry-hashed
`result.json` plus native velocity, pressure and occupancy arrays. The solver
path is runnable and its analytic duct validation passes, but the patient case
still needs mesh refinement and pressure-flow comparison before clinical use.

- [OpenCFD simpleFoam equations and input requirements](https://doc.openfoam.com/2306/tools/processing/solvers/rtm/incompressible/simpleFoam/)
- [OpenCFD Docker installation](https://www.openfoam.com/download/openfoam-installation-on-mac-using-docker)
- [OpenFOAM Foundation macOS installation](https://openfoam.org/version/macos/)

### Checks completed

The analytic square-duct regression converges with zero wall leak and second-
order refinement: 5.6% relative pressure-drop error at 8 cells across, 2.6% at
12, and 1.5% at 16. The screenshot-derived patient domain has passed the same
topology gate at the 0.7 mm reconstruction grid with two inlet patches and one
outlet; the current refined solve uses 0.35 mm finite-volume cells. Those checks
verify implementation and mesh integrity, not patient accuracy. The patient
case remains `meshIndependence: false` until a refinement comparison is complete.
Comparison with repeatable rhinomanometry is a separate clinical validation step.

The previous 300-iteration baseline has been withdrawn. It incorrectly reversed
X and Z when constructing the mesh, and classified convergence using final
linear-solver residuals from iterations later than the exported field. Its
248.5 Pa reading must not be treated as an accepted baseline.

Geometry version 5 uses world X/Y/Z for cell centers, surface vertices, field
indices and velocity components. The binary arrays are NumPy C order (Z
fastest). Version 2 of the solver gates SIMPLE's initial residuals at the exact
exported iteration, requires normal solver completion, and verifies the exported
surface against the native cell bounds. The viewer rejects older field schemas.
An explicit **Inspect unconverged field** action can show a finished numerical
iterate if its mesh, coordinates, mass balance and boundary flow checks pass.
That preview remains labeled unconverged in the lab and particle status; accepted
airway readings stay withheld. It never changes the result acceptance status.
The corrected `baseline-070-r4` run completed 2,400 iterations on 892,424 cells.
Mesh, coordinate alignment, mass balance, zero wall flux, requested flow,
pressure stability and normal completion passed. SIMPLE initial residuals
remained between 6.68e-5 and 7.96e-5, above the 1e-5 acceptance threshold.
Its status is therefore **unconverged**, with only the explicit preview available.
Mass imbalance was 1.78e-6 of outlet flow. These checks establish neither a
converged patient solution nor a full transient breathing cycle.
The asymmetric-domain regression independently raycasts seeded and advected
particles against the mesh exported by Python, which the symmetric duct did not
check. Convergence semantics follow [OpenCFD's SIMPLE termination documentation](https://doc.openfoam.com/2306/tools/processing/numerics/solvers/case-termination/).

Run the regression checks with:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_cfd*.py'
node --test tests/*.test.mjs
```

## Implementing the next stage

1. Review the airway segmentation slice by slice with an ENT/radiology reviewer.
   Explicitly label inlet patches, the neck outlet, walls, septum, turbinate bone,
   and mucosa wherever supportable. Record unresolved partial-volume boundaries,
   screenshot registration error and uncertain scale. Test a range of segmentation
   thresholds and spacing estimates. Do not turn a region-guide sphere into a cut.
2. Apply interventions to the reviewed voxel/surface domain. Preserve wall
   thickness and mucosa constraints; check connectivity, self-intersections,
   narrow passages and boundary patch integrity before meshing. The geometric
   preview must reference the exact domain hash used by the solver.
3. Generate a volume mesh with suitable near-wall resolution. Measure mesh
   quality and compare at least three successively refined meshes. Report the
   observed sensitivity of pressure drop, flow split and wall quantities rather
   than treating a fixed voxel count as sufficient.
4. Specify breathing direction and a boundary condition: equal ambient pressure
   at both nostrils and a prescribed outlet flow, or a measured pressure-driven
   condition. Use no-slip walls. Record air properties and whether inlet/outlet
   extensions affect the solution. Do not prescribe each nostril's split from
   the old potential-flow solution.
5. Run the solve in a worker process. Give each request a hash of geometry,
   posture assumptions, boundary conditions, mesh and solver version. Support
   cancel, progress and cache reuse. Preserve the visible scene while solving;
   attach results only if their request hash still matches the selected scenario.
6. Show residual histories, inlet/outlet mass imbalance, mesh statistics, flow
   regime assumptions and a mesh-refinement comparison. Mark unfinished,
   nonconverged and stale results explicitly; do not substitute the 1-D values
   while labeling them as CFD. Import velocity, pressure and wall-shear fields
   into the viewer only for their matching geometry.
7. Compare against repeatable nasal pressure-flow measurements in the relevant
   positions, with the nasal cycle/decongestion state documented. Report errors
   and uncertainty. Validate the posture model separately from the flow solver.
   Patient-reported blockage alone cannot uniquely identify compliance, reflex
   response, vascular engorgement and structural collapse.
8. Consider tissue mechanics/FSI only after material properties and support
   conditions can be constrained. A gravity animation is not tissue mechanics;
   a more accurate fluid solver cannot repair unsupported wall-motion inputs.

## Practical interaction contract

Choose explicitly between the estimated breathing cycle and asynchronous steady
3-D results. The steady field has one direction and one flow rate; it cannot
supply the time-dependent resting-breath waveform without a matching recorded
transient CFD result.
Full-resolution clinical-quality CFD should not be promised at animation frame
rate on a no-budget laptop. A reduced-order surrogate can eventually provide
near-real-time predictions, but it must be trained and validated against an
ensemble of full solves and must flag cases outside its validated range.

Every doctor-facing report should identify: original source and scan position;
reviewed versus inferred anatomy; operation parameters; posture assumptions;
geometry identity; solver and boundary conditions; convergence and mesh
sensitivity; comparison with measurements; and missing validation. Present
resistance, flow allocation and mucosal function together. Lower resistance
alone is not a measure of surgical success.

## Physiological evidence informing the boundaries of the model

- [Posture and nasal patency](https://pubmed.ncbi.nlm.nih.gov/6703492/): lateral
  posture effects include reflex vascular responses, not simply hydrostatic
  displacement of a flexible wall.
- [Simulating the nasal cycle with CFD](https://pmc.ncbi.nlm.nih.gov/articles/PMC4402730/):
  mucosal state changes can confound a pre/post comparison.
- [Predicting postsurgery physiology: challenges and limitations](https://pmc.ncbi.nlm.nih.gov/articles/PMC4405156/):
  virtual geometry and a single scan do not uniquely determine the healed outcome.
- [Regional peak mucosal cooling and perceived patency](https://pubmed.ncbi.nlm.nih.gov/23775640/):
  localized CFD heat loss correlated with unilateral patency ratings, while the
  paper still calls for individualized CFD and neurologic measures together.
- [Mucosal cooling after nasal surgery](https://pmc.ncbi.nlm.nih.gov/articles/PMC3917722/):
  the area exposed above 50 W/m2 correlated with NOSE scores in a small cohort.
