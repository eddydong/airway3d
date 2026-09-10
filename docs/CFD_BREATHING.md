# Geometry, comparisons and recorded breathing

The CFD path now connects prescribed intervention and body-position settings to
an exact 3-D solver mesh, a background OpenFOAM run, its numerical acceptance
checks, and the viewer. The source is still CT screenshots. Post-intervention
and non-supine anatomy are hypotheses, not additional observed scans.

## Using it

Start the read-only viewer with `make serve` and open
`http://127.0.0.1:8765/`. Run expensive work separately with the offline scripts
in [THERMAL_PLAYBACK.md](THERMAL_PLAYBACK.md).

- **CFD playback** selects exact precomputed geometry/flow recordings. Changing
  settings immediately invalidates mismatched results and never starts work.
- **Recorded breathing cycle** plays the saved transient field; the last cycle
  repeats. **Steady flow** requires a matching steady recording.
- **Compare plans × body positions with CFD** reads the catalog. Missing or
  unaccepted rows show no numbers and cannot be played. There are no queue,
  solve or cancel controls in the viewer.
- **Airway wall display** offers computed temperature and sensible heat flux
  when a thermal recording is available. It uses the final airflow cycle after
  offline thermal equilibration. Wall geometry/opacity, particles and playback
  are independent controls. The server exposes no calculation endpoints.

The existing saved “2 mm reduction” dataset and estimated 1-D report remain
available only in estimated mode. They are not substituted for the CFD plans.
“Post” in the CFD comparison means the proposed intervention settings.

## What is solved

`pipeline/cfd_transient.py` runs the pinned OpenFOAM 2412 `pimpleFoam` executable
with incompressible, laminar Navier–Stokes, backward time differencing, first-
order upwind convection, rigid no-slip walls and ambient static pressure at the
nostrils. Throat volume flux uses the smooth resting-beta-1 model: inspiration
occupies 40% of the period, passive expiration 60%, and both lobes have zero
slope at their reversals. Its signed integral is zero, and positive flow is out
of the throat during inspiration.

The signed Function1 table is supplied to `flowRateInletVelocity` with
`extrapolateProfile false`. Its negative inlet rate prescribes outward flux;
its positive rate prescribes inward flux. This sign handling was checked against
the source in the pinned container and exercised in the oscillating duct runs.
It is not an inspiration velocity field multiplied by a breathing sinusoid.

The adaptive time-step target is maximum Courant 0.5, with an acceptance limit
of 1.0 on the observed maximum. These are separate: acceleration can make the
observed value exceed the adaptive target. Up to 12 outer pressure–velocity
corrections are allowed per step. The checker requires initial residuals below
1e-5 in the final outer loop, including the first pressure correction there,
at **every** time step. Small final linear residuals cannot establish this.
Other gates cover completed physical duration, frame coverage, coordinates,
mesh, wall flux, and mass balance. Signed boundary flux must match the waveform;
relative errors use peak flow as denominator, including near reversal.

There are 32 saved intervals per cycle on the OpenFOAM recording. GPU lattice
recordings use 64 intervals per cycle and the smooth resting-beta-1 boundary
drive described in [CFD_GPU.md](CFD_GPU.md). Frames contain float32 XYZ velocity
values for each occupied cell, in native C-order. A fixed occupancy-to-cell map
connects compact frames to the exported surface. The browser retains only the
current frame and nearby frames. It pauses physical time while buffering and
never switches to estimated velocities. The spatial/time interpolation used
for visualization is distinct from the solver's much smaller time steps.

## Validation completed

The oscillatory square-duct test simulated two full four-second cycles. Both
normal and smaller maximum time-step runs passed the numerical gates. An
independent sine-eigenfunction Stokes solution predicts the harmonic central
pressure gradient. On the coarse six-cell-wide duct, the complex-response error
was **9.24%**, phase error below **0.00008 radians**, and the smaller time-step
run changed amplitude by **0.0010%**. This is an implementation check on a coarse
benchmark, not a patient convergence or accuracy claim.

Browser integration testing played all eight seconds from actual solver output,
observed both signed flow phases, retained all particles inside the duct and
stopped at the final recorded time. Unit tests also cover delayed frames,
request identity, matched comparison settings, solver residual semantics,
queue deduplication and cancellation.

Reproduce the transient checks after preparing a six-cell duct with
`cfd_validate.prepare(6)`:

```sh
.venv/bin/python pipeline/cfd_transient.py --case breathing-duct-6-r2 --copy-mesh duct-6 --q 1 --period 4 --cycles 2
.venv/bin/python pipeline/cfd_transient.py --case breathing-duct-6-dt --copy-mesh duct-6 --q 1 --period 4 --cycles 2 --max-dt .0025
.venv/bin/python pipeline/cfd_validate_transient.py
node --test tests/*.test.mjs
.venv/bin/python -m unittest discover -s tests -p 'test_cfd_*.py'
```

`viewer/tests/cfd-player.html` is a clearly labeled synthetic-duct integration
harness. It reads the exported benchmark assets under
`viewer/data/cfd-validation/breathing-duct/`; it never registers them as a
patient result.

## Limits still present

This is background computation and interactive playback, not real-time CFD.
Patient runs at the current 892,424-cell resolution can be much slower than the
small duct benchmark. A batch may require hours or longer; raw time frames also
consume substantial disk space. The worker checks available space before a
transient run and processes flow jobs sequentially.

Airway walls remain fixed during each breath. Moving tissue/fluid interaction,
patient-specific compliance, turbulence-model selection, mesh/time-step
independence, cycle-to-cycle airflow periodicity and clinical validation are not
established. The published thermal playback is a separate finite-capacity
air/tissue model with its own periodic temperature gate; it does not establish
patient-specific mucosal properties. Passing the transient numerical gates
means the finite recorded trajectory passed those checks, not that it is a
periodic or validated patient solution. The prior unconverged steady baseline
is still unaccepted.
