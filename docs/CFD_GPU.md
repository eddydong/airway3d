# GPU lattice recordings and the precomputed scenario library

The viewer's breathing animation in CFD mode is playback of a recorded 3-D
solution. The OpenFOAM path records such a solution in about nine hours per
0.35 simulated seconds on this Mac, so a library of scenarios was out of
reach. This backend records the same kind of trajectory on the Apple GPU with
a lattice-Boltzmann solver in tens of minutes, so the comparison grid
(body positions × plans) can be precomputed once and then replayed instantly.
Every scenario is still a separate, geometry-exact solve; nothing is
interpolated between different airways. Lookup does reuse a recording when
knobs change but the prescribed wall motion is identical (for example face-up
response sliders that do not displace the wall). The 1-D estimate is never
substituted for a missing 3-D field.

## What runs

`pipeline/gpu_cfd_build.py` compiles a private copy of
[FluidX3D](https://github.com/ProjectPhysX/FluidX3D) (pinned commit, license
kept under `work/gpu-cfd/`) with `pipeline/gpu_cfd_runner.cpp`. The kernel is
patched in five places, each checked against the pinned text:

- **Planar openings.** The throat is the inferior scan cut. Each nostril rim
  is an oblique stair-step surface facing down, forward and sideways;
  `cfd_geometry.flatten_nostrils` extends it through exterior air with a
  short collar of fluid cells (about 800 per nostril, 3 mm) to one plane
  normal to its dominant opening direction, and the displayed lattice
  geometry includes that collar. Every open face is then a flat face with one
  ghost cell outside it. Nostrils sit at ambient pressure, the throat at
  ambient minus `A·b(t)`, where `b(t)` is the smooth resting-beta-1 drive:
  40% of the period for inspiration and 60% for expiration. No flow rate and
  no left/right split is prescribed anywhere.
- **Opening rule by flow direction.** Where fluid enters, the ghost applies
  anti-bounce-back (`f_ī = −f_i + feq_i + feq_ī` at the reservoir density),
  which imposes the reservoir pressure exactly at the face and prescribes no
  velocity. Where fluid leaves, the ghost is an equilibrium at the reservoir
  density with the velocity read two cells inside — the standard open
  outlet. Both of the simpler choices failed on the patient: FluidX3D's stock
  equilibrium ghost copied its neighbour's velocity back into the inflow and
  grew 3–14 m/s jets on single stair corners of the nostril rim until the run
  diverged 0.25 s into expiration; anti-bounce-back alone reflected the
  eddies leaving through the throat and diverged in inspiration.
- **Momentum filter.** In passages one or two cells wide, bounce-back on both
  walls returns a cell's wall-normal momentum to the same cell (or its pair)
  every step: a period-2 mode that collisions conserve and the over-relaxed
  stress at τ→½ feeds. A Guo force `F = −α·ρ·(u − u_previous)` with α = 0.05
  damps period-2 momentum oscillations by 2α per step; smooth flow keeps its
  steady balance and acquires an unsteady inertia of (1+α)ρ, a 0.05 Pa
  effect on the inertial part of the breathing pressure.
- **LES closure.** Air at breathing speeds on a 0.35–0.7 mm lattice has a
  relaxation time within 1e-3 of the stability limit and the run diverged at
  0.09 s. FluidX3D's Smagorinsky–Lilly eddy viscosity (`SUBGRID`) is enabled.
  On Apple's OpenCL its stock loop compiles to a kernel that silently does
  nothing, so the non-equilibrium stress tensor is written out explicitly.
- **No clipping; flag checks.** FluidX3D's velocity clamp is removed; the
  runner aborts on any non-finite value, lattice Mach above 0.3 or density
  deviation above 5 %. The host-side SURFACE/TEMPERATURE flag checks are
  relaxed because their bits carry the opening direction.

Collision is D3Q19 TRT in FP32 with molecular air viscosity
(1.5e-5 m²/s, 1.2 kg/m³). The time step puts a 10 m/s design speed at
lattice speed 0.1 (Mach 0.17 there, 0.09 at the 5 m/s seen in the airway),
i.e. 7 µs at 0.7 mm and 3.5 µs at 0.35 mm. Walls are rigid halfway
bounce-back on the stair-step voxel surface, exactly the surface the viewer
shows. The result is a **large-eddy simulation**, not DNS and not the
laminar Navier–Stokes model of the OpenFOAM path.

## Measurements and gates

Flows are the face-normal velocities of the cells adjacent to each planar
opening times the face area, positive into the airway at the nostrils and
out at the throat. The nose→throat pressure difference is the mean pressure
of the cells adjacent to the nostrils minus that of the cells adjacent to the
throat.

The nominal amplitude `A` is the reservoir pressure. Inflow openings impose
it exactly; the outflow opening loses part of it (15 % of the nominal on the
duct below, 1 % on the patient at peak). The recorded `pressureDropPa` and
`flowMlS` are measured together and stay consistent with each other; across
scenarios at the same nominal `A`, compare the flow they achieve (the
comparison dialog shows total flow and its change against the
no-intervention row).

Gates in `result.json`: `mesh` (closed oriented voxel surface),
`coordinateAlignment` (surface bounds equal the field bounds), `completed`
and `frames` (all 64 saved intervals per cycle present at the requested
times), `stability` (finite, Mach and density limits held for every frame),
`massBalance` (|L+R−throat| below 3 % of peak throat flow on every frame; the
second-order face quadrature of the 0.7 mm lattice leaves 1–2 %). The viewer
accepts a GPU recording only if all pass; `residuals`, `courant`, `wallLeak`
and `requestedFlow` belong to the OpenFOAM path and are not claimed.

## Validation done

Square duct, 6 × 6 cells of 2/3 mm, 60 cells long, 0.08 Pa nominal, time
step 0.2 ms (`python pipeline/gpu_cfd.py` writes `work/gpu-cfd/validation-poiseuille/check.json`):

- The interior pressure gradient is linear and the flow follows the laminar
  square-duct solution for that gradient within **−3 %** (the Smagorinsky
  closure dissipates a little in laminar shear on a 6-cell duct). Opening
  loss 15.6 % of the nominal amplitude, at the equilibrium outflow; the
  anti-bounce-back inflow imposes its reservoir pressure exactly.
- The same duct at the breathing time step gave nonsense (0.04 mL/s): at
  0.08 Pa the lattice density deviation is 7e-6, at the FP32 floor. At
  breathing pressures (30 Pa → 7.5e-3) this is not a concern; the check
  therefore uses a time step matched to its own velocity scale.

Patient grid, 0.7 mm, one lattice cell per voxel (113,134 fluid cells with
the nostril collars, 2.4 M-cell box), 30 Pa nominal, 4 s period, one cycle
started in expiration (`work/gpu-cfd/diag-flat-hybrid`):

- The full cycle runs without a stability violation; max lattice Mach 0.054,
  max speed 3.1 m/s in the interior, no velocity spike at any opening.
- Peak flows 58 mL/s out and 61 mL/s in at measured drops of 28.8 and
  29.8 Pa (opening loss ≈1 % of the 30 Pa nominal); left/right split
  0.32/0.68 in both directions, the same dominant side as the OpenFOAM
  steady field on this patient.
- Mass imbalance at the openings 1.3 % of peak flow; flow returns to within
  3 mL/s of zero at the half-cycle crossings.
- The two-cycle scenario recorded through the service (supine, no
  intervention, 25 min) repeats its first cycle in the second to 0.03 mL/s
  at peak flow and 0.01 Pa at peak pressure, so the start from rest is
  forgotten within the first cycle; this is the reason the library records
  two cycles and plays back both.
- Before the planar openings and the direction-dependent rule the same
  recording diverged at 0.25 s of expiration on both the 0.7 mm and the
  0.35 mm lattice; this is documented above so that the boundary treatment
  is not simplified again without re-running that case.
- Throughput about 900 lattice steps/s alone at 0.7 mm: 10–17 min per
  breathing cycle depending on other GPU load, so a two-cycle scenario takes
  20–35 min. At two cells per voxel (892,424 fluid cells) about 200 steps/s,
  1.6 h per cycle.

Unit tests (`tests/test_gpu_cfd.py`) cover the ghost flag encoding, cell
order, blocked openings, opening flows on a forked channel, nostril
flattening, the full result schema and the mass gate without needing a GPU.

## Using the library

Start the read-only viewer with `make serve`. **Solved scenario** lists complete
accepted recordings. Four body-position buttons select a whole matching request;
if a period/intervention combination is unavailable in the new position, the
menu visibly switches to an existing recording there. Anatomy and breathing
inputs cannot be edited into unsupported combinations. Particle appearance,
playback speed, pause and wall display remain adjustable.

Body position has no time-after-turning dimension in the UI. Historical
`elapsed` and `tau` values remain in request metadata for immutable recording
identity; the existing default captures a fixed response factor of
`1 - exp(-10/5)`. Removing a control does not reinterpret already solved fields.

**Compare plans × body positions with CFD** reads separate exact results;
unavailable comparisons continue to withhold readings. Numerical acceptance
checks are retained. They do not establish mesh independence or clinical accuracy.

The small discrete expansion has four no-intervention positions and six
independent face-up changes: left/right head, body, or valve clearance at 1 mm.
All share 30 Pa, a 4 s breath, a 0.7 mm geometry grid and one lattice cell per
voxel by default. It does not imply support for combined interventions or for
these interventions in every position. Publish each additional recording only
after its own solve passes the numerical checks:

```sh
.venv/bin/python pipeline/gpu_cfd_build.py                      # once
.venv/bin/python pipeline/cfd_library.py --status               # what exists
.venv/bin/python pipeline/cfd_library.py --control-points --export-requests work/cfd/discrete-control-points.json
.venv/bin/python pipeline/cfd_library.py --control-points         # 4 existing baselines + 6 isolated 1 mm clearances
.venv/bin/python pipeline/cfd_library.py                        # 4 positions × no intervention
.venv/bin/python pipeline/cfd_library.py --plans plans.json     # + saved plans
.venv/bin/python pipeline/cfd_thermal.py --all                  # wall temperatures/heat flux
.venv/bin/python pipeline/cfd_catalog.py                        # static catalog
```

The batch publishes accepted airflow results individually as it progresses.
Refresh saved library to see them. New cases have airflow only until the separate
thermal script has completed; the viewer explicitly withholds missing wall temperatures.

Playback preloads and loops the second recorded cycle. Fractional playback time
is preserved across saved frames and the loop boundary is checked against the
strict thermal cycle-join tolerance before thermal fields are published. Inside
the airway the particles follow the recorded field with the same looks (glow,
vapour, streaklines…)
and colourings as the estimated engine. `result.json → openings` lists each
planar opening (centroid, normal, open faces), and from it the viewer draws
room air being pulled into an inhaling nostril and exhaled air leaving as a
puff; outside the lattice there is no computed velocity, so that air is a
kinematic sketch from the exit velocity and is labelled as such under the
particle controls.

`plans.json` is the viewer's saved-plan list (`localStorage['airway3d-plans-v1']`).
The CLI owns its offline worker queue. The browser and static server cannot
submit geometry, airflow or thermal jobs. See [THERMAL_PLAYBACK.md](THERMAL_PLAYBACK.md).

## Limits

A recording is a finite LES trajectory on a rigid stair-step wall from rest:
periodicity, mesh and time-step independence, the Smagorinsky constant, the
momentum filter constant, the opening rules and clinical validity are not
established. The nostril collars are virtual geometry: they close the
forward- and side-facing part of each nostril rim and open it on one plane
below. The 0.7 mm lattice leaves narrow passages one to three cells wide;
the 0.35 mm lattice is available but takes hours per scenario. Posture and intervention
effects remain prescribed wall displacements, not tissue mechanics. This is
still precomputed playback, not a live solver; the viewer shows a solved
field only for controls that have a recording.
