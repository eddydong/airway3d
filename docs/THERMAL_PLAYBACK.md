# Offline CFD and airway-wall thermal playback

The viewer is a read-only player. It cannot start geometry builds, airflow
solvers, thermal calculations, batches or background jobs. `serve.py` is a static
file server and rejects every POST with HTTP 405. The recording catalog is
`viewer/data/cfd/library.json`, created offline and keyed to the current source,
geometry version, solver version and complete physical request. Unsupported
settings show **Not precomputed**; the viewer never synthesizes a replacement CFD
field. The separate 1-D preview remains available with CFD playback switched off.

## Prepare and serve

```sh
# Offline airflow batch: runs only when explicitly launched from the terminal.
.venv/bin/python pipeline/cfd_library.py --positions supine,left,right,upright
# Also accepts --plans plans.json, using the saved-plan export shape.

# Offline thermal batch over accepted recordings in the current catalog.
.venv/bin/python pipeline/cfd_thermal.py --all

# Refresh the static catalog after externally prepared results.
.venv/bin/python pipeline/cfd_catalog.py

# Read-only viewer; no solver processes are started by this command.
make serve
```

Thermal parameters are chosen offline. For a specific recording use
`pipeline/cfd_thermal.py --result <result.json> --ambient 22 --body 37`.
`--dt .01 --output work/thermal-validation-half-step` generates a separate
validation recording. Generated binary assets stay in the ignored data/work
folders. Thermal assets use immutable parameter/version-hashed subdirectories;
the catalog and thermal metadata pointers are published atomically only after
completion and numerical acceptance. Old completed assets remain usable during
a new offline build.

## What the thermal model solves

The accepted velocity recording drives a separate three-dimensional energy
transport solve in every occupied fluid cell:

    rho_air cp_air (dT_air/dt + u . grad T_air) = k_air laplacian T_air

Each anatomical wall face has a finite-capacity tissue slab connected to a warm
deep-tissue boundary. Its exposed surface temperature is an unknown, obtained
from continuity of heat flux through the air half-cell and the tissue half-slab.
It is not held at a constant warm temperature and is not inferred from speed.
The slab node changes temperature as it loses heat to air and receives heat from
the deeper body boundary:

    C_wall dT_wall/dt = G_body (T_body - T_wall) - G_interface (T_wall - T_air)
    q_wall_to_air = G_interface (T_wall - T_air) / area
    T_surface = T_air + q_wall_to_air * h_air / (2 k_air)

The default deep tissue boundary is 37 °C, room inlet 22 °C, and incoming exhaled
throat air 34 °C. Air and tissue start at 34 °C. The assumed slab is 1 mm thick,
conductivity 0.5 W/(m K), density 1000 kg/m³ and specific heat 3600 J/(kg K).
Air conductivity is 0.026 W/(m K), density 1.2 kg/m³, and specific heat
1005 J/(kg K). These are explicit model inputs, not measurements of this patient.

Internal diffusion uses face conductance. Advection is first-order upwind in
advective form; time integration is backward Euler with a maximum 0.02 s step.
The air and slab unknowns are solved together by eliminating the slab into the
sparse air matrix. Incoming nares receive room-temperature air; incoming throat
flow receives exhaled air, with direction taken from each saved opening velocity.
Outflow has zero diffusive gradient. Artificial nostril extension walls are
adiabatic and excluded from the heated mucosal surface and reported wall area.

The final airflow cycle is repeated offline until the maximum fluid/slab
start/end temperature difference is below 1e-4 °C. Only the equilibrated thermal
cycle is published. The viewer aligns it with the final airflow cycle, not the
startup cycle. Temperature and heat flux interpolate at the same physical time.
A full compact thermal cycle is loaded once into a GPU texture, so playback
changes shader time uniforms, without a per-frame thermal solve or network fetch.
The body-part visibility checkbox hides the surface only. Particle visibility,
breath playback and thermal display are independent controls.

## Reading the visualization

In **Scenario lab → Airway wall display**, choose **Temperature · °C** or
**Mucosal sensible heat flux · W/m²**, then **View airway walls**. Temperature
defaults to **Detail bands**, with blue/cyan/green/yellow/red at
35.5/36.9/36.98/36.995/37 °C. This expands the near-body-temperature range
where most wall values lie. Heat flux detail bands use −25/0/1/30/1000 W/m²,
revealing both heat returning to the wall (negative) and small cooling fluxes
(positive). Both scales interpolate continuously between the labeled stops,
which stay fixed across scenarios and breath phases. Enhanced color contrast
does not imply additional model accuracy. **Linear scale** retains the original
35–37 °C and 0–1500 W/m² ranges. Values outside either scale saturate the palette
but remain available in the numeric readout.
Point at the surface for a local readout. Cut planes expose internal faces.
The global min/max/mean are interpolated frame summaries; a local cursor value
interpolates the actual face scalar. The playback control defaults to 1× in CFD
and preserves the independent legacy preview speed.

**Ember (speed, not temperature)** is a particle appearance palette. Particle
speed, air temperature, wall temperature and wall heat flux are different
quantities. The old **mucosal heat flux · inspiration** particle mode remains a
1-D proxy, available only in the estimated mode. The new CFD wall view uses the
computed thermal fields instead.

## Verification and limits

Unit tests check uniform-temperature invariance, cold-air warming, wall cooling,
warm expiration, port exclusion, interfacial thermal resistance, and refinement
towards an analytic coupled air/slab solution. Published results require finite
and bounded temperatures, a maximum scaled linear residual below 1e-4 °C,
combined air/tissue energy-ledger closure below 1e-4 W, and the periodic tolerance.
The refreshed recordings use a 1e-4 °C cycle-join tolerance; playback loads the
final recorded cycle before starting and carries fractional time across frame
boundaries, so no network fetch or time reset occurs at a loop.
The energy correction implied by divergence of recorded cell-centred velocity
is reported separately; a small linear residual does not establish physical
accuracy. A separate baseline time-step comparison (0.02 vs 0.01 s maximum step) is
recorded under `work/thermal-validation-half-step/comparison.json`: maximum
wall-temperature difference 0.0223 °C, RMS 0.00028 °C; maximum sensible-flux
difference 22.15 W/m², RMS 0.273 W/m². This is one time-step sensitivity check;
no mesh-independence or clinical-validation claim is made.

This is a **sensible heat transfer calculation**, not a complete nasal
conditioning model. Evaporation and humidity are absent, as are lateral tissue
conduction, a spatial perfusion model and thermal feedback into airflow.
The wall grid and tissue properties are uncalibrated. Temperatures are model
predictions, not measured patient temperatures. Heat-flux/temperature statistics
must not be interpreted as a patient congestion score.

Conducting-wall models with varying exposed surface temperature are described in
[the 2017 nasal air-conditioning study](https://pubmed.ncbi.nlm.nih.gov/28499215/)
and [the 2021 transient nasal heat-transfer study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8450908/).
Mucosal cooling is associated with perceived patency, while humidity/evaporation
also contribute: [2013 regional cooling study](https://pmc.ncbi.nlm.nih.gov/articles/PMC3841240/),
[2011 mucosal cooling study](https://pmc.ncbi.nlm.nih.gov/articles/PMC3192719/).
