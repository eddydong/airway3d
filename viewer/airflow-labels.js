// Shared user-facing names for the two airflow calculations.
// The 3-D picture is a drawing in both modes; these names describe the physics.

export const TUBE_ESTIMATE = 'Live tube estimate';
export const RECORDED_CFD = 'Recorded 3-D CFD';

export const AIRFLOW_INTRO = 'The 3-D picture is always a drawing. Choose the calculation behind the numbers and particles.';

export const TUBE_ESTIMATE_NOTE = 'Numbers and particles use a live tube model: one mean speed per cross-section along each passage (a 1-D pipe calculation). Paths are stored streamlines stretched with the wall. This is not a 3-D Navier–Stokes field.';

export const RECORDED_CFD_NOTE_TRANSIENT = 'Numbers and particles follow a recorded 3-D velocity in every cell. This is playback of an offline solve, not a live solver. Only saved scenarios have a field.';

export const RECORDED_CFD_NOTE_STEADY = 'Numbers and particles follow one recorded 3-D velocity field for the selected direction. This is playback, not a live solver. Choose Recorded breathing cycle for time-dependent fields.';

export const TUBE_GEOMETRY_NOTE = 'Experimental geometry with a live tube estimate. The 3-D drawing stretches stored anatomy; region labels are approximate and do not identify safe tissue to remove.';

export const CFD_GEOMETRY_NOTE = 'Changes prescribe a new 3-D airway for a separate recorded CFD solve. Region labels and tissue response are assumptions.';

export const MISSING_CFD = 'No recorded 3-D CFD for these settings. Choose a saved scenario, or switch to the live tube estimate.';
