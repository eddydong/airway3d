// Reduced-order inspiratory heat-transfer estimate.
// It uses the measured profile geometry and 1-D branch flow, not a 3-D
// Navier-Stokes/energy solve. Keep the output labelled accordingly in the UI.
const RHO = 1.2;       // kg/m3
const MU = 1.8e-5;     // Pa.s
const K_AIR = 0.026;   // W/(m.K)
const CP = 1006;       // J/(kg.K)
const PR = CP * MU / K_AIR;
const WALL_C = 32.6;   // representative nasal mucosa temperature
const INLET_C = 20;    // assumed inspired room-air temperature
const THRESHOLD = 50;  // W/m2, literature comparison threshold

const finite = x => Number.isFinite(x) ? x : 0;

export function heatFlux(profile, Qml, options={}) {
  if (!Number.isFinite(Qml) || Qml < 0) throw new Error('Flow must be finite and nonnegative');
  const closed = profile.area_mm2.some(a => a <= 1e-8);
  if (closed || Qml === 0) return {
    closed, Q: Qml, peakWm2: 0, meanWm2: 0, heatLossW: 0,
    areaAbove50Cm2: 0, surfaceAreaCm2: 0, outletTempC: INLET_C,
    fluxWm2: Array(profile.area_mm2.length).fill(0),
    thresholdWm2: options.thresholdWm2 || THRESHOLD,
  };
  const threshold = options.thresholdWm2 || THRESHOLD;
  const Q = Qml * 1e-6;
  let airC = options.inletTempC ?? INLET_C;
  let heatLoss = 0, surface = 0, above = 0, peak = 0;
  const fluxWm2 = new Array(profile.area_mm2.length).fill(0);
  for (let i=0; i<profile.area_mm2.length; i++) {
    const area = Math.max(profile.area_mm2[i] * 1e-6, 1e-12);
    const perimeter = Math.max(profile.perimeter_mm?.[i] * 1e-3 || Math.PI * profile.hyd_diam_mm[i] * 1e-3, 1e-7);
    const dh = Math.max(profile.hyd_diam_mm[i] * 1e-3, 1e-8);
    const ds = (i ? profile.s_mm[i] - profile.s_mm[i-1] : profile.s_mm[1] - profile.s_mm[0]) * 1e-3;
    const velocity = Q / area;
    const re = RHO * velocity * dh / MU;
    // Fully developed laminar tube correlation, with a standard turbulent
    // correlation for high-Re shells. This is a local heat-transfer estimate.
    const nu = re < 2300 ? 3.66 : 0.023 * re ** 0.8 * PR ** 0.4;
    const h = nu * K_AIR / dh;
    const flux = Math.max(0, h * (WALL_C - airC));
    fluxWm2[i] = finite(flux);
    const shellArea = perimeter * ds;
    heatLoss += flux * shellArea;
    surface += shellArea;
    if (flux >= threshold) above += shellArea;
    peak = Math.max(peak, flux);
    // Bulk-air warming along the shell. The exponential form remains stable
    // as Q becomes small and provides the outlet temperature for inspection.
    const gain = Math.min(1, h * shellArea / (RHO * Q * CP));
    airC += (WALL_C - airC) * (1 - Math.exp(-gain));
  }
  return {
    closed:false, Q:Qml, peakWm2:finite(peak), meanWm2:surface ? heatLoss / surface : 0,
    heatLossW:finite(heatLoss), areaAbove50Cm2:finite(above * 1e4), surfaceAreaCm2:finite(surface * 1e4),
    outletTempC:finite(airC), fluxWm2, thresholdWm2:threshold,
  };
}

export function evaluateHeatFlux(data, metrics, options={}) {
  const out={};
  for (const key of ['L','R','common']) {
    if (!data.sides[key] || !metrics[key]) continue;
    out[key]=heatFlux(data.sides[key].profile, metrics[key].Q, options);
  }
  out.total={
    heatLossW:['L','R','common'].reduce((sum,k)=>sum+(out[k]?.heatLossW||0),0),
    areaAbove50Cm2:['L','R','common'].reduce((sum,k)=>sum+(out[k]?.areaAbove50Cm2||0),0),
  };
  return out;
}
