// Quasi-steady 1-D estimate; SI internally. This is not a CFD/FSI solver.
const RHO = 1.2, MU = 1.8e-5;
export const isClosed = p => p.area_mm2.some(a => a <= 1e-8);

export function pressureDrop(profile, Qml) {
  if (!Number.isFinite(Qml) || Qml < 0) throw new Error('Flow must be finite and nonnegative');
  if (isClosed(profile)) return { dP: Qml ? Infinity : 0, friction: Qml ? Infinity : 0, local: 0, vmax: 0, vmean: 0, tauMax: 0, ReMax: 0, closed: true };
  const Q = Qml * 1e-6;
  let friction = 0, local = 0, vmax = 0, vsum = 0, tauMax = 0, ReMax = 0;
  const A = profile.area_mm2.map(a => a * 1e-6);
  for (let i = 0; i < A.length; i++) {
    const dh = Math.max(profile.hyd_diam_mm[i] * 1e-3, 1e-10);
    const v = Q / A[i], re = RHO * v * dh / MU;
    const ds = (i ? profile.s_mm[i] - profile.s_mm[i-1] : profile.s_mm[1] - profile.s_mm[0]) * 1e-3;
    // Algebraic laminar form stays linear all the way to zero flow.
    const stress = re < 2300 ? 32 * MU * v / dh : 0.316 / re ** 0.25 * 0.5 * RHO * v * v;
    friction += stress * ds / dh;
    tauMax = Math.max(tauMax, stress / 4);
    if (i) {
      const a0 = A[i-1], a1 = A[i];
      const K = a1 > a0 ? (1-a0/a1) ** 2 : 0.5 * (1-a1/a0);
      local += K * 0.5 * RHO * (a1 > a0 ? (Q/a0) ** 2 : v*v);
    }
    vmax = Math.max(vmax, v); vsum += v; ReMax = Math.max(ReMax, re);
  }
  local += 0.5 * RHO * (Q / A[0]) ** 2;
  return { dP: friction + local, friction, local, vmax, vmean: vsum / A.length, tauMax, ReMax, closed: false };
}

export function flowAt(profile, pressure) {
  if (isClosed(profile) || pressure <= 0) return 0;
  let lo = 0, hi = 5000;
  while (pressureDrop(profile, hi).dP < pressure && hi < 1e7) hi *= 2;
  for (let i=0; i<50; i++) { const q = (lo+hi)/2; if (pressureDrop(profile,q).dP > pressure) hi=q; else lo=q; }
  return (lo+hi)/2;
}

export function evaluate(data, q) {
  if (!Number.isFinite(q) || q < 0) throw new Error('Invalid total flow');
  const L=data.sides.L, R=data.sides.R, C=data.sides.common;
  const lc=isClosed(L.profile), rc=isClosed(R.profile), cc=C && isClosed(C.profile);
  const impossible=(lc && rc) || cc;
  let ql=0, qr=0;
  if (!impossible) {
    if (lc) qr=q;
    else if (rc) ql=q;
    else {
      let lo=0, hi=q;
      for (let i=0; i<50; i++) { const m=(lo+hi)/2; if (pressureDrop(L.profile,m).dP > pressureDrop(R.profile,q-m).dP) hi=m; else lo=m; }
      ql=(lo+hi)/2; qr=q-ql;
    }
  }
  const out={};
  for (const [key,side,flow] of [['L',L,ql],['R',R,qr],['common',C,impossible ? 0 : q]]) {
    if (!side) continue;
    const p=pressureDrop(side.profile,flow);
    const Q150=flowAt(side.profile,150);
    // Fixed-pressure resistance compares each passage independently of the
    // selected breathing effort and the opposite side. Not a symptom score.
    out[key]={ Q:flow, ...p, R:p.closed ? Infinity : flow ? p.dP/flow : null, Q150, R150:Q150 ? 150/Q150 : Infinity };
  }
  const nasal=lc ? out.R.dP : out.L.dP;
  const dP=impossible && q>0 ? Infinity : nasal+(out.common?.dP || 0);
  out.total={ dP, R:q ? dP/q : null, blocked:!!impossible, requestedQ:q, deliveredQ:impossible ? 0:q, nasalDP:nasal };
  return out;
}
