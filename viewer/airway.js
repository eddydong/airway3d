// Airway analytics panel: per-nostril table, cross-section profile chart, 3D centrelines
// and minimum-cross-section markers.  Data from data/<set>/airway.json (pipeline/airway.py).
import * as THREE from 'three';
import { evaluate } from './hydraulics.js';

const COL = { L: '#ff7a59', R: '#59b6ff', common: '#c9d1d9' };
const NAME = { L: 'Left', R: 'Right', common: 'Pharynx' };

const fmt = (v, d = 1) => (v === Infinity ? '∞' : v == null || !isFinite(v) ? '–' : (+v).toFixed(d));

export class AirwayPanel {
  constructor(scene, statsEl, chartEl) {
    this.scene = scene;
    this.statsEl = statsEl;
    this.chart = chartEl;
    this.group = new THREE.Group();
    scene.add(this.group);
    this.q = 250;
    this.data = null;
    this.ref = null;
    this.chartObserver=new ResizeObserver(()=>{if(this.chart.clientWidth>0)this.drawChart();});
    this.chartObserver.observe(this.chart);
  }

  setFlowRate(q) { this.q = q; this.renderStats(); }
  setCFD(result,message='') {
    this.cfdMode=true;this.cfdResult=result;this.cfdMessage=message;this.cfdFrame=null;
    this.chart.hidden=false;this.group.visible=false;this.renderStats();this.drawChart();
  }

  setCFDFrame(frame){this.cfdFrame=frame;this.renderStats();}

  setData(data, ref, datasetName) {
    this.data = data;
    this.ref = ref && ref !== data ? ref : null;
    this.datasetName = datasetName;
    this.group.traverse(o => { o.geometry?.dispose(); o.material?.dispose(); });
    this.group.clear();
    if (!data) { this.statsEl.innerHTML = '<span class="hint">no airway analysis yet — run pipeline/airway.py</span>'; return; }
    this.build3D();
    this.renderStats();
    this.drawChart();
  }

  build3D() {
    for (const side of ['L', 'R', 'common']) {
      const s = this.data.sides[side];
      if (!s || !s.profile?.centers?.length) continue;
      // preferred: the wall-avoiding minimal path (nostril → choana → nasopharynx meeting point
      // → outlet); fall back to the geodesic-shell centroids for older data
      const src = s.path || s.profile.centers.slice(0, s.profile.n_centerline ?? s.profile.centers.length);
      const pts = src.map((c) => new THREE.Vector3(...c));
      if (pts.length < 3) continue;
      const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
      const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(8, pts.length * 2), 0.35, 8, false),
        new THREE.MeshStandardMaterial({ color: COL[side], emissive: COL[side], emissiveIntensity: 0.25, roughness: 0.5 }));
      tube.renderOrder = 400;
      this.group.add(tube);
      if (s.min_area_pos && side !== 'common') {
        const ring = new THREE.Mesh(new THREE.TorusGeometry(Math.sqrt(s.min_area_mm2 / Math.PI) + 0.6, 0.35, 8, 40),
          new THREE.MeshStandardMaterial({ color: '#ffffff', emissive: COL[side], emissiveIntensity: 1.2 }));
        ring.position.set(...s.min_area_pos);
        // orient along the local flow direction (tangent of the centreline at the MCA)
        let i = 0, best = Infinity;
        pts.forEach((p, j) => { const d = p.distanceToSquared(ring.position); if (d < best) { best = d; i = j; } });
        const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)];
        ring.lookAt(ring.position.clone().add(b.clone().sub(a)));
        ring.renderOrder = 401;
        this.group.add(ring);
      }
    }
  }

  // 1-D pressure-loss model evaluated at the current flow rate (same model as the pipeline,
  // but the split between nostrils is re-solved here so the slider is live).
  renderStats() {
    if(this.cfdMode){
      const r=this.cfdResult;
      if(!r){this.statsEl.textContent=this.cfdMessage||'Build a matching CFD result';return;}
      if(r.solver.temporal==='transient'){
        const f=this.cfdFrame||r.field.frames[0];
        const lattice=r.solver.backend==='gpu-lbm';
        this.statsEl.innerHTML=`<p><strong>Recorded breathing · ${fmt(f.timeS,2)} s</strong></p><p>${lattice?'GPU lattice Boltzmann LES · pressure-driven':'OpenFOAM · laminar Navier–Stokes · flow-driven'}<br>Signed flow: + inspiration / − expiration</p><table><tbody>
          <tr><td>Left flow (mL/s)</td><td>${fmt(f.flowMlS.L)}</td></tr><tr><td>Right flow (mL/s)</td><td>${fmt(f.flowMlS.R)}</td></tr>
          <tr><td>Nose → throat ΔP (Pa)</td><td>${fmt(f.pressureDropPa)}</td></tr>
          <tr><td>Airway volume (cc)</td><td>${fmt(r.geometry.volumeCc,3)}</td></tr></tbody></table>
          <p class="hint">Velocity and these readings interpolate between the same saved CFD times. ${r.solver.cycles>1?'Two cycles':'One cycle'} from rest; periodicity, mesh/time-step independence and clinical validation are unestablished.${lattice?' ΔP is measured between the cells adjacent to the openings; the nominal throat amplitude was '+r.solver.pressureAmplitudePa+' Pa.':''}</p>`;
        return;
      }
      const rows=[['Solver','OpenFOAM · viscous Navier–Stokes'],['Direction',r.solver.direction],
        ['Airway volume (cc)',fmt(r.geometry.volumeCc,3)],['Left flow (mL/s)',fmt(r.flowMlS.L)],['Right flow (mL/s)',fmt(r.flowMlS.R)],
        ['Nose → throat ΔP (Pa)',fmt(r.pressureDropPa)],['Total resistance (Pa·s/mL)',fmt(r.resistancePaSMl,3)],
        ['Peak cell speed (m/s)',fmt(r.peakSpeedMS,2)],['99th percentile speed (m/s)',fmt(r.p99SpeedMS,2)],
        ['Solver grid (mm)',fmt(r.geometry.spacingMm,3)],['Volume cells',r.geometry.cells.toLocaleString()]];
      this.statsEl.innerHTML=`<table><tbody>${rows.map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join('')}</tbody></table><p class="hint">Measured on the solver domain and computed fields. Heat transfer and resistance at 150 Pa have not been solved in this case. Grid independence and clinical validation remain separate checks.</p>`;
      return;
    }
    const d = this.data; if (!d) return;
    const res = evaluate(d, this.q);
    const ref = this.ref ? evaluate(this.ref, this.q) : null;
    const rows = [
      ['Volume (cc)', (s) => s.volume_cc, 1],
      ['Length (mm)', (s) => s.length_mm, 0],
      ['Min cross-section (mm²)', (s) => s.min_area_mm2, 1],
      ['  at distance (mm)', (s) => s.min_area_at_mm, 0],
      ['Mean cross-section (mm²)', (s) => s.mean_area_mm2, 1],
      ['Flow share (mL/s)', (s, r) => r.Q, 0],
      ['Pressure drop (Pa)', (s, r) => r.dP, 1],
      ['Resistance (Pa·s/mL)', (s, r) => r.R, 3],
      ['Peak speed (m/s)', (s, r) => r.vmax, 2],
      ['Mean speed (m/s)', (s, r) => r.vmean, 2],
      ['Peak wall shear (Pa)', (s, r) => r.tauMax, 2],
      ['Max Reynolds', (s, r) => r.ReMax, 0],
      ['Flow @150 Pa (mL/s)', (s, r) => r.Q150, 0],
    ];
    const sides = ['L', 'R', 'common'].filter((k) => d.sides[k]);
    let html = `<table><tr><th>${this.datasetName === 'post' ? 'Saved virtual' : this.datasetName === 'scenario' ? 'What-if' : 'CT supine'}</th>${sides.map((k) => `<th style="color:${COL[k]}">${NAME[k]}${res[k].closed ? ' (closed)' : ''}</th>`).join('')}</tr>`;
    for (const [label, f, dec] of rows) {
      html += `<tr><td>${label}</td>`;
      for (const k of sides) {
        const v = f(d.sides[k], res[k]);
        let cell = res[k].closed && label === 'Pressure drop (Pa)' ? 'closed' : fmt(v, dec);
        if (ref) {
          const rv = f(this.ref.sides[k], ref[k]);
          if (rv != null && isFinite(rv) && isFinite(v) && rv !== v) {
            const dv = v - rv;
            cell += ` <span class="delta ${dv > 0 ? '' : 'neg'}">${dv > 0 ? '+' : ''}${fmt(dv, dec)}</span>`;
          }
        }
        html += `<td>${cell}</td>`;
      }
      html += '</tr>';
    }
    html += `<tr><td>Total nose + pharynx resistance</td><td colspan="${sides.length}">${fmt(res.total.R, 3)} Pa·s/mL · ΔP ${fmt(res.total.dP, 1)} Pa at ${this.q} mL/s</td></tr>`;
    html += '</table>';
    if(res.total.blocked)html += '<p class="model-note">Requested nasal flow is infeasible: no open path remains in this assumed state.</p>';
    html += `<div class="hint" style="margin-top:6px">Live tube estimate: Darcy friction (laminar 64/Re, Blasius) + Borda–Carnot expansion/contraction losses along each passage (one mean speed per cross-section; a 1-D pipe calculation). Stored potential-flow split (reference only for the drawing): L ${fmt(100 * (d.flow_split?.L ?? 0), 0)}% / R ${fmt(100 * (d.flow_split?.R ?? 0), 0)}%.</div>`;
    this.statsEl.innerHTML = html;
  }

  drawChart() {
    if(!this.data||!this.chart.clientWidth)return;
    const c = this.chart, ctx = c.getContext('2d');
    const W = c.width = c.clientWidth * 2, H = c.height = 170 * 2;
    ctx.clearRect(0, 0, W, H);
    const d = this.cfdMode?(this.ref||this.data):this.data;
    const reference=this.cfdMode?null:this.ref;
    const caption=document.querySelector('#chart-source');
    if(caption)caption.textContent=this.cfdMode?'Cross-section profile · CT reference anatomy (not the deformed CFD grid)':'Cross-section profile · current anatomy; dashed = CT reference';
    const series = ['L', 'R', 'common'].filter((k) => d.sides[k]?.profile);
    let smax = 0, amax = 0;
    for (const k of series) { const p = d.sides[k].profile; smax = Math.max(smax, p.s_mm[p.s_mm.length - 1] + (k === 'common' ? d.sides.L?.length_mm || 0 : 0)); amax = Math.max(amax, ...p.area_mm2); }
    if (reference) for (const k of series) { const p = reference.sides[k]?.profile; if (p) amax = Math.max(amax, ...p.area_mm2); }
    amax = Math.min(amax, 900);
    const pad = { l: 64, r: 12, t: 14, b: 34 };
    const X = (s) => pad.l + (s / smax) * (W - pad.l - pad.r);
    const Y = (a) => H - pad.b - (Math.min(a, amax) / amax) * (H - pad.t - pad.b);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 2; ctx.font = '20px Inter, system-ui'; ctx.fillStyle = '#8b93a3';
    for (let i = 0; i <= 4; i++) {
      const a = amax * i / 4, y = Y(a);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      ctx.fillText(a.toFixed(0), 8, y + 7);
    }
    for (let s = 0; s <= smax; s += 20) { const x = X(s); ctx.fillText(s.toFixed(0), x - 10, H - 8); }
    ctx.fillText('mm² vs mm from nostril', pad.l + 6, pad.t + 6);
    const plot = (data, k, alpha, dash) => {
      const p = data.sides[k]?.profile; if (!p) return;
      const off = k === 'common' ? (data.sides.L?.length_mm + data.sides.R?.length_mm) / 2 : 0;
      ctx.beginPath(); ctx.strokeStyle = COL[k]; ctx.globalAlpha = alpha; ctx.lineWidth = 3; ctx.setLineDash(dash ? [6, 6] : []);
      p.s_mm.forEach((s, i) => { const x = X(s + off), y = Y(p.area_mm2[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    };
    if (reference) for (const k of series) plot(reference, k, 0.45, true);
    for (const k of series) plot(d, k, 1, false);
    // MCA markers
    for (const k of ['L', 'R']) {
      const s = d.sides[k]; if (!s) continue;
      ctx.fillStyle = COL[k]; ctx.beginPath(); ctx.arc(X(s.min_area_at_mm), Y(s.min_area_mm2), 6, 0, Math.PI * 2); ctx.fill();
    }
  }
}
