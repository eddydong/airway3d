import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { evaluate, pressureDrop, flowAt } from '../viewer/hydraulics.js';
import { defaults, scenarioData, buildReport, noIntervention, POSITIONS } from '../viewer/scenario-model.js';
import { reportPDF } from '../viewer/report-pdf.js';
import { heatFlux, evaluateHeatFlux } from '../viewer/heatflux.js';

const base=JSON.parse(fs.readFileSync(new URL('../viewer/data/pre/airway.json',import.meta.url)));
const pipe=(radius=2,length=100)=>({s_mm:Array.from({length},(_,i)=>i+.5),area_mm2:Array(length).fill(Math.PI*radius**2),hyd_diam_mm:Array(length).fill(radius*2)});
const near=(actual,want,tol=1e-7)=>assert.ok(Math.abs(actual-want)<tol,`${actual} vs ${want}`);

test('viscous straight pipe matches analytic Hagen-Poiseuille pressure and wall shear',()=>{
  const p=pressureDrop(pipe(),10),r=.002,q=10e-6,mu=1.8e-5;
  near(p.friction,8*mu*.1*q/(Math.PI*r**4));
  near(p.tauMax,4*mu*q/(Math.PI*r**3));
  near(p.dP,p.friction+p.local);
});
test('zero-flow state is finite; invalid flow is rejected',()=>{
  const p=pressureDrop(pipe(),0);assert.equal(p.dP,0);assert.equal(p.vmax,0);
  assert.throws(()=>evaluate(base,NaN));assert.throws(()=>evaluate(base,-1));
});
test('parallel branches conserve flow and equalize pressure',()=>{
  const r=evaluate(base,250);near(r.L.Q+r.R.Q,250);near(r.L.dP,r.R.dP);
  near(r.total.dP,r.L.dP+r.common.dP);near(r.total.R,r.total.dP/250);
});
test('very narrow side is allowed below 1% flow, with no artificial 1% floor',()=>{
  const d=structuredClone(base);d.sides.L.profile=pipe(.05);d.sides.R.profile=pipe(2);
  const r=evaluate(d,250);assert.ok(r.L.Q/250<.001);near(r.L.dP,r.R.dP,1e-6);
});
test('unilateral closure routes all flow to the other nostril',()=>{
  const d=structuredClone(base);d.sides.L.profile.area_mm2[10]=0;
  const r=evaluate(d,250);assert.equal(r.L.Q,0);assert.equal(r.L.Q150,0);assert.equal(r.R.Q,250);
  assert.equal(r.L.R,Infinity);assert.ok(Number.isFinite(r.total.dP));assert.equal(r.total.blocked,false);
});
test('bilateral or pharyngeal closure reports impossible flow, never NaN',()=>{
  for(const keys of [['L','R'],['common']]){
    const d=structuredClone(base);keys.forEach(k=>d.sides[k].profile.area_mm2[10]=0);
    const r=evaluate(d,250);assert.equal(r.total.blocked,true);assert.equal(r.total.dP,Infinity);assert.equal(r.total.deliveredQ,0);
    assert.equal(r.L.Q+r.R.Q+r.common.Q,0);
  }
});
test('flow at fixed pressure is the inverse of pressure loss',()=>near(pressureDrop(pipe(),flowAt(pipe(),150)).dP,150,1e-6));
test('congestion proxy is resistance at a fixed 150 Pa, independent of selected breathing effort',()=>{
  const quiet=evaluate(base,50),strong=evaluate(base,1000),zero=evaluate(base,0);
  for(const k of ['L','R']){
    near(quiet[k].R150,150/quiet[k].Q150);
    near(quiet[k].R150,strong[k].R150);near(quiet[k].R150,zero[k].R150);
    near(pressureDrop(base.sides[k].profile,150/quiet[k].R150).dP,150,1e-6);
  }
  assert.ok(quiet.L.R150>quiet.R.R150,'absolute values expose baseline left/right asymmetry');
});
test('congestion proxy changes independently per side and reports closure without a false finite score',()=>{
  const baseline=evaluate(base,250),s=defaults();s.operations.L.head=1;
  const changed=evaluate(scenarioData(base,s),250);
  assert.ok(changed.L.R150<baseline.L.R150);near(changed.R.R150,baseline.R.R150);
  Object.assign(s,{position:'left',responseL:4,elapsed:15});s.operations.L.head=0;
  const closed=evaluate(scenarioData(base,s),250);assert.equal(closed.L.R150,Infinity);assert.ok(Number.isFinite(closed.R.R150));
  const reopened=evaluate(scenarioData(base,s,'right'),250);assert.ok(Number.isFinite(reopened.L.R150));
  const early=evaluate(scenarioData(base,{...s,elapsed:0}),250);assert.ok(early.L.R150<closed.L.R150);
});
test('reduced-order thermal metric is finite, positive during inspiration, and zero for closure',()=>{
  const metrics=evaluate(base,250),thermal=evaluateHeatFlux(base,metrics);
  for(const side of ['L','R','common']){
    assert.ok(thermal[side].peakWm2>0);assert.ok(thermal[side].heatLossW>0);assert.ok(thermal[side].areaAbove50Cm2>=0);
    assert.ok(thermal[side].outletTempC>20&&thermal[side].outletTempC<32.6);
  }
  const closed=structuredClone(base);closed.sides.L.profile.area_mm2[10]=0;
  const c=heatFlux(closed.sides.L.profile,0);assert.equal(c.closed,true);assert.equal(c.peakWm2,0);assert.equal(c.heatLossW,0);
  const s=defaults();s.operations.L.head=1;const altered=evaluateHeatFlux(scenarioData(base,s),evaluate(scenarioData(base,s),250));
  assert.notEqual(altered.L.areaAbove50Cm2,thermal.L.areaAbove50Cm2);
});
test('face-up reference is identity regardless of dependent response amplitudes',()=>{
  const s=defaults();s.responseL=4;s.responseR=2;
  const before=JSON.stringify(base),d=scenarioData(base,s);
  for(const side of ['L','R'])d.sides[side].profile.area_mm2.forEach((a,i)=>near(a,base.sides[side].profile.area_mm2[i]));
  assert.equal(JSON.stringify(base),before);
});
test('left-only intervention does not change right geometry; resetting is lossless',()=>{
  const s=defaults();s.operations.L.head=1;
  const d=scenarioData(base,s);assert.ok(d.sides.L.volume_cc>base.sides.L.volume_cc);
  d.sides.R.profile.area_mm2.forEach((a,i)=>near(a,base.sides.R.profile.area_mm2[i]));
  const reset=scenarioData(base,noIntervention(s));near(evaluate(reset,250).total.dP,evaluate(base,250).total.dP);
});
test('posture response evolves, can close the left branch, and reopens on the right',()=>{
  const s=defaults();Object.assign(s,{position:'left',responseL:4,elapsed:15});
  const closed=evaluate(scenarioData(base,s),250);assert.equal(closed.L.Q,0);assert.equal(closed.R.Q,250);
  const right=evaluate(scenarioData(base,s,'right'),250);assert.ok(right.L.Q>0&&right.R.Q>0);
  const early=evaluate(scenarioData(base,{...s,elapsed:0}),250);assert.ok(early.L.Q>0);
});
test('posture batch compares every arm and includes a matched baseline in each position',()=>{
  const s=defaults();s.operations.L.head=1;
  const other=defaults();other.operations.R.body=1;
  const report=buildReport(base,s,[{name:'Current',settings:s},{name:'Saved',settings:other}],250,{slice_mm:.625});
  assert.equal(report.rows.length,12);assert.equal(report.scanPosition,'supine');
  for(const r of report.rows){assert.ok(r.position in POSITIONS);assert.equal(r.settings.position,r.position);assert.ok(r.data.sides.L.profile.area_mm2.every(Number.isFinite));}
  assert.equal(report.rows.filter(r=>r.name==='No intervention').length,4);
  const pdf=reportPDF(report);assert.ok(new TextDecoder().decode(pdf).startsWith('%PDF-1.4'));assert.ok(new TextDecoder().decode(pdf).endsWith('%%EOF\n'));
});
test('duplicate plans are deduplicated and confirmed scan provenance is exported',()=>{
  const s=defaults();s.operations.L.body=1;
  const report=buildReport(base,s,[{name:'A',settings:s},{name:'A saved',settings:structuredClone(s)}],250);
  assert.equal(report.rows.length,8);
  assert.deepEqual(report.calibration.skull_window,{center:400,width:1500});
  assert.deepEqual(report.calibration.head_window,{center:40,width:350});
  assert.equal(report.simulation.viscous3D,false);
  assert.equal(report.simulation.clinicalValidation,'not performed');
  assert.equal(report.simulation.posture,'assumed response relative to face-up CT');
});
