import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { register } from 'node:module';
import { defaults, scenarioData, POSITIONS } from '../viewer/scenario-model.js';
import { evaluate } from '../viewer/hydraulics.js';
import { ScenarioTransport, warpPoint } from '../viewer/flow-scenario.js';
register('./three-loader.mjs',import.meta.url);
const THREE=await import('three');
const {FlowParticles}=await import('../viewer/flow.js');
const read=name=>fs.readFileSync(new URL(`../viewer/data/pre/${name}`,import.meta.url));
const base=JSON.parse(read('airway.json')),meta=JSON.parse(read('flow.json'));
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-6,`${a} vs ${b}`);

test('profile warp preserves longitudinal coordinates and scales transverse area',()=>{
  const out=[];warpPoint({center:[0,0,0],tangent:[0,0,1]},2,3,4,7,out);
  assert.deepEqual(out,[6,8,7]);
});
test('transport obeys branch flux / area scaling, zero closure, and restores reference width',()=>{
  const t=new ScenarioTransport(base,meta),s=defaults();s.operations.L.head=1;
  let d=scenarioData(base,s),m=evaluate(d,250);t.update(d,m,{immediate:true});
  for(const [j,f] of t.frames.entries()){
    const areaRatio=d.sides[f.side].profile.area_mm2[f.i]/base.sides[f.side].profile.area_mm2[f.i];
    near(t.gains[j]*areaRatio*meta.split[f.side],m[f.side].Q/250);
  }
  Object.assign(s,{position:'left',responseL:4,elapsed:15});s.operations.L.head=0;
  d=scenarioData(base,s);m=evaluate(d,250);t.update(d,m,{immediate:true});
  assert.equal(t.fractions.L,0);assert.equal(t.fractions.R,1);
  assert.ok(t.bySide.L.every(j=>t.gains[j]===0));
  d=scenarioData(base,defaults());t.update(d,evaluate(d,250),{immediate:true});
  assert.ok(t.ratios.every(r=>Math.abs(r-1)<1e-12));assert.ok(t.fractions.L>0);
});
test('real particle engine stays live across every posture and intervention without resetting phase or buffers',()=>{
  const flow=new FlowParticles(new THREE.Scene(),meta,read('flow.bin'));
  flow.setCount(1000);flow.setEnabled(true);
  const positions=flow.pos,geometry=flow.points.geometry;
  for(const position of Object.keys(POSITIONS))for(const gain of [0,1]) {
    const s=defaults();s.position=position;s.operations.L.head=gain;s.operations.R.body=gain;
    const data=scenarioData(base,s),metrics=evaluate(data,250),phase=flow.phase;
    flow.setScenario(base,data,metrics,{immediate:true});
    assert.equal(flow.phase,phase);assert.equal(flow.pos,positions);assert.equal(flow.points.geometry,geometry);
    const before=new Float32Array(flow.pos);
    for(let frame=0;frame<30;frame++)flow.step(1/60);
    assert.equal(flow.enabled,true);assert.ok(flow.phase>phase);
    assert.ok(flow.pos.some((v,i)=>v!==before[i]),`${position}/${gain} must move`);
    assert.ok(flow.life.some(v=>v>0));assert.ok(flow.renderPos.every(Number.isFinite));
    assert.ok(flow.lpos.every(Number.isFinite));
    near(flow.nostrils[0].frac,metrics.L.Q/250);
  }
  // Both breath directions remain active in a modified scenario.
  flow.phase=0.6;flow.step(1/60);assert.ok(flow.phaseInfo().q<0);
  flow.phase=0.1;flow.step(1/60);assert.ok(flow.phaseInfo().q>0);
  flow.setEnabled(false);const phase=flow.phase;flow.step(1/60);assert.equal(flow.phase,phase);
  flow.setEnabled(true);flow.step(1/60);assert.ok(flow.phase>phase);
  flow.dispose();
});
test('thermal colour mode is wired to live mucosal heat flux attributes',()=>{
  const flow=new FlowParticles(new THREE.Scene(),meta,read('flow.bin'));flow.setCount(1200);flow.setEnabled(true);
  const s=defaults(),d=scenarioData(base,s),m=evaluate(d,250);
  flow.setScenario(base,d,m,{immediate:true});flow.setColorMode('thermal');
  assert.equal(flow.pmat.uniforms.uColorMode.value,2);
  assert.equal(flow.lmat.uniforms.uColorMode.value,2);
  assert.ok(flow.points.geometry.getAttribute('thermal'));
  assert.ok(flow.lines.geometry.getAttribute('thermal'));
  flow.breathing=false;flow.step(1/60);
  assert.ok(flow.thermal.some(v=>v>0),'inspiratory particles must carry heat-flux values');
  assert.ok(flow.thermal.every(Number.isFinite));
  const before=flow.thermal.reduce((a,v)=>Math.max(a,v),0);
  const s2=defaults();s2.operations.L.head=1;const d2=scenarioData(base,s2),m2=evaluate(d2,250);
  flow.setScenario(base,d2,m2,{immediate:true});flow.step(1/60);
  assert.ok(flow.thermal.every(Number.isFinite));
  assert.notEqual(flow.thermalMax,500);
  assert.ok(before>0);
  flow.dispose();
});
test('closure removes left particles and seeding; bilateral closure stops flow; reopening recovers',()=>{
  const flow=new FlowParticles(new THREE.Scene(),meta,read('flow.bin'));flow.setCount(1000);flow.setEnabled(true);
  const s=defaults();Object.assign(s,{position:'left',responseL:4,elapsed:15});
  let d=scenarioData(base,s);flow.setScenario(base,d,evaluate(d,250),{immediate:true});
  for(let i=0;i<100;i++){const p=flow.pool[--flow.poolN];flow.seedRoom(p);assert.equal(flow.side[p],1);}
  for(let i=0;i<30;i++)flow.step(1/60);
  assert.ok(flow.mode.every((m,i)=>m===3||flow.side[i]!==0));
  d=structuredClone(d);d.sides.R.profile.area_mm2[10]=0;
  flow.setScenario(base,d,evaluate(d,250),{immediate:true});flow.step(1/60);
  assert.equal(flow.poolN,flow.n);assert.equal(flow.phaseInfo().q,0);
  d=scenarioData(base,defaults());flow.setScenario(base,d,evaluate(d,250),{immediate:true});
  for(let i=0;i<60;i++)flow.step(1/60);
  assert.ok(flow.poolN<flow.n);assert.ok(flow.life.some(v=>v>0));
  flow.dispose();
});
