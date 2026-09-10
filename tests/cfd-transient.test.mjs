import test from 'node:test';
import assert from 'node:assert/strict';
import {register} from 'node:module';
register('./three-loader.mjs',import.meta.url);
const THREE=await import('three');
const {CFDTransientParticles}=await import('../viewer/cfd-transient-particles.js');
const {comparisonRequests,comparisonReading}=await import('../viewer/cfd-comparison.js');
const {defaults}=await import('../viewer/scenario-model.js');
const {matchingResult}=await import('../viewer/cfd-lab.js');
const vector=x=>new Float32Array(Array.from({length:27},()=>[x,0,0]).flat()).buffer;
const frames=[0,1,2].map((timeS,i)=>({timeS,velocity:String(i),flowMlS:{L:[0,3,-3][i],R:[0,7,-7][i],outlet:[0,10,-10][i]},pressureDropPa:[0,2,-4][i]}));
const result={status:'converged',solver:{qMlS:10,periodS:2,durationS:2,temporal:'transient',direction:'cycle'},
  flowMlS:frames[0].flowMlS,p99SpeedMS:.01,field:{schemaVersion:3,axisOrder:'XYZ',storageOrder:'fluid-C',fluidCells:27,dims:[3,3,3],boxMin:[0,0,0],spacingMm:1,frames}};
const fluid=new Uint8Array(27).fill(1).buffer;
test('recorded velocity changes use distinct computed frames; readings follow identical time',async()=>{
  const p=new CFDTransientParticles(new THREE.Scene(),result,fluid,vector(0),async path=>vector(path==='1'?.001:-.003));
  await Promise.all(p.pending.values());p.setCount(100);p.setEnabled(true);p.timeScale=1;
  p.step(.5);assert.equal(p.time,.5);assert.equal(p.qNow,5);assert.equal(p.currentMetrics.pressureDropPa,1);
  assert.ok(Math.abs(p.sample([1.5,1.5,1.5])[0]-.0005)<1e-9);
  p.step(.5);p.step(.5);
  assert.equal(p.time,1.5);assert.equal(p.qNow,0);
  // At reversal the velocity is NOT forcibly zero: recorded inertia remains.
  assert.ok(Math.abs(p.sample([1.5,1.5,1.5])[0]+.001)<1e-9);
  p.step(.5);assert.equal(p.ended,true);assert.equal(p.qNow,-10);
  p.step(1);assert.equal(p.time,2);p.dispose();
});
test('two-cycle recordings loop their second cycle instead of ending',async()=>{
  const two=[0,1,2,3,4].map((timeS,i)=>({timeS,velocity:String(i),flowMlS:{L:0,R:0,outlet:[0,10,0,10,0][i]},pressureDropPa:0}));
  const r={...result,solver:{...result.solver,durationS:4,cycles:2},field:{...result.field,frames:two}};
  const loads=[];
  const p=new CFDTransientParticles(new THREE.Scene(),r,fluid,vector(0),async path=>{loads.push(path);return vector(.001);});
  assert.equal(p.loops,true);assert.equal(p.loopStart,2);
  p.setCount(10);p.setEnabled(true);p.timeScale=1;
  for(let i=0;i<8;i++){await Promise.all(p.pending.values());p.step(.5);}
  assert.equal(p.ended,false);assert.equal(p.time,2);assert.equal(p.frame,2);
  await Promise.all(p.pending.values());p.step(.5);assert.equal(p.time,2.5);assert.equal(p.qNow,5);
  assert.ok(loads.includes('2'),'loop start frame is fetched again before the wrap');p.dispose();
  // A single-cycle recording still ends.
  const q=new CFDTransientParticles(new THREE.Scene(),result,fluid,vector(0),async()=>vector(.001));
  assert.equal(q.loops,false);q.dispose();
});
test('with listed openings, exhaled air leaves the nostril as a puff and room air is drawn back in',async()=>{
  const vec=(x,y,z)=>new Float32Array(Array.from({length:27},()=>[x,y,z]).flat()).buffer;
  const faces=y=>[0,1,2].flatMap(i=>[0,1,2].map(k=>[i+.5,y,k+.5]));
  const openings=[{name:'left_nostril',side:'L',role:'nostril',centreMm:[1.5,3,1.5],normal:[0,1,0],radiusMm:1.7,facesMm:faces(3)},
    {name:'outlet',side:'outlet',role:'outlet',centreMm:[1.5,0,1.5],normal:[0,-1,0],radiusMm:1.7,facesMm:faces(0)}];
  const fr=[0,1,2].map((timeS,i)=>({timeS,velocity:String(i),flowMlS:{L:[0,-10,10][i],R:0,outlet:[0,-10,10][i]},pressureDropPa:0}));
  const r={...result,openings,field:{...result.field,frames:fr}};
  const p=new CFDTransientParticles(new THREE.Scene(),r,fluid,vec(0,0,0),async path=>path==='1'?vec(0,.5,0):vec(0,-.5,0));
  await Promise.all(p.pending.values());p.setCount(200);p.setEnabled(true);p.timeScale=1;
  assert.equal(p.exchange,true);
  const count=mode=>Array.from(p.mode).filter(m=>m===mode).length;
  const insideAll=()=>{for(let i=0;i<p.n;i++)if(p.mode[i]===0)assert.ok(p.inside(Array.from(p.pos.subarray(i*3,i*3+3))));};
  let puffs=0,room=0;
  for(let s=0;s<90;s++){await Promise.all(p.pending.values());p.step(.01);insideAll();puffs=Math.max(puffs,count(2));}
  assert.ok(puffs>0,'exhaled particles continue outside the nostril');
  for(let i=0;i<p.n;i++)if(p.mode[i]===2){assert.ok(p.pos[i*3+1]>3,'puff is outside the domain');assert.ok(p.puff[i]>0);}
  for(let s=0;s<110;s++){await Promise.all(p.pending.values());p.step(.01);insideAll();room=Math.max(room,count(1));}
  assert.ok(room>0,'room air is seeded outside the inhaling nostril');
  assert.ok(count(0)>0);p.dispose();
});
test('buffering stops physical time and disposal ignores late frames',async()=>{
  const promises=[];
  const p=new CFDTransientParticles(new THREE.Scene(),result,fluid,vector(0),()=>new Promise(resolve=>promises.push(resolve)));
  p.setEnabled(true);p.step(.1);assert.equal(p.time,0);assert.equal(p.buffering,true);
  p.dispose();promises.forEach(resolve=>resolve(vector(.001)));await Promise.resolve();assert.equal(p.cache.size,0);
});
test('comparison requests share physiology while every plan/position has its own solver request',()=>{
  const s=defaults();s.operations.L.head=1;
  const saved={name:'Other plan',settings:{...defaults(),responseL:4,operations:{L:{head:0,body:1,valve:0},R:{head:0,body:0,valve:0}}}};
  const rows=comparisonRequests({settings:s,mode:'transient',qMlS:250,periodS:4},[saved]);
  assert.equal(rows.length,12);assert.equal(rows.filter(r=>r.baseline).length,4);
  for(const row of rows){assert.equal(row.request.settings.responseL,s.responseL);assert.equal(row.request.periodS,4);}
  assert.equal(new Set(rows.map(r=>JSON.stringify(r.request))).size,12);
  assert.equal(rows.find(r=>r.baseline).request.settings.operations.L.head,0);
  assert.equal(s.operations.L.head,1);
});
test('transient comparison uses the same phase and never accepts failed numerical checks',()=>{
  const r={...result,requestId:'a',geometryHash:'g',geometry:{geometryHash:'g'},gates:Object.fromEntries(['mesh','massBalance','wallLeak','requestedFlow','coordinateAlignment','completed','frames','residuals','courant'].map(k=>[k,true]))};
  assert.ok(matchingResult(r,'a'));
  assert.ok(!matchingResult({...r,gates:{...r.gates,courant:false}},'a'));
  const reading=comparisonReading({...r,solver:{...r.solver,durationS:3,periodS:2}});
  assert.equal(reading.timeS,1); // closest saved frame to t=1.5, consistently selected
});
test('GPU lattice recordings are accepted on their own gate set and never on missing ones',()=>{
  const gpuGates=['mesh','massBalance','coordinateAlignment','completed','frames','stability'];
  const r={...result,requestId:'a',geometryHash:'g',geometry:{geometryHash:'g'},solver:{...result.solver,backend:'gpu-lbm',pressureAmplitudePa:30},
    gates:Object.fromEntries(gpuGates.map(k=>[k,true]))};
  assert.ok(matchingResult(r,'a'));
  for(const gate of gpuGates)assert.ok(!matchingResult({...r,gates:{...r.gates,[gate]:false}},'a'),gate);
  // An OpenFOAM recording still needs residual and Courant gates, which a lattice recording does not carry.
  assert.ok(!matchingResult({...r,solver:{...r.solver,backend:undefined}},'a'));
  // The lattice backend records breathing only; a steady lattice field is not displayable.
  assert.ok(!matchingResult({...r,solver:{...r.solver,temporal:'steady'},field:{...r.field,schemaVersion:2,storageOrder:'C'}},'a'));
  // Pressure-driven rows compare flow at the same nominal amplitude; the reading carries both.
  const reading=comparisonReading({...r,solver:{...r.solver,durationS:3,periodS:2}});assert.equal(reading.timeS,1);assert.ok('flow' in reading&&'pressure' in reading);
  const rows=comparisonRequests({settings:defaults(),backend:'gpu-lbm',pressurePa:30,spacingMm:.7,refinement:1,mode:'transient',periodS:4});
  assert.ok(rows.every(row=>row.request.backend==='gpu-lbm'&&row.request.pressurePa===30&&!('qMlS' in row.request)));
});

test('thermal playback advances with hidden particles and has its own pause control',async()=>{
  const p=new CFDTransientParticles(new THREE.Scene(),result,fluid,vector(0),async()=>vector(.001));
  await Promise.all(p.pending.values());p.timeScale=1;p.playWithoutParticles=true;
  assert.equal(p.enabled,false);const positions=p.pos.slice();
  p.step(.5);assert.equal(p.time,.5);assert.deepEqual(p.pos,positions);
  p.playbackPaused=true;p.step(.5);assert.equal(p.time,.5);
  p.playbackPaused=false;p.step(.5);assert.equal(p.time,1);p.dispose();
});

test('playback retains fractional time across saved frames and cycle wraps without further fetching',async()=>{
  const two=[0,1,2,3,4].map((timeS,i)=>({timeS,velocity:String(i),flowMlS:{L:0,R:0,outlet:0},pressureDropPa:0}));
  const r={...result,solver:{...result.solver,durationS:4},field:{...result.field,frames:two}};
  let loads=0;
  const p=new CFDTransientParticles(new THREE.Scene(),r,fluid,vector(0),async()=>{loads++;return vector(.001);});
  await p.prepareLoop();await Promise.all(p.pending.values());
  const initialLoads=loads;
  p.timeScale=1;p.playWithoutParticles=true;
  for(let i=0;i<77;i++)p.step(.037);
  assert.ok(Math.abs(p.time-(2+(77*.037)%2))<1e-9,'no time lost at frame or loop boundaries');
  assert.equal(p.buffering,false);assert.equal(loads,initialLoads);
  p.dispose();
});
