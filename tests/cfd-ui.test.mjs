import test from 'node:test';
import assert from 'node:assert/strict';
import {register} from 'node:module';
import {cfdAPI} from '../viewer/cfd-api.js';
import {TUBE_ESTIMATE,RECORDED_CFD,MISSING_CFD,TUBE_ESTIMATE_NOTE,AIRFLOW_INTRO} from '../viewer/airflow-labels.js';
register('./three-loader.mjs',import.meta.url);
const {CFDLab,cfdPanelMarkup}=await import('../viewer/cfd-lab.js');
const THREE=await import('three');

test('airflow mode switch names the calculation without calling the 3-D picture 1-D',()=>{
  const html=cfdPanelMarkup();
  assert.ok(html.includes(TUBE_ESTIMATE));
  assert.ok(html.includes(RECORDED_CFD));
  assert.ok(html.includes(AIRFLOW_INTRO));
  assert.equal(html.includes('1-D preview'),false);
  assert.ok(html.includes('id="cfd-recording-ui"'));
  assert.ok(html.includes('id="cfd-recording"'));
  assert.ok(html.includes('Solved scenario'));
  assert.ok(html.includes('data-cfd-position="left"'));
  assert.match(TUBE_ESTIMATE_NOTE,/1-D pipe calculation/);
  assert.match(MISSING_CFD,/live tube estimate/);
});
test('stalled CFD responses time out and abort the request',async t=>{
  let signal;
  t.mock.method(globalThis,'fetch',async (_,options)=>{
    signal=options.signal;
    return {ok:true,json:()=>new Promise((_,reject)=>signal.addEventListener('abort',()=>reject(Error('aborted'))))};
  });
  await assert.rejects(cfdAPI('/data/cfd/library.json',undefined, {timeoutMs:10}),/timed out/);
  assert.equal(signal.aborted,true);
});
test('unavailable service and HTTP failures are actionable',async t=>{
  const fetch=t.mock.method(globalThis,'fetch',async()=>{throw new TypeError('network');});
  await assert.rejects(cfdAPI('/api/cfd/status'),/make serve/);
  fetch.mock.mockImplementation(async()=>({ok:false,status:503,json:async()=>({error:'Queue unavailable'})}));
  await assert.rejects(cfdAPI('/data/cfd/library.json'),/Queue unavailable/);
});
test('CFD playback hides the four CT airway rows in the tissue list',t=>{
  const rows=Object.fromEntries(['airway','airway_L','airway_R','airway_common'].map(k=>[k,{hidden:false,style:{display:''}}]));
  const old=globalThis.document;
  globalThis.document={querySelector(sel){const m=sel.match(/data-k="([^"]+)"/);return m?rows[m[1]]:null;}};
  t.after(()=>{globalThis.document=old;});
  const lab=Object.create(CFDLab.prototype);
  lab.enabled=true;lab.syncReconstructionAirwayRows();
  assert.ok(Object.values(rows).every(r=>r.hidden&&r.style.display==='none'));
  lab.enabled=false;lab.syncReconstructionAirwayRows();
  assert.ok(Object.values(rows).every(r=>!r.hidden&&r.style.display===''));
});
test('CFD playback hides reconstruction airway meshes without stacking them on the solver surface',()=>{
  const mesh=new THREE.Mesh(),depth=new THREE.Mesh(),skin=new THREE.Mesh();
  mesh.visible=true;depth.visible=true;skin.visible=true;
  const lab=Object.create(CFDLab.prototype);
  Object.assign(lab,{hidden:[],ctx:{state:{tissues:{airway_L:{mesh,depthMesh:depth},skin:{mesh:skin}}}}});
  lab.hideReconstructionAirways();lab.hideReconstructionAirways();
  assert.equal(mesh.visible,false);assert.equal(depth.visible,false);assert.equal(skin.visible,true);
  assert.equal(lab.hidden.length,2);
});
test('hiding the CFD shell preserves the active field and simulation identity',t=>{
  const old=globalThis.document;
  globalThis.document={querySelector:()=>null};t.after(()=>{globalThis.document=old;});
  const lab=Object.create(CFDLab.prototype);
  Object.assign(lab,{enabled:true,surfaceVisible:false,surface:new THREE.Group(),generation:7,
    particles:{enabled:true},job:{id:'recording'},result:{requestId:'recording'}});
  lab.updateSurfaceVisibility();assert.equal(lab.surface.visible,false);
  assert.equal(lab.particles.enabled,true);assert.equal(lab.generation,7);assert.equal(lab.job.id,'recording');
  lab.surfaceVisible=true;lab.updateSurfaceVisibility();assert.equal(lab.surface.visible,true);
  lab.enabled=false;lab.updateSurfaceVisibility();assert.equal(lab.surface.visible,false);
});

test('viewer artifact loader refuses any job submission body',async()=>{await assert.rejects(cfdAPI('/api/cfd/jobs',{}),/only reads offline recordings/);});

function recordingLab(t){
  const elements=new Map(),old=globalThis.document;
  globalThis.document={querySelector(selector){
    if(!elements.has(selector))elements.set(selector,{textContent:'',hidden:false,replaceChildren(){},classList:{toggle(){}}});
    return elements.get(selector);
  }};
  t.after(()=>{globalThis.document=old;});
  let pending;
  t.mock.method(globalThis,'setTimeout',callback=>{pending=callback;return 1;});
  t.mock.method(globalThis,'clearTimeout',()=>{});
  const surface=new THREE.Group(),mesh=new THREE.Mesh(new THREE.BoxGeometry(),new THREE.MeshBasicMaterial());
  surface.add(mesh);surface.visible=false; // Thermal walls were showing instead.
  const surfaces=new THREE.Group();surfaces.add(surface);
  const tissue={visible:false},particles={dispose:t.mock.fn()},thermal={dispose:t.mock.fn()};
  const lab=Object.create(CFDLab.prototype);
  Object.assign(lab,{enabled:true,generation:0,surfaceVisible:true,surface,particles,thermal,hidden:[[tissue,true]],
    result:{requestId:'previous'},job:{id:'previous'},
    ctx:{surfaces,flow:particles,setFlow(value){this.flow=value;},airwayPanel:{setCFD:t.mock.fn()}},
    request:()=>({pressurePa:35}),library:{lookup:async()=>({state:'missing',message:'Not precomputed'})}});
  return {lab,elements,surface,mesh,tissue,particles,thermal,runLookup:()=>pending()};
}

test('missing recordings preserve a plain reference airway across repeated control changes',async t=>{
  const {lab,elements,surface,mesh,tissue,particles,thermal,runLookup}=recordingLab(t);
  const dispose=t.mock.method(mesh.geometry,'dispose');
  lab.changed();
  assert.match(elements.get('#flow-stats').textContent,new RegExp(MISSING_CFD.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')));
  assert.equal(surface.visible,true);
  assert.equal(surface.parent,lab.ctx.surfaces);
  assert.equal(dispose.mock.callCount(),0);
  assert.equal(particles.dispose.mock.callCount(),1);
  assert.equal(thermal.dispose.mock.callCount(),1);
  assert.equal(lab.ctx.flow,null);
  assert.equal(lab.result,null);
  assert.equal(tissue.visible,false);
  assert.match(elements.get('#cfd-surface-row .name .vol').textContent,/reference only/);
  assert.match(elements.get('#cfd-wall-probe').textContent,/unavailable/);
  await runLookup();
  assert.match(elements.get('#cfd-geometry-status').textContent,/No recorded 3-D CFD geometry.*Previous recording geometry/);
  lab.changed();await runLookup();
  assert.equal(lab.surface,surface);
  assert.equal(surface.visible,true);
  assert.equal(dispose.mock.callCount(),0);
  // A replacement recording or disabling CFD still fully releases the reference.
  lab.clear();
  assert.equal(surface.parent,null);
  assert.equal(dispose.mock.callCount(),1);
  assert.equal(tissue.visible,true);
  assert.equal(lab.referenceSurface,false);
});

test('failed catalog requests keep the reference and respect surface visibility',async t=>{
  const {lab,elements,surface,runLookup}=recordingLab(t);
  lab.library.lookup=async()=>{throw Error('Catalog offline');};
  lab.surfaceVisible=false;
  lab.changed();await runLookup();
  assert.equal(surface.visible,false);
  assert.equal(lab.surface,surface);
  assert.equal(elements.get('#cfd-status').textContent,'Catalog offline');
  assert.match(elements.get('#cfd-geometry-status').textContent,/reference/);
  lab.surfaceVisible=true;lab.updateSurfaceVisibility();
  assert.equal(surface.visible,true);
});
