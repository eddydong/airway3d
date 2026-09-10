import test from 'node:test';
import assert from 'node:assert/strict';
import {register} from 'node:module';
register('./three-loader.mjs',import.meta.url);
const {CFDThermalWalls,thermalFramePair,matchingThermal}=await import('../viewer/cfd-thermal.js');
const {libraryKey,normalizedRequest}=await import('../viewer/cfd-library.js');
const frames=[0,2,4].map(timeS=>({timeS}));
test('thermal playback interpolates physical values and repeats only at the breath boundary',()=>{
  assert.deepEqual(thermalFramePair(frames,1,4),{a:0,b:1,weight:.5,phase:1});
  assert.deepEqual(thermalFramePair(frames,7,4),{a:1,b:2,weight:.5,phase:3});
  const meta={periodS:4,wall:{faces:1},frames};
  const walls=new CFDThermalWalls(meta,new Float32Array(12),[[36,100],[34,500],[36,100]].map(x=>new Float32Array(x).buffer));
  walls.step(5);assert.deepEqual(walls.sampleFace(0),{temperatureC:35,heatFluxWm2:300});
  walls.setMode('flux');assert.equal(walls.material.uniforms.displayFlux.value,1);walls.dispose();
});
test('mismatched or unaccepted thermal recordings are never displayed',()=>{
  const r={requestId:'flow',geometryHash:'geo',solver:{periodS:4,durationS:8}};
  const t={schemaVersion:1,status:'complete',flowRequestId:'flow',geometryHash:'geo',periodS:4,wall:{faces:1,verticesPerFace:4},frames,
    gates:{finiteAndBounded:true,linearResidual:true,energyLedger:true,periodic:true}};
  assert.equal(matchingThermal(t,r),true);
  assert.equal(matchingThermal({...t,geometryHash:'stale'},r),false);
  assert.equal(matchingThermal({...t,gates:{...t.gates,periodic:false}},r),false);
});
test('static library matching uses solver inputs and ignores display/name settings',()=>{
  const request={backend:'gpu-lbm',mode:'transient',settings:{name:'My plan',position:'left'}};
  assert.equal(libraryKey(request),libraryKey(normalizedRequest(request)));
  assert.equal(libraryKey(request),libraryKey({...request,qMlS:400,settings:{...request.settings,name:'Renamed'}}));
  assert.notEqual(libraryKey(request),libraryKey({...request,pressurePa:35}));
  assert.notEqual(libraryKey(request),libraryKey({...request,settings:{...request.settings,operations:{L:{head:1}}}}));
});
test('face-up swelling knobs that do not move the wall reuse the recorded geometry',()=>{
  const request={backend:'gpu-lbm',mode:'transient',settings:{position:'supine',cycle:0}};
  assert.equal(libraryKey(request),libraryKey({...request,settings:{...request.settings,responseL:4,responseR:2,elapsed:30,tau:1,gravity:1,relief:50}}));
  assert.notEqual(libraryKey(request),libraryKey({...request,settings:{...request.settings,cycle:1}}));
  assert.notEqual(libraryKey(request),libraryKey({...request,settings:{...request.settings,position:'left'}}));
  const left={backend:'gpu-lbm',settings:{position:'left'}};
  assert.notEqual(libraryKey(left),libraryKey({...left,settings:{...left.settings,responseL:4}}));
});
