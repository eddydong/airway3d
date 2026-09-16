import test from 'node:test';
import assert from 'node:assert/strict';
import {register} from 'node:module';
import {availableRecordings,catalogRecording,recordingLabel,recordingDetails,selectRecording,recordingForPosition} from '../viewer/cfd-recordings.js';
import {normalizedRequest,libraryKey} from '../viewer/cfd-library.js';
import {defaults} from '../viewer/scenario-model.js';
register('./three-loader.mjs',import.meta.url);
const {CFDLab}=await import('../viewer/cfd-lab.js');
const entry=(id,change={})=>({id,state:'complete',resultUrl:`/data/cfd/${id}/result.json`,request:normalizedRequest({backend:'gpu-lbm',settings:defaults(),...change})});

test('only distinct complete recordings are selectable; exact current configuration is preferred',()=>{
  const baseline=entry('base'),short=entry('short',{periodS:2}),failed={...entry('failed',{pressurePa:60}),state:'unconverged'};
  const entries=availableRecordings([short,baseline,failed,{...baseline,id:'duplicate'},{state:'complete',id:'bad'}]);
  assert.equal(entries.length,2);assert.equal(selectRecording(entries,short.request).id,'short');
  assert.equal(selectRecording(entries,{...short.request,pressurePa:67}).id,'base');
  assert.equal(selectRecording([],baseline.request),null);
});
test('labels identify anatomy, intervention and breath without inventing an accepted combination',()=>{
  const settings=defaults();settings.position='left';settings.operations.L.valve=1;
  const r=entry('valve',{settings,periodS:2,pressurePa:60});
  assert.match(recordingLabel(r),/Left side down.*L valve \+1 mm.*60 Pa.*2 s breath/);
  assert.match(recordingDetails(r),/0.7 mm geometry.*Fixed tissue response.*static assumption/);
});
test('selection applies one complete immutable request, including breathing and solver settings',t=>{
  const elements=new Map(),old=globalThis.document;
  globalThis.document={querySelectorAll(){return [];},querySelector(selector){if(!elements.has(selector))elements.set(selector,{});return elements.get(selector);}};
  t.after(()=>{globalThis.document=old;});
  const settings=defaults();settings.position='right';settings.operations.R.body=1;
  const saved=entry('right',{settings,periodS:2,pressurePa:60,refinement:2,spacingMm:.5,cycles:1});
  const lab=Object.create(CFDLab.prototype);let changes=0;
  Object.assign(lab,{enabled:true,recordings:[saved],lab:{stopPlayback(){},sync(){}},modeControls(){},changed(){changes++;}});
  lab.chooseRecording(saved);
  assert.equal(elements.get('#cfd-period').value,2);assert.equal(elements.get('#cfd-pressure').value,60);
  assert.equal(elements.get('#cfd-refine').value,2);assert.equal(elements.get('#cfd-recording').value,'right');
  assert.equal(lab.lab.settings.operations.R.body,1);assert.equal(changes,1);
  assert.deepEqual(lab.request(),saved.request);
  const request=lab.request();request.settings.operations.R.body=0;
  assert.equal(lab.request().settings.operations.R.body,1,'request clones never mutate catalog');
  lab.chooseRecording({id:'right'});assert.equal(changes,2,'comparison jobs resolve by catalog id');
  lab.chooseRecording({...saved,id:'not-in-library'});assert.equal(changes,2);
});
test('unchanged physical request preserves playback time, particles and thermal state',()=>{
  const saved=entry('baseline');const lab=Object.create(CFDLab.prototype),particles={time:6.2};
  Object.assign(lab,{enabled:true,result:{request:saved.request},particles,selectedRecording:saved,clear(){throw Error('must not clear');}});
  lab.changed();assert.equal(lab.particles,particles);assert.equal(particles.time,6.2);
  assert.equal(libraryKey(lab.request()),libraryKey(saved.request));
});

test('four position shortcuts select only existing complete scenarios, even when breath differs',()=>{
  const short=entry('short',{periodS:2}),right=entry('right',{settings:{...defaults(),position:'right'}});
  const result=recordingForPosition([short,right],short.request,'right');
  assert.equal(result.id,'right');assert.equal(result.request.periodS,4);
  assert.equal(recordingForPosition([short,right],short.request,'left'),null);
});
test('comparison jobs map onto listed recordings by id',()=>{
  const saved=entry('base');
  assert.equal(catalogRecording([saved],{id:'base',state:'complete'}).id,'base');
  assert.equal(catalogRecording([saved],{id:'other'}),null);
});
