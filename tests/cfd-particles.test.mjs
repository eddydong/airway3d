import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {register} from 'node:module';
register('./three-loader.mjs',import.meta.url);
const THREE=await import('three');
const {CFDParticles}=await import('../viewer/cfd-particles.js');
const {CFDLab,matchingResult,matchingField}=await import('../viewer/cfd-lab.js');
const fixture=JSON.parse(execFileSync(new URL('../.venv/bin/python',import.meta.url).pathname,
  [new URL('./fixtures/cfd-domain.py',import.meta.url).pathname],{encoding:'utf8'}));
const result={field:{...fixture,schemaVersion:2,axisOrder:'XYZ',storageOrder:'C'},
  solver:{qMlS:250,direction:'inspiration'},flowMlS:{L:100,R:150},p99SpeedMS:.01};
const make=()=>new CFDParticles(new THREE.Scene(),result,
  new Float32Array(fixture.velocity).buffer,new Uint8Array(fixture.occupancy).buffer);
const geometry=new THREE.BufferGeometry();
geometry.setAttribute('position',new THREE.Float32BufferAttribute(fixture.vertices.flat(),3));
geometry.setIndex(fixture.faces.flat());
const surface=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({side:THREE.DoubleSide}));
surface.updateMatrixWorld();
function inSurface(point){
  const ray=new THREE.Raycaster(new THREE.Vector3(...point),new THREE.Vector3(1,.173,.319).normalize());
  return ray.intersectObject(surface).length%2===1;
}
test('native CFD cells and exported mesh agree in an asymmetric world-space domain',()=>{
  const particles=make();
  for(const p of fixture.centres){
    assert.ok(particles.inside(p));assert.ok(inSurface(p),`cell outside exported surface: ${p}`);
    const v=particles.sample(p);[.001,-.003,.002].forEach((x,j)=>assert.ok(Math.abs(v[j]-x)<1e-8));
  }
  particles.dispose();
});
test('seeded and advected CFD particles remain inside the independently exported surface',()=>{
  const particles=make();particles.setCount(100);particles.setEnabled(true);
  for(let frame=0;frame<100;frame++){
    particles.step(.03);
    for(let i=0;i<particles.n;i++){
      const p=Array.from(particles.pos.subarray(i*3,i*3+3));
      assert.ok(particles.inside(p));assert.ok(inSurface(p),`tracer outside exported surface: ${p}`);
    }
  }
  particles.dispose();
});
test('steady CFD never reverses or rescales its field through legacy breath controls',()=>{
  const particles=make(),velocity=particles.vel.slice();
  particles.setBreathing(true);particles.setPeriod(4);
  assert.equal(particles.phaseInfo().q,250);
  assert.deepEqual(particles.vel,velocity);
  particles.setEnabled(true);particles.setFlowRate(300);
  assert.equal(particles.enabled,false);
  particles.dispose();
});
test('CFD display rejects old coordinates, mismatched requests and failed convergence gates',()=>{
  const r={...result,status:'converged',requestId:'request-a',geometryHash:'geometry-a',
    geometry:{geometryHash:'geometry-a'},gates:Object.fromEntries(
      ['mesh','residuals','massBalance','wallLeak','pressureStable','requestedFlow','coordinateAlignment','completed'].map(k=>[k,true]))};
  assert.ok(matchingResult(r,'request-a'));
  assert.ok(!matchingResult(r,'request-b'));
  assert.ok(!matchingResult({...r,geometryHash:'different'},'request-a'));
  assert.ok(!matchingResult({...r,field:{...r.field,schemaVersion:1}},'request-a'));
  for(const gate of Object.keys(r.gates))assert.ok(!matchingResult({...r,gates:{...r.gates,[gate]:false}},'request-a'));
  assert.ok(!matchingResult({...r,status:'unconverged'},'request-a'));
  const preview={...r,status:'unconverged',gates:{...r.gates,residuals:false}};
  assert.ok(matchingField(preview,'request-a'));
  assert.ok(!matchingResult(preview,'request-a'));
  assert.ok(!matchingField({...preview,gates:{...preview.gates,coordinateAlignment:false}},'request-a'));
});
test('switching a saved dataset during CFD hides the newly attached legacy plume engine',()=>{
  const legacy={enabled:true,setEnabled(on){this.enabled=on;}};
  const lab=Object.create(CFDLab.prototype);
  Object.assign(lab,{enabled:true,particles:null,surface:null,hidden:[],
    ctx:{flow:legacy,setFlow(flow){this.flow=flow;}}});
  lab.clear();
  assert.equal(legacy.enabled,false);assert.equal(lab.originalFlow,legacy);assert.equal(lab.ctx.flow,null);
});
