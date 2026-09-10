import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {CFDTransientParticles} from '../cfd-transient-particles.js';
const base='../data/cfd-validation/breathing-duct/';
const status=document.querySelector('#status');
try {
  const r=await fetch(base+'result.json').then(x=>x.json());
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(35,innerWidth/innerHeight,.1,1000);
  camera.position.set(60,30,170);camera.lookAt(2,30,2);
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(innerWidth,innerHeight);document.body.append(renderer.domElement);
  const model=await new GLTFLoader().loadAsync(base+'airway.glb');
  model.scene.traverse(o=>{if(o.isMesh)o.material=new THREE.MeshBasicMaterial({color:0x68bedb,transparent:true,opacity:.15,depthWrite:false,side:THREE.DoubleSide});});scene.add(model.scene);
  const data=await Promise.all([fetch(base+'occupancy.u8').then(x=>x.arrayBuffer()),fetch(base+r.field.frames[0].velocity).then(x=>x.arrayBuffer())]);
  const p=new CFDTransientParticles(scene,r,...data,async path=>fetch(base+path).then(x=>x.arrayBuffer()));
  p.setCount(100);p.timeScale=1;p.setEnabled(true);
  let inspiration=false,expiration=false,escaped=0;const clock=new THREE.Clock();
  document.querySelector('#replay').onclick=()=>p.replay();
  renderer.setAnimationLoop(()=>{
    p.step(Math.min(clock.getDelta(),.05));
    inspiration ||= p.qNow>.5;expiration ||= p.qNow<-.5;
    for(let i=0;i<p.n;i++)if(!p.inside(p.pos.subarray(i*3,i*3+3)))escaped++;
    document.querySelector('#phase').textContent=p.phaseInfo().label;
    status.textContent=p.ended?`${inspiration&&expiration&&!escaped?'PASS':'FAIL'}: both signed flow phases observed; ${escaped} outside particles; playback stopped at ${p.time.toFixed(2)} s.`:
      `Native saved frames · Q ${p.qNow.toFixed(3)} mL/s · ΔP ${p.currentMetrics.pressureDropPa.toFixed(4)} Pa · ${escaped} outside particles`;
    renderer.render(scene,camera);
  });
}catch(e){status.textContent='ERROR: '+e.message;throw e;}
