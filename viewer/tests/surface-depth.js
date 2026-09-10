import * as THREE from 'three';
import {setSurfaceOpacity} from '../surface-material.js';
import {CFDThermalWalls} from '../cfd-thermal.js';

const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(128,128);
renderer.localClippingEnabled=true;renderer.outputColorSpace=THREE.LinearSRGBColorSpace;
document.body.append(renderer.domElement);
const target=new THREE.WebGLRenderTarget(128,128);
const scene=new THREE.Scene();scene.background=new THREE.Color(0);
const camera=new THREE.OrthographicCamera(-1,1,1,-1,.1,10);
camera.position.z=3;camera.lookAt(0,0,0);
const pixels=()=>{const data=new Uint8Array(128*128*4);renderer.setRenderTarget(target);renderer.render(scene,camera);renderer.readRenderTargetPixels(target,0,0,128,128,data);renderer.setRenderTarget(null);renderer.render(scene,camera);return data;};
const equal=(a,b)=>a.every((v,i)=>v===b[i]);
const fraction=(data,predicate)=>{let n=0;for(let i=0;i<data.length;i+=4)if(predicate(data[i],data[i+1],data[i+2]))n++;return n/(data.length/4);};
const red=data=>fraction(data,(r,g,b)=>r>180&&r>b*2);
const blue=data=>fraction(data,(r,g,b)=>b>180&&b>r*2);
let passed=0;
function check(name,condition){if(!condition)throw Error(name);passed++;const item=document.createElement('li');item.textContent='PASS · '+name;document.querySelector('#results').append(item);}
try{
  const near=new THREE.Mesh(new THREE.PlaneGeometry(2,2),new THREE.MeshBasicMaterial({color:'red',side:THREE.DoubleSide}));near.position.z=.3;
  const far=new THREE.Mesh(new THREE.PlaneGeometry(2,2),new THREE.MeshBasicMaterial({color:'blue',side:THREE.DoubleSide}));far.position.z=-.3;
  scene.add(near,far);
  // Negative control: this reproduces the original failure, so equality below
  // cannot pass merely because the two objects don't overlap on the GPU.
  for(const mesh of [near,far])Object.assign(mesh.material,{transparent:true,depthWrite:false,opacity:.95});
  near.renderOrder=1;far.renderOrder=2;const broken=pixels();
  near.renderOrder=2;far.renderOrder=1;
  check('old alpha blending reproduces draw-order dependence',!equal(broken,pixels()));
  for(const mesh of [near,far])setSurfaceOpacity(mesh.material,.95);
  near.renderOrder=0;far.renderOrder=0;const front=pixels();scene.remove(near,far);scene.add(far,near);
  check('transparent meshes use camera depth rather than insertion order',equal(front,pixels())&&red(front)>.85);
  check('paused translucent surfaces have identical pixels every frame',Array.from({length:20},()=>pixels()).every(p=>equal(front,p)));
  camera.position.z=-3;camera.lookAt(0,0,0);
  check('orbiting behind the surfaces reveals the newly nearer object',blue(pixels())>.85);
  camera.position.z=3;camera.lookAt(0,0,0);
  setSurfaceOpacity(near.material,0);setSurfaceOpacity(far.material,1);
  check('zero-opacity surfaces do not occlude geometry behind them',blue(pixels())===1);
  scene.remove(near,far);
  const quad=z=>[-1,-1,z,1,-1,z,1,1,z,-1,1,z];
  const frames=[0,2,4].map(timeS=>({timeS,minC:35.5,maxC:37,meanC:36.25,heatLossW:0}));
  const walls=new CFDThermalWalls({periodS:4,wall:{faces:2},frames},new Float32Array([...quad(.3),...quad(-.3)]),frames.map(()=>new Float32Array([37,0,35.5,0]).buffer));
  scene.add(walls.mesh);
  const index=walls.mesh.geometry.index;
  const forward=Array.from(index.array),reverse=[...forward.slice(6),...forward.slice(0,6)];
  walls.setOpacity(1);const opaque=pixels();
  index.array.set(reverse);index.needsUpdate=true;
  check('opaque CFD walls select the nearest face in either triangle order',equal(opaque,pixels())&&red(opaque)===1);
  walls.setOpacity(.5);const half=pixels();index.array.set(forward);index.needsUpdate=true;
  check('translucent CFD walls sort far faces before near faces',equal(half,pixels())&&fraction(half,(r,g,b)=>r>100&&r>b)===1);
  const ray=new THREE.Raycaster();ray.setFromCamera(new THREE.Vector2(),camera);
  check('probes retain the correct temperature after face sorting',walls.sampleHit(ray.intersectObject(walls.mesh)[0]).temperatureC===37);
  const frozen=pixels();
  check('paused thermal frames never flicker',Array.from({length:20},()=>pixels()).every(p=>equal(frozen,p)));
  camera.position.z=-3;camera.lookAt(0,0,0);const back=pixels();
  check('transparent wall ordering follows rotation',fraction(back,(r,g,b)=>b>100&&b>r)===1);
  camera.position.z=3;camera.lookAt(0,0,0);
  check('returning to a view reproduces its original pixels',equal(frozen,pixels()));
  walls.setOpacity(1);walls.material.clippingPlanes=[new THREE.Plane(new THREE.Vector3(0,0,-1),0)];
  check('cutting away the front wall reveals the rear wall',blue(pixels())===1);
  walls.material.clippingPlanes=[];walls.setOpacity(0);
  check('zero-opacity CFD walls leave neither color nor depth',fraction(pixels(),(r,g,b)=>r===0&&g===0&&b===0)===1);
  walls.setOpacity(1);pixels();
  document.querySelector('#status').textContent=`PASS · ${passed} GPU depth checks`;
}catch(error){document.querySelector('#status').textContent='FAIL · '+error.message;console.error(error);}
