import test from 'node:test';
import assert from 'node:assert/strict';
import {register} from 'node:module';
register('./three-loader.mjs',import.meta.url);
const THREE=await import('three');
const {sortTransparentFaces,nestedSurfaceOrder}=await import('../viewer/surface-order.js');
const {setSurfaceOpacity}=await import('../viewer/surface-material.js');

test('face order follows view direction, caches fixed views and refreshes edited geometry',()=>{
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute([-1,-1,1,1,-1,1,0,1,1,-1,-1,-1,1,-1,-1,0,1,-1],3));
  geometry.setIndex([0,1,2,3,4,5]);
  const mesh=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial());setSurfaceOpacity(mesh.material,.5);
  const camera=new THREE.PerspectiveCamera();camera.position.z=3;camera.lookAt(0,0,0);camera.updateMatrixWorld();mesh.updateMatrixWorld();
  sortTransparentFaces(mesh,camera);assert.deepEqual([...geometry.index.array],[3,4,5,0,1,2]);
  const version=geometry.index.version;
  for(let i=0;i<30;i++)sortTransparentFaces(mesh,camera);
  assert.equal(geometry.index.version,version);
  camera.position.z=5;camera.updateMatrixWorld();sortTransparentFaces(mesh,camera);
  assert.equal(geometry.index.version,version);
  camera.position.z=-3;camera.lookAt(0,0,0);camera.updateMatrixWorld();sortTransparentFaces(mesh,camera);
  assert.deepEqual([...geometry.index.array],[0,1,2,3,4,5]);
  for(let i=0;i<3;i++)geometry.attributes.position.setZ(i,-2);
  geometry.attributes.position.needsUpdate=true;sortTransparentFaces(mesh,camera);
  assert.deepEqual([...geometry.index.array],[3,4,5,0,1,2]);
  geometry.dispose();mesh.material.dispose();
});
test('nested shells draw inside to outside without a per-triangle sort',()=>{
  assert.ok(nestedSurfaceOrder('airway')<nestedSurfaceOrder('bone'));
  assert.ok(nestedSurfaceOrder('bone')<nestedSurfaceOrder('skin'));
});
test('counting sort keeps a stack of faces back to front',()=>{
  const geometry=new THREE.BufferGeometry();
  const positions=[],index=[];
  for(let i=0;i<8;i++){
    const z=1-i/3;
    positions.push(-1,-1,z,1,-1,z,0,1,z);
    index.push(i*3,i*3+1,i*3+2);
  }
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  geometry.setIndex(index);
  const mesh=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial());setSurfaceOpacity(mesh.material,.5);
  const camera=new THREE.PerspectiveCamera();camera.position.z=6;camera.lookAt(0,0,0);camera.updateMatrixWorld();mesh.updateMatrixWorld();
  sortTransparentFaces(mesh,camera);
  assert.deepEqual([...geometry.index.array],[21,22,23,18,19,20,15,16,17,12,13,14,9,10,11,6,7,8,3,4,5,0,1,2]);
  camera.position.z=-6;camera.lookAt(0,0,0);camera.updateMatrixWorld();
  sortTransparentFaces(mesh,camera);
  assert.deepEqual([...geometry.index.array],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]);
  geometry.dispose();mesh.material.dispose();
});
