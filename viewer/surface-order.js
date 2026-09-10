import * as THREE from 'three';

const caches=new WeakMap();
const view=new THREE.Matrix4();

// Nested translucent shells draw inside → outside (airway first, skin last).
// Per-triangle sorting is reserved for double-sided clipped tissues and for
// CFD/thermal walls that store front and back faces in one mesh.
export const NESTED_SURFACES=['skin','fat','muscle','soft','brain','eye','teeth','bone','sinus','airway','airway_L','airway_R','airway_common'];
export function nestedSurfaceOrder(key){
  const i=NESTED_SURFACES.indexOf(key);
  return 10*(NESTED_SURFACES.length-(i<0?0:i));
}

function countingSort(order,depth,scratch,offsets,buckets){
  const n=order.length;
  let min=Infinity,max=-Infinity;
  for(let i=0;i<n;i++){
    const d=depth[i];
    if(d<min)min=d;
    if(d>max)max=d;
  }
  if(!(max>min))return false;
  offsets.fill(0,0,buckets+1);
  const scale=(buckets-1)/(max-min);
  for(let i=0;i<n;i++)offsets[Math.min(buckets-1,((depth[i]-min)*scale)|0)]++;
  let sum=0;
  for(let b=0;b<buckets;b++){
    const count=offsets[b];
    offsets[b]=sum;
    sum+=count;
  }
  for(let face=0;face<n;face++)scratch[offsets[Math.min(buckets-1,((depth[face]-min)*scale)|0)]++]=face;
  let changed=false;
  for(let i=0;i<n;i++){
    if(order[i]!==scratch[i])changed=true;
    order[i]=scratch[i];
  }
  return changed;
}

// Stable back-to-front order inside a transparent mesh. Cache the centers and
// skip sorting when only the frame/time or camera distance changes. Reordering
// indices preserves positions, normals, thermal face IDs and all other fields.
export function sortTransparentFaces(mesh,camera,verticesPerFace=3){
  if(!mesh.material.transparent||mesh.material.opacity<=0)return;
  const geometry=mesh.geometry,index=geometry.index,position=geometry.getAttribute('position');
  if(!index||!position)return;
  const stride=verticesPerFace===4?6:3;
  let cache=caches.get(mesh);
  if(!cache||cache.geometry!==geometry||cache.index!==index||cache.indexVersion!==index.version||cache.positionVersion!==position.version){
    const count=Math.floor(index.count/stride),base=index.array.slice(),centers=new Float64Array(count*3);
    for(let face=0;face<count;face++){
      for(let j=0;j<stride;j++){
        const vertex=base[face*stride+j];
        centers[face*3]+=position.getX(vertex)/stride;
        centers[face*3+1]+=position.getY(vertex)/stride;
        centers[face*3+2]+=position.getZ(vertex)/stride;
      }
    }
    const buckets=Math.min(65536,Math.max(256,count));
    cache={geometry,index,indexVersion:index.version,positionVersion:position.version,base,centers,
      order:Uint32Array.from({length:count},(_,i)=>i),depth:new Float64Array(count),
      scratch:new Uint32Array(count),offsets:new Uint32Array(buckets+1),buckets,direction:null};
    caches.set(mesh,cache);
  }
  const e=view.multiplyMatrices(camera.matrixWorldInverse,mesh.matrixWorld).elements;
  if(cache.direction&&[e[2],e[6],e[10]].every((v,i)=>Math.abs(v-cache.direction[i])<1e-10))return;
  cache.direction=[e[2],e[6],e[10]];
  const {centers,depth,order,base}=cache;
  for(let i=0;i<order.length;i++)depth[i]=centers[i*3]*e[2]+centers[i*3+1]*e[6]+centers[i*3+2]*e[10];
  if(!countingSort(order,depth,cache.scratch,cache.offsets,cache.buckets))return;
  for(let i=0;i<order.length;i++){
    const source=order[i]*stride;
    for(let j=0;j<stride;j++)index.array[i*stride+j]=base[source+j];
  }
  index.needsUpdate=true;cache.indexVersion=index.version;
}
