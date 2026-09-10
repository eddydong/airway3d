import * as THREE from 'three';
import {cfdAPI} from './cfd-api.js';
import {setSurfaceOpacity} from './surface-material.js';
import {sortTransparentFaces,nestedSurfaceOrder} from './surface-order.js';

// Fixed across recordings and breath phases: enhance contrast without making
// the same physical value change color as the frame's extrema change.
export const WALL_COLOR_BANDS={
  detail:{temperature:[35.5,36.9,36.98,36.995,37],flux:[-25,0,1,30,1000]},
  linear:{temperature:[35,35.5,36,36.5,37],flux:[0,375,750,1125,1500]},
};

export function thermalFramePair(frames,time,period){
  const phase=((time%period)+period)%period;
  let a=0;while(a+1<frames.length-1&&frames[a+1].timeS<=phase)a++;
  const b=Math.min(a+1,frames.length-1),span=frames[b].timeS-frames[a].timeS;
  return {a,b,weight:span>0?(phase-frames[a].timeS)/span:0,phase};
}
export function matchingThermal(t,r){
  return t?.schemaVersion===1&&t.status==='complete'&&t.flowRequestId===r.requestId&&t.geometryHash===r.geometryHash&&
    ['finiteAndBounded','linearResidual','energyLedger','periodic'].every(k=>t.gates?.[k]===true)&&
    t.periodS===r.solver.periodS&&r.solver.durationS>=2*t.periodS&&Number.isInteger(t.wall?.faces)&&t.wall.faces>0&&t.wall.verticesPerFace===4&&
    Array.isArray(t.frames)&&t.frames.length>=3&&t.frames[0].timeS===0&&Math.abs(t.frames.at(-1).timeS-t.periodS)<1e-6&&
    t.frames.every((f,i)=>Number.isFinite(f.timeS)&&(!i||f.timeS>t.frames[i-1].timeS));
}
const VERT=`
#include <clipping_planes_pars_vertex>
attribute float faceId;
uniform sampler2D fields;
uniform vec2 textureSize;
uniform float faceCount, frameA, frameB, frameMix;
varying vec2 thermal;
vec2 sampleField(float frame){
  float k=frame*faceCount+faceId;
  return texture2D(fields,vec2(mod(k,textureSize.x)+.5,floor(k/textureSize.x)+.5)/textureSize).rg;
}
void main(){
  thermal=mix(sampleField(frameA),sampleField(frameB),frameMix);
  vec4 mvPosition=modelViewMatrix*vec4(position,1.);
  gl_Position=projectionMatrix*mvPosition;
  #include <clipping_planes_vertex>
}`;
const FRAG=`
#include <clipping_planes_pars_fragment>
varying vec2 thermal;
uniform float displayFlux, bands[5], opacity;
float colorPosition(float value){
  if(value<bands[1])return .25*clamp((value-bands[0])/(bands[1]-bands[0]),0.,1.);
  if(value<bands[2])return .25+.25*(value-bands[1])/(bands[2]-bands[1]);
  if(value<bands[3])return .5+.25*(value-bands[2])/(bands[3]-bands[2]);
  return .75+.25*clamp((value-bands[3])/(bands[4]-bands[3]),0.,1.);
}
vec3 spectrum(float t){
  if(t<.25)return mix(vec3(.03,.18,.95),vec3(0.,.85,.95),t*4.);
  if(t<.5)return mix(vec3(0.,.85,.95),vec3(.05,.78,.25),(t-.25)*4.);
  if(t<.75)return mix(vec3(.05,.78,.25),vec3(1.,.86,.05),(t-.5)*4.);
  return mix(vec3(1.,.86,.05),vec3(.95,.05,.03),(t-.75)*4.);
}
void main(){
  #include <clipping_planes_fragment>
  float value=mix(thermal.x,thermal.y,displayFlux);
  gl_FragColor=vec4(spectrum(colorPosition(value)),opacity);
}`;

export class CFDThermalWalls {
  static async load(url,result,isCurrent=()=>true){
    const meta=await cfdAPI(url);if(!matchingThermal(meta,result))throw Error('Thermal identity, timing or numerical checks do not match airflow');
    const base=url.slice(0,url.lastIndexOf('/')+1);
    const binary=path=>cfdAPI(base+path,undefined,{binary:true,timeoutMs:30000});
    const vertices=new Float32Array(await binary(meta.wall.geometry));
    if(vertices.length!==meta.wall.faces*12||vertices.some(v=>!Number.isFinite(v)))throw Error('Invalid thermal wall geometry');
    // Load a complete compact wall cycle before displaying it. There is no
    // thermal calculation or frame fetch on a playback tick.
    const buffers=[];
    for(let i=0;i<meta.frames.length;i+=4){
      if(!isCurrent())throw Error('Thermal selection changed');
      buffers.push(...await Promise.all(meta.frames.slice(i,i+4).map(f=>binary(f.file))));
    }
    if(!isCurrent())throw Error('Thermal selection changed');
    return new CFDThermalWalls(meta,vertices,buffers);
  }
  constructor(meta,vertices,buffers){
    this.meta=meta;const count=meta.wall.faces;
    const width=2048,height=Math.ceil(count*buffers.length/width),packed=new Float32Array(width*height*2);
    buffers.forEach((buffer,i)=>{const data=new Float32Array(buffer);
      if(data.length!==count*2||data.some(v=>!Number.isFinite(v)))throw Error('Invalid wall temperature/flux frame');
      packed.set(data,i*count*2);
    });
    this.data=packed;this.texture=new THREE.DataTexture(packed,width,height,THREE.RGFormat,THREE.FloatType);
    this.texture.minFilter=THREE.NearestFilter;this.texture.magFilter=THREE.NearestFilter;this.texture.needsUpdate=true;
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(vertices,3));
    const ids=new Float32Array(count*4),indices=new Uint32Array(count*6);
    for(let i=0;i<count;i++){ids.fill(i,i*4,i*4+4);indices.set([i*4,i*4+1,i*4+2,i*4,i*4+2,i*4+3],i*6);}
    geometry.setAttribute('faceId',new THREE.BufferAttribute(ids,1));geometry.setIndex(new THREE.BufferAttribute(indices,1));
    this.material=new THREE.ShaderMaterial({vertexShader:VERT,fragmentShader:FRAG,side:THREE.DoubleSide,clipping:true,
      transparent:true,depthWrite:false,toneMapped:false,uniforms:{fields:{value:this.texture},textureSize:{value:new THREE.Vector2(width,height)},
        faceCount:{value:count},frameA:{value:0},frameB:{value:1},frameMix:{value:0},displayFlux:{value:0},bands:{value:WALL_COLOR_BANDS.detail.temperature},opacity:{value:.95}}});
    this.setOpacity(1);
    // The faces are exported in voxel order. Three sorts whole meshes, not
    // faces within one mesh; sort the quads only when transparency needs it.
    this.material.forceSinglePass=true;
    this.mesh=new THREE.Mesh(geometry,this.material);this.mesh.frustumCulled=false;
    this.mesh.renderOrder=nestedSurfaceOrder('airway')+1;
    this.mesh.onBeforeRender=(_renderer,_scene,camera)=>sortTransparentFaces(this.mesh,camera,4);
    this.setMode('temperature');this.step(0);
  }
  setOpacity(opacity){setSurfaceOpacity(this.material,opacity);this.material.uniforms.opacity.value=opacity;}
  setMode(mode,scale='detail'){
    this.mode=mode;const u=this.material.uniforms;
    u.displayFlux.value=mode==='flux'?1:0;
    u.bands.value=WALL_COLOR_BANDS[scale][mode==='flux'?'flux':'temperature'];
  }
  step(time){
    this.pair=thermalFramePair(this.meta.frames,time,this.meta.periodS);
    const {a,b,weight}=this.pair,u=this.material.uniforms;u.frameA.value=a;u.frameB.value=b;u.frameMix.value=weight;
    const mix=key=>this.meta.frames[a][key]*(1-weight)+this.meta.frames[b][key]*weight;
    this.reading={minC:mix('minC'),maxC:mix('maxC'),meanC:mix('meanC'),heatLossW:mix('heatLossW')};
  }
  sampleFace(face){
    const {a,b,weight}=this.pair,n=this.meta.wall.faces;
    return {temperatureC:this.data[(a*n+face)*2]*(1-weight)+this.data[(b*n+face)*2]*weight,
      heatFluxWm2:this.data[(a*n+face)*2+1]*(1-weight)+this.data[(b*n+face)*2+1]*weight};
  }
  sampleHit(hit){
    const g=this.mesh.geometry,vertex=g.index.getX(hit.faceIndex*3);
    return this.sampleFace(g.getAttribute('faceId').getX(vertex));
  }
  dispose(){this.mesh.parent?.remove(this.mesh);this.mesh.geometry.dispose();this.material.dispose();this.texture.dispose();}
}
