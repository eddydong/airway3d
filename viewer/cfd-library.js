import {cfdAPI} from './cfd-api.js';
import {congestion} from './scenario-model.js';
import {MISSING_CFD} from './airflow-labels.js';
const defaults={position:'supine',elapsed:10,tau:5,responseL:.6,responseR:.6,cycle:0,gravity:.15,relief:0};
export function normalizedRequest(body){
  const s=body.settings||{},settings={};
  for(const [key,value] of Object.entries(defaults))settings[key]=s[key]??value;
  settings.operations=Object.fromEntries(['L','R'].map(side=>[side,Object.fromEntries(['head','body','valve'].map(k=>[k,s.operations?.[side]?.[k]??0]))]));
  const lattice=body.backend==='gpu-lbm',transient=lattice||body.mode==='transient';
  return {settings,...(lattice?{backend:'gpu-lbm',pressurePa:body.pressurePa??30,refinement:body.refinement??1}:{qMlS:body.qMlS??250}),spacingMm:body.spacingMm??.7,
    direction:transient?'cycle':body.direction??'inspiration',...(transient?{mode:'transient',periodS:body.periodS??4,cycles:lattice?body.cycles??2:2}:{})};
}
// Wall motion depends on position, operations, and the two effective swellings.
// Face-up response/time/gravity knobs that do not move the wall share a recording.
export function geometrySignature(settings){
  const swellL=congestion(settings,'L'),swellR=congestion(settings,'R');
  return {position:settings.position,operations:settings.operations,
    relief:(swellL>1e-8||swellR>1e-8)?settings.relief:0,
    swellL:Math.round(swellL*1e6)/1e6,swellR:Math.round(swellR*1e6)/1e6};
}
export function libraryKey(request){
  const n=normalizedRequest(request),{settings,...solver}=n;
  const canonical=value=>Array.isArray(value)?value.map(canonical):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(k=>[k,canonical(value[k])])):value;
  return JSON.stringify(canonical({...solver,geometry:geometrySignature(settings)}));
}
export class CFDLibrary {
  constructor(){this.promise=null;}
  async entries(){
    if(!this.promise)this.promise=cfdAPI('/data/cfd/library.json').then(c=>{
      if(c.schemaVersion!==1||!Array.isArray(c.entries))throw Error('Unsupported offline CFD catalog');return c.entries;
    }).catch(e=>{this.promise=null;throw e;});
    return this.promise;
  }
  async lookup(request){
    const entries=await this.entries(),key=libraryKey(request);
    return entries.find(e=>libraryKey(e.request)===key)||{state:'missing',request,message:MISSING_CFD};
  }
}
