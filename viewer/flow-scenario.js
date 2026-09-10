// Reduced-order transport on the stored field's reference coordinates.
// This warps existing streamlines; it does not solve a new 3-D velocity field.
import { heatFlux } from './heatflux.js';

export function shellFrame(profile,i) {
  const a=profile.centers[Math.max(0,i-1)],b=profile.centers[Math.min(profile.centers.length-1,i+1)];
  const tangent=b.map((v,k)=>v-a[k]),len=Math.hypot(...tangent)||1;
  return {center:profile.centers[i],tangent:tangent.map(v=>v/len)};
}

export function warpPoint(frame,ratio,x,y,z,out,offset=0) {
  const c=frame.center,t=frame.tangent,dx=x-c[0],dy=y-c[1],dz=z-c[2];
  const along=dx*t[0]+dy*t[1]+dz*t[2],gain=ratio-1;
  out[offset]=x+(dx-along*t[0])*gain;
  out[offset+1]=y+(dy-along*t[1])*gain;
  out[offset+2]=z+(dz-along*t[2])*gain;
}

export class ScenarioTransport {
  constructor(base,meta) {
    this.base=base;this.meta=meta;this.frames=[];this.bySide={L:[],R:[]};
    for(const side of ['L','R']) base.sides[side].profile.centers.forEach((_,i)=>{
      this.bySide[side].push(this.frames.length);
      this.frames.push({...shellFrame(base.sides[side].profile,i),side,i});
    });
    this.ratios=new Float64Array(this.frames.length).fill(1);
    this.target=new Float64Array(this.ratios);this.gains=new Float64Array(this.ratios);
    this.thermalBySide={L:null,R:null,common:null};
    this.lookup=new Int16Array(meta.dims.reduce((a,b)=>a*b,1)).fill(-2);
    this.commonLookup=new Int16Array(meta.dims.reduce((a,b)=>a*b,1)).fill(-2);
    this.commonFrames=(base.sides.common?.profile.centers||[]).map((_,i)=>shellFrame(base.sides.common.profile,i));
    this.voxel=meta.box_size.map((v,i)=>v/meta.dims[i]);
    this.midX=meta.nostrils.reduce((a,n)=>a+n.centre[0],0)/meta.nostrils.length;
    this.leftPositive=meta.nostrils.find(n=>n.name==='L').centre[0]>this.midX;
    this.fractions={L:0,R:0};this.delivered=0;
  }
  update(data,metrics,{immediate=false}={}) {
    const q=metrics.total.requestedQ;
    this.fractions={L:q?metrics.L.Q/q:0,R:q?metrics.R.Q/q:0};
    this.delivered=q?metrics.total.deliveredQ/q:0;
    this.closed={L:metrics.L.closed||metrics.total.blocked,R:metrics.R.closed||metrics.total.blocked};
    this.thermalBySide={
      L:heatFlux(data.sides.L.profile,metrics.L.Q),
      R:heatFlux(data.sides.R.profile,metrics.R.Q),
      common:data.sides.common&&metrics.common?heatFlux(data.sides.common.profile,metrics.common.Q):null,
    };
    this.frames.forEach((f,j)=>{
      const area=this.base.sides[f.side].profile.area_mm2[f.i];
      this.target[j]=Math.sqrt(data.sides[f.side].profile.area_mm2[f.i]/area);
    });
    this.remaining=immediate?0:0.3;
    if(immediate)this.ratios.set(this.target);
    this.refreshGains();
  }
  refreshGains() {
    this.frames.forEach((f,j)=>{
      const fraction=this.meta.split?.[f.side] ?? this.meta.nostrils.find(n=>n.name===f.side).frac;
      const areaRatio=this.ratios[j]**2;
      this.gains[j]=this.closed[f.side]||!areaRatio?0:this.fractions[f.side]/fraction/areaRatio;
    });
  }
  step(dt) {
    if(this.remaining<=0)return;
    this.remaining-=dt;
    const blend=this.remaining<=0?1:1-Math.exp(-dt*18);
    this.ratios.forEach((r,j)=>{this.ratios[j]+=(this.target[j]-r)*blend;});
    this.refreshGains();
  }
  indexAt(x,y,z) {
    if(z<this.base.choana_world_z)return -1;
    const m=this.meta,d=m.dims,v=this.voxel,b=m.box_min;
    const ix=Math.floor((x-b[0])/v[0]),iy=Math.floor((y-b[1])/v[1]),iz=Math.floor((z-b[2])/v[2]);
    if(ix<0||iy<0||iz<0||ix>=d[0]||iy>=d[1]||iz>=d[2])return -1;
    const voxel=ix+d[0]*(iy+d[1]*iz);
    if(this.lookup[voxel]!==-2)return this.lookup[voxel];
    const cx=b[0]+(ix+.5)*v[0],cy=b[1]+(iy+.5)*v[1],cz=b[2]+(iz+.5)*v[2];
    const side=(cx>this.midX)===this.leftPositive?'L':'R';
    let best=Infinity,found=-1;
    for(const j of this.bySide[side]) {
      const c=this.frames[j].center,dist=(cx-c[0])**2+(cy-c[1])**2+(cz-c[2])**2;
      if(dist<best){best=dist;found=j;}
    }
    this.lookup[voxel]=found;return found;
  }
  gainAt(x,y,z) { const j=this.indexAt(x,y,z);return j<0?this.delivered:this.gains[j]; }
  thermalAt(x,y,z) {
    const j=this.indexAt(x,y,z);
    if(j>=0) return this.thermalBySide[this.frames[j].side]?.fluxWm2?.[this.frames[j].i]||0;
    const common=this.thermalBySide.common;
    if(!common||!this.commonFrames.length)return 0;
    const m=this.meta,d=m.dims,v=this.voxel,b=m.box_min;
    const ix=Math.floor((x-b[0])/v[0]),iy=Math.floor((y-b[1])/v[1]),iz=Math.floor((z-b[2])/v[2]);
    if(ix<0||iy<0||iz<0||ix>=d[0]||iy>=d[1]||iz>=d[2])return 0;
    const voxel=ix+d[0]*(iy+d[1]*iz);
    let i=this.commonLookup[voxel];
    if(i===-2){
      const cx=b[0]+(ix+.5)*v[0],cy=b[1]+(iy+.5)*v[1],cz=b[2]+(iz+.5)*v[2];
      let best=Infinity; i=0;
      this.commonFrames.forEach((f,k)=>{const c=f.center,dist=(cx-c[0])**2+(cy-c[1])**2+(cz-c[2])**2;if(dist<best){best=dist;i=k;}});
      this.commonLookup[voxel]=i;
    }
    return common.fluxWm2?.[i]||0;
  }
  map(x,y,z,out,offset=0) {
    const j=this.indexAt(x,y,z);
    if(j<0){out[offset]=x;out[offset+1]=y;out[offset+2]=z;}
    else warpPoint(this.frames[j],this.ratios[j],x,y,z,out,offset);
  }
}
