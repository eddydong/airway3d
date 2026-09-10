import * as THREE from 'three';
import { PARTICLE_STYLES, PVERT, PFRAG, LVERT, LFRAG } from './flow.js';

// Tracers in the immutable native CFD cell field. No profile warping, branch
// gain, velocity clipping, or reversal of an inspiration field for expiration.
//
// Inside the airway every particle follows the recorded field. When the result
// lists its openings (result.openings, GPU lattice recordings), air is also
// exchanged with the room the way the estimated engine draws it: room air near
// a nostril is pulled in while that nostril inhales, and air leaving a nostril
// continues outside as an exhaled puff that slows, spreads and fades. Outside
// the CFD domain there is no computed velocity; that puff is a kinematic
// sketch driven only by the exit velocity, and is labelled as such.
const HIST = 8;
const IN = 0, ROOM = 1, PLUME = 2, DEAD = 3;
const PLUME_LEN = 28, PLUME_TIME = 1.2, ROOM_SHELL = 9, FILL = 0.55;

export class CFDParticles {
  constructor(scene,result,velocity,occupancy) {
    this.scene=scene;this.result=result;this.vel=new Float32Array(velocity);this.occ=new Uint8Array(occupancy);
    this.dims=result.field.dims;this.origin=result.field.boxMin;this.h=result.field.spacingMm;
    this.sparse=result.field.schemaVersion===3;
    if(![2,3].includes(result.field.schemaVersion)||result.field.axisOrder!=='XYZ'||result.field.storageOrder!==(this.sparse?'fluid-C':'C'))throw Error('Unsupported CFD coordinate format; rebuild the result');
    if(this.dims.length!==3||!this.dims.every(n=>Number.isInteger(n)&&n>0)||this.origin.length!==3||!this.origin.every(Number.isFinite)||!(this.h>0))throw Error('Invalid CFD grid');
    const cells=this.dims.reduce((a,b)=>a*b,1);
    if(this.occ.length!==cells||(!this.sparse&&this.vel.length!==cells*3))throw Error('CFD field size does not match its grid');
    this.fluid=[];this.occ.forEach((v,i)=>{if(v)this.fluid.push(i);});
    if(!this.fluid.length)throw Error('CFD field has no fluid cells');
    if(this.sparse){
      if(this.vel.length!==this.fluid.length*3||result.field.fluidCells!==this.fluid.length)throw Error('Sparse CFD field size mismatch');
      this.cellIndex=new Int32Array(cells).fill(-1);this.fluid.forEach((k,i)=>this.cellIndex[k]=i);
    }
    this.q=this.qRef=result.solver.qMlS;this.qNow=result.solver.direction==='cycle'?0:result.solver.direction==='expiration'?-this.q:this.q;
    this.meta={split:{L:result.flowMlS.L/this.q,R:result.flowMlS.R/this.q},method:'viscous CFD',geometryHash:result.geometryHash};
    this.enabled=false;this.timeScale=.06;this.maxSpeed=result.p99SpeedMS;this.thermalMax=0;
    this.styleName='glow';this.colorMode='speed';this.sizeMult=1;this.trail=.85;
    this.orderInside=45;this.orderOutside=500;
    this.volMm3=this.fluid.length*this.h**3;
    // Openings with their open faces (mm, world), outward normals and sizes.
    this.openings=(result.openings||[]).filter(o=>Array.isArray(o.facesMm)&&o.facesMm.length&&o.role).map(o=>({
      c:new THREE.Vector3(...o.centreMm),out:new THREE.Vector3(...o.normal).normalize(),r:o.radiusMm||3,faces:o.facesMm,
      side:o.side==='L'?0:o.side==='R'?1:2,role:o.role}));
    this.nostrils=this.openings.filter(o=>o.role==='nostril');this.outlets=this.openings.filter(o=>o.role==='outlet');
    this.exchange=this.nostrils.length>0&&this.outlets.length>0;
    this.midX=this.nostrils.length?this.nostrils.reduce((s,n)=>s+n.c.x,0)/this.nostrils.length:0;
    const left=this.nostrils.find(n=>n.side===0);this.leftIsPosX=left?left.c.x>this.midX:true;
    this.choanaZ=5;
    this.spawnAcc=0;
    this.group=new THREE.Group();scene.add(this.group);
    this.setCount(5000);
    this.setStyle(this.styleName);
  }

  // ---------------------------------------------------------------- appearance
  setStyle(name) {
    const s=PARTICLE_STYLES[name]||PARTICLE_STYLES.glow;
    this.styleName=PARTICLE_STYLES[name]?name:'glow';
    const pu=this.pmat.uniforms,lu=this.lmat.uniforms;
    pu.uRamp.value=s.ramp;lu.uRamp.value=s.ramp;
    pu.uCore.value=s.core;pu.uAlpha.value=s.alpha;pu.uGauss.value=s.gauss;
    pu.uNoise.value=s.noise||0;pu.uGrow.value=s.grow||0;
    pu.uPlumeAlpha.value=s.plumeAlpha??1;pu.uRoomAlpha.value=s.roomAlpha??1;
    pu.sizePx.value=s.size*this.sizeMult;
    lu.uTrailAlpha.value=s.trail;
    const blending=s.additive?THREE.AdditiveBlending:THREE.NormalBlending;
    for(const m of [this.pmat,this.pmatOut,this.lmat]){m.blending=blending;m.needsUpdate=true;}
  }
  setRenderOrder(inside,outside) {
    this.orderInside=inside;this.orderOutside=outside;
    if(!this.points)return;
    this.points.renderOrder=inside;this.lines.renderOrder=inside-1;this.pointsOut.renderOrder=outside;
  }
  setSize(mult){this.sizeMult=mult;this.pmat.uniforms.sizePx.value=(PARTICLE_STYLES[this.styleName]||PARTICLE_STYLES.glow).size*mult;}
  // Thermal colouring is a reduced-order estimate, not part of the CFD result;
  // the control is disabled in CFD mode and falls back to speed here.
  setColorMode(mode) {
    this.colorMode=mode==='side'?'side':'speed';
    const v=this.colorMode==='side'?1:0;
    this.pmat.uniforms.uColorMode.value=v;this.lmat.uniforms.uColorMode.value=v;
  }
  setSideColors(left,right,common) {
    for(const u of [this.pmat.uniforms,this.lmat.uniforms]){
      if(left)u.uColL.value.set(left);if(right)u.uColR.value.set(right);if(common)u.uColC.value.set(common);
    }
  }
  setTrail(t){this.trail=t;}
  setClipPlanes(planes){this.pmat.clippingPlanes=planes;this.pmatOut.clippingPlanes=planes;this.lmat.clippingPlanes=planes;}
  setEnabled(on){this.enabled=on;this.group.visible=on;}
  // A flow-driven field is only valid at its solved flow rate; a pressure-driven
  // recording has no prescribed rate, so the slider (hidden for it) cannot
  // invalidate it.
  setFlowRate(q){if(Number.isFinite(this.qRef)&&q!==this.qRef)this.setEnabled(false);}
  setBreathing(){} setPeriod(){} clearScenario(){}
  phaseInfo(){return {label:`${this.result.status==='unconverged'?'Unaccepted 3-D field':'Recorded 3-D CFD · steady'} ${this.qNow>0?'inspiration':'expiration'}`,q:this.qNow,frac:0};}
  // 0 = left nostril's air, 1 = right, 2 = pharynx / unknown
  sideAt(x,z){if(z<this.choanaZ)return 2;return (x>this.midX)===this.leftIsPosX?0:1;}

  setCount(count) {
    const n=this.n=Math.max(100,Math.min(20000,count));
    if(this.points){this.group.remove(this.points,this.pointsOut,this.lines);this.points.geometry.dispose();this.lines.geometry.dispose();}
    this.pos=new Float32Array(n*3);this.spd=new Float32Array(n);this.life=new Float32Array(n);this.age=new Float32Array(n);
    this.thermal=new Float32Array(n);this.side=new Float32Array(n);this.puff=new Float32Array(n);this.seedv=new Float32Array(n);
    this.mode=new Uint8Array(n).fill(DEAD);this.nos=new Uint8Array(n);this.aux=new Float32Array(n*4);
    this.stuck=new Uint16Array(n);this.slow=new Uint16Array(n);
    this.hist=new Float32Array(n*HIST*3);this.histHead=0;this.pool=new Int32Array(n);this.poolN=0;
    for(let i=0;i<n;i++)this.seedv[i]=Math.random();
    const colorUniforms=()=>({maxSpeed:{value:this.maxSpeed},uRamp:{value:0},uColorMode:{value:this.colorMode==='side'?1:0},uThermalMax:{value:1},
      uColL:{value:new THREE.Color('#ff6a3d')},uColR:{value:new THREE.Color('#59b6ff')},uColC:{value:new THREE.Color('#c9d1d9')}});
    const pg=new THREE.BufferGeometry();
    pg.setAttribute('position',new THREE.BufferAttribute(this.pos,3).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('speed',new THREE.BufferAttribute(this.spd,1).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('life',new THREE.BufferAttribute(this.life,1).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('side',new THREE.BufferAttribute(this.side,1).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('thermal',new THREE.BufferAttribute(this.thermal,1).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('puff',new THREE.BufferAttribute(this.puff,1).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('seed',new THREE.BufferAttribute(this.seedv,1));
    this.pmat=this.pmat||new THREE.ShaderMaterial({vertexShader:PVERT,fragmentShader:PFRAG,transparent:true,depthWrite:false,blending:THREE.AdditiveBlending,
      uniforms:{...colorUniforms(),sizePx:{value:2.2},uCore:{value:1},uAlpha:{value:.75},uGauss:{value:0},uNoise:{value:0},uGrow:{value:0},
        uPlumeAlpha:{value:1},uRoomAlpha:{value:1},uPass:{value:0}},clipping:true});
    // Air outside the head is drawn again after the tissues; shares every
    // uniform with pmat except the pass selector.
    this.pmatOut=this.pmatOut||new THREE.ShaderMaterial({vertexShader:PVERT,fragmentShader:PFRAG,transparent:true,depthWrite:false,blending:this.pmat.blending,
      uniforms:{...this.pmat.uniforms,uPass:{value:1}},clipping:true});
    this.points=new THREE.Points(pg,this.pmat);this.points.frustumCulled=false;
    this.pointsOut=new THREE.Points(pg,this.pmatOut);this.pointsOut.frustumCulled=false;
    const segs=n*(HIST-1);
    this.lpos=new Float32Array(segs*2*3);this.lspd=new Float32Array(segs*2);this.lthermal=new Float32Array(segs*2);
    this.lalpha=new Float32Array(segs*2);this.lside=new Float32Array(segs*2);
    const lg=new THREE.BufferGeometry();
    lg.setAttribute('position',new THREE.BufferAttribute(this.lpos,3).setUsage(THREE.DynamicDrawUsage));
    lg.setAttribute('speed',new THREE.BufferAttribute(this.lspd,1).setUsage(THREE.DynamicDrawUsage));
    lg.setAttribute('alpha',new THREE.BufferAttribute(this.lalpha,1).setUsage(THREE.DynamicDrawUsage));
    lg.setAttribute('side',new THREE.BufferAttribute(this.lside,1).setUsage(THREE.DynamicDrawUsage));
    lg.setAttribute('thermal',new THREE.BufferAttribute(this.lthermal,1).setUsage(THREE.DynamicDrawUsage));
    this.lmat=this.lmat||new THREE.ShaderMaterial({vertexShader:LVERT,fragmentShader:LFRAG,transparent:true,depthWrite:false,blending:THREE.AdditiveBlending,
      uniforms:{...colorUniforms(),uTrailAlpha:{value:.35}},clipping:true});
    this.lines=new THREE.LineSegments(lg,this.lmat);this.lines.frustumCulled=false;
    this.group.add(this.points,this.pointsOut,this.lines);
    this.setRenderOrder(this.orderInside,this.orderOutside);
    // Without openings the airway stays full; with them it starts half full
    // and the rest waits in the pool to enter with the breath.
    for(let i=0;i<n;i++){
      if(!this.exchange||i<n*.5){this.seed(i);this.age[i]=1;this.life[i]=1;}
      else this.kill(i);
    }
    this.group.visible=this.enabled;
  }

  // ---------------------------------------------------------------- seeding
  place(i,x,y,z,mode) {
    const j=i*3;this.pos[j]=x;this.pos[j+1]=y;this.pos[j+2]=z;
    this.mode[i]=mode;this.age[i]=0;this.life[i]=0;this.stuck[i]=0;this.slow[i]=0;this.spd[i]=0;this.thermal[i]=0;
    this.puff[i]=mode===ROOM?-1:0;
    for(let h=0;h<HIST;h++){const k=(i*HIST+h)*3;this.hist[k]=x;this.hist[k+1]=y;this.hist[k+2]=z;}
  }
  seed(i) {
    const k=this.fluid[Math.floor(Math.random()*this.fluid.length)],d=this.dims;
    // NumPy writes the native [X,Y,Z,component] array in C order, so Z is
    // the fastest index in the binary field.
    const z=k%d[2],y=Math.floor(k/d[2])%d[1],x=Math.floor(k/(d[1]*d[2]));
    const p=[x,y,z].map((v,j)=>(v+.25+Math.random()*.5)*this.h+this.origin[j]);
    this.place(i,p[0],p[1],p[2],IN);this.side[i]=this.sideAt(p[0],p[2]);
  }
  // Inside the cell that owns one open face of an opening.
  seedFace(i,o) {
    const f=o.faces[Math.floor(Math.random()*o.faces.length)],h=this.h,jit=()=>(Math.random()-.5)*.6*h;
    const x=f[0]-o.out.x*.5*h+jit()*(1-Math.abs(o.out.x)),y=f[1]-o.out.y*.5*h+jit()*(1-Math.abs(o.out.y)),z=f[2]-o.out.z*.5*h+jit()*(1-Math.abs(o.out.z));
    if(!this.inside([x,y,z])){this.kill(i);return;}
    this.place(i,x,y,z,IN);this.side[i]=o.role==='nostril'?o.side:this.sideAt(x,z);
  }
  // Room air in a shell outside a nostril, heading for one of its open faces.
  seedRoom(i,k) {
    const n=this.nostrils[k];
    let ux=0,uy=0,uz=0,l=0;
    do{ux=Math.random()*2-1;uy=Math.random()*2-1;uz=Math.random()*2-1;l=ux*ux+uy*uy+uz*uz;}while(l<.05||l>1);
    l=Math.sqrt(l);ux/=l;uy/=l;uz/=l;
    const dot=ux*n.out.x+uy*n.out.y+uz*n.out.z;
    if(dot<.15){const f=2*dot-.3;ux-=f*n.out.x;uy-=f*n.out.y;uz-=f*n.out.z;l=Math.hypot(ux,uy,uz);ux/=l;uy/=l;uz/=l;}
    const r=n.r+1.5+ROOM_SHELL*Math.pow(Math.random(),.7);
    this.place(i,n.c.x+ux*r,n.c.y+uy*r,n.c.z+uz*r,ROOM);
    this.nos[i]=k;this.side[i]=n.side;
    const f=n.faces[Math.floor(Math.random()*n.faces.length)],a=i*4,h=this.h;
    // target: just inside the owner cell of one open face
    this.aux[a]=f[0]-n.out.x*.5*h;this.aux[a+1]=f[1]-n.out.y*.5*h;this.aux[a+2]=f[2]-n.out.z*.5*h;
  }
  kill(i){this.mode[i]=DEAD;this.life[i]=0;this.spd[i]=0;this.thermal[i]=0;this.pool[this.poolN++]=i;}
  nearOpening(x,y,z) {
    let best=-1,bd=Infinity;
    for(let k=0;k<this.openings.length;k++){
      const o=this.openings[k],lim=2.5*o.r+4,d2=(x-o.c.x)**2+(y-o.c.y)**2+(z-o.c.z)**2;
      if(d2<lim*lim&&d2<bd){bd=d2;best=k;}
    }
    return best;
  }
  pick(weights) {
    const total=weights.reduce((a,b)=>a+b,0);let r=Math.random()*total;
    for(let k=0;k<weights.length;k++){r-=weights[k];if(r<=0)return k;}
    return weights.length-1;
  }

  // ---------------------------------------------------------------- field access
  inBox(p) {
    for(let c=0;c<3;c++){const k=(p[c]-this.origin[c])/this.h;if(k<0||k>=this.dims[c])return false;}
    return true;
  }
  inside(p) {
    const d=this.dims,x=Math.floor((p[0]-this.origin[0])/this.h),
      y=Math.floor((p[1]-this.origin[1])/this.h),z=Math.floor((p[2]-this.origin[2])/this.h);
    return x>=0&&x<d[0]&&y>=0&&y<d[1]&&z>=0&&z<d[2]&&this.occ[(x*d[1]+y)*d[2]+z]===1;
  }
  sample(p,out=[0,0,0],offset=0) {
    const d=this.dims,fx=(p[0]-this.origin[0])/this.h-.5,
      fy=(p[1]-this.origin[1])/this.h-.5,fz=(p[2]-this.origin[2])/this.h-.5;
    const x0=Math.floor(fx),y0=Math.floor(fy),z0=Math.floor(fz),tx=fx-x0,ty=fy-y0,tz=fz-z0;
    out[0]=out[1]=out[2]=0;
    for(let dx=0;dx<2;dx++)for(let dy=0;dy<2;dy++)for(let dz=0;dz<2;dz++){
      const x=x0+dx,y=y0+dy,z=z0+dz;
      if(x<0||x>=d[0]||y<0||y>=d[1]||z<0||z>=d[2])continue;
      const w=(dx?tx:1-tx)*(dy?ty:1-ty)*(dz?tz:1-tz),cell=(x*d[1]+y)*d[2]+z;
      const index=this.sparse?this.cellIndex[cell]:cell;if(index<0)continue;
      const n=index*3;
      // Solid-cell zeros are retained in the interpolation weights.
      const a=this.frameWeight?this.frameWeight(offset):0,b=1-a;
      out[0]+=w*(b*this.vel[n]+(a? a*this.velNext[n]:0));
      out[1]+=w*(b*this.vel[n+1]+(a? a*this.velNext[n+1]:0));
      out[2]+=w*(b*this.vel[n+2]+(a? a*this.velNext[n+2]:0));
    }
    return out;
  }
  // Signed flows now (mL/s): + into the airway at a nostril, + out at the throat.
  flowsNow() {
    if(this.currentMetrics)return this.currentMetrics.flowMlS;
    const s=this.meta.split;return {L:this.qNow*(s.L||0),R:this.qNow*(s.R||0),outlet:this.qNow};
  }

  // ---------------------------------------------------------------- dynamics
  step(dt) {
    if(!this.enabled)return;
    this.advance(dt*this.timeScale);
  }
  // physical: seconds of recorded airflow covered by this frame
  advance(physical) {
    const hs=physical,dt=this.timeScale>0?physical/this.timeScale:physical,n=this.n,pos=this.pos;
    const flows=this.flowsNow(),qOut=flows.outlet,qAbs=Math.abs(qOut);
    const ps=PARTICLE_STYLES[this.styleName]||PARTICLE_STYLES.glow;
    const pLen=ps.plumeLen??PLUME_LEN,pTime=ps.plumeTime??PLUME_TIME,pFloor=ps.decayFloor??.5;
    const pSpread=ps.spread??.18,pBuoy=ps.buoy??0,pFade=ps.fadePow??1.5;
    const nostrilFlow=o=>o.side===0?flows.L:flows.R;
    if(this.exchange){
      // inject so the concentration inside stays ~FILL n / V whatever the flow
      this.spawnAcc+=FILL*n/this.volMm3*qAbs*1000*hs;
      let k=Math.floor(this.spawnAcc);this.spawnAcc-=k;
      const inflow=this.nostrils.map(o=>Math.max(nostrilFlow(o),0));
      while(k-->0&&this.poolN>0){
        const i=this.pool[--this.poolN];
        if(qOut>=0){if(inflow.some(w=>w>0))this.seedRoom(i,this.pick(inflow));else this.kill(i);}
        else this.seedFace(i,this.outlets[Math.floor(Math.random()*this.outlets.length)]);
      }
    }
    const p=[0,0,0],v=[0,0,0],mid=[0,0,0],vm=[0,0,0],next=[0,0,0],t=[0,0,0];
    this.histHead=(this.histHead+1)%HIST;
    for(let i=0;i<n;i++){
      const m=this.mode[i];if(m===DEAD)continue;
      const j=i*3;let x=pos[j],y=pos[j+1],z=pos[j+2];
      this.age[i]+=dt;
      if(m===IN){
        p[0]=x;p[1]=y;p[2]=z;
        this.sample(p,v);let speed=Math.hypot(v[0],v[1],v[2]);
        const steps=Math.min(24,Math.max(1,Math.ceil(speed*1000*hs/(this.h*.25)))),sub=hs/steps;
        let alive=true;
        for(let s=0;s<steps&&alive;s++){
          if(s){this.sample(p,v,s*sub);speed=Math.hypot(v[0],v[1],v[2]);}
          for(let c=0;c<3;c++)mid[c]=p[c]+v[c]*sub*500;
          if(this.inside(mid)){this.sample(mid,vm,(s+.5)*sub);for(let c=0;c<3;c++)next[c]=p[c]+vm[c]*sub*1000;}
          else for(let c=0;c<3;c++)next[c]=p[c]+v[c]*sub*1000;
          if(this.inside(next)){p[0]=next[0];p[1]=next[1];p[2]=next[2];this.stuck[i]=0;continue;}
          // stepped out of the airway
          if(!this.exchange){this.seed(i);alive=false;break;}
          const k=this.nearOpening(next[0],next[1],next[2]),o=k>=0?this.openings[k]:null;
          if(o&&o.role==='nostril'&&nostrilFlow(o)<0){                                   // exhaled through a nostril
            this.mode[i]=PLUME;this.nos[i]=this.nostrils.indexOf(o);this.age[i]=0;this.side[i]=o.side;
            const a=i*4;this.aux[a]=0;this.aux[a+1]=v[0];this.aux[a+2]=v[1];this.aux[a+3]=v[2];
            p[0]=next[0];p[1]=next[1];p[2]=next[2];break;
          }
          if(o||!this.inBox(next)){this.kill(i);alive=false;break;}                       // left through the throat
          // slide along the wall: keep the components that stay inside
          t[0]=next[0];t[1]=p[1];t[2]=p[2];const ax=this.inside(t);
          t[0]=p[0];t[1]=next[1];const ay=this.inside(t);
          t[1]=p[1];t[2]=next[2];const az=this.inside(t);
          t[0]=ax?next[0]:p[0];t[1]=ay?next[1]:p[1];t[2]=az?next[2]:p[2];
          if((ax||ay||az)&&this.inside(t)){p[0]=t[0];p[1]=t[1];p[2]=t[2];}
          else if(ax)p[0]=next[0];else if(ay)p[1]=next[1];else if(az)p[2]=next[2];
          else{this.stuck[i]++;break;}
        }
        if(!alive)continue;
        x=p[0];y=p[1];z=p[2];
        if(this.mode[i]===IN){
          // wedged in a corner, or stagnant in a dead-end pocket
          if(speed<.005)this.slow[i]++;else this.slow[i]=0;
          if(this.stuck[i]>30||this.slow[i]>240){if(this.exchange)this.kill(i);else this.seed(i);continue;}
          this.spd[i]=speed;this.life[i]=Math.min(this.age[i]/.3,1);
          if(this.side[i]===2&&z>=this.choanaZ)this.side[i]=this.sideAt(x,z);
        }
      }else if(m===ROOM){
        const nn=this.nostrils[this.nos[i]],a=i*4,qn=nostrilFlow(nn);
        const rx=x-this.aux[a],ry=y-this.aux[a+1],rz=z-this.aux[a+2];
        const r2=rx*rx+ry*ry+rz*rz,r=Math.sqrt(r2);
        // half-space point sink at the target face (source while that nostril exhales)
        const vmag=Math.min(Math.abs(qn)*1000/(2*Math.PI*Math.max(r2,4)),4000);           // mm/s
        const sgn=qn>=0?-1:1;
        const stepMm=sgn<0?Math.min(vmag*hs,Math.max(r-.05,.02)):vmag*hs;
        const f=sgn*stepMm/Math.max(r,1e-3);
        const nx=x+rx*f,ny=y+ry*f,nz=z+rz*f;
        this.spd[i]=vmag/1000;
        t[0]=nx;t[1]=ny;t[2]=nz;
        if(this.inside(t)){this.mode[i]=IN;this.stuck[i]=0;this.puff[i]=0;}
        else if(sgn<0&&r<.1){this.kill(i);continue;}
        x=nx;y=ny;z=nz;
        if(sgn<0)this.life[i]=Math.min(this.age[i]/.4,1);
        else{this.life[i]-=dt*2.5;if(this.life[i]<=0||r>nn.r+ROOM_SHELL+6){this.kill(i);continue;}}
      }else{ // PLUME: outside the domain, kinematic only
        const a=i*4,nn=this.nostrils[this.nos[i]],qn=nostrilFlow(nn);
        let d=this.aux[a],vx=this.aux[a+1],vy=this.aux[a+2],vz=this.aux[a+3];
        const v0=Math.hypot(vx,vy,vz)||1e-3;
        const decay=Math.max(pFloor,1-(1-pFloor)*d/pLen);
        const stepMm=v0*decay*1000*hs;
        const jit=pSpread*Math.sqrt(Math.max(stepMm,1e-6)/10);
        vx+=(Math.random()-.5)*jit*v0;vy+=(Math.random()-.5)*jit*v0;vz+=(Math.random()-.5)*jit*v0;
        const l=Math.hypot(vx,vy,vz)||1e-3;vx*=v0/l;vy*=v0/l;vz*=v0/l;
        this.aux[a+1]=vx;this.aux[a+2]=vy;this.aux[a+3]=vz;
        const prog=Math.min(d/pLen,1);
        let nx=x+vx/v0*stepMm,ny=y+vy/v0*stepMm+pBuoy*1000*hs*prog,nz=z+vz/v0*stepMm;
        if(qn>0){ // the next breath draws the puff back toward the nostril
          const rx=nx-nn.c.x,ry=ny-nn.c.y,rz=nz-nn.c.z,r2=rx*rx+ry*ry+rz*rz,r=Math.sqrt(r2);
          const vmag=Math.min(qn*1000/(2*Math.PI*Math.max(r2,4)),4000),f=-vmag*hs/Math.max(r,1e-3);
          nx+=rx*f;ny+=ry*f;nz+=rz*f;
          t[0]=nx;t[1]=ny;t[2]=nz;
          if(this.inside(t)){this.mode[i]=IN;this.stuck[i]=0;this.puff[i]=0;}
        }
        d+=stepMm;this.aux[a]=d;
        this.spd[i]=v0*decay;
        if(this.mode[i]===PLUME)this.puff[i]=Math.max(prog,1e-3);
        this.life[i]=Math.max(0,1-d/pLen)**pFade*Math.max(0,1-this.age[i]/pTime);
        if(this.life[i]<=0){this.kill(i);continue;}
        x=nx;y=ny;z=nz;
      }
      pos[j]=x;pos[j+1]=y;pos[j+2]=z;
      const hk=(i*HIST+this.histHead)*3;this.hist[hk]=x;this.hist[hk+1]=y;this.hist[hk+2]=z;
    }
    // trail segments
    const lp=this.lpos,ls=this.lspd,la=this.lalpha,lsd=this.lside,lth=this.lthermal,hist=this.hist;
    const nseg=Math.max(1,Math.round((HIST-1)*this.trail));
    let q=0,qa=0;
    for(let i=0;i<n;i++){
      const sp=this.spd[i],lf=this.life[i],sd=this.side[i];
      for(let sgi=0;sgi<HIST-1;sgi++){
        const a=(this.histHead-sgi+HIST)%HIST,b=(this.histHead-sgi-1+HIST)%HIST;
        const ka=(i*HIST+a)*3,kb=(i*HIST+b)*3;
        lp[q++]=hist[ka];lp[q++]=hist[ka+1];lp[q++]=hist[ka+2];
        lp[q++]=hist[kb];lp[q++]=hist[kb+1];lp[q++]=hist[kb+2];
        ls[qa]=sp;ls[qa+1]=sp;lsd[qa]=sd;lsd[qa+1]=sd;lth[qa]=0;lth[qa+1]=0;
        const fade=sgi<nseg&&lf>0?lf*(1-sgi/nseg):0;
        la[qa++]=fade;la[qa++]=fade*.8;
      }
    }
    const pg=this.points.geometry,lg=this.lines.geometry;
    for(const k of ['position','speed','life','side','thermal','puff'])pg.attributes[k].needsUpdate=true;
    for(const k of ['position','speed','alpha','side','thermal'])lg.attributes[k].needsUpdate=true;
    this.pmat.uniforms.maxSpeed.value=this.maxSpeed;this.lmat.uniforms.maxSpeed.value=this.maxSpeed;
  }
  dispose(){this.scene.remove(this.group);this.points.geometry.dispose();this.lines.geometry.dispose();this.pmat.dispose();this.pmatOut.dispose();this.lmat.dispose();}
}
