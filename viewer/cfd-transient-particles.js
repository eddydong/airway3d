import { CFDParticles } from './cfd-particles.js';

// Retain one bounded breathing cycle when it fits; larger recordings stream.
// Time stops when a computed frame is unavailable; no estimated fallback.
export class CFDTransientParticles extends CFDParticles {
  constructor(scene,result,occupancy,firstFrame,loadFrame) {
    super(scene,result,firstFrame,occupancy);
    this.frames=result.field.frames;
    if(!Array.isArray(this.frames)||this.frames.length<3||this.frames[0].timeS!==0||
      this.frames.some((f,i)=>!Number.isFinite(f.timeS)||(i&&f.timeS<=this.frames[i-1].timeS)))throw Error('Invalid CFD frame times');
    this.loadFrame=loadFrame;this.cache=new Map([[0,this.vel]]);this.pending=new Map();
    this.time=0;this.frame=0;this.buffering=true;this.ended=false;this.disposed=false;
    // Loop the final cycle, excluding startup from rest. This is playback;
    // repeating a finite LES trajectory does not establish periodic convergence.
    // Shorter recordings stop at their last frame.
    const period=result.solver.periodS,last=this.frames.at(-1).timeS;
    this.loopStart=Number.isFinite(period)&&period>0&&last>=2*period-1e-6?
      this.frames.findIndex(f=>Math.abs(f.timeS-(last-period))<1e-6):-1;
    this.ensure(1);this.ensure(2);this.updateMetrics();
  }
  get loops(){return this.loopStart>=0;}
  async prepareLoop(){
    if(!this.loops)return;
    this.loadingCycle=true;
    const bytes=(this.frames.length-this.loopStart)*this.fluid.length*12;
    this.retainCycle=bytes<=128*1024*1024;
    const end=this.retainCycle?this.frames.length:Math.min(this.loopStart+4,this.frames.length);
    try{
      for(let i=this.loopStart;i<end;i+=4){
        if(this.disposed)return;
        const jobs=[];for(let j=i;j<Math.min(i+4,end);j++){this.ensure(j);jobs.push(this.pending.get(j));}
        await Promise.all(jobs);if(this.error)throw Error(this.error);
      }
      if(this.disposed)return;
      this.time=this.frames[this.loopStart].timeS;this.frame=this.loopStart;
      this.vel=this.cache.get(this.frame);this.velNext=this.cache.get(this.frame+1);
      this.buffering=false;this.updateMetrics();
      for(const key of this.cache.keys())if(key<this.loopStart)this.cache.delete(key);
    }finally{this.loadingCycle=false;}
  }
  ensure(index){
    if(index>=this.frames.length||this.cache.has(index)||this.pending.has(index)||this.disposed)return;
    const task=this.loadFrame(this.frames[index].velocity).then(buffer=>{
      if(this.disposed)return;
      const values=new Float32Array(buffer);
      if(values.length!==this.fluid.length*3||values.some(x=>!Number.isFinite(x)))throw Error('Invalid velocity frame');
      this.cache.set(index,values);this.pending.delete(index);
    }).catch(e=>{this.pending.delete(index);this.error=e.message;this.buffering=true;});
    this.pending.set(index,task);
  }
  frameWeight(offset=0){
    const a=this.frames[this.frame].timeS,b=this.frames[Math.min(this.frame+1,this.frames.length-1)].timeS;
    return b>a?Math.max(0,Math.min(1,(this.time+offset-a)/(b-a))):0;
  }
  updateMetrics(){
    const a=this.frames[this.frame],b=this.frames[Math.min(this.frame+1,this.frames.length-1)],w=this.frameWeight();
    const mix=(x,y)=>x+(y-x)*w;
    this.currentMetrics={timeS:this.time,pressureDropPa:mix(a.pressureDropPa,b.pressureDropPa),
      flowMlS:Object.fromEntries(['L','R','outlet'].map(k=>[k,mix(a.flowMlS[k],b.flowMlS[k])]))};
    this.qNow=this.currentMetrics.flowMlS.outlet;
    this.meta.split={L:Math.abs(this.qNow)>1e-8?this.currentMetrics.flowMlS.L/this.qNow:0,
                     R:Math.abs(this.qNow)>1e-8?this.currentMetrics.flowMlS.R/this.qNow:0};
  }
  step(dt){
    if((!this.enabled&&!this.playWithoutParticles)||this.loadingCycle||this.playbackPaused||this.ended||this.error)return;
    let remaining=Math.max(0,dt*this.timeScale);
    while(remaining>1e-10&&!this.ended){
    while(this.frame+1<this.frames.length-1&&this.time>=this.frames[this.frame+1].timeS-1e-10)this.frame++;
    const keep=[this.frame,this.frame+1,this.frame+2];
    // Near the end of a looping recording, prefetch the loop start so the wrap
    // does not buffer.
    if(this.loops&&this.frame+2>=this.frames.length-1)keep.push(this.loopStart,this.loopStart+1);
    for(const index of keep)this.ensure(index);
    this.vel=this.cache.get(this.frame);this.velNext=this.cache.get(this.frame+1);
    this.buffering=!this.vel||!this.velNext;if(this.buffering)return;
    if(!this.retainCycle)for(const key of this.cache.keys())if(!keep.includes(key))this.cache.delete(key);
    const physical=Math.min(remaining,this.frames[this.frame+1].timeS-this.time);
    if(this.enabled)this.advance(physical);this.time+=physical;this.updateMetrics();
    remaining-=physical;
    if(this.time>=this.frames.at(-1).timeS-1e-10){
      if(this.loops){this.time=this.frames[this.loopStart].timeS;this.frame=this.loopStart;this.updateMetrics();}
      else this.ended=true;
    }
    }
    // Keep sampling references in sync even when a tick ends exactly at wrap.
    this.vel=this.cache.get(this.frame)||this.vel;this.velNext=this.cache.get(this.frame+1)||this.velNext;
  }
  phaseInfo(){
    const prefix=this.result.status==='unconverged'?'Unaccepted 3-D field':'Recorded 3-D CFD';
    const state=this.error?'frame error: '+this.error:this.buffering?'buffering':this.ended?'recording ended — Replay to restart':
      Math.abs(this.qNow)<1e-5?'flow reversal':this.qNow>0?'inspiration':'expiration';
    return {label:`${prefix} · ${state} · ${this.time.toFixed(2)} s`,q:this.qNow,flow:this.currentMetrics.flowMlS,frac:(this.time%this.result.solver.periodS)/this.result.solver.periodS};
  }
  replay(){
    this.frame=this.loops?this.loopStart:0;this.time=this.frames[this.frame].timeS;
    this.ended=false;this.error=null;this.buffering=true;
    this.ensure(this.frame);this.ensure(this.frame+1);for(let i=0;i<this.n;i++)this.seed(i);this.updateMetrics();
  }
  dispose(){this.disposed=true;this.cache.clear();super.dispose();}
}
