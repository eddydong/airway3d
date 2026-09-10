import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { CFDParticles } from './cfd-particles.js';
import { CFDTransientParticles } from './cfd-transient-particles.js';
import { CFDComparison } from './cfd-comparison.js';
import { cfdAPI } from './cfd-api.js';
import { CFDLibrary } from './cfd-library.js';
import { CFDThermalWalls, WALL_COLOR_BANDS } from './cfd-thermal.js';
import { setSurfaceOpacity } from './surface-material.js';
import { sortTransparentFaces, nestedSurfaceOrder } from './surface-order.js';
import { TUBE_ESTIMATE, RECORDED_CFD, AIRFLOW_INTRO, TUBE_ESTIMATE_NOTE, RECORDED_CFD_NOTE_TRANSIENT, RECORDED_CFD_NOTE_STEADY, TUBE_GEOMETRY_NOTE, CFD_GEOMETRY_NOTE, MISSING_CFD } from './airflow-labels.js';
const $=s=>document.querySelector(s);
const fmt=(v,n=2)=>Number.isFinite(v)?v.toFixed(n):'—';

export function cfdPanelMarkup(){
  return `<header><h2>Airflow model</h2><p>${AIRFLOW_INTRO}</p></header>
      <div class="seg" id="cfd-mode" role="group" aria-label="Airflow model" style="display:flex;width:100%;margin:12px 0">
        <button type="button" data-cfd="off" style="flex:1">${TUBE_ESTIMATE}</button>
        <button type="button" data-cfd="on" class="on" style="flex:1">${RECORDED_CFD}</button>
      </div>
      <p id="cfd-mode-note" class="model-note">${RECORDED_CFD_NOTE_TRANSIENT}</p>
      <div id="cfd-recording-ui">
      <div class="cfd-options"><label>Solver<select id="cfd-backend"><option value="gpu-lbm">GPU lattice · LES, pressure-driven</option><option value="openfoam">OpenFOAM · CPU, flow-driven</option></select></label>
      <label>Simulation<select id="cfd-temporal"><option value="steady">Steady flow</option><option value="transient">Recorded breathing cycle</option></select></label>
      <label id="cfd-period-row" hidden>Breath period (s)<input id="cfd-period" type="number" min="2" max="10" step="0.5" value="4"></label>
      <label id="cfd-pressure-row" hidden>Throat pressure amplitude (Pa)<input id="cfd-pressure" type="number" min="5" max="200" step="5" value="30"></label>
      <label>Geometry grid<select id="cfd-grid"><option value="0.7">0.7 mm · initial</option><option value="0.5">0.5 mm · finer</option></select></label>
      <label id="cfd-refine-row" hidden>Lattice<select id="cfd-refine"><option value="1">1 cell per voxel</option><option value="2">2 cells per voxel</option></select></label>
      <label id="cfd-direction-row">Flow direction<select id="cfd-direction"><option value="inspiration">Inspiration</option><option value="expiration">Expiration</option></select></label></div>
      <p class="hint" id="cfd-method-note">Each geometry cell is subdivided for the solver. Rigid, no-slip walls; laminar Navier–Stokes. Both nostrils share ambient pressure.</p>
      <p id="cfd-geometry-status" class="hint" role="status"></p>
      <p class="hint">The viewer only reads saved recordings. Changing settings never starts a geometry build or a solver. Knobs that leave the wall unchanged reuse the same recording. Missing scenarios must be prepared offline.</p>
      <div id="cfd-wall-controls"><label>Airway wall display<select id="cfd-wall-mode"><option value="temperature">Temperature · °C</option><option value="flux">Mucosal sensible heat flux · W/m²</option><option value="surface">Plain surface</option></select></label>
      <p id="cfd-thermal-status" class="hint" role="status">Checking recorded 3-D wall temperatures…</p>
      <label id="cfd-wall-scale-row">Color scale<select id="cfd-wall-scale"><option value="detail">Detail bands · enhanced contrast</option><option value="linear">Linear scale</option></select></label>
      <div id="cfd-wall-legend" hidden><div class="units"></div><i></i><div class="ticks"><span></span><span></span><span></span><span></span><span></span></div><p class="hint"></p></div>
      <output id="cfd-wall-reading"></output><p id="cfd-wall-probe" class="hint">Point at a wall to inspect its local temperature and heat flux.</p>
      <label class="cfd-switch"><input id="cfd-wall-play" type="checkbox" checked> Play recorded breath</label>
      <button id="cfd-view-walls" class="small">View airway walls</button></div>
      <button id="cfd-check" class="small">Refresh saved library</button>
      <button id="cfd-compare" class="small">Compare plans × body positions with recorded 3-D CFD</button>
      <button id="cfd-replay" class="small" hidden>Replay recorded breathing</button>
      <p id="cfd-status" role="status">Checking for a matching result…</p><div id="cfd-result"></div>
      <details><summary>Anatomy and breathing assumptions</summary><p class="hint">Only the face-up screenshot anatomy is observed. Intervention and position controls prescribe wall motion; widening is limited to the lateral soft wall and preserves visible bone. Recorded 3-D CFD calculates airflow through that changed domain. It does not predict how your tissue moves.</p>
      <p class="hint">Steady particles follow one solved field. Recorded breathing uses time-dependent Navier–Stokes velocity frames and interpolates between saved times. Each run starts at rest and records two cycles; playback loops the second cycle, and Replay restarts the recording. With thermal results, both start on the final airflow cycle after thermal equilibration. Airway walls remain rigid during each breath. Periodicity and time-step independence are not established. CT images remain the scan reference.</p></details>
      </div>`;
}

export function requestSignature(request){return JSON.stringify(request);}
export const isLattice=result=>result?.solver?.backend==='gpu-lbm';
// Gates that make a field displayable at all (identity, coordinates, mass)
// and the extra gates an accepted reading needs, per backend.
export const FIELD_GATES={openfoam:['mesh','massBalance','wallLeak','requestedFlow','coordinateAlignment','completed'],
  'gpu-lbm':['mesh','massBalance','coordinateAlignment','completed','frames']};
export const RESULT_GATES={openfoam:{steady:['residuals','pressureStable'],transient:['residuals','courant']},'gpu-lbm':{transient:['stability']}};
export function matchingField(result,requestId){
  const backend=isLattice(result)?'gpu-lbm':'openfoam';
  if(backend==='gpu-lbm'&&result.solver?.temporal!=='transient')return false;
  return ['converged','unconverged'].includes(result?.status)&&result.requestId===requestId&&result.geometryHash===result.geometry?.geometryHash
    &&result.field?.axisOrder==='XYZ'&&((result.field.schemaVersion===2&&result.field.storageOrder==='C')||(result.field.schemaVersion===3&&result.field.storageOrder==='fluid-C'&&result.solver?.temporal==='transient'&&result.gates?.frames===true))
    &&FIELD_GATES[backend].every(k=>result.gates?.[k]===true);
}
export function matchingResult(result,requestId){
  if(!matchingField(result,requestId)||result.status!=='converged'||!Object.values(result.gates).every(v=>v===true))return false;
  const backend=isLattice(result)?'gpu-lbm':'openfoam',temporal=result.solver?.temporal==='transient'?'transient':'steady';
  return (RESULT_GATES[backend][temporal]||[]).every(k=>result.gates[k]===true)&&(backend!=='gpu-lbm'||temporal==='transient');
}

export class CFDLab {
  constructor(lab) {
    this.lab=lab;this.ctx=lab.ctx;this.enabled=false;this.generation=0;this.originalFlow=this.ctx.flow;this.hidden=[];this.surfaceVisible=true;this.surfaceOpacity=.1;this.library=new CFDLibrary();
    const panel=document.createElement('section');panel.id='cfd-panel';
    panel.innerHTML=cfdPanelMarkup();
    $('#scenario-lab').prepend(panel);
    $('#cfd-mode').onclick=e=>{
      const b=e.target.closest('[data-cfd]');if(!b)return;
      const on=b.dataset.cfd==='on';if(on===this.enabled)return;this.toggle(on);
    };
    $('#cfd-backend').onchange=()=>{this.modeControls();this.changed();};
    $('#cfd-temporal').onchange=()=>{this.modeControls();this.changed();};
    $('#cfd-period').oninput=()=>this.changed();$('#cfd-pressure').oninput=()=>this.changed();$('#cfd-refine').onchange=()=>this.changed();
    $('#cfd-wall-mode').onchange=()=>this.updateThermalDisplay();
    $('#cfd-wall-scale').onchange=()=>this.updateThermalDisplay();
    $('#cfd-wall-play').onchange=e=>{if(this.particles)this.particles.playbackPaused=!e.target.checked;};
    $('#cfd-view-walls').onclick=()=>{this.ctx.applyPreset('Airway');this.surfaceVisible=true;this.mountSurfaceControl();this.updateThermalDisplay();};
    this.raycaster=new THREE.Raycaster();this.pointer=new THREE.Vector2();this.probeAt=0;
    $('#gl').addEventListener('pointermove',e=>{
      if(!this.thermal?.mesh.visible||performance.now()-this.probeAt<100)return;this.probeAt=performance.now();
      const box=e.target.getBoundingClientRect();this.pointer.set((e.clientX-box.left)/box.width*2-1,-(e.clientY-box.top)/box.height*2+1);
      this.raycaster.setFromCamera(this.pointer,this.ctx.camera);
      const hit=this.raycaster.intersectObject(this.thermal.mesh).find(h=>!(this.thermal.material.clippingPlanes||[]).some(p=>p.distanceToPoint(h.point)<0));
      if(hit){const reading=this.thermal.sampleHit(hit);$('#cfd-wall-probe').textContent=`At cursor: ${reading.temperatureC.toFixed(2)} °C · ${reading.heatFluxWm2.toFixed(1)} W/m² (positive = cooling)`;}
    });
    this.comparison=new CFDComparison(this);
    $('#cfd-compare').onclick=()=>this.comparison.open();
    $('#cfd-replay').onclick=()=>{if(this.thermal){this.particles.time=this.particles.frames[this.particles.loopStart].timeS;this.particles.frame=this.particles.loopStart;this.particles.ended=false;}else this.particles?.replay?.();};
    $('#cfd-grid').onchange=()=>this.changed();$('#cfd-direction').onchange=()=>this.changed();
    $('#cfd-check').onclick=()=>{this.library.promise=null;this.changed();};
    this.mountSurfaceControl();
    this.toggle(true);

  }
  backend(){return $('#cfd-backend').value;}
  request(){
    const lattice=this.backend()==='gpu-lbm',mode=lattice?'transient':$('#cfd-temporal').value;
    if(lattice)return JSON.parse(JSON.stringify({settings:this.lab.settings,backend:'gpu-lbm',pressurePa:+$('#cfd-pressure').value,
      spacingMm:+$('#cfd-grid').value,refinement:+$('#cfd-refine').value,mode,periodS:+$('#cfd-period').value}));
    return JSON.parse(JSON.stringify({settings:this.lab.settings,qMlS:this.ctx.q(),spacingMm:+$('#cfd-grid').value,
      direction:mode==='transient'?'cycle':$('#cfd-direction').value,
      ...(mode==='transient'?{mode,periodS:+$('#cfd-period').value}: {})}));
  }
  modeControls(){
    const lattice=this.backend()==='gpu-lbm';
    if(lattice)$('#cfd-temporal').value='transient';
    $('#cfd-temporal').disabled=lattice;
    const transient=$('#cfd-temporal').value==='transient';
    $('#cfd-period-row').hidden=!transient;$('#cfd-direction-row').hidden=transient;
    $('#cfd-pressure-row').hidden=!lattice;$('#cfd-refine-row').hidden=!lattice;
    $('#cfd-method-note').textContent=lattice?'GPU lattice Boltzmann (FluidX3D, D3Q19 TRT) with Smagorinsky eddy viscosity: a large-eddy simulation on the rigid voxel wall. Recorded throat pressure drives flow against ambient nostrils; flow rates and the left/right split are results, not inputs. Two recorded cycles from rest.'
      :transient?'Time-dependent, laminar Navier–Stokes. Smooth asymmetric resting-breath drive, ambient nostril pressure; two recorded cycles from rest. Rigid, no-slip walls.':'Rigid, no-slip walls; steady laminar Navier–Stokes. Both nostrils share ambient pressure.';
    this.syncModeCopy(transient);
    const flowRow=$('#flow-q').closest('.row');flowRow.hidden=this.enabled&&lattice;
    flowRow.querySelector('label').textContent=this.enabled?(transient?'Recorded peak flow':'Recorded flow rate'):'Peak flow';
  }
  syncModeCopy(transient=$('#cfd-temporal')?.value==='transient'||this.backend?.()==='gpu-lbm'){
    document.querySelectorAll('#cfd-mode [data-cfd]').forEach(b=>{
      const on=(b.dataset.cfd==='on')===this.enabled;
      b.classList.toggle('on',on);b.setAttribute('aria-pressed',String(on));
    });
    $('#cfd-mode-note').textContent=this.enabled?(transient?RECORDED_CFD_NOTE_TRANSIENT:RECORDED_CFD_NOTE_STEADY):TUBE_ESTIMATE_NOTE;
    $('#scenario-model-note').textContent=this.enabled?CFD_GEOMETRY_NOTE:TUBE_GEOMETRY_NOTE;
  }
  api(path,body){return cfdAPI(path,body);}
  binary(path){return cfdAPI(path,undefined,{binary:true,timeoutMs:30000});}
  async model(path){return new GLTFLoader().parseAsync(await this.binary(path),path.slice(0,path.lastIndexOf('/')+1));}
  mountSurfaceControl(){
    const root=$('#tissues');if(!root)return;
    $('#cfd-surface-row')?.remove();
    const row=document.createElement('div');row.id='cfd-surface-row';row.className='tissue';
    row.innerHTML=`<input id="cfd-surface-visible" class="vis" type="checkbox" aria-label="Show CFD airway surface">
      <div class="name"><span>CFD airway · solver surface</span><span class="vol">Display only · recorded airflow and temperature continue</span></div>
      <div class="ctrl"><div><label for="cfd-surface-opacity">opacity</label><input id="cfd-surface-opacity" type="range" min="0" max="1" step="0.01"></div></div>`;
    root.append(row);
    const visible=$('#cfd-surface-visible');visible.checked=this.surfaceVisible;
    visible.onchange=()=>{this.surfaceVisible=visible.checked;this.updateSurfaceVisibility();};
    const opacity=$('#cfd-surface-opacity');opacity.value=this.surfaceOpacity;
    opacity.oninput=()=>{this.thermalOpacityInitialized=true;this.surfaceOpacity=+opacity.value;this.surface?.traverse(o=>{if(o.isMesh)setSurfaceOpacity(o.material,this.surfaceOpacity);});this.thermal?.setOpacity(this.surfaceOpacity);};
    this.updateSurfaceVisibility();
    this.syncReconstructionAirwayRows();
  }
  syncReconstructionAirwayRows(){
    for(const key of ['airway','airway_L','airway_R','airway_common']){
      const row=document.querySelector(`#tissues .tissue[data-k="${key}"]`);
      if(row){row.hidden=this.enabled;row.style.display=this.enabled?'none':'';}
    }
  }
  updateSurfaceVisibility(){
    if(this.surface)this.surface.visible=this.enabled&&this.surfaceVisible&&(!this.thermal||$('#cfd-wall-mode').value==='surface');
    if(this.thermal)this.thermal.mesh.visible=this.enabled&&this.surfaceVisible&&$('#cfd-wall-mode').value!=='surface';
    $('#cfd-surface-row')?.classList.toggle('off',!this.enabled||!this.surfaceVisible);
    const label=$('#cfd-surface-row .name .vol');
    if(label)label.textContent=this.referenceSurface?'Previous recording geometry · reference only':'Display only · recorded airflow and temperature continue';
  }
  toggle(on){
    const speed=$('#flow-speed');
    if(on&&!this.enabled){this.legacyTimeScale=+speed.value;speed.value=this.cfdTimeScale??1;}
    if(!on&&this.enabled){this.cfdTimeScale=+speed.value;speed.value=this.legacyTimeScale??.06;}
    this.enabled=on;document.body.classList.toggle('cfd-mode',on);
    if(on)this.hideReconstructionAirways();
    this.syncReconstructionAirwayRows();
    this.lab.stopPlayback();this.lab.restoreGeometry();this.modeControls();
    // The recording dictates the breath; look, size, count, speed and trail
    // remain the user's.
    for(const id of ['flow-breathe','flow-period','flow-thermal-walls']){
      $('#'+id).disabled=on;$('#'+id).closest('.row').hidden=on;
    }
    $('#flow-q').closest('.row').querySelector('label').textContent=on?'Recorded flow rate':'Peak flow';
    $('#flow-q').closest('.row').hidden=false;
    $('#flow-q').max=on?'500':'1000';
    if(on&&+$('#flow-q').value>500){$('#flow-q').value=500;$('#flow-q-val').textContent='500 mL/s';}
    $('#flow-n').max=on?'20000':'40000';
    if(on&&+$('#flow-n').value>20000)$('#flow-n').value=20000;
    $('#cfd-recording-ui').hidden=!on;
    $('#cfd-panel .cfd-options').hidden=!on;
    $('#cfd-compare').hidden=!on;
    if(!on){clearTimeout(this.geometryTimer);$('#cfd-geometry-status').textContent='';}
    $('#cfd-check').hidden=!on;$('#cfd-wall-controls').hidden=!on;
    this.updateSurfaceVisibility();
    this.syncModeCopy();
    $('#flow-color').querySelector('[value="thermal"]').disabled=on;
    $('#flow-thermal-walls').disabled=on;
    if(on){
      this.legacyColorMode=$('#flow-color').value;
      if(this.legacyColorMode==='thermal'){
        $('#flow-color').value='speed';$('#flow-color').dispatchEvent(new Event('change'));
      }
      this.originalFlow=this.ctx.flow;this.originalFlow?.setEnabled(false);
      this.ctx.setFlow(null);this.ctx.airwayPanel.setCFD(null,'No matching recorded 3-D field');
      if(this.ctx.state.dataset!=='pre')this.ctx.loadDataset('pre').then(()=>this.changed());else this.changed();
    }else{
      ++this.generation;this.clear();this.ctx.setFlow(this.originalFlow);this.ctx.airwayPanel.cfdMode=false;
      if(this.legacyColorMode==='thermal'){$('#flow-color').value='thermal';$('#flow-color').dispatchEvent(new Event('change'));}
      this.applyPlaybackControls(this.originalFlow);
      clearTimeout(this.timer);clearTimeout(this.geometryTimer);this.ctx.airwayPanel.chart.hidden=false;this.lab.update();
      $('#scenario-flow-toggle').disabled=false;
      $('#cfd-result').replaceChildren();$('#cfd-status').textContent=TUBE_ESTIMATE;$('#cfd-replay').hidden=true;
    }
    this.modeControls();$('#flow-color').dispatchEvent(new Event('change'));
  }
  applyPlaybackControls(flow){
    if(!flow)return;
    flow.setCount(+$('#flow-n').value);flow.setSize(+$('#flow-size').value);
    flow.setColorMode($('#flow-color').value);flow.timeScale=+$('#flow-speed').value;
    flow.setBreathing($('#flow-breathe').checked);flow.setPeriod(+$('#flow-period').value);
    flow.setStyle($('#flow-style').value);flow.setTrail(+$('#flow-trail').value);
    flow.setEnabled($('#flow-on').checked);
  }
  clear({preserveSurface=false}={}){
    // Dataset changes may attach a cached legacy engine before notifying the
    // lab. Retain and hide it before clearing the active pointer, including its
    // exterior room-air/plume particles.
    if(this.enabled&&this.ctx.flow&&this.ctx.flow!==this.particles){
      this.originalFlow=this.ctx.flow;this.originalFlow.setEnabled(false);
    }
    this.thermal?.dispose();this.thermal=null;
    this.particles?.dispose();this.particles=null;
    if(!preserveSurface){
      if(this.surface){this.ctx.surfaces.remove(this.surface);this.disposeModel(this.surface);this.surface=null;}
      for(const [mesh,visible] of this.hidden)mesh.visible=visible;this.hidden=[];
    }
    this.referenceSurface=preserveSurface&&!!this.surface;
    if(this.enabled)this.ctx.setFlow(null);
  }
  changed(){
    if(!this.enabled)return;
    // Invalidate the simulated readings immediately, but keep the last airway
    // as a plain reference while looking for a replacement recording.
    const generation=++this.generation;this.clear({preserveSurface:true});this.result=null;this.job=null;
    this.updateSurfaceVisibility();
    const referenceNote=this.referenceSurface?' Previous recording geometry remains visible as a reference; it does not represent the selected settings.':'';
    $('#cfd-replay').hidden=true;clearTimeout(this.geometryTimer);
    $('#cfd-wall-legend').hidden=true;$('#cfd-wall-reading').textContent='';$('#cfd-thermal-status').textContent='Checking recorded 3-D wall temperatures…';
    $('#cfd-wall-probe').textContent='Wall readings unavailable until a matching thermal recording loads.';
    $('#cfd-geometry-status').textContent='Checking recorded 3-D geometry…'+referenceNote;
    this.ctx.airwayPanel.setCFD(null,'Controls changed · looking for a recorded 3-D field');
    $('#cfd-result').replaceChildren();$('#cfd-status').textContent='No matching recorded 3-D field · checking saved results…';
    $('#flow-stats').textContent=MISSING_CFD;
    $('#scenario-flow-status').textContent='No matching recorded 3-D field';
    $('#scenario-flow-toggle').disabled=true;
    $('#flow-sec h2 .hint').textContent='recorded 3-D CFD · checking library';
    $('#flow-phase').textContent='';
    clearTimeout(this.timer);
    this.timer=setTimeout(async()=>{
      try{
        const request=this.request();const job=await this.library.lookup(request);
        if(generation!==this.generation)return;
        await this.observe(job,generation);
        if(job.state==='missing'){$('#cfd-geometry-status').textContent='No recorded 3-D CFD geometry for these settings.'+referenceNote;$('#cfd-thermal-status').textContent='No recorded 3-D wall temperatures for these settings.';}
      }
      catch(e){if(generation===this.generation)$('#cfd-status').textContent=e.message;}
    },0);
  }
  disposeModel(scene){scene.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});}
  installSurface(scene){
    this.surface=scene;this.surface.traverse(o=>{
      if(!o.isMesh)return;o.material.dispose();o.material=new THREE.MeshStandardMaterial({color:0x53bfe9,roughness:.6,side:THREE.FrontSide});
      setSurfaceOpacity(o.material,this.surfaceOpacity);o.renderOrder=nestedSurfaceOrder('airway');
      o.onBeforeRender=(_renderer,_scene,camera)=>sortTransparentFaces(o,camera);
    });this.ctx.surfaces.add(this.surface);this.updateSurfaceVisibility();
    this.hideReconstructionAirways();
  }
  hideReconstructionAirways(){
    for(const key of ['airway','airway_L','airway_R','airway_common']){
      const tissue=this.ctx.state.tissues[key];if(!tissue)continue;
      for(const mesh of [tissue.mesh,tissue.depthMesh]){
        if(!mesh)continue;
        if(!this.hidden.some(([item])=>item===mesh))this.hidden.push([mesh,mesh.visible]);
        mesh.visible=false;
      }
    }
  }
  async observe(job,generation){
    if(generation!==this.generation||!this.enabled)return;
    this.job=job;$('#cfd-status').textContent=job.message;
    if(job.state==='complete'){
      try{await this.load(job,generation);}catch(e){if(generation===this.generation)$('#cfd-status').textContent=e.message;}
    }
  }
  async load(job,generation,preview=false){
    const r=await this.api(job.resultUrl);
    if(!(preview?matchingField(r,job.id):matchingResult(r,job.id)))throw Error('Result identity, coordinates or required numerical checks do not match');
    const base=job.resultUrl.slice(0,job.resultUrl.lastIndexOf('/')+1),transient=r.solver.temporal==='transient';
    const velocityPath=transient?r.field.frames[0].velocity:r.field.velocity;
    const [v,o,model]=await Promise.all([this.binary(base+velocityPath),
      this.binary(base+r.field.occupancy),this.model(base+'airway.glb')]);
    if(generation!==this.generation||!this.enabled){model.scene.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});return;}
    const box=new THREE.Box3().setFromObject(model.scene),bounds=r.field.fluidBoundsMm;
    if(!bounds||[...box.min.toArray(),...box.max.toArray()].some((x,i)=>Math.abs(x-bounds.flat()[i])>1e-3)){
      model.scene.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});
      throw Error('CFD surface and field occupy different coordinates; animation withheld');
    }
    this.clear();this.result=r;this.statsTick=-1;
    this.installSurface(model.scene);
    this.particles=transient?new CFDTransientParticles(this.ctx.scene,r,o,v,path=>this.binary(base+path)):new CFDParticles(this.ctx.scene,r,v,o);
    if(transient){
      const particles=this.particles;
      $('#cfd-status').textContent='Loading saved breathing cycle…';
      await particles.prepareLoop();
      if(generation!==this.generation||!this.enabled)return;
    }
    if(this.originalFlow?.choanaZ!=null)this.particles.choanaZ=this.originalFlow.choanaZ;
    this.applyPlaybackControls(this.particles);
    this.ctx.setFlow(this.particles);
    this.particles.playbackPaused=!$('#cfd-wall-play').checked;
    this.ctx.airwayPanel.setCFD(preview?null:r,preview?'Unconverged field preview · accepted CFD readings withheld':'');
    $('#scenario-flow-toggle').disabled=false;
    $('#cfd-replay').hidden=!transient;
    $('#cfd-geometry-status').textContent='Displayed airway matches the CFD field geometry';
    const lattice=isLattice(r);
    $('#cfd-status').textContent=preview?'UNACCEPTED PREVIEW · required numerical checks failed':lattice?'GPU lattice checks passed · recorded LES trajectory; periodicity not established':transient?'Transient numerical checks passed · recorded trajectory; periodicity not established':'Converged · mesh and fields match these controls';
    $('#cfd-result').innerHTML=preview?`<p class="model-note">Particles show numerical fields that failed acceptance checks. Failed checks: ${Object.entries(r.gates).filter(([,v])=>!v).map(([k])=>k).join(', ')}.</p><a href="${job.resultUrl}" target="_blank">Iteration, residuals & diagnostic readings</a>`:`<div class="cfd-reading"><strong>${fmt(r.pressureDropPa,1)} <small>Pa</small></strong><span>nose → throat pressure difference at ${r.solver.qMlS} mL/s</span></div>
      <div class="cfd-numbers"><span>Left <b>${fmt(r.flowMlS.L,1)} mL/s</b></span><span>Right <b>${fmt(r.flowMlS.R,1)} mL/s</b></span></div>
      <p class="hint">${r.geometry.cells.toLocaleString()} cells · ${r.geometry.spacingMm} mm solver grid · ${r.iterations} iterations<br>Mass imbalance ${fmt(r.massImbalanceFraction*100,4)}% · wall leakage ${fmt(r.wallLeakMlS,6)} mL/s</p>
      <p class="model-note">${r.meshIndependence?'Mesh comparison recorded.':'Mesh independence has not been established.'} Screenshot anatomy and prescribed tissue response remain unvalidated.</p>
      <a href="${job.resultUrl}" target="_blank">Result, assumptions & diagnostics</a>`;
    if(transient&&!preview){
      $('#cfd-result').replaceChildren();
      const note=document.createElement('p');note.className='model-note';
      note.textContent=lattice?`${r.solver.cycles} cycle${r.solver.cycles>1?'s':''} · ${r.solver.periodS} s/breath · ${r.solver.pressureAmplitudePa} Pa nominal throat amplitude · ${r.geometry.spacingMm} mm lattice · ${(r.diagnostics.computeSeconds/60).toFixed(0)} min GPU. Peak measured nose → throat ΔP ${fmt(r.pressureDropPa,1)} Pa (openings lose part of the nominal amplitude). Large-eddy closure on a stair-step wall; mesh independence and periodicity are not established.`
        :`${r.solver.cycles} cycles · ${r.solver.periodS} s/breath · starts at rest. Airway readings follow playback time. Mesh/time-step independence and periodicity are not established.`;
      const a=document.createElement('a');a.href=job.resultUrl;a.target='_blank';a.textContent='Recorded times, boundary conditions & diagnostics';
      $('#cfd-result').append(note,a);
      if(r.solver.breathModel==='resting-beta-1'){
        const pattern=document.createElement('p');pattern.className='hint';
        pattern.textContent=`Resting-breath model · ${(r.solver.periodS*.4).toFixed(1)} s inspiratory drive / ${(r.solver.periodS*.6).toFixed(1)} s expiratory drive, with a rounded rise and longer expiratory tail. Model pattern, not a patient measurement.`;
        $('#cfd-result').prepend(pattern);
      }
    }
    if(!preview&&!$('#flow-on').checked){
      // The recording is loaded but particles are off (the default), so nothing
      // moves yet; offer the same action as the lab's View airflow button here.
      const view=document.createElement('button');view.type='button';view.className='small';view.id='cfd-view-airflow';
      view.textContent=transient?'View breathing airflow':'View airflow';
      view.onclick=()=>{this.ctx.applyPreset('Airway');$('#flow-on').checked=true;this.ctx.flow?.setEnabled(true);view.remove();};
      $('#flow-on').addEventListener('change',()=>view.remove(),{once:true});
      $('#cfd-result').append(view);
    }
    $('#flow-stats').textContent=`${preview?'UNACCEPTED 3-D field':lattice?'GPU lattice LES':'Recorded 3-D CFD'} · ${r.solver.direction} · velocity in every cell. Speed controls playback time only.`
      +(this.particles.exchange?' Air outside the nostrils (room air drawn in, exhaled puffs) is a kinematic sketch from the exit velocity, not part of the CFD domain.':'');
    $('#flow-sec h2 .hint').textContent=preview?'unaccepted 3-D field':lattice?'recorded 3-D CFD · GPU lattice':'recorded 3-D CFD';
    $('#cfd-thermal-status').textContent=job.thermalUrl?'Loading recorded 3-D wall temperatures…':'No recorded 3-D wall temperatures. Airflow only; speed colour is not temperature.';
    if(job.thermalUrl){
      try{
        const thermal=await CFDThermalWalls.load(job.thermalUrl,r,()=>generation===this.generation&&this.enabled);
        if(generation!==this.generation||!this.enabled){thermal.dispose();return;}
        this.thermal=thermal;this.ctx.surfaces.add(thermal.mesh);
        // Thermal cycle was equilibrated on the final airflow cycle. Start
        // both there; never align it with the CFD startup-from-rest cycle.
        this.particles.time=this.particles.frames[this.particles.loopStart].timeS;
        this.particles.frame=this.particles.loopStart;this.particles.ensure(this.particles.frame);this.particles.ensure(this.particles.frame+1);
        this.particles.updateMetrics();this.particles.playWithoutParticles=true;if(!this.thermalOpacityInitialized){this.surfaceOpacity=1;this.thermalOpacityInitialized=true;}thermal.setOpacity(this.surfaceOpacity);this.mountSurfaceControl();
        const p=thermal.meta.settings;
        $('#cfd-thermal-status').textContent=`Computed 3-D sensible heat transfer · room ${p.ambientC} °C, deep tissue ${p.bodyC} °C, exhaled air ${p.exhaledC} °C; ${p.tissueThicknessMm} mm assumed tissue layer. Evaporation is not included. Model temperatures, not patient measurements.`;
        $('#cfd-wall-probe').textContent='Point at a wall to inspect its local temperature and heat flux.';
        this.updateThermalDisplay();$('#flow-color').dispatchEvent(new Event('change'));
      }catch(e){if(generation===this.generation)$('#cfd-thermal-status').textContent=e.message;}
    }
  }
  updateThermalDisplay(){
    const mode=$('#cfd-wall-mode').value,scale=$('#cfd-wall-scale').value;
    this.thermal?.setMode(mode,scale);this.updateSurfaceVisibility();
    $('#cfd-wall-scale-row').hidden=mode==='surface';
    const legend=$('#cfd-wall-legend');legend.hidden=!this.thermal||mode==='surface';
    legend.querySelector('.units').textContent=mode==='flux'?'Heat flux · W/m² · positive = wall cooling':'Wall temperature · °C';
    const bands=WALL_COLOR_BANDS[scale][mode==='flux'?'flux':'temperature'];
    legend.querySelectorAll('.ticks span').forEach((tick,i)=>{tick.textContent=bands[i];});
    legend.querySelector('p').textContent=scale==='detail'?'Fixed nonlinear bands reveal small changes; extra color detail does not imply extra model accuracy.':'Fixed linear scale. Values outside the range use the endpoint colors.';
  }
  step(){
    if(!this.enabled)return;
    if(this.thermal&&this.particles){
      this.thermal.step(this.particles.time);const r=this.thermal.reading;
      if(Math.floor(this.particles.time*10)!==this.thermalStatsTick){this.thermalStatsTick=Math.floor(this.particles.time*10);$('#cfd-wall-reading').textContent=`Wall ${r.minC.toFixed(2)}–${r.maxC.toFixed(2)} °C · mean ${r.meanC.toFixed(2)} °C · sensible heat loss ${r.heatLossW.toFixed(2)} W`;}
      this.thermal.material.clippingPlanes=Object.values(this.ctx.state.tissues)[0]?.mat.clippingPlanes||[];
    }
    if(this.particles instanceof CFDTransientParticles&&this.result?.status==='converged'){
      const tick=Math.floor(this.particles.time*10);
      if(tick!==this.statsTick){this.statsTick=tick;this.ctx.airwayPanel.setCFDFrame(this.particles.currentMetrics);}
    }
    for(const [mesh] of this.hidden)mesh.visible=false;
    if(this.surface){const planes=Object.values(this.ctx.state.tissues)[0]?.mat.clippingPlanes||[];this.surface.traverse(o=>{if(o.isMesh)o.material.clippingPlanes=planes;});}
  }
}
