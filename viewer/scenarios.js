import { defaults, clone, REGIONS, POSITIONS, congestion, scenarioData, noIntervention, hasPreview, buildReport } from './scenario-model.js';
import { evaluate } from './hydraulics.js';
import { evaluateHeatFlux } from './heatflux.js';
import { ScenarioView } from './scenario-view.js';
import { reportPDF, download, number } from './report-pdf.js';
import { CFDLab } from './cfd-lab.js';
import { TUBE_ESTIMATE, TUBE_ESTIMATE_NOTE, TUBE_GEOMETRY_NOTE } from './airflow-labels.js';

const $=s=>document.querySelector(s);
const html=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,d=1)=>v===Infinity?'∞':number(v,d);
const slider=(key,label,min,max,step,value,unit='mm')=>`<label class="lab-slider"><span>${label}</span><output data-output="${key}">${value} ${unit}</output><input aria-label="${label}" data-setting="${key}" data-unit="${unit}" type="range" min="${min}" max="${max}" step="${step}" value="${value}"></label>`;

export class ScenarioLab {
  constructor(ctx) {
    this.ctx=ctx;this.settings=defaults();this.saved=[];this.previewActive=false;this.view=new ScenarioView(ctx);this.queueId=0;
    this.ctMetrics=evaluate(ctx.base,250);
    this.ctThermal=evaluateHeatFlux(ctx.base,this.ctMetrics);
    try {
      const saved=JSON.parse(localStorage.getItem('airway3d-plans-v1')||'[]');
      this.saved=saved.filter(p=>typeof p.name==='string' && ['L','R'].every(side=>Object.keys(REGIONS).every(k=>Number.isFinite(p.settings?.operations?.[side]?.[k])&&p.settings.operations[side][k]>=0&&p.settings.operations[side][k]<=2)) && Number.isFinite(p.settings.relief)&&p.settings.relief>=0&&p.settings.relief<=100).slice(0,3);
    }catch{}
    this.build();
    this.cfd=new CFDLab(this);
  }
  build() {
    const root=$('#scenario-lab');root.innerHTML=`
      <div id="congestion-monitor" role="region" aria-label="Live left and right congestion proxy">
        <div class="congestion-heading"><strong>Congestion proxy</strong><button id="congestion-about" class="small" aria-label="About the congestion indicator">About</button></div>
        <p class="congestion-caption">Live tube estimate · higher = harder airflow</p>
        <div id="congestion-values" class="congestion-values"></div>
        <p id="congestion-context" class="congestion-context"></p>
      </div>
      <div id="thermal-monitor" role="region" aria-label="Live left and right mucosal heat flux estimate">
        <div class="congestion-heading"><strong>Mucosal heat flux</strong><button id="thermal-about" class="small" aria-label="About the heat flux estimate">About</button></div>
        <p class="congestion-caption">Live tube estimate · cooling area above 50 W/m²</p>
        <div id="thermal-values" class="congestion-values"></div>
        <p id="thermal-context" class="congestion-context"></p>
      </div>
      <div class="lab-controls">
      <header><h1>Scenario lab</h1><p class="lab-intro">Explore a change and test it against the face-up CT reference.</p></header>
      <div class="dataset-buttons"><button class="small on" data-source="pre">CT · face up</button><button class="small" data-source="post">Saved 2 mm reduction</button></div>
      <p id="lab-source-status" class="hint" role="status">CT reference · face up</p>
      <p class="model-note" id="scenario-model-note">${TUBE_GEOMETRY_NOTE}</p>
      <section><h2>Intervention</h2>
        <label class="lab-name">Plan name<input id="plan-name" maxlength="48" value="Untitled intervention"></label>
        <div class="region-tabs" role="group" aria-label="Region to highlight">${Object.entries(REGIONS).map(([k,r])=>`<button class="small ${k==='head'?'on':''}" data-region="${k}" style="--region-color:${r.color}">${r.short}</button>`).join('')}</div>
        <p id="region-help" class="hint">${REGIONS.head.help}</p>
        <div class="lab-actions"><label><input id="region-guides" type="checkbox"> Show region guides</label><button id="focus-region" class="small">Focus region</button></div>
        <p class="hint">Assumed equivalent-radius gain. Head/body represent clearance after tissue change; valve represents widening/support.</p>
        <div class="operation-grid"><span>Clearance (mm)</span><strong class="side-left">Left</strong><strong class="side-right">Right</strong>
          ${Object.entries(REGIONS).map(([k,r])=>`<label>${r.short}</label>${['L','R'].map(side=>`<div><input type="range" aria-label="${side==='L'?'Left':'Right'} ${r.short.toLowerCase()} clearance" min="0" max="2" step="0.1" value="0" data-op="${side}.${k}"><output data-op-output="${side}.${k}">0.0</output></div>`).join('')}`).join('')}
        </div>
        ${slider('relief','Assumed swelling suppression',0,100,5,0,'%')}
        <p class="hint">Suppression is a separate intervention hypothesis. It is not an established effect of bone reduction.</p>
        <div class="lab-actions"><button id="clear-intervention" class="small">Clear intervention</button><button id="save-plan" class="small">Save plan</button></div>
        <div id="saved-plans"></div>
      </section>
      <section><h2>Body position & tissue response</h2>
        <div class="position-buttons" role="group" aria-label="Body position">${Object.entries(POSITIONS).map(([k,v])=>`<button class="small" data-position="${k}" aria-pressed="${k==='supine'}">${v}</button>`).join('')}</div>
        <p class="hint">Face up is the observed CT. Other positions explore assumed swelling and displacement; these responses are not measured by the scan.</p>
        ${slider('elapsed','Time in position',0,30,0.5,10,'min')}
        <button id="play-posture" class="small" aria-pressed="false">Play tissue response</button>
        <p class="hint">Plays 0–15 minutes in 15 animation seconds. Particle breathing continues independently.</p>
        <details><summary>Adjust unmeasured tissue response</summary>
          ${slider('responseL','Left dependent swelling',0,5,0.1,0.6)}
          ${slider('responseR','Right dependent swelling',0,5,0.1,0.6)}
          ${slider('gravity','Immediate gravity displacement',0,1,0.05,0.15)}
          ${slider('cycle','Nasal cycle bias (positive = left)',-2,2,0.1,0)}
          ${slider('tau','Vascular settling time',0.5,15,0.5,5,'min')}
          <button id="reported-blockage" class="small">Explore reported left-side blockage</button>
          <p class="hint">This preset assumes stronger left swelling to explore your observation. It does not establish its cause or severity.</p>
        </details><p id="posture-state" class="posture-state"></p>
      </section>
      <section><h2>Live comparison</h2><p class="hint" id="live-context">Selected scenario vs no intervention in the face-up CT reference.</p>
        <details class="fidelity"><summary>Flow calculation & clinical readiness</summary><p class="hint">${TUBE_ESTIMATE}: viscous friction and local losses along each passage, with pressure-balanced left/right flow. ${TUBE_ESTIMATE_NOTE} Recorded 3-D CFD and wall heat transfer are a separate saved library. Anatomy review, mesh independence and comparison with measured nasal pressure/flow: not yet performed.</p><p class="hint">A useful doctor-facing result needs a reviewed airway and boundary conditions, a converged 3-D solve, and validation against measurements. Posture compliance cannot be recovered from one CT. The PDF records these gaps.</p></details>
        <p id="scenario-flow-status" class="posture-state" aria-label="Live particle breathing"></p>
        <div class="lab-actions"><button id="scenario-flow-toggle" class="small" aria-pressed="true">Pause particles</button><button id="scenario-flow-focus" class="small">View airflow</button></div>
        <div id="scenario-live"></div><p id="preview-note" class="model-note" role="status">CT reference. No preview deformation.</p>
        <button id="compare-scenarios" class="lab-primary">Compare plans × body positions</button>
        <p class="hint">Current plan + up to three saved plans. Interactive report, full PDF and reproducible data.</p>
        <button id="reset-scenario" class="small">Reset to CT reference</button>
      </section></div>`;
    const explanation=document.createElement('dialog');explanation.id='congestion-explanation';explanation.setAttribute('aria-labelledby','congestion-explanation-title');
    explanation.innerHTML=`<header><h2 id="congestion-explanation-title">What does congestion feel like?</h2><button class="small" id="close-congestion-explanation">Close</button></header>
      <p>The direct measure is your own rating for each side: <strong>0 = completely clear, 10 = completely blocked</strong>. A model cannot assign that rating on your behalf.</p>
      <p>Research links perceived openness to <strong>cooling of the nasal lining</strong>. Heat loss and the area of lining cooled by airflow are promising predictors, but no single CFD measure reliably predicts every patient's symptoms. This app does not yet calculate heat transfer or sensory nerve response.</p>
      <p>The live indicator therefore shows <strong>estimated nasal resistance at 150 Pa</strong>: R = 150 ÷ Q at that pressure, in Pa·s/mL. Higher means less air passes for the same pressure. Left and right are evaluated separately, excluding the shared throat.</p>
      <p>The percentage compares each side with its own face-up CT reference. It is a change in resistance, <strong>not a percentage of blockage or a symptom severity score</strong>. No clinical mild/moderate/severe thresholds are assigned. Lower resistance alone does not establish a better surgical outcome.</p>
      <p>Intervention, posture, swelling and tissue-response playback update the numbers. The test pressure stays at 150 Pa, so changing particle speed, breath phase or peak flow does not change this comparison. A closed branch has infinite resistance and is shown as “Closed.”</p>
      <p>These live numbers are a tube estimate from screenshot anatomy, not measured rhinomanometry: one mean speed per cross-section along each passage (a 1-D pipe calculation). The 3-D picture is a drawing of that estimate. The indicators follow the selected scenario even when CT slices or volume rendering still show the saved scan.</p>
      <p class="congestion-sources"><a href="https://pubmed.ncbi.nlm.nih.gov/22022361/" target="_blank" rel="noreferrer">Cooling and perceived nasal patency (2011)</a><br><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3917722/" target="_blank" rel="noreferrer">Mucosal cooling and symptoms after surgery (2014)</a></p>`;
    document.body.appendChild(explanation);
    $('#congestion-about').onclick=()=>explanation.showModal();
    $('#close-congestion-explanation').onclick=()=>explanation.close();
    const thermalExplanation=document.createElement('dialog');thermalExplanation.id='thermal-explanation';
    thermalExplanation.innerHTML=`<header><h2>What is this heat-flux number?</h2><button class="small" id="close-thermal-explanation">Close</button></header>
      <p>This live indicator is the tube-model heat flux: local convection from the same station-by-station inspiratory flow, hydraulic diameter, assumed mucosa temperature and bulk-air warming. It updates with every slider. Recorded 3-D CFD wall temperatures are a separate saved library, shown only in that mode.</p>
      <p>The most useful displayed quantity is the <strong>mucosal area exposed above 50 W/m²</strong>, with peak heat flux beside it. A clinical CFD study found this type of cooling signal correlated with patient-reported nasal patency, but that does not make this screenshot-based estimate a patient prediction.</p>
      <p>In the View &amp; flow panel, choose <strong>mucosal heat flux · inspiration</strong> to color both particles and airway walls blue → cyan → green → yellow → red. The wall overlay and particle colors use the same per-shell estimate; uncheck <strong>Thermal walls</strong> if you want particles only.</p>
      <p>This live indicator is not 3-D CFD heat transfer. It does not solve Navier–Stokes plus energy transport, measured wall temperature, humidity or near-wall boundary layers. Those live in the recorded 3-D CFD library when a matching thermal recording exists.</p>
      <p>All values update when you change an intervention, body position, swelling response or saved dataset. They are model outputs, not a 0–10 symptom score. Rate your left and right congestion separately if you want to compare the calculation with how you feel.</p>
      <p class="congestion-sources"><a href="https://pubmed.ncbi.nlm.nih.gov/23775640/" target="_blank" rel="noreferrer">Peak mucosal cooling and nasal patency (2014)</a><br><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3917722/" target="_blank" rel="noreferrer">Mucosal cooling after surgery (2014)</a></p>`;
    document.body.appendChild(thermalExplanation);
    $('#thermal-about').onclick=()=>thermalExplanation.showModal();$('#close-thermal-explanation').onclick=()=>thermalExplanation.close();
    // Panel stowing lives in app.js (the rail); body.lab-open mirrors whether
    // this panel is open so labels dock to the right column.
    root.addEventListener('input',e=>{
      const t=e.target;
      if(t.id==='plan-name'){this.settings.name=t.value;return;}
      if(t.dataset.op){const [side,k]=t.dataset.op.split('.');this.settings.operations[side][k]=+t.value;}
      if(t.dataset.setting)this.settings[t.dataset.setting]=+t.value;
      if(t.dataset.op || t.dataset.setting){this.sync();this.queue();}
    });
    root.querySelectorAll('[data-region]').forEach(b=>b.onclick=()=>{
      this.view.select(b.dataset.region,$('#region-guides').checked);
      $('#region-help').textContent=REGIONS[b.dataset.region].help;
      root.querySelectorAll('[data-region]').forEach(x=>x.classList.toggle('on',x===b));
    });
    $('#region-guides').onchange=e=>this.view.select(this.view.selected,e.target.checked);
    $('#scenario-flow-toggle').onclick=()=>{const cb=$('#flow-on');cb.checked=!cb.checked;this.ctx.flow?.setEnabled(cb.checked);};
    $('#scenario-flow-focus').onclick=()=>{this.ctx.applyPreset('Airway');$('#flow-on').checked=true;this.ctx.flow?.setEnabled(true);};
    root.querySelectorAll('[data-position]').forEach(b=>b.onclick=()=>{this.settings.position=b.dataset.position;this.sync();this.queue();});
    $('#play-posture').onclick=()=>{
      if(this.playing){this.stopPlayback();return;}
      this.settings.elapsed=0;this.playing=true;this.playbackTick=0;
      $('#play-posture').textContent='Pause tissue response';$('#play-posture').setAttribute('aria-pressed','true');this.sync();this.queue();
    };
    $('#reported-blockage').onclick=()=>{this.stopPlayback();Object.assign(this.settings,{position:'left',responseL:4,responseR:0.6,gravity:0.15,cycle:0,tau:5,elapsed:15});this.sync();this.queue();};
    $('#focus-region').onclick=()=>{$('#region-guides').checked=true;this.view.focus(this.view.selected,'L');};
    root.querySelectorAll('[data-source]').forEach(b=>b.onclick=async()=>{
      this.stopPlayback(); ++this.queueId;
      const loaded=await this.ctx.loadDataset(b.dataset.source);
      if(loaded && b.dataset.source==='pre') {this.settings=defaults();this.sync();this.update();}
    });
    $('#clear-intervention').onclick=()=>{this.settings=noIntervention(this.settings);this.sync();this.queue();};
    $('#reset-scenario').onclick=async()=>{this.stopPlayback();this.settings=defaults();this.sync();await this.ctx.loadDataset('pre');this.update();};
    $('#save-plan').onclick=()=>{
      if(this.saved.length>=3){$('#plan-feedback').textContent='Three plans saved. Remove a plan to save another.';return;}
      const name=this.settings.name.trim() || `Intervention ${this.saved.length+1}`;
      this.saved.push({name:name.slice(0,48),settings:clone(this.settings)});this.persist();this.savedUI();
    };
    $('#compare-scenarios').onclick=()=>{this.stopPlayback();this.openReport();};
    this.savedUI();this.sync();this.update();
  }
  persist(){try{localStorage.setItem('airway3d-plans-v1',JSON.stringify(this.saved));this.storageError=false;}catch{this.storageError=true;}}
  savedUI(){
    $('#saved-plans').innerHTML=this.saved.map((p,i)=>`<div class="saved-plan"><button class="small" data-recall="${i}">${html(p.name)}</button><button class="small" data-remove="${i}" aria-label="Remove ${html(p.name)}">Remove</button></div>`).join('')+`<p id="plan-feedback" class="hint" role="status">${this.storageError?'Saved for this session; browser storage is unavailable.':`${this.saved.length}/3 plans saved locally. Saved plans store interventions; comparisons share current physiology assumptions.`}</p>`;
    $('#saved-plans').querySelectorAll('[data-recall]').forEach(b=>b.onclick=()=>{const p=this.saved[+b.dataset.recall];this.settings.operations=clone(p.settings.operations);this.settings.relief=p.settings.relief;this.settings.name=p.name;this.sync();this.queue();});
    $('#saved-plans').querySelectorAll('[data-remove]').forEach(b=>b.onclick=()=>{this.saved.splice(+b.dataset.remove,1);this.persist();this.savedUI();});
  }
  sync(){
    const root=$('#scenario-lab');$('#plan-name').value=this.settings.name;
    root.querySelectorAll('[data-op]').forEach(e=>{const [s,k]=e.dataset.op.split('.');e.value=this.settings.operations[s][k];root.querySelector(`[data-op-output="${e.dataset.op}"]`).textContent=(+e.value).toFixed(1);});
    root.querySelectorAll('[data-setting]').forEach(e=>{e.value=this.settings[e.dataset.setting];root.querySelector(`[data-output="${e.dataset.setting}"]`).textContent=`${e.value} ${e.dataset.unit}`;});
    root.querySelectorAll('[data-position]').forEach(b=>{const on=b.dataset.position===this.settings.position;b.classList.toggle('on',on);b.setAttribute('aria-pressed',String(on));});
    $('#posture-state').textContent=`Assumed radial response before suppression: L ${congestion(this.settings,'L').toFixed(2)} mm · R ${congestion(this.settings,'R').toFixed(2)} mm. Positive narrows; negative opens.`;
  }
  async queue(){
    const id=++this.queueId;
    if(this.ctx.state.dataset!=='pre') {if(!await this.ctx.loadDataset('pre'))return;}
    requestAnimationFrame(()=>{if(id===this.queueId)this.update();});
  }
  restoreGeometry(){this.view.restore();this.previewActive=false;this.ctx.flow?.clearScenario();}
  datasetChanged(name,data){
    if(this.cfd?.enabled){this.cfd.changed();return;}
    this.restoreGeometry();this.view.select(this.view.selected,false);$('#region-guides').checked=false;
    $('#scenario-lab').querySelectorAll('[data-source]').forEach(b=>b.classList.toggle('on',b.dataset.source===name));
    if(name==='post'){
      this.renderCongestion(evaluate(data,this.ctx.q()),'Saved 2 mm reduction');
      $('#flow-sec h2 .hint').textContent='saved streamlines · tube-model split';
      $('#flow-stats').textContent=this.ctx.flow?.meta.summary_html || 'No stored flow field';
      this.renderHeatFlux(evaluateHeatFlux(data,evaluate(data,this.ctx.q())), 'Saved 2 mm reduction');
      $('#preview-note').textContent='Saved geometric reduction is displayed. Editing a what-if control returns to the CT reference. CT labels and stored particles match this saved dataset.';
      $('#scenario-live').textContent='Saved dataset metrics are in the Airway panel. The comparison report uses the configurable plan above, not this saved mesh.';
    }
  }
  update(){
    if(this.cfd?.enabled){this.cfd.changed();return;}
    if(this.ctx.state.dataset!=='pre')return;
    const s=this.settings,base=this.ctx.base;
    this.data=scenarioData(base,s);this.metrics=evaluate(this.data,this.ctx.q());
    const refData=scenarioData(base,noIntervention(s));const ref=evaluate(refData,this.ctx.q());
    this.previewActive=hasPreview(s);
    if(this.previewActive)this.view.update(this.data);else this.view.restore();
    this.syncParticles();
    this.ctx.flow?.setEnabled($('#flow-on').checked);
    this.ctx.airwayPanel.setData(this.previewActive?this.data:base,this.previewActive?refData:null,this.previewActive?'scenario':'pre');
    this.ctx.airwayPanel.setFlowRate(this.ctx.q());this.ctx.updateCuts();
    this.renderLive(ref);
    $('#preview-note').textContent=this.previewActive?'Live tube estimate: the 3-D drawing stretches stored anatomy; particle width, speed and nostril split follow the tube model. CT slices/volume stay the saved scan. Not a 3-D Navier–Stokes field.':'Face-up CT reference. Particles follow stored streamlines; speed uses the live tube-model flow split.';
    $('#flow-sec h2 .hint').textContent=this.previewActive?'live tube estimate · particles':'stored streamlines · tube-model split';
    $('#live-context').textContent=`Selected scenario vs no intervention · ${POSITIONS[s.position]} · ${s.elapsed.toFixed(1)} min.`;
  }
  renderLive(ref){
    const m=this.metrics,d=this.data;
    this.renderCongestion(m,`${POSITIONS[this.settings.position]} · ${this.settings.elapsed.toFixed(1)} min`);
    this.renderHeatFlux(evaluateHeatFlux(d,m),`${POSITIONS[this.settings.position]} · ${this.settings.elapsed.toFixed(1)} min`);
    const reference=scenarioData(this.ctx.base,noIntervention(this.settings));
    $('#scenario-live').innerHTML=`<div class="live-pressure"><strong>${m.total.blocked?'Infeasible':`${fmt(m.total.dP)} <small>Pa</small>`}</strong><span>nose → pharynx at ${this.ctx.q()} mL/s</span></div>
      <div class="flow-balance" aria-label="Flow distribution"><i style="width:${m.total.deliveredQ?m.L.Q/m.total.deliveredQ*100:0}%"></i></div>
      <div class="live-sides"><span class="side-left">L ${fmt(m.L.Q)} mL/s${m.L.closed?' · closed':''}</span><span class="side-right">R ${fmt(m.R.Q)} mL/s${m.R.closed?' · closed':''}</span></div>
      <table class="lab-table"><thead><tr><th>Metric</th><th>No intervention</th><th>Current</th></tr></thead><tbody>
        <tr><td>Resistance<br><small>Pa·s/mL</small></td><td>${fmt(ref.total.R,3)}</td><td>${fmt(m.total.R,3)}</td></tr>
        <tr><td>Min area L / R<br><small>mm²</small></td><td>${fmt(reference.sides.L.min_area_mm2)} / ${fmt(reference.sides.R.min_area_mm2)}</td><td>${fmt(d.sides.L.min_area_mm2)} / ${fmt(d.sides.R.min_area_mm2)}</td></tr>
      </tbody></table>${m.total.blocked?'<p class="model-note">No open nasal path can carry the requested flow in this assumed state.</p>':''}`;
  }
  renderHeatFlux(thermal,context){
    this.thermal=thermal;
    $('#thermal-values').innerHTML=['L','R'].map(side=>{
      const h=thermal[side]||{};
      return `<div class="congestion-side ${side==='L'?'side-left':'side-right'}" data-thermal-side="${side}" data-peak="${h.peakWm2||0}">
        <div><span>${side==='L'?'Left':'Right'}</span><output aria-label="${side==='L'?'Left':'Right'} cooled area">${h.closed?'Closed':`${fmt(h.areaAbove50Cm2,1)} cm²`}</output></div>
        <span class="congestion-change">${h.closed?'No nasal passage flow':`peak ${fmt(h.peakWm2,0)} W/m² · loss ${fmt(h.heatLossW,2)} W`}</span>
      </div>`;
    }).join('');
    $('#thermal-context').textContent=`Area above 50 W/m² · ${context}`;
    $('#thermal-monitor').dataset.context=context;
  }
  renderCongestion(metrics,context){
    $('#congestion-values').innerHTML=['L','R'].map(side=>{
      const value=metrics[side].R150,reference=this.ctMetrics[side].R150;
      const delta=Number.isFinite(value)&&Number.isFinite(reference)&&reference>0?100*(value/reference-1):null;
      const change=delta===null?'No finite comparison':Math.abs(delta)<0.05?'Same as face-up CT':`${Math.abs(delta).toFixed(1)}% ${delta>0?'higher':'lower'} vs CT`;
      const closed=metrics[side].closed;
      return `<div class="congestion-side ${side==='L'?'side-left':'side-right'}" data-congestion-side="${side}" data-resistance="${value}" data-closed="${closed}">
        <div><span>${side==='L'?'Left':'Right'}</span><output aria-label="${side==='L'?'Left':'Right'} estimated nasal resistance">${closed?'Closed':value>=1000&&Number.isFinite(value)?value.toExponential(1):fmt(value,3)}</output></div>
        <span class="congestion-change">${closed?'No nasal passage flow':html(change)}</span>
      </div>`;
    }).join('');
    $('#congestion-context').textContent=`R at 150 Pa · Pa·s/mL · ${context}`;
    $('#congestion-monitor').dataset.context=context;
    $('#congestion-monitor').dataset.blocked=String(metrics.total.blocked);
    $('#congestion-monitor').title='Calculated airflow resistance, not a predicted patient symptom score. Higher means less airflow at the same pressure.';
  }
  syncParticles(){
    this.ctx.flow?.setScenario(this.ctx.base,this.data,this.metrics,{immediate:!this.previewActive||matchMedia('(prefers-reduced-motion: reduce)').matches});
    this.ctx.updateThermalWalls?.(this.data,this.metrics);
    $('#flow-stats').textContent=`${TUBE_ESTIMATE} · peak split L ${fmt(this.metrics.L.Q)} / R ${fmt(this.metrics.R.Q)} mL/s. Breathing scales and reverses this split along stored streamlines stretched with the wall. Not a 3-D Navier–Stokes field.`;
  }
  flowChanged(){if(this.cfd?.enabled){this.cfd.changed();return;}if(this.ctx.state.dataset==='pre'){this.metrics=evaluate(this.data,this.ctx.q());this.syncParticles();this.renderLive(evaluate(scenarioData(this.ctx.base,noIntervention(this.settings)),this.ctx.q()));}}
  stopPlayback(){this.playing=false;$('#play-posture').textContent='Play tissue response';$('#play-posture').setAttribute('aria-pressed','false');}
  step(dt){
    this.view.step(dt);
    if(this.playing){
      this.settings.elapsed=Math.min(15,this.settings.elapsed+dt);this.playbackTick+=dt;
      if(this.playbackTick>=0.1){this.playbackTick=0;this.sync();this.queue();}
      if(this.settings.elapsed>=15){this.stopPlayback();this.sync();this.queue();}
    }
  }
  openReport(){
    const current={name:this.settings.name.trim()||'Current intervention',settings:clone(this.settings)};
    this.report=buildReport(this.ctx.base,this.settings,[current,...this.saved],this.ctx.q(),this.ctx.state.meta);
    const dialog=$('#scenario-report');dialog.innerHTML=`
      <header class="report-header"><div><h1>Intervention comparison</h1><p>Matched what-if comparisons · ${this.report.q} mL/s · CT reference: face up</p></div><button class="small" id="close-report">Close</button></header>
      <div class="report-body"><p class="report-warning">Sensitivity analysis of assumed interventions and posture responses, not a prediction of surgical benefit. The observed CT is face up; the shared pharynx stays fixed.</p>
      <div class="report-toolbar"><label>Body position <select id="report-position"><option value="all">All positions</option>${Object.entries(POSITIONS).map(([k,v])=>`<option value="${k}">${v}</option>`).join('')}</select></label><label>Sort by <select id="report-sort"><option value="position">Plan order</option><option value="pressure">Total pressure</option><option value="area">Smallest nasal area</option></select></label>
      <button class="small" id="download-pdf">Download full PDF</button><button class="small" id="download-report-data">Download data</button></div>
      <p class="hint">Select a row to inspect cross-sections and per-side losses, then preview its particle flow. ΔR uses no intervention in the same position. ∞ means prescribed flow is infeasible.</p>
      <div class="report-table-wrap"><table class="report-table"><thead><tr><th>Plan</th><th>Min area L / R<br>mm²</th><th>Total pressure<br>Pa</th><th>Resistance<br>Pa·s/mL</th><th>ΔR<br>Pa·s/mL</th><th>Flow L / R<br>mL/s</th></tr></thead><tbody id="report-rows"></tbody></table></div>
      <section id="report-detail"></section>
      <details class="report-methods"><summary>Full assumptions, equations and evidence</summary><p>All plans share the same flow rate, elapsed time and tissue-response assumptions. Equivalent radius changes where the selected region overlaps the airway; area scales with radius², perimeter and hydraulic diameter with radius. The downloadable data contains every transformed profile.</p><ul>${this.report.limitations.map(s=>`<li>${html(s)}</li>`).join('')}</ul>${this.report.sources.map(s=>`<p><a href="${s.url}" target="_blank" rel="noreferrer">${html(s.title)}</a></p>`).join('')}</details>
      </div>`;
    $('#close-report').onclick=()=>dialog.close();
    $('#report-sort').onchange=()=>this.reportRows();
    $('#report-position').onchange=()=>this.reportRows();
    $('#download-pdf').onclick=()=>download(reportPDF(this.report),'airway3d-scenario-report.pdf','application/pdf');
    $('#download-report-data').onclick=()=>download(JSON.stringify(this.report,(_,v)=>typeof v==='number'&&!Number.isFinite(v)?String(v):v,2),'airway3d-scenario-report.json','application/json');
    dialog.showModal();this.reportRows();
  }
  reportRows(){
    let rows=this.report.rows.map((r,i)=>({...r,id:i}));
    if($('#report-position').value!=='all')rows=rows.filter(r=>r.position===$('#report-position').value);
    if($('#report-sort').value==='pressure')rows.sort((a,b)=>a.metrics.total.dP-b.metrics.total.dP);
    if($('#report-sort').value==='area')rows.sort((a,b)=>Math.min(a.data.sides.L.min_area_mm2,a.data.sides.R.min_area_mm2)-Math.min(b.data.sides.L.min_area_mm2,b.data.sides.R.min_area_mm2));
    $('#report-rows').innerHTML=rows.map(r=>`<tr data-report-row="${r.id}"><td><button data-inspect="${r.id}">${html(r.name)}</button><small>${html(r.positionLabel)}</small></td><td>${fmt(r.data.sides.L.min_area_mm2)} / ${fmt(r.data.sides.R.min_area_mm2)}</td><td>${fmt(r.metrics.total.dP)}</td><td>${fmt(r.metrics.total.R,3)}</td><td>${fmt(r.deltaResistance,3)}</td><td>${fmt(r.metrics.L.Q)} / ${fmt(r.metrics.R.Q)}</td></tr>`).join('');
    $('#report-rows').querySelectorAll('tr').forEach(tr=>tr.onclick=()=>this.reportDetail(+tr.dataset.reportRow));
    if(rows.length)this.reportDetail(rows[0].id);
  }
  reportDetail(id){
    this.selectedReport=id;const r=this.report.rows[id];
    $('#report-rows').querySelectorAll('tr').forEach(tr=>{const on=+tr.dataset.reportRow===id;tr.classList.toggle('selected',on);tr.querySelector('button').setAttribute('aria-pressed',String(on));});
    const o=r.settings.operations;
    $('#report-detail').innerHTML=`<h2>${html(r.name)} · ${html(r.positionLabel)}</h2><p class="hint">Clearance head / body / valve: L ${o.L.head} / ${o.L.body} / ${o.L.valve} mm; R ${o.R.head} / ${o.R.body} / ${o.R.valve} mm. Assumed swelling suppression ${r.settings.relief}%.</p>
      <button id="preview-report-flow" class="lab-primary">Preview this scenario’s particle flow</button>
      <canvas id="report-chart" width="1000" height="270" aria-label="Cross-section area versus distance from nostril"></canvas>
      <p class="legend"><span class="side-left">Left</span> · <span class="side-right">Right</span> · solid: selected plan · dashed: no intervention in the same position</p>
      <div class="report-table-wrap"><table class="report-table"><thead><tr><th>Passage</th><th>Flow<br>mL/s</th><th>Pressure<br>Pa</th><th>Wall friction<br>Pa</th><th>Local losses<br>Pa</th><th>Peak speed<br>m/s</th><th>Peak shear<br>Pa</th><th>Isolated Q @150 Pa<br>mL/s</th><th>Heat area &gt;50<br>W/m²</th><th>Peak heat flux<br>W/m²</th></tr></thead><tbody>${['L','R','common'].map(k=>{const m=r.metrics[k],h=r.thermal?.[k]||{};return `<tr><td>${k==='common'?'Pharynx':k==='L'?'Left':'Right'}${m.closed?' (closed)':''}</td><td>${fmt(m.Q)}</td><td>${m.closed?'closed':fmt(m.dP)}</td><td>${m.closed?'n/a':fmt(m.friction)}</td><td>${fmt(m.local)}</td><td>${fmt(m.vmax,2)}</td><td>${fmt(m.tauMax,2)}</td><td>${fmt(m.Q150)}</td><td>${m.closed?'n/a':fmt(h.areaAbove50Cm2,1)} cm²</td><td>${m.closed?'n/a':fmt(h.peakWm2,0)}</td></tr>`;}).join('')}</tbody></table></div>
      <p class="hint">Local losses include entry, expansion and contraction. Heat columns are the live tube-model inspiration estimate, not recorded 3-D thermal CFD. Q @150 Pa is each isolated branch, not combined nose-to-neck capacity. A closed branch has zero flow; its resistance is infinite.</p>`;
    this.drawReportChart(r);
    $('#preview-report-flow').onclick=async()=>{
      this.settings={...clone(r.settings),name:r.name};this.sync();$('#scenario-report').close();
      $('#flow-on').checked=true;this.ctx.applyPreset('Airway');await this.queue();
    };
  }
  drawReportChart(row){
    const canvas=$('#report-chart'),c=canvas.getContext('2d'),W=1000,H=270;
    const ref=this.report.rows.find(r=>r.position===row.position&&r.name==='No intervention').data;
    const maxA=Math.max(...['L','R'].flatMap(k=>[...row.data.sides[k].profile.area_mm2,...ref.sides[k].profile.area_mm2]));
    const maxS=Math.max(row.data.sides.L.length_mm,row.data.sides.R.length_mm);
    const X=s=>55+s/maxS*(W-80),Y=a=>H-40-a/maxA*(H-70);
    c.clearRect(0,0,W,H);c.font='13px system-ui';c.fillStyle='#aebbc9';c.strokeStyle='#334553';
    for(let i=0;i<=4;i++){const a=maxA*i/4,y=Y(a);c.fillText(a.toFixed(0),5,y+4);c.beginPath();c.moveTo(55,y);c.lineTo(W-25,y);c.stroke();}
    for(let s=0;s<=maxS;s+=10)c.fillText(String(s),X(s)-5,H-20);
    c.fillText('Area (mm²)',55,15);c.fillText('Distance from nostril (mm)',W/2-85,H-2);
    for(const data of [ref,row.data])for(const side of ['L','R']){
      const p=data.sides[side].profile;c.strokeStyle=side==='L'?'#ff987c':'#71c4ff';c.lineWidth=2;c.setLineDash(data===ref?[6,5]:[]);c.beginPath();
      p.s_mm.forEach((s,i)=>{i?c.lineTo(X(s),Y(p.area_mm2[i])):c.moveTo(X(s),Y(p.area_mm2[i]));});c.stroke();
    }
    c.setLineDash([]);
    canvas.onmousemove=e=>{const rect=canvas.getBoundingClientRect(),s=((e.clientX-rect.left)/rect.width*W-55)/(W-80)*maxS;
      canvas.title=['L','R'].map(k=>{const p=row.data.sides[k].profile,i=p.s_mm.reduce((best,x,j)=>Math.abs(x-s)<Math.abs(p.s_mm[best]-s)?j:best,0);return `${k}: ${p.area_mm2[i].toFixed(1)} mm² at ${p.s_mm[i]} mm`;}).join(' · ');};
  }
}
