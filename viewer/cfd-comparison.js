import { clone, noIntervention, POSITIONS } from './scenario-model.js';
import { matchingResult } from './cfd-lab.js';
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=v=>Number.isFinite(v)?v.toFixed(2):'—';

// Only interventions vary between arms; all arms share physiology, rate,
// numerical grid and, for transient runs, the imposed breathing waveform.
export function comparisonRequests(request,saved=[]){
  const s=request.settings,seen=new Set();
  const plans=[{name:'Pre · no intervention',settings:noIntervention(s)},
    {name:s.name?.trim()||'Current intervention',settings:s},...saved];
  const unique=plans.filter(p=>{
    const id=JSON.stringify([p.settings.operations,p.settings.relief]);
    if(seen.has(id))return false;seen.add(id);return true;
  });
  return Object.entries(POSITIONS).flatMap(([position,label])=>unique.map((p,i)=>({
    name:p.name,positionLabel:label,baseline:i===0,
    request:{...clone(request),settings:{...clone(s),position,operations:clone(p.settings.operations),relief:p.settings.relief}}
  })));
}
export const pressureDriven=result=>result?.solver?.backend==='gpu-lbm';
export function comparisonReading(result){
  if(result.solver.temporal!=='transient')return {pressure:result.pressureDropPa,flow:result.flowMlS};
  // Compare the same phase in the last recorded cycle, not each run's
  // unrelated maximum or the startup cycle. No periodicity claim is made.
  const t=result.solver.durationS-result.solver.periodS*.75;
  const frame=result.field.frames.reduce((best,f)=>Math.abs(f.timeS-t)<Math.abs(best.timeS-t)?f:best);
  return {pressure:frame.pressureDropPa,flow:frame.flowMlS,timeS:frame.timeS};
}
export class CFDComparison {
  constructor(cfd){
    this.cfd=cfd;this.dialog=document.createElement('dialog');this.dialog.id='cfd-comparison';
    document.body.append(this.dialog);this.dialog.addEventListener('close',()=>{clearTimeout(this.timer);this.generation++;});
    this.generation=0;
  }
  open(){
    clearTimeout(this.timer);this.generation++;
    this.rows=comparisonRequests(this.cfd.request(),this.cfd.lab.saved).map(r=>({...r,job:null,result:null}));
    const transient=this.rows[0].request.mode==='transient',lattice=this.rows[0].request.backend==='gpu-lbm';
    this.lattice=lattice;
    this.dialog.innerHTML=`<header><div><h1>Recorded 3-D CFD comparison</h1><p>Pre-intervention reference and proposed plans in each body position</p></div><button data-close>Close</button></header>
      <p class="model-note">Each row uses its own rebuilt 3-D airway and ${lattice?'GPU lattice recording':'OpenFOAM job'}. Post-intervention geometry and posture responses are hypotheses based on the face-up CT screenshots. This is not an observed post-op scan or a prediction of surgical benefit.</p>
      <p>${lattice?`Two recorded breathing cycles per geometry at the same nominal throat pressure (${this.rows[0].request.pressurePa} Pa). Rows compare the flow they achieve at peak inspiration of the last cycle, the physically comparable quantity for a pressure-driven run; periodicity is unverified.`
        :transient?'Two recorded breathing cycles per geometry. Values compare peak inspiration in the last cycle; periodicity is unverified.':'Steady fields at the same rate and direction.'} Unaccepted or missing results show no comparison readings.</p>
      <div class="lab-actions"><button data-export>Download CFD comparison data</button></div>
      <p class="hint">Only saved recordings are available. Missing scenarios are prepared offline; this viewer never submits jobs.</p>
      <p data-status role="status">Checking saved results…</p><div class="report-table-wrap"><table><thead><tr><th>Position / plan</th><th>Solver status</th><th>${transient?'Peak-inspiration':'Steady'} ΔP (Pa)</th>${lattice?'<th>Total flow (mL/s)</th><th>Flow vs pre (mL/s)</th>':'<th>ΔP vs pre (Pa)</th>'}<th>Flow L / R (mL/s)</th><th>Actions</th></tr></thead><tbody></tbody></table></div>`;
    this.dialog.querySelector('[data-close]').onclick=()=>this.dialog.close();
    this.dialog.querySelector('[data-export]').onclick=()=>{
      const data={source:'CT screenshots; prescribed intervention/posture geometry',method:'3-D viscous CFD',generatedAt:new Date().toISOString(),
        rows:this.rows.map(r=>({...r,accepted:r.result&&matchingResult(r.result,r.job.id),reading:r.result?comparisonReading(r.result):null}))};
      const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
      a.download='airway-cfd-comparison.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
    };
    this.dialog.showModal();this.render();this.refresh(this.generation);
  }
  async refresh(generation){
    try{
      await Promise.all(this.rows.map(async row=>{
        const job=await this.cfd.library.lookup(row.request);
        if(generation!==this.generation)return;
        row.job=job;
        if(job.state==='complete'&&!row.result){
          const result=await this.cfd.api(job.resultUrl);
          if(generation===this.generation&&matchingResult(result,job.id))row.result=result;
        }
      }));
      if(generation!==this.generation)return;
      this.dialog.querySelector('[data-status]').textContent=`${this.rows.filter(r=>r.result).length}/${this.rows.length} results passed numerical checks.`;
      this.render();
    }catch(e){if(generation===this.generation)this.dialog.querySelector('[data-status]').textContent=e.message;}
  }
  render(){
    const tbody=this.dialog.querySelector('tbody');tbody.replaceChildren();
    this.rows.forEach((row,i)=>{
      const reading=row.result?comparisonReading(row.result):null;
      const base=this.rows.find(r=>r.baseline&&r.request.settings.position===row.request.settings.position)?.result;
      const ref=base?comparisonReading(base):null;
      const total=r=>r?r.flow.L+r.flow.R:NaN;
      const delta=reading&&ref?(this.lattice?total(reading)-total(ref):reading.pressure-ref.pressure):null;
      const tr=document.createElement('tr');
      tr.innerHTML=`<td>${escape(row.positionLabel)}<br><strong>${escape(row.name)}</strong></td><td>${escape(row.job?.message||'Checking…')}</td><td>${fmt(reading?.pressure)}</td>${this.lattice?`<td>${fmt(total(reading))}</td><td>${fmt(delta)}</td>`:`<td>${fmt(delta)}</td>`}<td>${fmt(reading?.flow.L)} / ${fmt(reading?.flow.R)}</td><td></td>`;
      const actions=tr.lastElementChild,view=document.createElement('button');view.textContent='View geometry / flow';
      view.onclick=()=>{this.dialog.close();this.cfd.lab.settings=clone(row.request.settings);this.cfd.lab.sync();this.cfd.changed();};actions.append(view);
      view.disabled=!row.result;tbody.append(tr);
    });
  }
}
