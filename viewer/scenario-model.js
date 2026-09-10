import { evaluate } from './hydraulics.js';
import { evaluateHeatFlux } from './heatflux.js';

export const MODEL_VERSION = 'airway-what-if-2';
export const CT_POSTURE = 'Face up · CT reference';
export const POSITIONS = {supine:CT_POSTURE,left:'Left side down',right:'Right side down',upright:'Upright'};
export const REGIONS = {
  head:{ label:'Inferior turbinate head region', short:'Turbinate head', t:0.38, width:0.16, color:'#f2ad73', help:'Anterior lateral-wall region near the turbinate head. Location is inferred from passage geometry; inspect the CT before interpreting it.' },
  body:{ label:'Inferior turbinate body region', short:'Turbinate body', t:0.64, width:0.22, color:'#c7a3ef', help:'Longer lateral-wall region behind the head. Clearance is an assumed airway gain, not a segmented bone cut or a tissue-removal volume.' },
  valve:{ label:'Anterior nasal valve region', short:'Valve support', t:0.17, width:0.12, color:'#86d6c1', help:'Anterior passage envelope for a widening/support what-if. This is not a recommendation to remove valve tissue.' },
};
export function defaults() {
  return { name:'Untitled intervention', position:'supine', elapsed:10, tau:5, responseL:0.6, responseR:0.6, cycle:0, gravity:0.15, relief:0,
    operations:{ L:{head:0,body:0,valve:0}, R:{head:0,body:0,valve:0} } };
}
export const clone = x => JSON.parse(JSON.stringify(x));
export function weight(t,key) {
  const r=REGIONS[key], u=Math.abs((t-r.t)/r.width);
  return u>=1 ? 0 : (1-u*u)**2;
}
export function congestion(s,side,position=s.position) {
  const dependent=position===(side==='L'?'left':'right');
  const opposite=position===(side==='L'?'right':'left');
  const amplitude=side==='L'?s.responseL:s.responseR;
  const vascular=dependent?amplitude:opposite?-0.25*amplitude:position==='upright'?-0.3*amplitude:0;
  const settled=1-Math.exp(-s.elapsed/Math.max(0.1,s.tau));
  return (vascular+s.cycle*(side==='L'?1:-1))*settled+(dependent?s.gravity:opposite?-s.gravity:0);
}
export function radialChange(s,side,t,position=s.position) {
  const op=s.operations[side];
  const swelling=congestion(s,side,position);
  return Object.keys(REGIONS).reduce((sum,k)=>sum+op[k]*weight(t,k),0)
    -swelling*Math.max(weight(t,'head'),weight(t,'body'))*(swelling>0?1-s.relief/100:1);
}
export function hasPreview(s) {
  return ['L','R'].some(side=>Object.values(s.operations[side]).some(v=>v!==0)||Math.abs(congestion(s,side))>1e-8);
}
export function scenarioData(base,settings,position=settings.position) {
  const d=structuredClone(base);
  d.dataset='scenario';
  d.scenario={version:MODEL_VERSION,position,settings:clone(settings)};
  for (const side of ['L','R']) {
    const src=base.sides[side], out=d.sides[side], p=out.profile;
    let oldVolume=0, newVolume=0;
    const valid=[];
    for (let i=0;i<p.area_mm2.length;i++) {
      const t=p.s_mm[i]/src.length_mm;
      const r=Math.sqrt(src.profile.area_mm2[i]/Math.PI);
      const next=Math.max(0,r+radialChange(settings,side,t,position));
      const scale=r?next/r:0;
      p.area_mm2[i]=src.profile.area_mm2[i]*scale*scale;
      p.perimeter_mm[i]=src.profile.perimeter_mm[i]*scale;
      p.hyd_diam_mm[i]=src.profile.hyd_diam_mm[i]*scale;
      const ds=i ? p.s_mm[i]-p.s_mm[i-1] : p.s_mm[1]-p.s_mm[0];
      oldVolume+=src.profile.area_mm2[i]*ds; newVolume+=p.area_mm2[i]*ds;
      if(p.s_mm[i]>=3 && p.s_mm[i]<=p.s_mm.at(-1)-4) valid.push(i);
    }
    const mi=valid.reduce((a,b)=>p.area_mm2[a]<=p.area_mm2[b]?a:b);
    out.min_area_index=mi; out.min_area_mm2=p.area_mm2[mi]; out.min_area_at_mm=p.s_mm[mi];
    // Keep the wall-avoiding marker when the narrowest shell has not moved.
    out.min_area_pos=mi===src.min_area_index?src.min_area_pos:p.centers[mi];
    out.mean_area_mm2=valid.reduce((a,i)=>a+p.area_mm2[i],0)/valid.length;
    out.volume_cc=src.volume_cc*newVolume/oldVolume;
  }
  d.total.volume_cc=Object.values(d.sides).reduce((a,s)=>a+s.volume_cc,0);
  return d;
}
export function noIntervention(settings) {
  const s=clone(settings); s.operations=defaults().operations; s.relief=0; return s;
}
export function buildReport(base,settings,plans,q,meta={}) {
  const seen=new Set();
  const variants=[{name:'No intervention',settings:noIntervention(settings)},...plans].filter(plan=>{
    const signature=JSON.stringify([plan.settings.operations,plan.settings.relief]);
    if(seen.has(signature))return false;seen.add(signature);return true;
  });
  const rows=[];
  for(const [position,label] of Object.entries(POSITIONS)) {
    const baseline=scenarioData(base,noIntervention(settings),position), ref=evaluate(baseline,q);
    for(const plan of variants) {
      // All arms share the current posture/physiology assumptions for a matched comparison.
      const s={...clone(settings),position,operations:clone(plan.settings.operations),relief:plan.settings.relief};
      const data=scenarioData(base,s,position), metrics=evaluate(data,q);
      rows.push({position,positionLabel:label,name:plan.name,settings:s,data,metrics,thermal:evaluateHeatFlux(data,metrics),
        deltaResistance:Number.isFinite(ref.total.R)&&Number.isFinite(metrics.total.R)?metrics.total.R-ref.total.R:null});
    }
  }
  return {version:MODEL_VERSION,generatedAt:new Date().toISOString(),q,scanPosition:'supine',
    calibration:{slice_mm:meta.slice_mm,px_mm:meta.px_mm,ipd_mm:68,skull_window:meta.window || {center:400,width:1500},head_window:meta.soft_window || {center:40,width:350},source:'PNG screenshots only'},
    simulation:{method:'live tube estimate: quasi-steady Darcy + local losses along each passage (1-D pipe model)',recalculated:true,viscous3D:false,clinicalValidation:'not performed',geometryValidation:'heuristic segmentation; no clinician review recorded',posture:'assumed response relative to face-up CT',particles:'stored streamlines warped by profile radius, speed scaled by branch flow / area; no new 3-D Navier–Stokes field'},
    congestionProxy:{quantity:'isolated nasal resistance',pressurePa:150,unit:'Pa·s/mL',field:'R150',reference:'face-up CT, same side',predictsPatientSymptoms:false},
    thermalProxy:{quantity:'mucosal heat flux',method:'tube-model local convection and bulk-air warming (1-D pipe heat transfer)',thresholdWm2:50,units:['W/m2','cm2','W'],clinicalValidation:false},
    assumptions:clone(settings),rows,
    sources:[
      {title:'Posture and nasal patency (1984)',url:'https://pubmed.ncbi.nlm.nih.gov/6703492/'},
      {title:'Simulating the nasal cycle with computational fluid dynamics (2015)',url:'https://pmc.ncbi.nlm.nih.gov/articles/PMC4402730/'},
      {title:'Predicting post-surgery nasal physiology: challenges and limitations (2015)',url:'https://pmc.ncbi.nlm.nih.gov/articles/PMC4405156/'},
    ],
    limitations:[
      'Exploratory sensitivity model, not a surgical recommendation or a validated patient-specific prediction.',
      'The only observed anatomy is the face-up CT. Other positions use adjustable assumptions. The reported left-side blockage preset is a sensitivity test, not a measured response or a fit to a second scan.',
      'Screenshot CT has approximate HU and IPD-calibrated scale. Turbinate/valve regions are geometric envelopes, not verified bone or mucosa segmentations.',
      'Clearance changes equivalent radii with smooth region weights. Cross-section shape is assumed similar; adjacent-wall deformation is illustrative and may intersect tissue. No tissue mechanics, blood circulation, healing, neural response or fluid-structure interaction is solved.',
      'Posture response uses assumed radial displacement plus exponential vascular settling. The amplitudes, time constant, cycle bias and suppression are not inferred from CT or validated for this patient.',
      'Pressure uses a live tube estimate: Darcy friction and local losses along each passage (one mean speed per cross-section; a 1-D pipe model), with two parallel nasal passages plus the shared pharynx. Minimum area excludes incomplete inlet/outlet shells.',
      'Closed passages carry zero flow; bilateral closure makes prescribed nasal flow infeasible. Near-closure pressures are extrapolations. No mouth-breathing compensation or pressure-limited ventilation is modeled.',
      'Live particles reuse stored potential-flow streamlines, deform their cross-sections, and scale speed by the peak tube-model branch-flow ratio divided by local area ratio. Nostril seeding uses that split. The breathing waveform scales/reverses this approximation; new 3-D pressure, jets, recirculation and flow rerouting are not solved. The 3-D picture is a drawing of the tube estimate. CT slices/volume stay at the saved anatomy.',
      'Mucosal heat flux is a tube-model local convection estimate using assumed wall temperature 32.6 C, inspired air 20 C, air properties and the transformed passage profiles. It is not recorded 3-D CFD heat transfer; wall temperature, humidity, mucosal properties and 3-D near-wall gradients are not measured.',
      'The @150 Pa metric is isolated branch capacity, not combined nose-to-neck capacity. Wall shear is a tube-model proxy; humidification and mucosal cooling are not modeled. Lower resistance does not establish clinical benefit.',
    ]};
}
