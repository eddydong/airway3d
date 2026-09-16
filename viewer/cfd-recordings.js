import {libraryKey,normalizedRequest} from './cfd-library.js';
import {POSITIONS,congestion} from './scenario-model.js';

export function recordingLabel(entry){
  const r=normalizedRequest(entry.request),s=r.settings,operations=[];
  for(const side of ['L','R'])for(const region of ['head','body','valve']){
    const gain=s.operations[side][region];if(gain)operations.push(`${side} ${region} +${gain} mm`);
  }
  if(s.relief)operations.push(`${s.relief}% swelling suppression`);
  const lattice=r.backend==='gpu-lbm',transient=r.mode==='transient';
  return `${POSITIONS[s.position]} · ${operations.join(', ')||'No intervention'} · ${lattice?`${r.pressurePa} Pa`:`${r.qMlS} mL/s`} · ${transient?`${r.periodS} s breath`:r.direction}`;
}
export function recordingDetails(entry){
  const r=normalizedRequest(entry.request),s=r.settings;
  return `${r.backend==='gpu-lbm'?'GPU lattice LES':'OpenFOAM'} · ${r.spacingMm} mm geometry${r.backend==='gpu-lbm'?` · ${r.refinement} cell${r.refinement===1?'':'s'} per voxel`:''}. `+
    `Fixed tissue response before suppression: L ${congestion(s,'L').toFixed(2)} mm / R ${congestion(s,'R').toFixed(2)} mm; positive narrows. Body position is a static assumption.`;
}
export function availableRecordings(entries){
  const seen=new Set();
  return entries.filter(e=>{
    if(e.state!=='complete'||typeof e.id!=='string'||typeof e.resultUrl!=='string'||!e.request)return false;
    const key=libraryKey(e.request);if(seen.has(key))return false;seen.add(key);return true;
  }).sort((a,b)=>recordingLabel(a).localeCompare(recordingLabel(b))||a.id.localeCompare(b.id));
}
export function selectRecording(entries,request){
  const key=libraryKey(request);
  return entries.find(e=>libraryKey(e.request)===key)||entries.find(e=>{
    const r=normalizedRequest(e.request);
    return r.backend==='gpu-lbm'&&r.settings.position==='supine'&&r.pressurePa===30&&r.periodS===4&&
      ['L','R'].every(side=>Object.values(r.settings.operations[side]).every(v=>v===0));
  })||entries[0]||null;
}

export function recordingForPosition(entries,request,position){
  const options=entries.filter(e=>normalizedRequest(e.request).settings.position===position);
  const target=normalizedRequest(request);target.settings.position=position;
  // Preserve every setting when that exact posture variant exists; otherwise
  // choose a whole recorded configuration and update the visible menu with it.
  return selectRecording(options,target);
}

export function catalogRecording(entries,job){
  return entries?.find(e=>e.id===job?.id)||null;
}
