// Small offline PDF writer: built-in Helvetica, vector charts, paginated tables.
// Every string is normalized to ASCII so byte offsets are exact without font downloads.
const ascii=s=>String(s).replace(/Δ/g,'Delta ').replace(/²/g,'2').replace(/³/g,'3').replace(/·/g,' / ').replace(/[–—−]/g,'-').replace(/→/g,' to ').normalize('NFKD').replace(/[^\x20-\x7e\n]/g,'');
const esc=s=>ascii(s).replace(/([\\()])/g,'\\$1');
export const number=(v,d=1)=>v===Infinity?'infeasible':v==null || !Number.isFinite(v)?'n/a':v.toFixed(d);
export function reportPDF(report) {
  const pages=[]; let lines=[],y=0,sectionTitle='';
  const text=(str,x,yy,size=10,color='0.12 0.17 0.23')=>lines.push(`BT /F1 ${size} Tf ${color} rg 1 0 0 1 ${x} ${yy} Tm (${esc(str)}) Tj ET`);
  const page=(title,continued=false)=>{if(!continued)sectionTitle=title;if(lines.length)pages.push(lines.join('\n'));lines=[];y=750;text('airway3d / Scenario report',40,798,11,'0.2 0.35 0.45');text(title,40,772,19);lines.push('0.7 0.77 0.8 RG 40 762 m 555 762 l S');};
  const room=n=>{if(y-n<55)page(`${sectionTitle} (continued)`,true);};
  const para=(str,size=10)=>{
    const max=Math.floor(510/(size*0.52));
    for(const source of ascii(str).split('\n')) {
      let line='';
      for(const word of source.split(' ')) {
        if((line+' '+word).length>max && line) {room(size+5);text(line,40,y,size);y-=size+5;line='';}
        line+=(line?' ':'')+word;
      }
      room(size+5);text(line,40,y,size);y-=size+5;
    }
    y-=5;
  };
  const table=(headers,rows,widths)=>{
    const row=(values,header=false)=>{room(21);let x=40;values.forEach((s,i)=>{text(s,x+3,y,header?8:9,header?'0.2 0.35 0.45':'0.12 0.17 0.23');x+=widths[i];});y-=20;lines.push(`0.85 0.88 0.9 RG 40 ${y+12} m 555 ${y+12} l S`);};
    row(headers,true);rows.forEach(r=>row(r));y-=8;
  };
  page('Intervention and posture comparison');
  para(`Created ${report.generatedAt.replace('T',' ').slice(0,19)} UTC. Model ${report.version}.`);
      para('Exploratory what-if analysis. These results are the live tube estimate, not surgical advice, patient-specific predictions, or a recorded 3-D CFD field.',12);
  para(`Observed reference: face-up (supine) CT screenshots. Slice spacing ${report.calibration.slice_mm ?? 'unknown'} mm; in-plane source pixel ${report.calibration.px_mm ?? 'unknown'} mm; reported pupil distance 68 mm.`);
  para(`PNG display windows: skull C${report.calibration.skull_window.center}/W${report.calibration.skull_window.width}; head C${report.calibration.head_window.center}/W${report.calibration.head_window.width}. Intensities clipped outside these windows cannot be recovered. Real-time calculation: live tube estimate (one mean speed per cross-section along each passage; a 1-D pipe model). Recorded 3-D CFD and clinical validation: not performed.`,10);
  para(`Prescribed inspiratory flow ${report.q} mL/s. All interventions are compared with no intervention in each position; face up is the observed CT reference.`);
  const settings=report.assumptions;
  para(`Shared response assumptions: elapsed ${number(settings.elapsed)} min; settling time ${settings.tau} min; dependent swelling L ${settings.responseL} / R ${settings.responseR} mm; immediate displacement ${settings.gravity} mm; cycle bias ${settings.cycle} mm.`);
  para('Model equations',12);
  para('Region weight w = (1 - u^2)^2 for |u| < 1, otherwise zero. Region center / half-width as a fraction of passage length: head 0.38 / 0.16; body 0.64 / 0.22; valve 0.17 / 0.12. These are geometric assumptions.');
  para('Equivalent radius r = sqrt(A/pi); new r = max(0, r + sum(clearance*w) - swelling*max(w_head,w_body)). Positive swelling is reduced by the assumed suppression percentage. Area scales with r^2, perimeter and hydraulic diameter with r.');
  para('Swelling = (vascular + signed cycle bias)*(1-exp(-elapsed/tau)) + immediate displacement. Vascular response is +amplitude on the dependent side, -0.25*amplitude on the opposite side, -0.3*amplitude upright, zero face up. Immediate displacement is +gravity dependent, -gravity opposite, zero otherwise. These are sensitivity assumptions, not measured tissue mechanics.');
  para('Pressure = Darcy friction + entry / expansion / contraction losses. Air density 1.2 kg/m3, viscosity 1.8e-5 Pa.s. Flow split equalizes nasal pressure drops; the pharynx is in series. Closed branches have zero flow. Bilateral closure makes prescribed flow infeasible.');
  for(const position of [...new Set(report.rows.map(r=>r.position))]) {
    const rows=report.rows.filter(r=>r.position===position);
    page(rows[0].positionLabel);
    para(`Matched comparison at ${report.q} mL/s. Delta resistance is relative to no intervention in this same posture.`,10);
    table(['Plan','MCA L / R mm2','Total Pa','R Pa.s/mL','Delta R'],rows.map((r,i)=>[`${i+1}. ${r.name.slice(0,22)}`,`${number(r.data.sides.L.min_area_mm2)} / ${number(r.data.sides.R.min_area_mm2)}`,number(r.metrics.total.dP),number(r.metrics.total.R,3),number(r.deltaResistance,3)]),[150,125,75,85,80]);
    for(const [i,r] of rows.entries()) {
      room(170);para(`${i+1}. ${r.name}`,12);
      const o=r.settings.operations;
      para(`Clearance mm (head/body/valve): L ${o.L.head}/${o.L.body}/${o.L.valve}; R ${o.R.head}/${o.R.body}/${o.R.valve}. Assumed swelling suppression ${r.settings.relief}%.`,9);
      table(['Passage','Q mL/s','dP Pa','Friction Pa','Local Pa','Peak m/s','Shear Pa'],['L','R','common'].map(k=>{
        const m=r.metrics[k];return [k==='common'?'Pharynx':k==='L'?'Left':'Right',number(m.Q),m.closed?'closed':number(m.dP),m.closed?'n/a':number(m.friction),number(m.local),number(m.vmax,2),number(m.tauMax,2)];
      }),[85,65,65,80,75,75,70]);
      para(['L','R'].map(k=>{const h=r.thermal?.[k]||{};return `${k}: MCA ${number(r.data.sides[k].min_area_mm2)} mm2 at ${number(r.data.sides[k].min_area_at_mm)} mm; volume ${number(r.data.sides[k].volume_cc)} cc; isolated Q@150Pa ${number(r.metrics[k].Q150)} mL/s; tube-model heat area >50 W/m2 ${number(h.areaAbove50Cm2)} cm2; peak heat flux ${number(h.peakWm2)} W/m2`}).join('. '),9);
      if(r.metrics.total.blocked)para('Nasal flow is infeasible: there is no patent nose-to-pharynx path in this assumed state.',10);
    }
  }
  page('Cross-section profiles');
  para('Left and right nasal passage area (mm2) versus distance from nostril (mm). Curves show all compared plans; colors identify plan order. These are transformed tube-model profiles, not newly segmented CT measurements.');
  for(const position of [...new Set(report.rows.map(r=>r.position))]) {
    const rows=report.rows.filter(r=>r.position===position);
    room(215);para(rows[0].positionLabel,12);
    const top=y;
    for(const [col,side] of ['L','R'].entries()) {
      const x=55+col*265,w=225,h=125,bottom=top-145;
      const maxA=Math.max(...rows.flatMap(r=>r.data.sides[side].profile.area_mm2)),maxS=rows[0].data.sides[side].length_mm;
      text(`${side==='L'?'Left':'Right'} / area mm2`,x,top,10);
      lines.push(`0.65 0.7 0.75 RG ${x} ${bottom} m ${x+w} ${bottom} l S ${x} ${bottom} m ${x} ${bottom+h} l S`);
      text('0',x-10,bottom-12,8);text(`${number(maxS,0)} mm`,x+w-35,bottom-12,8);text(number(maxA,0),x, bottom+h+3,8);
      const colors=['0.4 0.45 0.5','0.05 0.4 0.65','0.65 0.3 0.1','0.4 0.2 0.6','0.1 0.5 0.3'];
      rows.forEach((r,i)=>{
        const p=r.data.sides[side].profile;
        lines.push(`${colors[i%colors.length]} RG 1.2 w`);
        lines.push(p.s_mm.map((s,j)=>`${(x+s/maxS*w).toFixed(2)} ${(bottom+p.area_mm2[j]/maxA*h).toFixed(2)} ${j?'l':'m'}`).join(' ')+' S');
      });
    }
    y=top-175;
    const colorNames=['grey','blue','brown','purple','green'];
    para(rows.map((r,i)=>`${i+1} (${colorNames[i]}): ${r.name}`).join(' | '),8);
  }
  page('Assumptions, limits and evidence');
  report.limitations.forEach((s,i)=>para(`${i+1}. ${s}`,10));
  para('Sources establish context; none supplies patient-specific parameter values.',11);
  report.sources.forEach(s=>{para(s.title,10);para(s.url,8);});
  para('For reproducibility, use the on-screen Download data button for complete settings, per-shell profiles and unrounded metrics. Non-finite values are encoded as strings in JSON.',10);
  pages.push(lines.join('\n'));
  const objects=['','<< /Type /Catalog /Pages 2 0 R >>','', '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>'];
  const kids=[];
  pages.forEach((content,i)=>{
    content+=`\nBT /F1 8 Tf 0.4 0.45 0.5 rg 1 0 0 1 40 28 Tm (Exploratory model / ${i+1} of ${pages.length} / CT reference: supine) Tj ET`;
    const id=objects.length;kids.push(`${id} 0 R`);
    objects.push(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 3 0 R >> >> /Contents ${id+1} 0 R >>`);
    objects.push(`<< /Length ${content.length} >>\nstream\n${content}\nendstream`);
  });
  objects[2]=`<< /Type /Pages /Kids [${kids.join(' ')}] /Count ${pages.length} >>`;
  let pdf='%PDF-1.4\n',offsets=[0];
  for(let i=1;i<objects.length;i++){offsets.push(pdf.length);pdf+=`${i} 0 obj\n${objects[i]}\nendobj\n`;}
  const xref=pdf.length;
  pdf+=`xref\n0 ${objects.length}\n0000000000 65535 f \n`+offsets.slice(1).map(x=>`${String(x).padStart(10,'0')} 00000 n \n`).join('');
  pdf+=`trailer\n<< /Size ${objects.length} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return new Uint8Array([...pdf].map(c=>c.charCodeAt(0)));
}
export function download(data,name,type) {
  const url=URL.createObjectURL(new Blob([data],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();
  setTimeout(()=>URL.revokeObjectURL(url),10000);
}
