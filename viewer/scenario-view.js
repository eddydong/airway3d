import * as THREE from 'three';
import { REGIONS } from './scenario-model.js';
import { shellFrame, warpPoint } from './flow-scenario.js';

// Schematic surface preview. The scan and its volume textures are never modified.
export class ScenarioView {
  constructor(ctx) {
    this.ctx=ctx; this.entries=[]; this.markers=[]; this.active=false;
    this.group=new THREE.Group(); ctx.scene.add(this.group);
    this.buildGuides();
  }
  buildGuides() {
    const ns='http://www.w3.org/2000/svg';
    this.leaders=document.createElementNS(ns,'svg');this.leaders.classList.add('anatomy-leaders');
    this.leaders.setAttribute('aria-hidden','true');
    const defs=document.createElementNS(ns,'defs');this.leaders.appendChild(defs);
    for(const [key,r] of Object.entries(REGIONS)) {
      const arrow=document.createElementNS(ns,'marker');arrow.id=`region-arrow-${key}`;
      for(const [k,v] of Object.entries({viewBox:'0 0 6 6',refX:6,refY:3,markerWidth:6,markerHeight:6,orient:'auto',markerUnits:'userSpaceOnUse'}))arrow.setAttribute(k,v);
      const tip=document.createElementNS(ns,'path');tip.setAttribute('d','M 0 0 L 6 3 L 0 6');tip.setAttribute('fill','none');tip.setAttribute('stroke',r.color);tip.setAttribute('stroke-width','1');
      arrow.appendChild(tip);defs.appendChild(arrow);
    }
    document.body.appendChild(this.leaders);
    this.labelDock=document.createElement('div');this.labelDock.className='region-callout-dock';this.labelDock.hidden=true;
    this.labelDock.setAttribute('aria-label','Region callouts');document.body.appendChild(this.labelDock);
    for(const side of ['L','R']) for(const [key,r] of Object.entries(REGIONS)) {
      const p=this.ctx.base.sides[side].profile;
      const i=p.s_mm.reduce((best,s,j)=>Math.abs(s/this.ctx.base.sides[side].length_mm-r.t)<Math.abs(p.s_mm[best]/this.ctx.base.sides[side].length_mm-r.t)?j:best,0);
      const center=new THREE.Vector3(...p.centers[i]);
      center.x+=(side==='L'?1:-1)*2;
      const mesh=new THREE.Mesh(new THREE.SphereGeometry(1,24,16),new THREE.MeshBasicMaterial({color:r.color,transparent:true,opacity:0.22,wireframe:true,depthWrite:false}));
      mesh.position.copy(center); mesh.scale.set(5,key==='body'?8:5,key==='body'?11:6); mesh.renderOrder=410;
      this.group.add(mesh);
      const label=document.createElement('button'); label.className='anatomy-label';
      const title=document.createElement('span');title.textContent=`${side==='L'?'Left':'Right'} ${r.short.toLowerCase()}`;
      const note=document.createElement('small');note.textContent='Approximate region';label.append(title,note);
      label.setAttribute('aria-label',`${title.textContent} · approximate · focus region`);
      label.style.setProperty('--region-color',r.color);
      label.addEventListener('click',()=>this.focus(key,side)); document.body.appendChild(label);
      const leader=document.createElementNS(ns,'g'),line=document.createElementNS(ns,'path'),point=document.createElementNS(ns,'circle');
      line.setAttribute('fill','none');line.setAttribute('stroke',r.color);line.setAttribute('stroke-width','1');line.setAttribute('marker-end',`url(#region-arrow-${key})`);
      point.setAttribute('r','2.5');point.setAttribute('fill',r.color);point.setAttribute('stroke','#0b0d12');point.setAttribute('stroke-width','1');
      leader.append(line,point);this.leaders.appendChild(leader);
      this.markers.push({key,side,mesh,label,center,leader,line,point});
    }
    this.select('head',false);
  }
  select(key,visible=true) { this.selected=key; this.guidesVisible=visible; }
  focus(key,side) {
    this.select(key,true); this.ctx.applyPreset('Airway');
    const m=this.markers.find(m=>m.key===key && m.side===side);
    this.ctx.controls.target.copy(m.center);
    this.ctx.camera.position.copy(m.center).add(new THREE.Vector3(side==='L'?65:-65,18,125).multiplyScalar(1/Math.min(1,this.ctx.camera.aspect)));
    this.ctx.controls.update();
  }
  attach() {
    if(this.entries.length) return;
    const centers=[];
    for(const side of ['L','R']) this.ctx.base.sides[side].profile.centers.forEach((c,i)=>centers.push({c,side,i,...shellFrame(this.ctx.base.sides[side].profile,i)}));
    for(const key of ['airway_L','airway_R','soft']) {
      const tissue=this.ctx.state.tissues[key]; if(!tissue) continue;
      const original=tissue.mesh.geometry, geometry=original.clone();
      const base=new Float32Array(geometry.attributes.position.array), target=new Float32Array(base);
      const index=new Uint16Array(base.length/3), influence=new Float32Array(index.length);
      const candidates=key.startsWith('airway_')?centers.map((c,i)=>({...c,index:i})).filter(c=>c.side===key.slice(-1)):centers.map((c,i)=>({...c,index:i}));
      for(let v=0;v<index.length;v++) {
        const x=base[v*3],y=base[v*3+1],z=base[v*3+2]; let best=Infinity, found=0;
        for(const p of candidates) { const d=(x-p.c[0])**2+(y-p.c[1])**2+(z-p.c[2])**2; if(d<best){best=d;found=p.index;} }
        index[v]=found;
        // Only the nearby soft envelope moves; other tissues retain the observed scan.
        influence[v]=key==='soft'?Math.max(0,1-Math.max(0,Math.sqrt(best)-7)/7):1;
      }
      tissue.mesh.geometry=geometry; tissue.depthMesh.geometry=geometry;
      this.entries.push({tissue,original,geometry,base,target,index,influence});
    }
    this.centers=centers;
  }
  update(data) {
    this.attach(); this.active=true; this.moving=true;
    for(const e of this.entries) {
      for(let v=0;v<e.index.length;v++) {
        const p=this.centers[e.index[v]], src=this.ctx.base.sides[p.side].profile;
        const ratio=Math.sqrt(data.sides[p.side].profile.area_mm2[p.i]/src.area_mm2[p.i]);
        const scale=1+(ratio-1)*e.influence[v];
        warpPoint(p,scale,e.base[v*3],e.base[v*3+1],e.base[v*3+2],e.target,v*3);
      }
    }
    this.remaining=matchMedia('(prefers-reduced-motion: reduce)').matches?0:0.3;
  }
  restore() {
    for(const e of this.entries) { e.tissue.mesh.geometry=e.original; e.tissue.depthMesh.geometry=e.original; e.geometry.dispose(); }
    this.entries=[]; this.active=false; this.moving=false;
  }
  projectedHead(w,h) {
    // Use the complete outer head even when skin is transparent or hidden.
    // A conservative envelope also avoids labels landing on deeper anatomy.
    const mesh=this.ctx.state.tissues.skin?.mesh;
    if(!mesh)return {left:0,right:w,top:0,bottom:h};
    if(!mesh.geometry.boundingBox)mesh.geometry.computeBoundingBox();
    const b=mesh.geometry.boundingBox,out={left:Infinity,right:-Infinity,top:Infinity,bottom:-Infinity};
    mesh.updateWorldMatrix(true,false);
    for(let i=0;i<8;i++) {
      const p=new THREE.Vector3(i&1?b.max.x:b.min.x,i&2?b.max.y:b.min.y,i&4?b.max.z:b.min.z).applyMatrix4(mesh.matrixWorld).project(this.ctx.camera);
      if(p.z< -1||p.z>1)return {left:0,right:w,top:0,bottom:h};
      const x=(p.x+1)*w/2,y=(1-p.y)*h/2;
      out.left=Math.min(out.left,x);out.right=Math.max(out.right,x);out.top=Math.min(out.top,y);out.bottom=Math.max(out.bottom,y);
    }
    return out;
  }
  layoutGuides() {
    const w=innerWidth,h=innerHeight,compact=w<=760;
    const labOpen=document.body.classList.contains('lab-open');
    // All panels sit in the dock on the left; the scene starts after the
    // right-most open one. Labels dock into the lab when it is open, else the
    // first open panel.
    const open=['#scenario-lab','#left','#right'].map(s=>document.querySelector(s)).filter(p=>!p.classList.contains('stowed'));
    const panel=open[0]||null,dockRight=open.reduce((r,p)=>Math.max(r,p.getBoundingClientRect().right),document.querySelector('#rail').getBoundingClientRect().right);
    const space={left:compact?54:dockRight+10,right:w-10,top:12,bottom:h-12};
    // On a phone an open panel covers the scene; suppress scene annotations until it is stowed.
    const available=this.guidesVisible&&(!compact||!panel);
    this.leaders.setAttribute('viewBox',`0 0 ${w} ${h}`);
    // Leaders stay in the scene, so they never cross controls in either sidebar.
    const leaderLeft=compact?space.left:dockRight-12,leaderBottom=space.bottom;
    this.leaders.style.clipPath=`inset(${space.top}px ${w-space.right}px ${h-leaderBottom}px ${leaderLeft}px)`;
    const selected=[];
    for(const m of this.markers) {
      const show=available&&m.key===this.selected;
      m.mesh.visible=show;
      m.mesh.material.clippingPlanes=Object.values(this.ctx.state.tissues)[0]?.mat.clippingPlanes || [];
      const p=m.center.clone().project(this.ctx.camera);
      const x=(p.x+1)*w/2,y=(1-p.y)*h/2;
      const visible=show&&p.z>=-1&&p.z<=1&&x>=space.left&&x<=space.right&&y>=space.top&&y<=space.bottom;
      m.label.hidden=!visible;m.leader.style.display=visible?'':'none';
      if(visible)selected.push({m,x,y});
    }
    if(!selected.length){this.labelDock.hidden=true;return;}
    const head=this.projectedHead(w,h),width=compact?136:168,height=46,gap=14,placed=[];
    const overlaps=(a,b)=>a.left<b.right+gap&&a.right>b.left-gap&&a.top<b.bottom+gap&&a.bottom>b.top-gap;
    const clamp=(v,min,max)=>Math.max(min,Math.min(max,v));
    selected.sort((a,b)=>a.x-b.x);
    let dock=false;
    for(const item of selected) {
      const {x,y}=item,vertical=clamp(y-height/2,space.top,space.bottom-height),horizontal=clamp(x-width/2,space.left,space.right-width);
      const left={left:space.left,top:vertical},right={left:space.right-width,top:vertical};
      const candidates=[...(x<(head.left+head.right)/2?[left,right]:[right,left]),{left:horizontal,top:head.top-height-gap},{left:horizontal,top:head.bottom+gap}];
      const spot=candidates.map(p=>({...p,right:p.left+width,bottom:p.top+height})).find(p=>p.left>=space.left&&p.right<=space.right&&p.top>=space.top&&p.bottom<=space.bottom&&!overlaps(p,head)&&!placed.some(other=>overlaps(p,other)));
      if(!spot){dock=true;break;}item.spot=spot;placed.push(spot);
    }
    if(dock&&!panel){for(const {m} of selected){m.label.hidden=true;m.leader.style.display='none';}this.labelDock.hidden=true;return;}
    this.labelDock.hidden=!dock;
    if(dock) {
      // When zoom leaves no clear margin, keep labels in a dedicated panel rail
      // rather than moving text onto the head or changing the user's camera.
      const before=labOpen&&!compact?panel.querySelector('.lab-controls'):panel.firstChild;
      if(this.labelDock.parentNode!==panel||this.dockCompact!==compact)panel.insertBefore(this.labelDock,before);
      this.dockCompact=compact;
      this.labelDock.title='Approximate region locations. Select a label to focus.';
      selected.sort((a,b)=>compact?a.x-b.x:a.y-b.y);
      const order=selected.map(({m})=>m.key+m.side).join('-');
      if(order!==this.dockOrder){for(const {m} of selected)this.labelDock.appendChild(m.label);this.dockOrder=order;}
    }
    for(const {m,x,y,spot} of selected) {
      const parent=dock?this.labelDock:document.body;
      if(m.label.parentNode!==parent)parent.appendChild(m.label);
      m.label.classList.toggle('docked',dock);
      let sx,sy,ex,ey;
      if(dock) {
        m.label.style.removeProperty('left');m.label.style.removeProperty('top');
        const rect=m.label.getBoundingClientRect();
        if(compact){sx=rect.left+rect.width/2;sy=rect.top;ex=sx;ey=Math.min(space.bottom-12,y+20);}
        else{sx=rect.right-10;sy=rect.top+rect.height/2;ex=space.left+12;ey=sy;}
      } else {
        m.label.style.left=`${spot.left}px`;m.label.style.top=`${spot.top}px`;
        sx=clamp(x,spot.left,spot.right);sy=clamp(y,spot.top,spot.bottom);
        const horizontal=x<spot.left||x>spot.right;
        ex=horizontal?sx+(x>sx?12:-12):sx;ey=horizontal?sy:sy+(y>sy?12:-12);
      }
      const distance=Math.hypot(x-ex,y-ey)||1;
      // Leave a small separation between the open arrowhead and its target dot.
      const endX=x-(x-ex)/distance*5,endY=y-(y-ey)/distance*5;
      m.line.setAttribute('d',`M ${sx} ${sy} L ${ex} ${ey} L ${endX} ${endY}`);
      m.point.setAttribute('cx',x);m.point.setAttribute('cy',y);
      m.label.dataset.placement=dock?'panel':'outside-head';
    }
  }
  step(dt) {
    if(this.moving) {
      this.remaining-=dt; const last=this.remaining<=0;
      const blend=last?1:1-Math.exp(-dt*18);
      for(const e of this.entries) {
        const a=e.geometry.attributes.position.array;
        for(let i=0;i<a.length;i++) a[i]+=(e.target[i]-a[i])*blend;
        e.geometry.attributes.position.needsUpdate=true;
        if(last) {e.geometry.computeVertexNormals();e.geometry.computeBoundingSphere();}
      }
      if(last)this.moving=false;
    }
    this.layoutGuides();
  }
}
