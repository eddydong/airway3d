// Particle visualisation of the airway velocity field (potential flow solved offline).
// flow.json: { dims:[X,Y,Z], box_min, box_size, scale (m/s per unit), q_ref_mL_s, max_speed_m_s,
//              inlets:[[x,y,z,w],...], outlets:[[x,y,z,w],...],
//              nostrils:[{centre:[x,y,z], out:[dx,dy,dz], frac, radius_mm}, ...] }
// flow.bin : int16 x 3 per voxel, order [Z][Y][X], velocity in world axes at q_ref (inhalation).
//
// The field is linear in the flow rate, so a breathing cycle is just a time-varying multiplier
// (negative while exhaling).  Every particle is in one of four modes:
//   IN     inside the airway, advected by the field (sub-stepped, slides along the walls)
//   ROOM   room air near a nostril: drawn in by a point-sink model, pushed away when exhaling
//   PLUME  exhaled air: leaves the nostril along its exit velocity, slows, spreads and fades
//   DEAD   parked in the pool; seeded again at a rate proportional to |Q(t)| so the particle
//          concentration stays uniform from the nostrils to the trachea (flux conservation)
import * as THREE from 'three';
import { ScenarioTransport } from './flow-scenario.js';
import { restingBreath } from './breathing.js';

// colour shared by points and trails: a speed ramp (uRamp) or the nostril the air belongs to
const COLOR_GLSL = /* glsl */`
uniform float maxSpeed;
uniform int uRamp;       // 0 spectral, 1 heat, 2 ice, 3 smoke (grey->white), 4 white
uniform int uColorMode;  // 0 by speed, 1 by side (L / R / pharynx), 2 by mucosal heat flux
uniform float uThermalMax;
uniform vec3 uColL, uColR, uColC;
vec3 stops(vec3 a, vec3 b, vec3 c, vec3 d, float t) {
  return t < 0.33 ? mix(a, b, t / 0.33) : (t < 0.66 ? mix(b, c, (t - 0.33) / 0.33) : mix(c, d, (t - 0.66) / 0.34));
}
vec3 ramp(float t) {
  if (uRamp == 1) return stops(vec3(0.10, 0.02, 0.18), vec3(0.70, 0.08, 0.30), vec3(1.00, 0.50, 0.10), vec3(1.00, 0.95, 0.70), t);
  if (uRamp == 2) return stops(vec3(0.08, 0.20, 0.75), vec3(0.25, 0.60, 1.00), vec3(0.70, 0.90, 1.00), vec3(1.00, 1.00, 1.00), t);
  if (uRamp == 3) return vec3(mix(0.50, 1.00, t));
  if (uRamp == 4) return vec3(1.0);
  if (uRamp == 5) return vec3(0.86, 0.91, 1.00);  // vapour: pale, slightly cold white
  return stops(vec3(0.10, 0.25, 0.95), vec3(0.20, 0.85, 1.00), vec3(1.00, 0.90, 0.30), vec3(1.00, 0.35, 0.15), t);
}
vec3 thermalRamp(float t) {
  if (t < 0.25) return mix(vec3(0.03, 0.18, 0.95), vec3(0.00, 0.85, 0.95), t / 0.25);
  if (t < 0.50) return mix(vec3(0.00, 0.85, 0.95), vec3(0.05, 0.78, 0.25), (t - 0.25) / 0.25);
  if (t < 0.75) return mix(vec3(0.05, 0.78, 0.25), vec3(1.00, 0.86, 0.05), (t - 0.50) / 0.25);
  return mix(vec3(1.00, 0.86, 0.05), vec3(0.95, 0.05, 0.03), (t - 0.75) / 0.25);
}
vec3 particleColor(float speed, float side, float thermal) {
  float t = clamp(speed / maxSpeed, 0.0, 1.0);
  if (uColorMode == 1) {
    vec3 c = side < 0.5 ? uColL : (side < 1.5 ? uColR : uColC);
    return c * (0.55 + 0.6 * t);
  }
  if (uColorMode == 2) return thermalRamp(clamp(thermal / max(uThermalMax, 1.0), 0.0, 1.0));
  return ramp(t);
}
`;
export const PVERT = /* glsl */`
#include <clipping_planes_pars_vertex>
attribute float speed;
attribute float life;
attribute float side;
attribute float thermal;
attribute float puff;   // plume progress 0..1 (grows the sprite); -1 = room air
attribute float seed;   // per-particle constant for the puff texture
varying float vSpeed;
varying float vLife;
varying float vSide;
varying float vThermal;
varying float vPuff;
varying float vSeed;
uniform float sizePx;
uniform float uGrow;    // sprite size multiplier reached at the end of the plume
uniform int uPass;      // 0: air inside the airway (puff == 0), 1: room air + exhaled plume (puff != 0)
void main() {
  // the two passes are drawn at different points of the frame: inside air before the tissues
  // around the airway (so a translucent skin dims it), outside air after them (it is in front
  // of the face)
  float l = ((uPass == 1) == (puff != 0.0)) ? life : 0.0;
  vSpeed = speed; vLife = l; vSide = side; vThermal = thermal; vPuff = puff; vSeed = seed;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  float grow = 1.0 + uGrow * sqrt(max(puff, 0.0));  // puffs expand quickly after leaving the nostril
  gl_PointSize = l > 0.0 ? sizePx * grow * (300.0 / max(-mvPosition.z, 1.0)) : 0.0;
  #include <clipping_planes_vertex>
}
`;
export const PFRAG = /* glsl */`
#include <clipping_planes_pars_fragment>
varying float vSpeed;
varying float vLife;
varying float vSide;
varying float vThermal;
varying float vPuff;
varying float vSeed;
uniform float uCore;        // falloff sharpness: <1 soft halo, >1 tight core
uniform float uAlpha;
uniform float uGauss;       // 1 = gaussian sprite (smoke), 0 = smoothstep disc
uniform float uNoise;       // 1 = cloudy puff texture
uniform float uPlumeAlpha;  // alpha multiplier for exhaled (plume) particles
uniform float uRoomAlpha;   // alpha multiplier for room air being inhaled
${COLOR_GLSL}
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}
void main() {
  #include <clipping_planes_fragment>
  if (vLife <= 0.0) discard;
  vec2 p = gl_PointCoord - 0.5;
  float r = length(p);
  if (r > 0.5) discard;
  float disc = pow(smoothstep(0.5, 0.0, r), uCore);
  float gauss = exp(-r * r * mix(14.0, 7.0, uNoise)) * smoothstep(0.5, 0.4, r);  // cloudy puffs are wider
  float glow = mix(disc, gauss, uGauss);
  if (uNoise > 0.0) {
    // two octaves of value noise, offset per particle, so puffs read as little clouds
    vec2 q = gl_PointCoord * 3.0 + vSeed * 37.0;
    float nz = 0.65 * vnoise(q) + 0.35 * vnoise(q * 2.3 + 11.0);
    glow *= mix(1.0, 0.55 + 0.75 * nz, uNoise);
  }
  vec3 col = particleColor(vSpeed, vSide, vThermal);
  // vapour: lit from above, a touch of blue in the shadowed lower half
  if (uRamp == 5) col *= 0.82 + 0.4 * (0.5 - gl_PointCoord.y);
  float a = glow * vLife * uAlpha * (vPuff > 0.0 ? uPlumeAlpha : (vPuff < 0.0 ? uRoomAlpha : 1.0));
  gl_FragColor = vec4(col * mix(0.5 + 0.5 * glow, 1.0, uGauss), min(a, 1.0));
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;
export const LVERT = /* glsl */`
#include <clipping_planes_pars_vertex>
attribute float speed;
attribute float alpha;
attribute float side;
attribute float thermal;
varying float vSpeed;
varying float vAlpha;
varying float vSide;
varying float vThermal;
void main() {
  vSpeed = speed; vAlpha = alpha; vSide = side; vThermal = thermal;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}
`;
export const LFRAG = /* glsl */`
#include <clipping_planes_pars_fragment>
varying float vSpeed;
varying float vAlpha;
varying float vSide;
varying float vThermal;
uniform float uTrailAlpha;
${COLOR_GLSL}
void main() {
  #include <clipping_planes_fragment>
  if (vAlpha <= 0.0) discard;
  gl_FragColor = vec4(particleColor(vSpeed, vSide, vThermal), vAlpha * uTrailAlpha);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

// Particle looks. size in px at 300 mm, core = falloff exponent, alpha = point alpha, trail = trail
// alpha, additive = light-emitting (additive) vs. occluding (normal) blending, gauss = smoke sprite,
// noise = cloudy texture.  Optional plume overrides: plumeLen (mm), plumeTime (s), decayFloor (speed
// fraction kept at the end), spread (jet widening), buoy (m/s of upward drift), grow (sprite growth),
// plumeAlpha / roomAlpha (alpha multipliers outside the airway), fadePow (fade curve).
export const PARTICLE_STYLES = {
  glow:    { label: 'Glow (spectral)',  ramp: 0, size: 2.2, core: 1.0, alpha: 0.75, trail: 0.35, additive: true,  gauss: 0 },
  heat:    { label: 'Ember (speed, not temperature)', ramp: 1, size: 2.0, core: 1.3, alpha: 0.85, trail: 0.40, additive: true,  gauss: 0 },
  ice:     { label: 'Ice',              ramp: 2, size: 1.9, core: 1.5, alpha: 0.85, trail: 0.35, additive: true,  gauss: 0 },
  smoke:   { label: 'Smoke',            ramp: 3, size: 10.0, core: 0.6, alpha: 0.28, trail: 0.08, additive: false, gauss: 1 },
  vapour:  { label: 'Vapour (winter breath)', ramp: 5, size: 9.0, core: 0.6, alpha: 0.24, trail: 0.03, additive: false, gauss: 1, noise: 1,
             plumeLen: 140, plumeTime: 2.6, decayFloor: 0.12, spread: 0.32, buoy: 0.3, grow: 5.0, plumeAlpha: 2.8, roomAlpha: 0.25, fadePow: 0.8 },
  streaks: { label: 'Streaklines',      ramp: 4, size: 0.9, core: 2.0, alpha: 0.35, trail: 0.30, additive: true,  gauss: 0 },
  tracer:  { label: 'Tracer dots',      ramp: 4, size: 2.6, core: 3.0, alpha: 1.00, trail: 0.20, additive: false, gauss: 0 },
};

const HIST = 8; // trail history length
const IN = 0, ROOM = 1, PLUME = 2, DEAD = 3;
const PLUME_LEN = 28;   // mm the exhaled plume is followed before it fades out (default look)
const PLUME_TIME = 1.2; // s (animation) after which a plume particle is recycled regardless
const ROOM_SHELL = 9;   // mm of room air seeded around each nostril
const FILL = 0.55;      // target share of the particle budget inside the airway

export class FlowParticles {
  constructor(scene, meta, buf) {
    this.scene = scene;
    this.meta = meta;
    this.dims = meta.dims;
    this.boxMin = new THREE.Vector3(...meta.box_min);
    this.boxSize = new THREE.Vector3(...meta.box_size);
    this.vox = new THREE.Vector3(this.boxSize.x / this.dims[0], this.boxSize.y / this.dims[1], this.boxSize.z / this.dims[2]);
    this.vel = new Int16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
    this.scale = meta.scale; // m/s per unit at q_ref
    this.qRef = meta.q_ref_mL_s;
    this.q = this.qRef;      // peak flow rate (mL/s) - the UI slider
    this.qNow = this.q;      // instantaneous, signed (negative = exhaling)
    const v99 = meta.color_speed_m_s || meta.max_speed_m_s || 10;  // 99th-percentile speed at q_ref
    this.maxSpeed = 0.6 * v99;    // top of the colour ramp: the nasal valve saturates, the meatuses use the mid range
    this.clampSpeed = 1.5 * v99;  // single-voxel hot spots are numerical: cap them
    // occupancy (fluid voxels) + list for uniform seeding; airway volume for the seeding rate
    const [d0, d1, d2] = this.dims, nv = d0 * d1 * d2;
    this.occ = new Uint8Array(nv);
    const fluid = [];
    for (let i = 0; i < nv; i++) { const o = i * 3; if (this.vel[o] !== 0 || this.vel[o + 1] !== 0 || this.vel[o + 2] !== 0) { this.occ[i] = 1; fluid.push(i); } }
    this.fluid = Int32Array.from(fluid);
    this.volMm3 = fluid.length * this.vox.x * this.vox.y * this.vox.z;
    this.outletY = this.boxMin.y + 1.5 * this.vox.y; // below this plane a particle has left through the trachea
    this.maxStep = 0.4 * Math.min(this.vox.x, this.vox.y, this.vox.z); // mm per sub-step
    // nostrils (room-air seeding / plume) and outlet voxels (seeding while exhaling)
    this.nostrils = (meta.nostrils || []).map((n, k) => ({ c: new THREE.Vector3(...n.centre), out: new THREE.Vector3(...n.out).normalize(), frac: n.frac, r: n.radius_mm || 5,
      side: n.name === 'R' ? 1 : (n.name === 'L' ? 0 : Math.min(k, 1)) }));
    // side colouring: which side of the midline is the left nostril; the choanal plane separates nose from pharynx
    this.midX = this.nostrils.length ? this.nostrils.reduce((s, n) => s + n.c.x, 0) / this.nostrils.length : 0;
    const nl = this.nostrils.find((n) => n.side === 0);
    this.leftIsPosX = nl ? nl.c.x > this.midX : true;
    this.choanaZ = 5;
    this.nostrilCdf = [];
    let acc = 0;
    for (const n of this.nostrils) { acc += n.frac; this.nostrilCdf.push(acc); }
    this.inlets = meta.inlets || [];
    this.outlets = meta.outlets || this.inlets;
    this.inletCdf = []; acc = 0;
    for (const p of this.inlets) { acc += p[3]; this.inletCdf.push(acc); }
    // inlet voxels grouped per nostril: room-air particles are drawn toward one of them (a real
    // fluid voxel), so they always find the opening
    for (const n of this.nostrils) { n.inlets = []; n.cdf = []; }
    for (const p of this.inlets) {
      let best = -1, bd = Infinity;
      this.nostrils.forEach((n, k) => { const d = (p[0] - n.c.x) ** 2 + (p[1] - n.c.y) ** 2 + (p[2] - n.c.z) ** 2; if (d < bd) { bd = d; best = k; } });
      if (best >= 0) { const n = this.nostrils[best]; n.inlets.push(p); n.cdf.push((n.cdf.length ? n.cdf[n.cdf.length - 1] : 0) + p[3]); }
    }
    this.outletCdf = []; acc = 0;
    for (const p of this.outlets) { acc += p[3]; this.outletCdf.push(acc); }
    // breathing cycle (animation seconds per breath; inspiration takes inspFrac of it)
    this.breathing = true;
    this.period = 4;
    this.inspFrac = 0.4;
    this.phase = 0;
    this.enabled = false;
    this.trail = 0.85;
    this.timeScale = 0.06; // 1 s of animation = 60 ms of real airflow
    this.spawnAcc = 0;
    this.styleName = 'glow';
    this.colorMode = 'speed';
    this.thermalMax = 500;
    this.sizeMult = 1;
    // Particle layers follow surfaces and test against opaque tissue depth,
    // including the exterior plume.
    this.orderInside = 45;
    this.orderOutside = 500;
    this.group = new THREE.Group();
    scene.add(this.group);
    this.setCount(8000);
    this.setStyle(this.styleName);
  }

  // ---------------------------------------------------------------- appearance
  setStyle(name) {
    const s = PARTICLE_STYLES[name] || PARTICLE_STYLES.glow;
    this.styleName = PARTICLE_STYLES[name] ? name : 'glow';
    const pu = this.pmat.uniforms, lu = this.lmat.uniforms;
    const ramp = this.colorMode === 'thermal' ? 1 : s.ramp;
    pu.uRamp.value = ramp; lu.uRamp.value = ramp;
    pu.uCore.value = s.core; pu.uAlpha.value = s.alpha; pu.uGauss.value = s.gauss;
    pu.uNoise.value = s.noise || 0; pu.uGrow.value = s.grow || 0;
    pu.uPlumeAlpha.value = s.plumeAlpha ?? 1; pu.uRoomAlpha.value = s.roomAlpha ?? 1;
    pu.sizePx.value = s.size * this.sizeMult;
    lu.uTrailAlpha.value = s.trail;
    const blending = s.additive && this.colorMode !== 'thermal' ? THREE.AdditiveBlending : THREE.NormalBlending;
    for (const m of [this.pmat, this.pmatOut, this.lmat]) { m.blending = blending; m.needsUpdate = true; }
  }
  // Order among the transparent particle layers; surface occlusion uses depth.
  setRenderOrder(inside, outside) {
    this.orderInside = inside; this.orderOutside = outside;
    if (!this.points) return;
    this.points.renderOrder = inside; this.lines.renderOrder = inside - 1; this.pointsOut.renderOrder = outside;
  }
  setSize(mult) { this.sizeMult = mult; this.pmat.uniforms.sizePx.value = (PARTICLE_STYLES[this.styleName] || PARTICLE_STYLES.glow).size * mult; }
  setColorMode(mode) {
    this.colorMode = mode === 'thermal' || mode === 'side' ? mode : 'speed';
    const v = this.colorMode === 'side' ? 1 : (this.colorMode === 'thermal' ? 2 : 0);
    this.pmat.uniforms.uColorMode.value = v; this.lmat.uniforms.uColorMode.value = v;
    const style = PARTICLE_STYLES[this.styleName] || PARTICLE_STYLES.glow;
    const ramp = this.colorMode === 'thermal' ? 1 : style.ramp;
    this.pmat.uniforms.uRamp.value = ramp; this.lmat.uniforms.uRamp.value = ramp;
    this.setStyle(this.styleName);
  }
  setSideColors(left, right, common) {
    for (const u of [this.pmat.uniforms, this.lmat.uniforms]) {
      if (left) u.uColL.value.set(left);
      if (right) u.uColR.value.set(right);
      if (common) u.uColC.value.set(common);
    }
  }
  // 0 = left nostril's air, 1 = right, 2 = pharynx / unknown
  sideAt(x, z) {
    if (z < this.choanaZ) return 2;
    return (x > this.midX) === this.leftIsPosX ? 0 : 1;
  }

  setCount(n) {
    this.n = n;
    if (this.points) { this.group.remove(this.points, this.pointsOut, this.lines); this.points.geometry.dispose(); this.lines.geometry.dispose(); }
    this.pos = new Float32Array(n * 3);
    this.renderPos = new Float32Array(n * 3);
    this.spd = new Float32Array(n);
    this.thermal = new Float32Array(n);
    this.life = new Float32Array(n);
    this.age = new Float32Array(n);
    this.mode = new Uint8Array(n).fill(DEAD);
    this.nos = new Uint8Array(n);        // nostril index (ROOM / PLUME)
    this.aux = new Float32Array(n * 4);  // PLUME: travelled mm, exit velocity (m/s)
    this.stuck = new Uint16Array(n);
    this.slow = new Uint16Array(n);
    this.hist = new Float32Array(n * HIST * 3);
    this.histHead = 0;
    this.pool = new Int32Array(n);
    this.poolN = 0;
    this.side = new Float32Array(n);
    this.puff = new Float32Array(n);
    this.seed = new Float32Array(n);
    for (let i = 0; i < n; i++) this.seed[i] = Math.random();
    const colorUniforms = () => ({ maxSpeed: { value: this.maxSpeed }, uRamp: { value: 0 }, uColorMode: { value: this.colorMode === 'thermal' ? 2 : (this.colorMode === 'side' ? 1 : 0) }, uThermalMax: { value: this.thermalMax },
      uColL: { value: new THREE.Color('#ff6a3d') }, uColR: { value: new THREE.Color('#59b6ff') }, uColC: { value: new THREE.Color('#c9d1d9') } });
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(this.renderPos, 3).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('speed', new THREE.BufferAttribute(this.spd, 1).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('life', new THREE.BufferAttribute(this.life, 1).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('side', new THREE.BufferAttribute(this.side, 1).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('thermal', new THREE.BufferAttribute(this.thermal, 1).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('puff', new THREE.BufferAttribute(this.puff, 1).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('seed', new THREE.BufferAttribute(this.seed, 1));
    this.pmat = this.pmat || new THREE.ShaderMaterial({ vertexShader: PVERT, fragmentShader: PFRAG, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { ...colorUniforms(), sizePx: { value: 2.2 }, uCore: { value: 1 }, uAlpha: { value: 0.75 }, uGauss: { value: 0 }, uNoise: { value: 0 }, uGrow: { value: 0 },
        uPlumeAlpha: { value: 1 }, uRoomAlpha: { value: 1 }, uPass: { value: 0 } }, clipping: true });
    // second draw of the same points for the air outside the head; shares every uniform object
    // with pmat (so style / colour changes apply to both) except the pass selector
    this.pmatOut = this.pmatOut || new THREE.ShaderMaterial({ vertexShader: PVERT, fragmentShader: PFRAG, transparent: true, depthWrite: false, blending: this.pmat.blending,
      uniforms: { ...this.pmat.uniforms, uPass: { value: 1 } }, clipping: true });
    this.points = new THREE.Points(pg, this.pmat);
    this.points.frustumCulled = false;
    this.pointsOut = new THREE.Points(pg, this.pmatOut);
    this.pointsOut.frustumCulled = false;
    // trails: (HIST-1) segments per particle
    const segs = n * (HIST - 1);
    this.lpos = new Float32Array(segs * 2 * 3);
    this.lspd = new Float32Array(segs * 2);
    this.lthermal = new Float32Array(segs * 2);
    this.lalpha = new Float32Array(segs * 2);
    this.lside = new Float32Array(segs * 2);
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.BufferAttribute(this.lpos, 3).setUsage(THREE.DynamicDrawUsage));
    lg.setAttribute('speed', new THREE.BufferAttribute(this.lspd, 1).setUsage(THREE.DynamicDrawUsage));
    lg.setAttribute('alpha', new THREE.BufferAttribute(this.lalpha, 1).setUsage(THREE.DynamicDrawUsage));
    lg.setAttribute('side', new THREE.BufferAttribute(this.lside, 1).setUsage(THREE.DynamicDrawUsage));
    lg.setAttribute('thermal', new THREE.BufferAttribute(this.lthermal, 1).setUsage(THREE.DynamicDrawUsage));
    this.lmat = this.lmat || new THREE.ShaderMaterial({ vertexShader: LVERT, fragmentShader: LFRAG, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { ...colorUniforms(), uTrailAlpha: { value: 0.35 } }, clipping: true });
    this.lines = new THREE.LineSegments(lg, this.lmat);
    this.lines.frustumCulled = false;
    this.group.add(this.points, this.pointsOut, this.lines);
    this.setRenderOrder(this.orderInside, this.orderOutside);
    // start with the airway half filled (uniform), the rest waits in the pool
    for (let i = 0; i < n; i++) {
      if (i < n * 0.5 && this.fluid.length) { this.seedInside(i); this.age[i] = 1; }
      else this.kill(i);
    }
    this.group.visible = this.enabled;
  }

  setEnabled(on) { this.enabled = on; this.group.visible = on; }
  setScenario(base,data,metrics,options={}) {
    if(!this.transport)this.transport=new ScenarioTransport(base,this.meta);
    this.transport.update(data,metrics,options);
    const peaks=['L','R'].map(k=>this.transport.thermalBySide[k]?.peakWm2||0);
    this.thermalMax=Math.max(250, ...peaks)*1.1;
    this.pmat.uniforms.uThermalMax.value=this.thermalMax;
    this.lmat.uniforms.uThermalMax.value=this.thermalMax;
    let acc=0;
    this.nostrils.forEach((n,i)=>{n.frac=this.transport.fractions[n.side===0?'L':'R'];this.nostrilCdf[i]=(acc+=n.frac);});
    // Recycle only particles in closed branches. Keep phase, live positions,
    // styles and GPU objects through ordinary scenario edits.
    for(let i=0;i<this.n;i++)if(this.mode[i]!==DEAD) {
      const side=this.mode[i]===IN?this.sideAt(this.pos[i*3],this.pos[i*3+2]):this.nostrils[this.nos[i]]?.side;
      if(!this.transport.delivered||(side<2&&this.transport.closed[side===0?'L':'R']))this.kill(i);
    }
  }
  clearScenario() {
    this.transport=null;
    let acc=0;
    this.nostrils.forEach((n,i)=>{n.frac=this.meta.nostrils[i].frac;this.nostrilCdf[i]=(acc+=n.frac);});
  }
  setFlowRate(q) { this.q = q; }
  setTrail(t) { this.trail = t; }
  setBreathing(on) { this.breathing = on; if (!on) this.phase = 0; }
  setPeriod(T) { this.period = T; }
  setClipPlanes(planes) { this.pmat.clippingPlanes = planes; this.pmatOut.clippingPlanes = planes; this.lmat.clippingPlanes = planes; }
  dispose() { this.scene.remove(this.group); this.points.geometry.dispose(); this.lines.geometry.dispose(); }

  // normalised breathing waveform: +1 peak inspiration, negative expiration with the same tidal volume
  wave(ph) {
    return restingBreath(ph,this.inspFrac);
  }
  phaseInfo() {
    if(this.transport&&!this.transport.delivered)return {label:'no nasal flow',q:0,frac:this.phase};
    if (!this.breathing) return { label: 'steady inhalation', q: this.qNow, frac: 1 };
    return { label: this.qNow >= 0 ? 'inhaling' : 'exhaling', q: this.qNow, frac: this.phase };
  }

  // ---------------------------------------------------------------- seeding
  pick(cdf) {
    const r = Math.random() * cdf[cdf.length - 1];
    let lo = 0, hi = cdf.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (cdf[m] <= r) lo = m + 1; else hi = m; }
    return lo;
  }
  place(i, x, y, z, mode) {
    const j = i * 3;
    this.pos[j] = x; this.pos[j + 1] = y; this.pos[j + 2] = z;
    this.mode[i] = mode; this.age[i] = 0; this.life[i] = 0; this.stuck[i] = 0; this.slow[i] = 0; this.spd[i] = 0; this.thermal[i] = 0;
    this.puff[i] = mode === ROOM ? -1 : 0;
    for (let h = 0; h < HIST; h++) { const k = (i * HIST + h) * 3; this.hist[k] = x; this.hist[k + 1] = y; this.hist[k + 2] = z; }
  }
  seedInside(i) {
    const [d0, d1] = this.dims;
    const vi = this.fluid[(Math.random() * this.fluid.length) | 0];
    const ix = vi % d0, iy = ((vi / d0) | 0) % d1, iz = (vi / (d0 * d1)) | 0;
    this.place(i, this.boxMin.x + (ix + Math.random()) * this.vox.x, this.boxMin.y + (iy + Math.random()) * this.vox.y, this.boxMin.z + (iz + Math.random()) * this.vox.z, IN);
    this.side[i] = this.sideAt(this.pos[i * 3], this.pos[i * 3 + 2]);
  }
  seedVoxel(i, p) {
    this.place(i, p[0] + (Math.random() - 0.5) * this.vox.x, p[1] + (Math.random() - 0.5) * this.vox.y, p[2] + (Math.random() - 0.5) * this.vox.z, IN);
    this.side[i] = this.sideAt(this.pos[i * 3], this.pos[i * 3 + 2]);
  }
  seedRoom(i) {
    if(this.transport&&!this.transport.delivered){this.kill(i);return;}
    if (!this.nostrils.length) { if (this.inlets.length) this.seedVoxel(i, this.inlets[this.pick(this.inletCdf)]); else this.seedInside(i); return; }
    const k = this.pick(this.nostrilCdf), n = this.nostrils[k];
    // random direction in the outward hemisphere, radius biased toward the opening
    let ux = 0, uy = 0, uz = 0, l = 0;
    do { ux = Math.random() * 2 - 1; uy = Math.random() * 2 - 1; uz = Math.random() * 2 - 1; l = ux * ux + uy * uy + uz * uz; } while (l < 0.05 || l > 1);
    l = Math.sqrt(l); ux /= l; uy /= l; uz /= l;
    const dot = ux * n.out.x + uy * n.out.y + uz * n.out.z;
    if (dot < 0.15) { const f = 2 * dot - 0.3; ux -= f * n.out.x; uy -= f * n.out.y; uz -= f * n.out.z; l = Math.hypot(ux, uy, uz); ux /= l; uy /= l; uz /= l; }
    const r = n.r + 1.5 + ROOM_SHELL * Math.pow(Math.random(), 0.7);
    this.place(i, n.c.x + ux * r, n.c.y + uy * r, n.c.z + uz * r, ROOM);
    this.nos[i] = k;
    this.side[i] = n.side;  // this air belongs to that nostril for the rest of its journey
    // target: one of this nostril's inlet voxels (weighted by inflow speed), jittered inside the voxel
    const a = i * 4;
    if (n.inlets.length) {
      const p = n.inlets[this.pick(n.cdf)];
      this.aux[a] = p[0] + (Math.random() - 0.5) * 0.6 * this.vox.x; this.aux[a + 1] = p[1] + (Math.random() - 0.5) * 0.6 * this.vox.y; this.aux[a + 2] = p[2] + (Math.random() - 0.5) * 0.6 * this.vox.z;
    } else { this.aux[a] = n.c.x; this.aux[a + 1] = n.c.y; this.aux[a + 2] = n.c.z; }
  }
  seedOutlet(i) {
    if (this.outlets.length) this.seedVoxel(i, this.outlets[this.pick(this.outletCdf)]); else this.seedInside(i);
  }
  kill(i) {
    this.mode[i] = DEAD; this.life[i] = 0; this.spd[i] = 0; this.thermal[i] = 0;
    this.pool[this.poolN++] = i;
  }
  nearNostril(x, y, z) {
    for (let k = 0; k < this.nostrils.length; k++) {
      const n = this.nostrils[k], lim = 2.5 * n.r + 4;
      if ((x - n.c.x) ** 2 + (y - n.c.y) ** 2 + (z - n.c.z) ** 2 < lim * lim) return k;
    }
    return -1;
  }

  // ---------------------------------------------------------------- field access
  occAt(x, y, z) {
    const d = this.dims;
    const ix = Math.floor((x - this.boxMin.x) / this.vox.x), iy = Math.floor((y - this.boxMin.y) / this.vox.y), iz = Math.floor((z - this.boxMin.z) / this.vox.z);
    if (ix < 0 || iy < 0 || iz < 0 || ix >= d[0] || iy >= d[1] || iz >= d[2]) return false;
    return this.occ[ix + d[0] * (iy + d[1] * iz)] === 1;
  }
  // trilinear sample of the velocity field (m/s at the current signed flow rate); returns speed, fills out
  sample(x, y, z, out) {
    const d = this.dims;
    const fx = (x - this.boxMin.x) / this.vox.x - 0.5, fy = (y - this.boxMin.y) / this.vox.y - 0.5, fz = (z - this.boxMin.z) / this.vox.z - 0.5;
    if (fx < -0.5 || fy < -0.5 || fz < -0.5 || fx > d[0] - 0.5 || fy > d[1] - 0.5 || fz > d[2] - 0.5) { out[0] = out[1] = out[2] = 0; return 0; }
    // clamp the 2x2x2 neighbourhood to the box so the outer half-voxel (e.g. the outlet layer) is still sampled
    const x0 = Math.min(Math.max(Math.floor(fx), 0), d[0] - 2), y0 = Math.min(Math.max(Math.floor(fy), 0), d[1] - 2), z0 = Math.min(Math.max(Math.floor(fz), 0), d[2] - 2);
    const tx = Math.min(Math.max(fx - x0, 0), 1), ty = Math.min(Math.max(fy - y0, 0), 1), tz = Math.min(Math.max(fz - z0, 0), 1);
    const v = this.vel, sx = 3, sy = d[0] * 3, sz = d[0] * d[1] * 3;
    let vx = 0, vy = 0, vz = 0, wsum = 0;
    for (let k = 0; k < 8; k++) {
      const ix = x0 + (k & 1), iy = y0 + ((k >> 1) & 1), iz = z0 + (k >> 2);
      const w = ((k & 1) ? tx : 1 - tx) * (((k >> 1) & 1) ? ty : 1 - ty) * ((k >> 2) ? tz : 1 - tz);
      const o = ix * sx + iy * sy + iz * sz;
      const a = v[o], b = v[o + 1], c = v[o + 2];
      if (a === 0 && b === 0 && c === 0) continue; // wall / outside: do not drag toward zero
      vx += w * a; vy += w * b; vz += w * c; wsum += w;
    }
    if (wsum < 0.1) { out[0] = out[1] = out[2] = 0; return 0; }
    let s = this.scale * (this.qNow / this.qRef) / wsum;
    let mag = Math.hypot(vx, vy, vz) * Math.abs(s);
    const cap = this.clampSpeed * Math.abs(this.qNow) / this.qRef;
    if (mag > cap) { s *= cap / mag; mag = cap; }
    // Reference-coordinate transport speed: branch Q / local area. Applying
    // the profile warp at draw time keeps particles with the preview surfaces.
    const gain=this.transport?.gainAt(x,y,z) ?? 1;
    s*=gain;mag*=gain;
    out[0] = vx * s; out[1] = vy * s; out[2] = vz * s;
    return mag;
  }

  // ---------------------------------------------------------------- dynamics
  step(dt, camera) {
    this.transport?.step(dt);
    if (!this.enabled) return;
    if (this.breathing) { this.phase = (this.phase + dt / this.period) % 1; this.qNow = this.q * this.wave(this.phase); }
    else this.qNow = this.q;
    this.qNow*=this.transport?.delivered ?? 1;
    const hs = dt * this.timeScale; // seconds of real airflow covered by this frame
    const n = this.n, pos = this.pos, v1 = [0, 0, 0];
    const qAbs = Math.abs(this.qNow), qRel = Math.max(qAbs / this.qRef, 1e-4);
    // exhaled-plume behaviour depends on the look (the vapour look follows the puff much further)
    const ps = PARTICLE_STYLES[this.styleName] || PARTICLE_STYLES.glow;
    const pLen = ps.plumeLen ?? PLUME_LEN, pTime = ps.plumeTime ?? PLUME_TIME, pFloor = ps.decayFloor ?? 0.5;
    const pSpread = ps.spread ?? 0.18, pBuoy = ps.buoy ?? 0, pFade = ps.fadePow ?? 1.5;
    // inject particles so the concentration inside stays ~FILL n / V whatever the flow rate
    this.spawnAcc += FILL * n / this.volMm3 * qAbs * 1000 * hs;
    let k = Math.floor(this.spawnAcc); this.spawnAcc -= k;
    while (k-- > 0 && this.poolN > 0) { const i = this.pool[--this.poolN]; if (this.qNow >= 0) this.seedRoom(i); else this.seedOutlet(i); }

    this.histHead = (this.histHead + 1) % HIST;
    for (let i = 0; i < n; i++) {
      const m = this.mode[i];
      if (m === DEAD) continue;
      const j = i * 3;
      let x = pos[j], y = pos[j + 1], z = pos[j + 2];
      this.age[i] += dt;
      if (m === IN) {
        if(this.qNow===0){ this.thermal[i]=0; continue; } // preserve particles at breath reversal / zero flow
        let s = this.sample(x, y, z, v1);
        if (s === 0) { this.kill(i); continue; }
        const nsub = Math.min(24, Math.max(1, Math.ceil(s * 1000 * hs / this.maxStep)));
        const hh = hs / nsub;
        let alive = true;
        for (let sub = 0; sub < nsub && alive; sub++) {
          if (sub > 0) { s = this.sample(x, y, z, v1); if (s === 0) { this.kill(i); alive = false; break; } }
          const nx = x + v1[0] * 1000 * hh, ny = y + v1[1] * 1000 * hh, nz = z + v1[2] * 1000 * hh;
          if (this.occAt(nx, ny, nz)) { x = nx; y = ny; z = nz; this.stuck[i] = 0; continue; }
          // stepped out of the airway
          if (ny < this.outletY) { this.kill(i); alive = false; break; }            // left through the trachea
          if (this.qNow < 0 && this.nearNostril(nx, ny, nz) >= 0) {                   // exhaled through a nostril
            this.mode[i] = PLUME; this.nos[i] = this.nearNostril(nx, ny, nz); this.age[i] = 0;
            const a = i * 4; this.aux[a] = 0; this.aux[a + 1] = v1[0]; this.aux[a + 2] = v1[1]; this.aux[a + 3] = v1[2];
            x = nx; y = ny; z = nz; break;
          }
          // slide along the wall: keep the components that stay inside
          const ax = this.occAt(nx, y, z), ay = this.occAt(x, ny, z), az = this.occAt(x, y, nz);
          const tx = ax ? nx : x, ty = ay ? ny : y, tz = az ? nz : z;
          if ((ax || ay || az) && this.occAt(tx, ty, tz)) { x = tx; y = ty; z = tz; }
          else if (ax) x = nx; else if (ay) y = ny; else if (az) z = nz;
          else { this.stuck[i]++; break; }
        }
        if (!alive) continue;
        // wedged in a corner, or stagnant in a dead-end pocket (< 2 cm/s at the reference flow)
        if (s / qRel < 0.02) this.slow[i]++; else this.slow[i] = 0;
        if (this.stuck[i] > 30 || this.slow[i] > 240) { this.kill(i); continue; }
        this.spd[i] = s;
        this.thermal[i] = this.qNow > 0 && this.transport
          ? this.transport.thermalAt(x, y, z) * Math.min(Math.abs(this.qNow) / Math.max(Math.abs(this.q), 1e-6), 1)
          : 0;
        this.life[i] = Math.min(this.age[i] / 0.3, 1);
        // pharynx air entering a nasal passage (exhalation) takes that side's colour
        if (this.side[i] === 2 && z >= this.choanaZ) this.side[i] = this.sideAt(x, z);
      } else if (m === ROOM) {
        const nn = this.nostrils[this.nos[i]], a = i * 4;
        const rx = x - this.aux[a], ry = y - this.aux[a + 1], rz = z - this.aux[a + 2];
        const r2 = rx * rx + ry * ry + rz * rz, r = Math.sqrt(r2);
        // half-space point sink at the target inlet voxel (source when exhaling): v = Q / (2 pi r^2)
        const vmag = Math.min(qAbs * 1000 * nn.frac / (2 * Math.PI * Math.max(r2, 4)), 4000); // mm/s
        const sgn = this.qNow >= 0 ? -1 : 1;
        const stepMm = sgn < 0 ? Math.min(vmag * hs, Math.max(r - 0.05, 0.02)) : vmag * hs; // never overshoot the target
        const f = sgn * stepMm / Math.max(r, 1e-3);
        const nx = x + rx * f, ny = y + ry * f, nz = z + rz * f;
        this.spd[i] = vmag / 1000;
        this.thermal[i] = 0;
        if (this.occAt(nx, ny, nz)) { this.mode[i] = IN; this.stuck[i] = 0; this.puff[i] = 0; }
        else if (sgn < 0 && r < 0.1) { this.kill(i); continue; }                       // target voxel not fluid after all
        x = nx; y = ny; z = nz;
        if (sgn < 0) this.life[i] = Math.min(this.age[i] / 0.4, 1);
        else {                                                                          // exhaling: the waiting room air disperses
          this.life[i] -= dt * 2.5;
          if (this.life[i] <= 0 || r > nn.r + ROOM_SHELL + 6) { this.kill(i); continue; }
        }
      } else { // PLUME
        const a = i * 4, nn = this.nostrils[this.nos[i]];
        let d = this.aux[a], vx = this.aux[a + 1], vy = this.aux[a + 2], vz = this.aux[a + 3];
        const v0 = Math.hypot(vx, vy, vz) || 1e-3;
        const decay = Math.max(pFloor, 1 - (1 - pFloor) * d / pLen);
        const stepMm = v0 * decay * 1000 * hs;
        // jet spreading: random walk of the direction (default ~10 degrees per 10 mm)
        const jit = pSpread * Math.sqrt(Math.max(stepMm, 1e-6) / 10);
        vx += (Math.random() - 0.5) * jit * v0; vy += (Math.random() - 0.5) * jit * v0; vz += (Math.random() - 0.5) * jit * v0;
        const l = Math.hypot(vx, vy, vz) || 1e-3;
        vx *= v0 / l; vy *= v0 / l; vz *= v0 / l;
        this.aux[a + 1] = vx; this.aux[a + 2] = vy; this.aux[a + 3] = vz;
        const prog = Math.min(d / pLen, 1);
        // warm breath is buoyant: once the jet has slowed it drifts upward
        let nx = x + vx / v0 * stepMm, ny = y + vy / v0 * stepMm + pBuoy * 1000 * hs * prog, nz = z + vz / v0 * stepMm;
        if (this.qNow > 0 && nn) { // the next breath draws the plume back toward the nostril
          const rx = nx - nn.c.x, ry = ny - nn.c.y, rz = nz - nn.c.z, r2 = rx * rx + ry * ry + rz * rz, r = Math.sqrt(r2);
          const vmag = Math.min(qAbs * 1000 * nn.frac / (2 * Math.PI * Math.max(r2, 4)), 4000);
          const f = -vmag * hs / Math.max(r, 1e-3);
          nx += rx * f; ny += ry * f; nz += rz * f;
          if (this.occAt(nx, ny, nz)) { this.mode[i] = IN; this.stuck[i] = 0; this.puff[i] = 0; }
        }
        d += stepMm; this.aux[a] = d;
        this.spd[i] = v0 * decay;
        this.thermal[i] = 0;
        if (this.mode[i] === PLUME) this.puff[i] = Math.max(prog, 1e-3);
        // fades with distance, and with time so slow exits do not hog the particle budget
        this.life[i] = Math.max(0, 1 - d / pLen) ** pFade * Math.max(0, 1 - this.age[i] / pTime);
        if (this.life[i] <= 0) { this.kill(i); continue; }
        x = nx; y = ny; z = nz;
      }
      pos[j] = x; pos[j + 1] = y; pos[j + 2] = z;
      const hk = (i * HIST + this.histHead) * 3;
      this.hist[hk] = x; this.hist[hk + 1] = y; this.hist[hk + 2] = z;
    }
    // Reference coordinates retain the wall-aware stored trajectories. Both
    // points and complete trail histories are mapped to the current preview.
    this.renderPos.set(pos);
    if(this.transport)for(let i=0;i<n;i++)if(this.mode[i]===IN) {
      const j=i*3;this.transport.map(pos[j],pos[j+1],pos[j+2],this.renderPos,j);
    }
    // rebuild trail segments
    const lp = this.lpos, ls = this.lspd, la = this.lalpha, lsd = this.lside, lth = this.lthermal, hist = this.hist;
    const nseg = Math.max(1, Math.round((HIST - 1) * this.trail));
    let q = 0, qa = 0;
    for (let i = 0; i < n; i++) {
      const sp = this.spd[i], lf = this.life[i], sd = this.side[i];
      for (let sgi = 0; sgi < HIST - 1; sgi++) {
        const a = (this.histHead - sgi + HIST) % HIST, b = (this.histHead - sgi - 1 + HIST) % HIST;
        const ka = (i * HIST + a) * 3, kb = (i * HIST + b) * 3;
        lp[q++] = hist[ka]; lp[q++] = hist[ka + 1]; lp[q++] = hist[ka + 2];
        lp[q++] = hist[kb]; lp[q++] = hist[kb + 1]; lp[q++] = hist[kb + 2];
        if(this.transport&&this.mode[i]===IN&&lf>0) {
          this.transport.map(hist[ka],hist[ka+1],hist[ka+2],lp,q-6);
          this.transport.map(hist[kb],hist[kb+1],hist[kb+2],lp,q-3);
        }
        ls[qa] = sp; ls[qa + 1] = sp;
        lsd[qa] = sd; lsd[qa + 1] = sd;
        lth[qa] = this.thermal[i]; lth[qa + 1] = this.thermal[i];
        const fade = sgi < nseg && lf > 0 ? lf * (1 - sgi / nseg) : 0;
        la[qa++] = fade; la[qa++] = fade * 0.8;
      }
    }
    const pg = this.points.geometry, lg = this.lines.geometry;
    pg.attributes.position.needsUpdate = true; pg.attributes.speed.needsUpdate = true; pg.attributes.life.needsUpdate = true; pg.attributes.side.needsUpdate = true; pg.attributes.thermal.needsUpdate = true; pg.attributes.puff.needsUpdate = true;
    lg.attributes.position.needsUpdate = true; lg.attributes.speed.needsUpdate = true; lg.attributes.alpha.needsUpdate = true; lg.attributes.side.needsUpdate = true; lg.attributes.thermal.needsUpdate = true;
    // colour scale fixed to the peak flow so the breathing cycle reads as a change of colour
    this.pmat.uniforms.maxSpeed.value = this.maxSpeed * (this.q / this.qRef);
    this.lmat.uniforms.maxSpeed.value = this.maxSpeed * (this.q / this.qRef);
  }
}
