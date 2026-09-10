// airway3d viewer — surfaces (PBR), GPU volume rendering, cut planes with CT slices,
// airway analytics and particle airflow.  Plain ES modules, no build step.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { VolumeRenderer } from './volren.js';
import { SlicePlanes } from './slices.js';
import { FlowParticles, PARTICLE_STYLES } from './flow.js';
import { AirwayPanel } from './airway.js';
import { ScenarioLab } from './scenarios.js';
import { heatFlux } from './heatflux.js';
import { evaluate } from './hydraulics.js';
import { setSurfaceOpacity } from './surface-material.js';
import { sortTransparentFaces, nestedSurfaceOrder, NESTED_SURFACES as ORDER } from './surface-order.js';

const $ = (s) => document.querySelector(s);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// ---------------------------------------------------------------- material styles
const STYLES = {
  matte: { roughness: 0.75, metalness: 0.0, clearcoat: 0.0, sheen: 0.0 },
  satin: { roughness: 0.45, metalness: 0.0, clearcoat: 0.15, sheen: 0.3 },
  soft: { roughness: 0.6, metalness: 0.0, clearcoat: 0.04, clearcoatRoughness: 0.5, sheen: 0.1 }, // photo-textured skin: no plastic glare
  wet: { roughness: 0.25, metalness: 0.0, clearcoat: 0.9, clearcoatRoughness: 0.2 },
  glossy: { roughness: 0.12, metalness: 0.05, clearcoat: 0.6, clearcoatRoughness: 0.08 },
  glass: { roughness: 0.05, metalness: 0.0, transmission: 0.85, thickness: 3.0, ior: 1.4 },
};
// per-tissue defaults (colour / opacity come from meta.labels, these refine the look)
const LOOK = {
  skin: { style: 'satin', opacity: 0.35, color: '#d9a683' },
  fat: { style: 'wet', opacity: 0.55 },
  muscle: { style: 'wet', opacity: 0.75 },
  soft: { style: 'satin', opacity: 0.55 },
  bone: { style: 'matte', opacity: 1.0 },
  teeth: { style: 'glossy', opacity: 1.0 },
  brain: { style: 'wet', opacity: 0.9 },
  eye: { style: 'glossy', opacity: 1.0 },
  airway: { style: 'satin', opacity: 0.85, color: '#2fa9e8' },
  airway_L: { style: 'satin', opacity: 0.85, color: '#ff6a3d', name: 'Airway — left nostril' },
  airway_R: { style: 'satin', opacity: 0.85, color: '#2f8fe8', name: 'Airway — right nostril' },
  airway_common: { style: 'satin', opacity: 0.85, color: '#9aa4b2', name: 'Airway — pharynx' },
  sinus: { style: 'satin', opacity: 0.4, color: '#4fd1a5' },
};
const AIRWAY_KEYS = ['airway', 'airway_L', 'airway_R', 'airway_common'];

// Thermal-camera overlay for the airway walls. The scalar is attached to each
// airway vertex once (nearest measured profile shell), then only the colour
// attribute changes as a scenario changes. This keeps the overlay responsive
// without rebuilding the airway mesh or reloading the scene.
const THERMAL_WALL_VERTEX = /* glsl */`
#include <clipping_planes_pars_vertex>
attribute vec3 thermalColor;
varying vec3 vThermalColor;
void main() {
  vThermalColor = thermalColor;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}
`;
const THERMAL_WALL_FRAGMENT = /* glsl */`
#include <clipping_planes_pars_fragment>
varying vec3 vThermalColor;
uniform float uOpacity;
void main() {
  #include <clipping_planes_fragment>
  gl_FragColor = vec4(vThermalColor, uOpacity);
}
`;
const thermalWall = { meshes: new Map(), visible: false, max: 500 };
const thermalStops = [
  [0.00, [0.03, 0.18, 0.95]], // blue
  [0.25, [0.00, 0.85, 0.95]], // cyan
  [0.50, [0.05, 0.78, 0.25]], // green
  [0.75, [1.00, 0.86, 0.05]], // yellow
  [1.00, [0.95, 0.05, 0.03]], // red
];
function thermalRGB(value, max, out) {
  const t = clamp((Number.isFinite(value) ? value : 0) / Math.max(max, 1), 0, 1);
  let a = thermalStops[0], b = thermalStops[thermalStops.length - 1];
  for (let i = 1; i < thermalStops.length; i++) { if (t <= thermalStops[i][0]) { a = thermalStops[i - 1]; b = thermalStops[i]; break; } }
  const f = (t - a[0]) / Math.max(b[0] - a[0], 1e-6);
  out.setRGB(a[1][0] + (b[1][0] - a[1][0]) * f, a[1][1] + (b[1][1] - a[1][1]) * f, a[1][2] + (b[1][2] - a[1][2]) * f);
}

const PRESETS = {
  Portrait: { show: ['skin', 'eye'], op: { skin: 1.0, eye: 1.0 }, photo: 1.0 },
  Dissect: { show: ['skin', 'fat', 'muscle', 'soft', 'eye', 'bone', 'brain'], op: { skin: 0.18, fat: 0.35, muscle: 0.8, soft: 0.5 } },
  Muscles: { show: ['muscle', 'bone', 'eye', 'skin'], op: { skin: 0.08, muscle: 1.0, bone: 1.0 } },
  Skeleton: { show: ['bone', 'teeth', 'skin'], op: { skin: 0.08, bone: 1.0 } },
  Airway: { show: ['bone', 'airway', 'airway_L', 'airway_R', 'airway_common', 'sinus', 'skin'], op: { skin: 0.05, bone: 0.07, sinus: 0.25, airway: 0.45, airway_L: 0.45, airway_R: 0.45, airway_common: 0.4 } },
  Brain: { show: ['brain', 'eye', 'bone', 'skin'], op: { skin: 0.06, bone: 0.2, brain: 1.0 } },
};

// ---------------------------------------------------------------- renderer / scene
const canvas = $('#gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.localClippingEnabled = true;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.debug.onShaderError = (gl, program, vs, fs) => {
  const log = gl.getShaderInfoLog(fs) || gl.getShaderInfoLog(vs) || gl.getProgramInfoLog(program);
  console.error('shader error', log);
  const hud = document.getElementById('hud');
  if (hud) hud.textContent = 'shader error: ' + String(log).slice(0, 160);
};

const scene = new THREE.Scene();
scene.background = new THREE.Color($('#bg').value);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.55;

const camera = new THREE.PerspectiveCamera(32, 1, 1, 3000);
camera.position.set(120, 40, 420);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);

const key = new THREE.DirectionalLight(0xfff4e6, 1.2);
key.position.set(1.5, 2.0, 2.5);
scene.add(key);
const rim = new THREE.DirectionalLight(0x9fc5ff, 0.6);
rim.position.set(-2, 0.5, -2);
scene.add(rim);
scene.add(new THREE.HemisphereLight(0xffffff, 0x334, 0.25));

const surfaces = new THREE.Group();
scene.add(surfaces);

// ---------------------------------------------------------------- state
const state = {
  dataset: 'pre',
  meta: null,
  tissues: {}, // key -> { mesh, mat, color, opacity, visible, style, volume_cc }
  mode: 'surface',
  cuts: [
    { axis: 0, name: 'Sagittal', on: false, pos: 0, flip: false }, // X
    { axis: 1, name: 'Axial', on: false, pos: 0, flip: false },    // Y
    { axis: 2, name: 'Coronal', on: false, pos: 0, flip: false },  // Z
  ],
  sliceMode: 'ct',
  window: 'soft',
};
const planes = [new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0), new THREE.Plane(new THREE.Vector3(0, -1, 0), 0), new THREE.Plane(new THREE.Vector3(0, 0, -1), 0)];
let activePlanes = [];
let volren = null;
let slices = null;
let flow = null;
let airwayPanel = null;
let lab = null;
const datasetCache = new Map();
const geometryCache = new Map();
let datasetRequest = 0;
const assetURL = (name, file) => new URL(`./data/${name}/${file}`, location.href).href;
let volumeTex = null, volumeSoftTex = null, labelTex = null;

// ---------------------------------------------------------------- loading helpers
const loading = $('#loading'), loadbar = $('#loadbar'), loadtext = $('#loadtext');
function progress(f, text) {
  loadbar.style.width = `${Math.round(f * 100)}%`;
  if (text) loadtext.textContent = text;
}
// data files are regenerated by the pipeline: always revalidate with the server (a 304 costs
// nothing) instead of trusting the browser's heuristic freshness
async function fetchBin(url) {
  const r = await fetch(url, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}
async function fetchJSON(url) {
  const r = await fetch(url, { cache: 'no-cache' });
  if (!r.ok) return null;
  return r.json();
}

function makeTexture3D(data, dims, linear = true) {
  const t = new THREE.Data3DTexture(data, dims[0], dims[1], dims[2]);
  t.format = THREE.RedFormat;
  t.type = THREE.UnsignedByteType;
  t.minFilter = t.magFilter = linear ? THREE.LinearFilter : THREE.NearestFilter;
  t.unpackAlignment = 1;
  t.needsUpdate = true;
  return t;
}

function makeMaterial(t) {
  const st = STYLES[t.style] || STYLES.satin;
  const mat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(t.color),
    roughness: st.roughness,
    metalness: st.metalness,
    clearcoat: st.clearcoat || 0,
    clearcoatRoughness: st.clearcoatRoughness || 0.1,
    sheen: st.sheen || 0,
    sheenColor: new THREE.Color(t.color).offsetHSL(0, 0, 0.2),
    transmission: st.transmission || 0,
    thickness: st.thickness || 0,
    ior: st.ior || 1.45,
    transparent: t.opacity < 0.999,
    opacity: t.opacity,
    side: THREE.DoubleSide,
    depthWrite: t.opacity >= 0.999,
    clippingPlanes: activePlanes,
    clipShadows: true,
  });
  if (t.photo) addPhotoToMaterial(mat, t.photo);
  setSurfaceOpacity(mat,t.opacity);
  return mat;
}

// ---------------------------------------------------------------- skin photo (projective textures)
// skin_photo.py projects the patient's photos (one frontal, optionally more from the sides)
// through their fitted cameras onto the skin mesh and stores, per photo and per vertex, the
// texture coordinate plus a weight (how well that photo sees the vertex: 0 = occluded / turned
// away / off the picture).  The material blends the photos by those weights, normalised per
// pixel, and fades to the flat skin colour where their sum is small (back and top of the head,
// under the chin, the cut faces).
async function loadSkinPhoto(geom, dir) {
  const info = await fetchJSON(`${dir}/skin_photo.json`);
  if (!info) return null;
  const n = geom.attributes.position.count;
  if (info.vertices !== n) {
    console.warn(`skin photo: UVs are for ${info.vertices} vertices but skin.glb has ${n} — re-run pipeline/skin_photo.py`);
    return null;
  }
  const views = info.views || [{ file: info.file, source: info.source }]; // older single-photo files
  const raw = await fetchBin(`${dir}/${info.uv_file || 'skin_uv.bin'}`);
  const all = new Float32Array(raw.buffer, raw.byteOffset, n * 3 * views.length);
  views.forEach((v, i) => {
    const buf = new THREE.InterleavedBuffer(all.subarray(i * n * 3, (i + 1) * n * 3), 3);
    geom.setAttribute(`photoUv${i}`, new THREE.InterleavedBufferAttribute(buf, 2, 0));
    geom.setAttribute(`photoW${i}`, new THREE.InterleavedBufferAttribute(buf, 1, 2));
  });
  const texs = await Promise.all(views.map((v) => new Promise((res, rej) => new THREE.TextureLoader().load(`${dir}/${v.file}?v=${info.vertices}`, (tex) => {
    tex.flipY = false; // UVs use the glTF convention (v down from the top-left corner)
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    tex.generateMipmaps = true;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    res(tex);
  }, undefined, rej))));
  // weights sum to weight_scale where one photo sees the surface head-on; the old format stored
  // a 0..1 mask instead
  return { texs, views, info, strength: 1.0, gain: 1.0, scale: 1 / (info.weight_scale || 1) };
}

function addPhotoToMaterial(mat, photo) {
  const n = photo.texs.length;
  mat.userData.photo = { uPhoto: { value: photo.strength }, uPhotoGain: { value: photo.gain }, uPhotoScale: { value: photo.scale } };
  photo.texs.forEach((tex, i) => { mat.userData.photo[`uPhotoTex${i}`] = { value: tex }; });
  let vDecl = '', vBody = '', fDecl = '', fBody = '';
  for (let i = 0; i < n; i++) {
    vDecl += `attribute vec2 photoUv${i};\nattribute float photoW${i};\nvarying vec2 vPhotoUv${i};\nvarying float vPhotoW${i};\n`;
    vBody += `  vPhotoUv${i} = photoUv${i};\n  vPhotoW${i} = photoW${i};\n`;
    fDecl += `uniform sampler2D uPhotoTex${i};\nvarying vec2 vPhotoUv${i};\nvarying float vPhotoW${i};\n`;
    fBody += `  { float w = max( vPhotoW${i}, 0.0 ); if ( w > 0.0 ) { pAcc += w * texture2D( uPhotoTex${i}, vPhotoUv${i} ).rgb; pW += w; } }\n`;
  }
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, mat.userData.photo);
    shader.vertexShader = shader.vertexShader.replace('void main() {', `${vDecl}void main() {\n${vBody}`);
    shader.fragmentShader = shader.fragmentShader
      .replace('void main() {', `${fDecl}uniform float uPhoto;\nuniform float uPhotoGain;\nuniform float uPhotoScale;\nvoid main() {`)
      .replace('#include <map_fragment>', `#include <map_fragment>
        vec3 pAcc = vec3( 0.0 ); float pW = 0.0;
        ${fBody}
        if ( pW > 0.0 ) {
          float pm = clamp( pW * uPhotoScale, 0.0, 1.0 ) * uPhoto;
          diffuseColor.rgb = mix( diffuseColor.rgb, pAcc / pW * uPhotoGain, pm );
        }`);
  };
  mat.customProgramCacheKey = () => `skin-photo-${n}`;
}

// particles coloured "by nostril" reuse the airway mesh colours so both legends agree
function syncFlowSideColors() {
  if (!flow) return;
  const c = (k) => state.tissues[k]?.color || LOOK[k]?.color;
  flow.setSideColors(c('airway_L'), c('airway_R'), c('airway_common'));
}

function profileIndexAttribute(geometry, profile) {
  const positions = geometry.getAttribute('position');
  let index = geometry.getAttribute('thermalIndex');
  if (index && index.count === positions.count) return index;
  const values = new Float32Array(positions.count);
  const p = profile?.centers || [];
  const xyz = new THREE.Vector3();
  for (let v = 0; v < positions.count; v++) {
    xyz.fromBufferAttribute(positions, v);
    let best = Infinity, found = 0;
    for (let i = 0; i < p.length; i++) {
      const c = p[i], dx = xyz.x - c[0], dy = xyz.y - c[1], dz = xyz.z - c[2], d = dx * dx + dy * dy + dz * dz;
      if (d < best) { best = d; found = i; }
    }
    values[v] = found;
  }
  index = new THREE.BufferAttribute(values, 1);
  geometry.setAttribute('thermalIndex', index);
  return index;
}

function thermalWallMesh(key, tissue, profile) {
  if (!tissue?.mesh || !profile?.centers?.length) return null;
  const geometry = tissue.mesh.geometry;
  let entry = thermalWall.meshes.get(key);
  if (!entry) {
    const material = new THREE.ShaderMaterial({
      vertexShader: THERMAL_WALL_VERTEX, fragmentShader: THERMAL_WALL_FRAGMENT,
      transparent: true, depthWrite: false, depthTest: true, side: THREE.DoubleSide,
      clipping: true, toneMapped: false, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
      uniforms: { uOpacity: { value: 0.92 } },
    });
    setSurfaceOpacity(material,.92);
    entry = { mesh: new THREE.Mesh(geometry, material), key, material };
    material.forceSinglePass=true;
    entry.mesh.renderOrder=nestedSurfaceOrder(key)+1;
    entry.mesh.onBeforeRender=(_renderer,_scene,camera)=>sortTransparentFaces(entry.mesh,camera);
    entry.mesh.frustumCulled = false;
    surfaces.add(entry.mesh);
    thermalWall.meshes.set(key, entry);
  }
  if (entry.mesh.geometry !== geometry) entry.mesh.geometry = geometry;
  const index = profileIndexAttribute(geometry, profile);
  let colors = geometry.getAttribute('thermalColor');
  if (!colors || colors.count !== index.count) {
    colors = new THREE.BufferAttribute(new Float32Array(index.count * 3), 3).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('thermalColor', colors);
  }
  entry.mesh.visible = thermalWall.visible && tissue.visible && state.mode !== 'volume';
  entry.mesh.renderOrder = (tissue.mesh.renderOrder || nestedSurfaceOrder(key)) + 1;
  entry.material.clippingPlanes = activePlanes;
  return { entry, index, colors };
}

function updateThermalWalls(data, metrics) {
  if (!data?.sides) return;
  const fields = {};
  let peak = 0;
  for (const side of ['L', 'R', 'common']) {
    if (!data.sides[side]?.profile || !metrics?.[side]) continue;
    fields[side] = heatFlux(data.sides[side].profile, metrics[side].Q);
    peak = Math.max(peak, fields[side].peakWm2 || 0);
  }
  thermalWall.max = Math.max(250, peak) * 1.1;
  for (const [key, side] of [['airway_L', 'L'], ['airway_R', 'R'], ['airway_common', 'common'], ['airway', 'common']]) {
    const tissue = state.tissues[key];
    const field = fields[side];
    if (!tissue || !field) continue;
    const mapped = thermalWallMesh(key, tissue, data.sides[side].profile);
    if (!mapped) continue;
    const { index, colors } = mapped;
    const color = new THREE.Color();
    for (let v = 0; v < index.count; v++) {
      thermalRGB(field.fluxWm2?.[Math.round(index.getX(v))] || 0, thermalWall.max, color);
      colors.setXYZ(v, color.r, color.g, color.b);
    }
    colors.needsUpdate = true;
  }
  for (const [key, entry] of thermalWall.meshes) {
    const tissue = state.tissues[key];
    entry.mesh.visible = thermalWall.visible && !!tissue?.visible && state.mode !== 'volume';
    entry.material.clippingPlanes = activePlanes;
  }
}

function setThermalWallVisibility(on) {
  thermalWall.visible = !!on;
  for (const [key, entry] of thermalWall.meshes) {
    const tissue = state.tissues[key];
    entry.mesh.visible = thermalWall.visible && !!tissue?.visible && state.mode !== 'volume';
  }
}

function applyTissue(t) {
  if (t.key && t.key.startsWith('airway_')) syncFlowSideColors();
  if (!t.mesh) return;
  const st = STYLES[t.style] || STYLES.satin;
  const m = t.mat;
  m.color.set(t.color);
  m.sheenColor.copy(m.color).offsetHSL(0, 0, 0.2);
  m.roughness = st.roughness; m.metalness = st.metalness;
  m.clearcoat = st.clearcoat || 0; m.clearcoatRoughness = st.clearcoatRoughness || 0.1;
  m.sheen = st.sheen || 0; m.transmission = st.transmission || 0; m.thickness = st.thickness || 0; m.ior = st.ior || 1.45;
  setSurfaceOpacity(m,t.opacity);
  if (t.photo && m.userData.photo) {
    m.userData.photo.uPhoto.value = t.photo.strength;
    m.userData.photo.uPhotoGain.value = t.photo.gain;
  }
  // translucent shells: front faces only (back faces would double the layers and wash out);
  // while a cut plane is active we need the inner surfaces, so go double-sided then
  const cutting = state.cuts.some((c) => c.on);
  m.side = (cutting || t.opacity >= 0.999) ? THREE.DoubleSide : THREE.FrontSide;
  m.needsUpdate = true;
  const cfdAirway = lab?.cfd?.enabled && AIRWAY_KEYS.includes(t.key);
  t.mesh.visible = t.visible && state.mode !== 'volume' && !cfdAirway;
  const thermal = thermalWall.meshes.get(t.key);
  if (thermal) {
    thermal.mesh.geometry = t.mesh.geometry;
    thermal.mesh.visible = thermalWall.visible && t.visible && state.mode !== 'volume' && !cfdAirway;
  }
  // Nested shells draw inside → outside. An invisible solid skin pre-pass
  // would incorrectly occlude other translucent objects, so it stays off.
  // Clip planes expose back faces, so those shells need a triangle sort.
  t.mesh.renderOrder = nestedSurfaceOrder(t.key);
  t.mesh.onBeforeRender = (cutting && t.opacity < 0.999)
    ? (_renderer,_scene,camera)=>sortTransparentFaces(t.mesh,camera)
    : ()=>{};
  if(t.depthMesh)t.depthMesh.visible=false;
  // centrelines / minimum-area markers follow the airway's visibility
  if (airwayPanel && t.key.startsWith('airway')) {
    airwayPanel.group.visible = !airwayPanel.cfdMode && Object.values(state.tissues).some((x) => x.key.startsWith('airway') && x.visible);
  }
}

// ---------------------------------------------------------------- dataset loading
async function loadInitialDataset(name) {
  loading.style.display = '';
  loading.classList.remove('hide');
  progress(0, `loading ${name}…`);
  // dispose old, but remember the user's per-tissue choices so switching datasets keeps them
  const keep = {};
  for (const k in state.tissues) {
    const t = state.tissues[k];
    keep[k] = { color: t.color, opacity: t.opacity, visible: t.visible, style: t.style };
    if (t.photo) keep[k].photoSettings = { strength: t.photo.strength, gain: t.photo.gain };
    if (t.mesh) { surfaces.remove(t.mesh, t.depthMesh); t.mesh.geometry.dispose(); t.mat.dispose(); t.depthMesh.material.dispose(); t.photo?.texs.forEach((x) => x.dispose()); }
  }
  state.tissues = {};
  const base = `./data/${name}`;
  const meta = await fetchJSON(`${base}/meta.json`);
  if (!meta) { loadtext.textContent = `dataset ${name} not found`; return; }
  state.meta = meta;
  state.dataset = name;

  const keys = Object.keys(meta.meshes).sort((a, b) => ORDER.indexOf(a) - ORDER.indexOf(b));
  const labelByKey = {};
  for (const id in meta.labels) labelByKey[meta.labels[id].key] = { id: +id, ...meta.labels[id] };
  const loader = new GLTFLoader();
  let done = 0;
  const jobs = keys.map((k) => new Promise((resolve) => {
    const file = meta.meshes[k].file ? `${base}/${meta.meshes[k].file}` : `${base}/${k}.glb`;
    loader.load(file, async (g) => {
      let geom = null;
      g.scene.traverse((o) => { if (o.isMesh && !geom) geom = o.geometry; });
      geom.computeVertexNormals();
      geometryCache.set(new URL(file, location.href).href, geom);
      const lab = labelByKey[k] || labelByKey[k.split('_')[0]] || { color: '#cccccc', opacity: 0.8, name: k };
      const look = LOOK[k] || {};
      const stat = meta.stats?.classes?.[lab.id];
      const t = {
        key: k, name: look.name || lab.name, color: look.color || lab.color, opacity: look.opacity ?? lab.opacity,
        visible: true, style: look.style || 'satin', volume_cc: stat?.volume_cc, labelId: lab.id,
      };
      // when the airway is split by side, hide the combined one
      if (k === 'airway' && keys.includes('airway_L')) t.visible = false;
      if (k === 'skin') {
        // the patient's photo projected onto the skin (sidecar next to the mesh file, if present)
        try { t.photo = await loadSkinPhoto(geom, file.slice(0, file.lastIndexOf('/'))); } catch (e) { console.warn('skin photo', e); t.photo = null; }
        if (t.photo) {
          if (!keep[k]) {
            if (t.photo.info.skin_tone) t.color = t.photo.info.skin_tone; // untextured parts match the photo
            t.style = 'soft'; // a satin clearcoat reads as wet plastic on a photographed face
          }
          if (keep[k]?.photoSettings) Object.assign(t.photo, keep[k].photoSettings);
        }
      }
      if (keep[k]) { const { photoSettings, ...rest } = keep[k]; Object.assign(t, rest); }
      t.mat = makeMaterial(t);
      t.mesh = new THREE.Mesh(geom, t.mat);
      t.mat.forceSinglePass=true;
      t.mesh.frustumCulled = false;
      // Retained for cached geometry/visibility bookkeeping; never rendered.
      t.depthMesh = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ colorWrite: false, transparent: true, clippingPlanes: activePlanes }));
      t.depthMesh.frustumCulled = false;
      surfaces.add(t.mesh, t.depthMesh);
      state.tissues[k] = t;
      applyTissue(t);
      done++;
      progress(0.1 + 0.6 * done / keys.length, `meshes ${done}/${keys.length}`);
      resolve();
    }, undefined, () => { done++; resolve(); });
  }));
  await Promise.all(jobs);

  // volumes
  progress(0.75, 'volume textures…');
  const dims = meta.volume.dims;
  const files = meta.files || {};
  const gray = await fetchBin(`${base}/${files.volume || 'volume_u8.bin'}`);
  const labels = await fetchBin(`${base}/labels_u8.bin`);
  let soft = null;
  if (meta.soft_window) { try { soft = await fetchBin(`${base}/${files.volume_soft || 'volume_soft_u8.bin'}`); } catch (e) { soft = null; } }
  volumeTex?.dispose(); labelTex?.dispose(); volumeSoftTex?.dispose();
  volumeTex = makeTexture3D(gray, dims, true);
  labelTex = makeTexture3D(labels, dims, false);
  volumeSoftTex = soft ? makeTexture3D(soft, dims, true) : volumeTex;
  const box = new THREE.Box3(new THREE.Vector3(...meta.volume.box_min), new THREE.Vector3().addVectors(new THREE.Vector3(...meta.volume.box_min), new THREE.Vector3(...meta.volume.box_size)));

  if (!volren) { volren = new VolumeRenderer(renderer, scene, camera); }
  volren.setVolume({ gray: volumeTex, soft: volumeSoftTex, labels: labelTex, box, dims, labelsMeta: meta.labels });
  if (!slices) { slices = new SlicePlanes(scene); }
  slices.setVolume({ gray: volumeTex, soft: volumeSoftTex, labels: labelTex, box, labelsMeta: meta.labels });

  // cut plane ranges
  for (const c of state.cuts) { c.min = box.min.getComponent(c.axis); c.max = box.max.getComponent(c.axis); if (c.pos === 0) c.pos = (c.min + c.max) / 2; }
  buildCutUI();
  updateCuts();

  // airway analytics + flow
  progress(0.9, 'airway data…');
  const aw = await fetchJSON(`${base}/airway.json`);
  if (!airwayPanel) airwayPanel = new AirwayPanel(scene, $('#airway-stats'), $('#chart'));
  airwayPanel.setData(aw, await fetchJSON(`./data/pre/airway.json`), name);
  airwayPanel.group.visible = !airwayPanel.cfdMode && Object.values(state.tissues).some((x) => x.key.startsWith('airway') && x.visible);
  if (aw?.sides) {
    // per-side volumes for the split airway meshes (the label stat is the whole airway)
    for (const [k, side] of [['airway_L', 'L'], ['airway_R', 'R'], ['airway_common', 'common']]) {
      if (state.tissues[k] && aw.sides[side]) state.tissues[k].volume_cc = aw.sides[side].volume_cc;
    }
    if (state.tissues.airway && aw.total) state.tissues.airway.volume_cc = aw.total.volume_cc;
  }
  if (flow) { flow.dispose(); flow = null; }
  const flowMeta = await fetchJSON(`${base}/flow.json`);
  if (flowMeta) {
    const buf = await fetchBin(`${base}/flow.bin`);
    flow = new FlowParticles(scene, flowMeta, buf);
    flow.setEnabled($('#flow-on').checked);
    flow.setCount(+$('#flow-n').value);
    flow.setFlowRate(+$('#flow-q').value);
    flow.setTrail(+$('#flow-trail').value);
    flow.setBreathing($('#flow-breathe').checked);
    flow.setPeriod(+$('#flow-period').value);
    flow.timeScale = +$('#flow-speed').value;
    flow.setStyle($('#flow-style').value);
    flow.setSize(+$('#flow-size').value);
    flow.setColorMode($('#flow-color').value);
    updateFlowColorNote($('#flow-color').value);
    if (aw?.choana_world_z != null) flow.choanaZ = aw.choana_world_z;
    syncFlowSideColors();
    $('#flow-sec').style.display = '';
    $('#flow-stats').innerHTML = flowMeta.summary_html || '';
  } else {
    $('#flow-sec').style.display = 'none';
  }
  updateThermalWalls(aw, aw ? evaluate(aw, +$('#flow-q').value) : null);
  buildTissueUI();
  refreshTransfer();
  datasetCache.set(name, Promise.resolve({ meta, aw, flow, volumeTex, volumeSoftTex, labelTex, geometries: Object.fromEntries(Object.entries(state.tissues).map(([k,t]) => [k,t.mesh.geometry])) }));
  progress(1, 'ready');
  setTimeout(() => {
    loading.classList.add('hide');
    setTimeout(() => { if (loading.classList.contains('hide')) loading.style.display = 'none'; }, 600);
  }, 250);
}

// Stage all changed assets while the current scene continues rendering. Commit in one turn.
async function prepareDataset(name) {
  if (datasetCache.has(name)) return datasetCache.get(name);
  const promise = (async () => {
    const meta = await fetchJSON(`./data/${name}/meta.json`);
    if (!meta) throw new Error(`Dataset ${name} is unavailable`);
    const reference = await datasetCache.get('pre');
    const geometries = {};
    await Promise.all(Object.keys(meta.meshes).map(async k => {
      const url = assetURL(name, meta.meshes[k].file || `${k}.glb`);
      if (geometryCache.has(url)) { geometries[k] = geometryCache.get(url); return; }
      const gltf = await new GLTFLoader().loadAsync(url);
      let geom; gltf.scene.traverse(o => { if (o.isMesh && !geom) geom=o.geometry; });
      if (!geom) throw new Error(`No geometry in ${k}`);
      geom.computeVertexNormals(); geometryCache.set(url,geom); geometries[k]=geom;
    }));
    const [aw, fm, labels] = await Promise.all([
      fetchJSON(`./data/${name}/airway.json`), fetchJSON(`./data/${name}/flow.json`), fetchBin(`./data/${name}/labels_u8.bin`),
    ]);
    if (!aw) throw new Error('Airway metrics missing');
    const tex = async (file, refFile, existing, linear) => assetURL(name,file) === assetURL('pre',refFile)
      ? existing : makeTexture3D(await fetchBin(assetURL(name,file)),meta.volume.dims,linear);
    const vt = await tex(meta.files?.volume || 'volume_u8.bin','volume_u8.bin',reference.volumeTex,true);
    const st = await tex(meta.files?.volume_soft || 'volume_soft_u8.bin','volume_soft_u8.bin',reference.volumeSoftTex,true);
    const fb = fm ? await fetchBin(`./data/${name}/flow.bin`) : null;
    return {meta,aw,geometries,volumeTex:vt,volumeSoftTex:st,labelTex:makeTexture3D(labels,meta.volume.dims,false),fm,fb,flow:null};
  })();
  datasetCache.set(name,promise);
  promise.catch(()=> { if (datasetCache.get(name)===promise) datasetCache.delete(name); });
  return promise;
}
async function loadDataset(name) {
  if (!['pre','post'].includes(name)) throw new Error('Unknown saved dataset');
  if (!state.meta) return loadInitialDataset(name);
  const request=++datasetRequest;
  const status={set textContent(value){$('#dataset-status').textContent=value;const labStatus=$('#lab-source-status');if(labStatus)labStatus.textContent=value;}};
  status.textContent=`Loading ${name==='pre'?'CT reference':'saved virtual reduction'}; current model remains interactive…`;
  try {
    const next=await prepareDataset(name);
    if (request!==datasetRequest) return false;
    lab?.restoreGeometry();
    flow?.setEnabled(false);
    for (const [k,geom] of Object.entries(next.geometries)) {
      const t=state.tissues[k]; if (!t) continue;
      if(t.mesh.geometry!==geom) {t.mesh.geometry=geom; t.depthMesh.geometry=geom;}
      const side=k.replace('airway_','');
      t.volume_cc=next.aw.sides[side]?.volume_cc ?? next.meta.stats?.classes?.[t.labelId]?.volume_cc;
    }
    state.meta=next.meta; state.dataset=name;
    volumeTex=next.volumeTex; volumeSoftTex=next.volumeSoftTex; labelTex=next.labelTex;
    const meta=next.meta;
    const box=new THREE.Box3(new THREE.Vector3(...meta.volume.box_min),new THREE.Vector3(...meta.volume.box_min).add(new THREE.Vector3(...meta.volume.box_size)));
    const volumes={gray:volumeTex,soft:volumeSoftTex,labels:labelTex,box,dims:meta.volume.dims,labelsMeta:meta.labels};
    volren.setVolume(volumes); slices.setVolume(volumes);
    if (!next.flow && next.fm) { next.flow=new FlowParticles(scene,next.fm,next.fb); next.fb=null; }
    flow=next.flow;
    if(flow) {
      flow.clearScenario();
      if(flow.n!==+$('#flow-n').value)flow.setCount(+$('#flow-n').value);
      flow.setFlowRate(+$('#flow-q').value); flow.setTrail(+$('#flow-trail').value);
      flow.setBreathing($('#flow-breathe').checked); flow.setPeriod(+$('#flow-period').value); flow.timeScale=+$('#flow-speed').value;
      flow.setStyle($('#flow-style').value); flow.setSize(+$('#flow-size').value); flow.setColorMode($('#flow-color').value);
      flow.choanaZ=next.aw.choana_world_z; flow.setEnabled($('#flow-on').checked); syncFlowSideColors();
    }
    airwayPanel.setData(next.aw,(await datasetCache.get('pre')).aw,name);
    airwayPanel.setFlowRate(+$('#flow-q').value);
    updateThermalWalls(next.aw, evaluate(next.aw, +$('#flow-q').value));
    buildTissueUI(); refreshTransfer(); updateCuts();
    status.textContent=name==='pre'?'CT reference · face up':'Saved geometric reduction · 2 mm bilateral';
    canvas.dataset.sceneId=scene.uuid;canvas.dataset.skinGeometry=state.tissues.skin?.mesh.geometry.uuid;canvas.dataset.volumeTexture=volumeTex.uuid;canvas.dataset.surfaceCount=surfaces.children.length;
    lab?.datasetChanged(name,next.aw);
    return true;
  } catch(error) {
    if(request===datasetRequest) status.textContent=`Could not load ${name}: ${error.message}. Current model retained.`;
    return false;
  }
}

// ---------------------------------------------------------------- UI: tissues
function buildTissueUI() {
  const root = $('#tissues');
  root.innerHTML = '';
  const keys = Object.keys(state.tissues).sort((a, b) => ORDER.indexOf(a) - ORDER.indexOf(b));
  for (const k of keys) {
    const t = state.tissues[k];
    const row = document.createElement('div');
    row.dataset.k = k;
    row.className = 'tissue' + (t.visible ? '' : ' off');
    row.innerHTML = `
      <input type="checkbox" ${t.visible ? 'checked' : ''} data-k="${k}" class="vis" />
      <div class="name"><span>${t.name}</span><span class="vol">${t.volume_cc != null ? t.volume_cc.toFixed(0) + ' cc' : ''}</span></div>
      <input type="color" value="${t.color}" class="col" data-k="${k}" />
      <div class="ctrl">
        <div><label>opacity ${Math.round(t.opacity * 100)}%</label><input type="range" min="0" max="1" step="0.01" value="${t.opacity}" class="op" data-k="${k}" /></div>
        <div><label>hue / light</label><input type="range" min="-0.5" max="0.5" step="0.01" value="0" class="tint" data-k="${k}" /></div>
        <div><label>material</label><select class="style" data-k="${k}">${Object.keys(STYLES).map((s) => `<option ${s === t.style ? 'selected' : ''}>${s}</option>`).join('')}</select></div>
        ${t.photo ? `
        <div><label>photo ${Math.round(t.photo.strength * 100)}%</label><input type="range" min="0" max="1" step="0.01" value="${t.photo.strength}" class="photo" /></div>
        <div><label>photo light</label><input type="range" min="0.5" max="2" step="0.01" value="${t.photo.gain}" class="photo-gain" /></div>` : ''}
      </div>`;
    root.appendChild(row);
    if (t.photo) {
      row.querySelector('.photo').addEventListener('input', (e) => { t.photo.strength = +e.target.value; e.target.previousElementSibling.textContent = `photo ${Math.round(t.photo.strength * 100)}%`; applyTissue(t); });
      row.querySelector('.photo-gain').addEventListener('input', (e) => { t.photo.gain = +e.target.value; applyTissue(t); });
      const fit = t.photo.info;
      const more = t.photo.views.slice(1).map((v) => `${v.source} (${v.fit?.median_mm != null ? `median ${v.fit.median_mm.toFixed(1)} mm to the frontal photo` : 'side view'})`);
      row.querySelector('.name').title = `${fit.source}: camera ${fit.camera.distance_mm.toFixed(0)} mm from the face, landmark fit RMS ${fit.rms_mm.toFixed(1)} mm`
        + (more.length ? `\n+ ${more.join(', ')}` : '');
    }
    row.querySelector('.vis').addEventListener('change', (e) => {
      t.visible = e.target.checked; row.classList.toggle('off', !t.visible); applyTissue(t);
      // the combined airway and the per-side airways are the same surface in two colourings:
      // showing both makes coincident translucent faces flip order as the camera moves
      if (t.visible && AIRWAY_KEYS.includes(k)) {
        const others = k === 'airway' ? AIRWAY_KEYS.filter((x) => x !== 'airway') : ['airway'];
        for (const o of others) {
          const u = state.tissues[o];
          if (!u || !u.visible) continue;
          u.visible = false; applyTissue(u);
          const r = root.querySelector(`.vis[data-k="${o}"]`);
          if (r) { r.checked = false; r.closest('.tissue').classList.add('off'); }
        }
      }
      refreshTransfer();
    });
    row.querySelector('.op').addEventListener('input', (e) => { t.opacity = +e.target.value; e.target.previousElementSibling.textContent = `opacity ${Math.round(t.opacity * 100)}%`; applyTissue(t); refreshTransfer(); });
    row.querySelector('.col').addEventListener('input', (e) => { t.color = e.target.value; applyTissue(t); refreshTransfer(); });
    row.querySelector('.tint').addEventListener('input', (e) => {
      const c = new THREE.Color(row.querySelector('.col').value); c.offsetHSL(0, 0, +e.target.value * 0.6);
      t.color = '#' + c.getHexString(); applyTissue(t); refreshTransfer();
    });
    row.querySelector('.style').addEventListener('change', (e) => { t.style = e.target.value; applyTissue(t); });
  }
  lab?.cfd?.mountSurfaceControl();
}

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  for (const k in state.tissues) {
    const t = state.tissues[k];
    t.visible = p.show.includes(k);
    if (k === 'airway' && state.tissues.airway_L) t.visible = false;
    const look = LOOK[k] || {};
    t.opacity = p.op[k] ?? look.opacity ?? t.opacity;
    if (t.photo && p.photo != null) t.photo.strength = p.photo;
    applyTissue(t);
  }
  document.querySelectorAll('#presets button').forEach((b) => b.classList.toggle('on', b.textContent === name));
  buildTissueUI();
  refreshTransfer();
}

// transfer function for the volume renderer: label -> colour/opacity (from the tissue UI)
function refreshTransfer() {
  if (!volren || !state.meta) return;
  const tf = {};
  for (const id in state.meta.labels) {
    const key = state.meta.labels[id].key;
    let t = state.tissues[key];
    if (key === 'airway' && !(t && t.visible)) t = AIRWAY_KEYS.map((k) => state.tissues[k]).find((x) => x && x.visible);
    if (t && t.visible) tf[id] = { color: t.color, opacity: t.opacity };
  }
  volren.setTransfer(tf);
  slices?.setTransfer(tf, state.meta.labels);
}

// ---------------------------------------------------------------- UI: cuts
function buildCutUI() {
  const root = $('#cuts');
  root.innerHTML = '';
  for (const c of state.cuts) {
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = `<label><input type="checkbox" ${c.on ? 'checked' : ''} class="on"/> ${c.name}</label>
      <input type="range" min="${c.min}" max="${c.max}" step="0.25" value="${c.pos}" class="pos"/>
      <button class="small flip" title="flip side">⇄</button>`;
    root.appendChild(row);
    row.querySelector('.on').addEventListener('change', (e) => { c.on = e.target.checked; updateCuts(); });
    row.querySelector('.pos').addEventListener('input', (e) => { c.pos = +e.target.value; updateCuts(); });
    row.querySelector('.flip').addEventListener('click', () => { c.flip = !c.flip; updateCuts(); });
  }
}

function updateCuts() {
  activePlanes.length = 0;
  for (const c of state.cuts) {
    const p = planes[c.axis];
    const n = new THREE.Vector3(); n.setComponent(c.axis, c.flip ? 1 : -1);
    p.normal.copy(n);
    p.constant = c.flip ? -c.pos : c.pos; // keeps points with n·x + constant >= 0
    if (c.on) activePlanes.push(p);
  }
  for (const k in state.tissues) {
    const t = state.tissues[k];
    t.mat.clippingPlanes = activePlanes; t.depthMesh.material.clippingPlanes = activePlanes;
    applyTissue(t);
  }
  volren?.setClipPlanes(state.cuts);
  slices?.update(state.cuts, state.sliceMode, state.window);
  flow?.setClipPlanes(activePlanes);
  for (const entry of thermalWall.meshes.values()) entry.material.clippingPlanes = activePlanes;
}

// ---------------------------------------------------------------- UI: misc
for (const name in PRESETS) {
  const b = document.createElement('button');
  b.textContent = name;
  b.addEventListener('click', () => applyPreset(name));
  $('#presets').appendChild(b);
}
$('#mode').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  state.mode = b.dataset.v;
  document.querySelectorAll('#mode button').forEach((x) => x.classList.toggle('on', x === b));
  for (const k in state.tissues) applyTissue(state.tissues[k]);
  setThermalWallVisibility(thermalWall.visible);
  volren?.setMode(state.mode);
});
$('#slicemode').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  state.sliceMode = b.dataset.v;
  document.querySelectorAll('#slicemode button').forEach((x) => x.classList.toggle('on', x === b));
  updateCuts();
});
$('#window')?.addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  state.window = b.dataset.v;
  document.querySelectorAll('#window button').forEach((x) => x.classList.toggle('on', x === b));
  updateCuts();
  volren?.setWindow(state.window);
});
$('#exposure').addEventListener('input', (e) => { renderer.toneMappingExposure = +e.target.value; });
$('#density').addEventListener('input', (e) => volren?.setDensity(+e.target.value));
$('#quality').addEventListener('input', (e) => volren?.setQuality(+e.target.value));
$('#bg').addEventListener('input', (e) => scene.background.set(e.target.value));
$('#shot').addEventListener('click', () => {
  render();
  const a = document.createElement('a');
  a.download = `airway3d-${state.dataset}-${Date.now()}.png`;
  a.href = renderer.domElement.toDataURL('image/png');
  a.click();
});
$('#flow-on').addEventListener('change', (e) => flow?.setEnabled(e.target.checked));
$('#flow-q').addEventListener('input', (e) => { $('#flow-q-val').textContent = `${e.target.value} mL/s`; flow?.setFlowRate(+e.target.value); airwayPanel?.setFlowRate(+e.target.value); lab?.flowChanged(); });
$('#flow-n').addEventListener('input', (e) => flow?.setCount(+e.target.value));
$('#flow-style').innerHTML = Object.entries(PARTICLE_STYLES).map(([k, s]) => `<option value="${k}">${s.label}</option>`).join('');
$('#flow-style').addEventListener('change', (e) => flow?.setStyle(e.target.value));
function updateFlowColorNote(mode) {
  const note = $('#flow-color-note');
  const legend = $('#flow-thermal-legend');
  if (!note) return;
  if (lab?.cfd?.enabled) {
    setThermalWallVisibility(false);if(legend)legend.style.display='none';
    note.textContent=lab.cfd.thermal?'Particles show speed or origin. Recorded 3-D CFD walls show independently calculated wall temperature (°C) or sensible heat flux (W/m²), synchronized to this breath.':'Particles show speed or origin. No recorded 3-D wall temperatures for this scenario; speed colours are not temperature.';
    return;
  }
  if (legend) legend.style.display = mode === 'thermal' ? 'flex' : 'none';
  const walls = $('#flow-thermal-walls');
  const wallOn = mode === 'thermal' && !!walls?.checked;
  setThermalWallVisibility(wallOn);
  note.textContent = mode === 'thermal'
    ? `Thermal colour: particles${wallOn ? ' and airway walls' : ''} use the live tube-model mucosal heat flux during inspiration; exhaled and room particles are neutral.`
    : mode === 'side' ? 'Side mode: particle origin is carried from each nostril through the stored field.'
      : 'Speed mode: colour follows local particle speed.';
}
$('#flow-color').addEventListener('change', (e) => { flow?.setColorMode(e.target.value); updateFlowColorNote(e.target.value); });
$('#flow-thermal-walls').addEventListener('change', () => updateFlowColorNote($('#flow-color').value));
$('#flow-size').addEventListener('input', (e) => flow?.setSize(+e.target.value));
$('#flow-breathe').addEventListener('change', (e) => flow?.setBreathing(e.target.checked));
$('#flow-period').addEventListener('input', (e) => { $('#flow-period-val').textContent = `${e.target.value} s`; flow?.setPeriod(+e.target.value); });
$('#flow-speed').addEventListener('input', (e) => { if (flow) flow.timeScale = +e.target.value; });
$('#flow-trail').addEventListener('input', (e) => flow?.setTrail(+e.target.value));

// ---------------------------------------------------------------- panel rail
// Each panel stows into a tab on the left edge; open panels sit side by side.
// body.lab-open mirrors the scenario lab so labels dock to the right column.
{
  const KEY = 'airway3d.panels';
  const tabs = [...document.querySelectorAll('#rail .tab')];
  const saved = (() => { try { return JSON.parse(localStorage.getItem(KEY)); } catch { return null; } })();
  const setOpen = (id, open) => {
    document.getElementById(id).classList.toggle('stowed', !open);
    tabs.find((t) => t.dataset.panel === id)?.setAttribute('aria-pressed', String(open));
    if (id === 'scenario-lab') document.body.classList.toggle('lab-open', open);
  };
  // One panel at a time, like a tab strip; clicking the open tab stows it.
  const show = (id) => { for (const t of tabs) setOpen(t.dataset.panel, t.dataset.panel === id); };
  show(typeof saved === 'string' || (saved === null && localStorage.getItem(KEY) === 'null') ? saved : 'scenario-lab');
  for (const t of tabs) t.addEventListener('click', () => {
    const id = t.dataset.panel, open = !document.getElementById(id).classList.contains('stowed');
    show(open ? null : id);
    localStorage.setItem(KEY, JSON.stringify(open ? null : id));
    window.dispatchEvent(new Event('resize')); // recentre the head in the free area
  });
}

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'r' || e.key === 'R') { controls.reset(); camera.position.set(120, 40, 420).multiplyScalar(portraitScale); controls.target.set(0, 0, 0); }
  const names = Object.keys(PRESETS);
  if (/^[1-6]$/.test(e.key) && names[+e.key - 1]) applyPreset(names[+e.key - 1]);
  if (e.key === 'f' || e.key === 'F') { const cb = $('#flow-on'); cb.checked = !cb.checked; flow?.setEnabled(cb.checked); }
});

// ---------------------------------------------------------------- resize / loop
let portraitScale = 1;
function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  const nextScale=1/Math.min(1,camera.aspect);
  camera.position.sub(controls.target).multiplyScalar(nextScale/portraitScale).add(controls.target);
  portraitScale=nextScale;
  // Centre the head in the area left of the rail and the open panel.
  const open=document.querySelector('#dock .panel:not(.stowed)');
  const covered=open?open.getBoundingClientRect().right:document.getElementById('rail').getBoundingClientRect().right;
  camera.setViewOffset(w,h,-Math.round(covered/2),w<=760?Math.round(h*.22):0,w,h);
  camera.updateProjectionMatrix();
  volren?.resize(w, h);
}
window.addEventListener('resize', resize);
resize();

const clock = new THREE.Clock();
const phaseEl = $('#flow-phase');
let phaseText = '';
function render() {
  const dt = Math.min(clock.getDelta(), 0.05);
  controls.update();
  lab?.step(dt);
  if (flow) {
    flow.step(dt, camera);
    const ph = flow.enabled ? flow.phaseInfo() : null;
    const txt = ph ? `${ph.label} ${Math.abs(ph.q).toFixed(0)} mL/s` : '';
    if (txt !== phaseText) { phaseText = txt; phaseEl.textContent = txt; }
    const status=$('#scenario-flow-status');
    if(status){
      const fractions=flow.transport?.fractions || flow.meta.split;
      status.textContent=ph?`${txt} · L ${Math.abs(ph.flow?.L??ph.q*fractions.L).toFixed(0)} / R ${Math.abs(ph.flow?.R??ph.q*fractions.R).toFixed(0)} mL/s`:'Particles paused';
      status.dataset.enabled=String(flow.enabled);status.dataset.position=state.dataset==='post'?'supine':lab?.settings.position || 'supine';
      $('#scenario-flow-toggle').textContent=flow.enabled?'Pause particles':'Resume particles';
      $('#scenario-flow-toggle').setAttribute('aria-pressed',String(flow.enabled));
    }
  }
  lab?.cfd?.step(); // sample thermal walls and CFD readings after the same airflow clock tick
  if(volren&&state.mode!=='surface')volren.render(surfaces,state.mode);
  else renderer.render(scene,camera);
}
renderer.setAnimationLoop(render);

// ---------------------------------------------------------------- boot
window.__errors = [];
window.addEventListener('error', (e) => window.__errors.push(String(e.message)));
const _cerr = console.error.bind(console);
console.error = (...a) => { window.__errors.push(a.map(String).join(' ').slice(0, 2000)); _cerr(...a); };
window.__app = { THREE, surfaces, state, render: () => render(), get volren() { return volren; }, get slices() { return slices; }, get flow() { return flow; }, get lab() { return lab; }, thermalWall, renderer, scene, camera, controls, applyPreset, loadDataset, updateCuts };
(async () => {

  const names = [];
  for (const n of ['pre', 'post']) { if (await fetchJSON(`./data/${n}/meta.json`)) names.push(n); }
  if (!names.length) { loadtext.textContent = 'no datasets in viewer/data — run the pipeline first'; return; }

  await loadDataset(names[0]);
  applyPreset('Dissect');
  const pre=await datasetCache.get('pre');
  lab=new ScenarioLab({scene,surfaces,camera,controls,state,base:pre.aw,airwayPanel,loadDataset,applyPreset,updateCuts,updateThermalWalls, get flow(){return flow;},setFlow(value){flow=value;syncFlowSideColors();updateCuts();}, q:()=>+$('#flow-q').value});
  $('#dataset-status').textContent='CT reference · face up';
  canvas.dataset.sceneId=scene.uuid;canvas.dataset.skinGeometry=state.tissues.skin?.mesh.geometry.uuid;canvas.dataset.volumeTexture=volumeTex.uuid;canvas.dataset.surfaceCount=surfaces.children.length;
})();
