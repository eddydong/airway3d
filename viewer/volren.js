// GPU volume renderer: single-pass ray marching through the CT volume with a per-tissue
// transfer function (label volume -> colour/opacity from the tissue UI), gradient shading,
// cut planes and depth-correct compositing with the surface meshes (hybrid mode).
import * as THREE from 'three';

const VERT = /* glsl */`
out vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const FRAG = /* glsl */`
precision highp sampler3D;
layout(location = 0) out highp vec4 pc_fragColor;
#define gl_FragColor pc_fragColor
in vec2 vUv;

uniform sampler2D tColor;
uniform sampler2D tDepth;
uniform sampler3D tGray;      // bone window
uniform sampler3D tSoft;      // soft-tissue window
uniform sampler3D tLabels;    // uint8 labels (nearest)
uniform sampler2D tTF;        // 256 x 1: rgb colour, a opacity, per label
uniform sampler2D tWin;       // 256 x 1: r = 1 -> use bone window for this label
uniform vec3 boxMin, boxSize, texel;
uniform vec3 cameraPos, camForward;
uniform mat4 invProjView;
uniform float cameraNear, cameraFar;
uniform float stepMM, density, time;
uniform int nPlanes;
uniform vec4 planes[3];
uniform vec3 lightDir;
uniform int useSoft;          // 0: bone window everywhere, 1: soft where flagged, 2: soft everywhere

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

float viewZFromDepth(float d) {
  // perspective: reconstruct view-space z (negative) from the non-linear depth buffer
  float z = d * 2.0 - 1.0;
  return (2.0 * cameraNear * cameraFar) / (cameraFar + cameraNear - z * (cameraFar - cameraNear));
}

float grayAt(vec3 uvw, float useS) {
  return mix(texture(tGray, uvw).r, texture(tSoft, uvw).r, useS);
}

void main() {
  vec4 sceneCol = texture(tColor, vUv);
  float depth = texture(tDepth, vUv).r;
  vec2 ndc = vUv * 2.0 - 1.0;
  vec4 pn = invProjView * vec4(ndc, -1.0, 1.0); pn /= pn.w;
  vec4 pf = invProjView * vec4(ndc, 1.0, 1.0); pf /= pf.w;
  vec3 dir = normalize(pf.xyz - pn.xyz);
  vec3 o = cameraPos;

  // slab intersection with the volume box
  vec3 inv = 1.0 / dir;
  vec3 ta = (boxMin - o) * inv, tb = (boxMin + boxSize - o) * inv;
  vec3 tmin = min(ta, tb), tmax = max(ta, tb);
  float t0 = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
  float t1 = min(min(tmax.x, tmax.y), tmax.z);
  // cut planes keep n.x + d >= 0
  for (int i = 0; i < 3; i++) {
    if (i >= nPlanes) break;
    vec3 n = planes[i].xyz; float d = planes[i].w;
    float den = dot(n, dir);
    float num = dot(n, o) + d;
    if (abs(den) < 1e-6) { if (num < 0.0) { t1 = -1.0; } }
    else { float t = -num / den; if (den > 0.0) t0 = max(t0, t); else t1 = min(t1, t); }
  }
  // stop at scene geometry
  float sceneDist = viewZFromDepth(depth) / max(dot(dir, camForward), 1e-4);
  if (depth < 1.0) t1 = min(t1, sceneDist);
  // background pixels (nothing rendered) bypass tone mapping so the page colour is exact
  vec3 surf = depth < 1.0 ? sceneCol.rgb : vec3(0.0);
  vec3 bg = depth < 1.0 ? vec3(0.0) : sceneCol.rgb;
  if (t1 <= t0) {
    gl_FragColor = vec4(surf, 1.0);
    #include <tonemapping_fragment>
    gl_FragColor.rgb += bg;
    #include <colorspace_fragment>
    return;
  }

  float jitter = hash(vUv * 1024.0 + fract(time)) * stepMM;
  float t = t0 + jitter;
  vec3 acc = vec3(0.0);
  float A = 0.0;
  int guard = 0;
  while (t < t1 && guard < 1400) {
    guard++;
    vec3 p = o + dir * t;
    vec3 uvw = (p - boxMin) / boxSize;
    float lab = texture(tLabels, uvw).r * 255.0;
    int li = int(lab + 0.5);
    vec4 tf = texelFetch(tTF, ivec2(li, 0), 0);
    if (tf.a <= 0.001) { t += stepMM * 1.5; continue; }
    float useS = useSoft == 0 ? 0.0 : (useSoft == 2 ? 1.0 : 1.0 - texelFetch(tWin, ivec2(li, 0), 0).r);
    float g = grayAt(uvw, useS);
    // gradient (central differences) for shading; air classes are shaded by their label boundary instead
    vec3 gx = vec3(grayAt(uvw + vec3(texel.x, 0, 0), useS) - grayAt(uvw - vec3(texel.x, 0, 0), useS),
                   grayAt(uvw + vec3(0, texel.y, 0), useS) - grayAt(uvw - vec3(0, texel.y, 0), useS),
                   grayAt(uvw + vec3(0, 0, texel.z), useS) - grayAt(uvw - vec3(0, 0, texel.z), useS));
    float gm = length(gx);
    vec3 n = gm > 1e-4 ? normalize(-gx / boxSize * boxSize.x) : vec3(0.0);
    float isAir = (li == 7 || li == 8) ? 1.0 : 0.0;
    float shade = mix(mix(0.55, 1.15, clamp(g * 1.6, 0.0, 1.0)), 1.0, isAir);
    float ndl = gm > 1e-4 ? max(dot(n, lightDir), 0.0) : 0.6;
    float spec = gm > 1e-4 ? pow(max(dot(reflect(-lightDir, n), -dir), 0.0), 40.0) : 0.0;
    float edge = clamp(gm * 6.0, 0.0, 1.0);
    vec3 col = tf.rgb * shade * (0.45 + 0.75 * ndl) + spec * 0.35 * edge;
    // opacity: the tissue's UI opacity is reached across ~3 mm of tissue (density 1),
    // boosted at edges so interfaces read as surfaces
    float k = -log(1.0 - min(tf.a, 0.995)) / 3.0;
    float a = 1.0 - exp(-k * density * stepMM * (0.6 + 1.2 * edge));
    acc += (1.0 - A) * a * col;
    A += (1.0 - A) * a;
    if (A > 0.985) break;
    t += stepMM;
  }
  gl_FragColor = vec4(acc + (1.0 - A) * surf, 1.0);
  #include <tonemapping_fragment>
  gl_FragColor.rgb += (1.0 - A) * bg;
  #include <colorspace_fragment>
}
`;

export class VolumeRenderer {
  constructor(renderer, scene, camera) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.mode = 'surface';
    this.window = 'soft';
    const size = renderer.getSize(new THREE.Vector2());
    const pr = renderer.getPixelRatio();
    this.rt = new THREE.WebGLRenderTarget(Math.floor(size.x * pr), Math.floor(size.y * pr), {
      type: THREE.HalfFloatType,
      depthTexture: new THREE.DepthTexture(Math.floor(size.x * pr), Math.floor(size.y * pr), THREE.UnsignedIntType),
    });
    this.rt.texture.colorSpace = THREE.LinearSRGBColorSpace;
    this.tf = new THREE.DataTexture(new Uint8Array(256 * 4), 256, 1);
    this.tf.needsUpdate = true;
    this.win = new THREE.DataTexture(new Uint8Array(256 * 4), 256, 1);
    this.win.needsUpdate = true;
    this.quality = 1;
    this.mat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VERT,
      fragmentShader: FRAG,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        tColor: { value: this.rt.texture }, tDepth: { value: this.rt.depthTexture },
        tGray: { value: null }, tSoft: { value: null }, tLabels: { value: null }, tTF: { value: this.tf }, tWin: { value: this.win },
        boxMin: { value: new THREE.Vector3() }, boxSize: { value: new THREE.Vector3(1, 1, 1) }, texel: { value: new THREE.Vector3(0.01, 0.01, 0.01) },
        cameraPos: { value: new THREE.Vector3() }, camForward: { value: new THREE.Vector3() }, invProjView: { value: new THREE.Matrix4() },
        cameraNear: { value: 1 }, cameraFar: { value: 3000 },
        stepMM: { value: 0.8 }, density: { value: 1.2 }, time: { value: 0 },
        nPlanes: { value: 0 }, planes: { value: [new THREE.Vector4(), new THREE.Vector4(), new THREE.Vector4()] },
        lightDir: { value: new THREE.Vector3(0.4, 0.7, 0.6).normalize() },
        useSoft: { value: 1 },
      },
    });
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.mat);
    this.quad.frustumCulled = false;
    this.quadScene = new THREE.Scene();
    this.quadScene.add(this.quad);
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  }

  setVolume({ gray, soft, labels, box, dims, labelsMeta }) {
    const u = this.mat.uniforms;
    u.tGray.value = gray; u.tSoft.value = soft; u.tLabels.value = labels;
    u.boxMin.value.copy(box.min);
    u.boxSize.value.subVectors(box.max, box.min);
    u.texel.value.set(1 / dims[0], 1 / dims[1], 1 / dims[2]);
    // labels that should be shaded with the bone window
    const w = this.win.image.data; w.fill(0);
    for (const id in labelsMeta) if (['bone', 'teeth', 'airway', 'sinus'].includes(labelsMeta[id].key)) w[+id * 4] = 255;
    this.win.needsUpdate = true;
    this.setQuality(this.quality);
  }

  setTransfer(tf) {
    const d = this.tf.image.data; d.fill(0);
    const c = new THREE.Color();
    for (const id in tf) {
      c.set(tf[id].color);  // hex is sRGB; Color.set() already converts to the linear working space
      const i = +id * 4;
      d[i] = Math.round(c.r * 255); d[i + 1] = Math.round(c.g * 255); d[i + 2] = Math.round(c.b * 255); d[i + 3] = Math.round(tf[id].opacity * 255);
    }
    this.tf.needsUpdate = true;
  }

  setMode(m) { this.mode = m; }
  setWindow(w) { this.window = w; this.mat.uniforms.useSoft.value = w === 'bone' ? 0 : (w === 'soft' ? 2 : 1); }
  setDensity(d) { this.mat.uniforms.density.value = d; }
  setQuality(q) {
    this.quality = q;
    const bs = this.mat.uniforms.boxSize.value;
    const diag = bs.length() || 300;
    this.mat.uniforms.stepMM.value = Math.max(0.3, diag / (420 * q));
  }
  setClipPlanes(cuts) {
    const u = this.mat.uniforms;
    let n = 0;
    for (const c of cuts) {
      if (!c.on) continue;
      const v = u.planes.value[n++];
      v.set(0, 0, 0, 0);
      v.setComponent(c.axis, c.flip ? 1 : -1);
      v.w = c.flip ? -c.pos : c.pos;
    }
    u.nPlanes.value = n;
  }
  resize(w, h) {
    const pr = this.renderer.getPixelRatio();
    this.rt.setSize(Math.floor(w * pr), Math.floor(h * pr));
  }

  render(surfaces, mode) {
    const r = this.renderer, cam = this.camera, u = this.mat.uniforms;
    const prevVis = surfaces.visible;
    if (mode === 'volume') surfaces.visible = false;
    r.setRenderTarget(this.rt);
    r.render(this.scene, cam);
    surfaces.visible = prevVis;
    r.setRenderTarget(null);
    u.cameraPos.value.copy(cam.position);
    cam.getWorldDirection(u.camForward.value);
    u.invProjView.value.copy(cam.projectionMatrix).multiply(cam.matrixWorldInverse).invert();
    u.cameraNear.value = cam.near; u.cameraFar.value = cam.far;
    u.time.value = (u.time.value + 0.61803) % 1.0;
    r.render(this.quadScene, this.quadCam);
  }
}
