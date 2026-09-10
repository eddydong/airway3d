// CT slice quads shown at the cut planes: bone / soft-tissue window or label colours.
import * as THREE from 'three';

const VERT = /* glsl */`
#include <clipping_planes_pars_vertex>
out vec3 vWorld;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorld = wp.xyz;
  vec4 mvPosition = viewMatrix * wp;
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}
`;

const FRAG = /* glsl */`
precision highp sampler3D;
layout(location = 0) out highp vec4 pc_fragColor;
#define gl_FragColor pc_fragColor
#include <clipping_planes_pars_fragment>
in vec3 vWorld;
uniform sampler3D tGray, tSoft, tLabels;
uniform sampler2D tTF;
uniform vec3 boxMin, boxSize;
uniform int mode;      // 0 CT bone, 1 CT soft, 2 labels
void main() {
  #include <clipping_planes_fragment>
  vec3 uvw = (vWorld - boxMin) / boxSize;
  // stay one voxel inside the box: the border voxels hold the screenshot frame line
  vec3 margin = 1.0 / boxSize;  // ~1 mm on a 0.55 mm grid
  if (any(lessThan(uvw, margin)) || any(greaterThan(uvw, 1.0 - margin))) discard;
  float g = mode == 0 ? texture(tGray, uvw).r : texture(tSoft, uvw).r;
  int li = int(texture(tLabels, uvw).r * 255.0 + 0.5);
  if (li == 0 && g < 0.04) discard;  // air outside the head: see through the cut
  // the PNG grey levels are display (sRGB) values: linearise so the CT comes out exactly
  // as scanned after the output colour-space conversion (material is not tone mapped)
  vec3 col;
  if (mode == 2) {
    vec4 tf = texelFetch(tTF, ivec2(li, 0), 0);
    float gs = pow(texture(tSoft, uvw).r, 2.2);
    col = li == 0 ? vec3(0.002) : mix(vec3(gs) * 0.6, tf.rgb * (0.3 + 1.2 * gs), 0.7);
    if (li == 7 || li == 8) col = tf.rgb * 0.6;
  } else {
    col = vec3(pow(g, 2.2));
  }
  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

export class SlicePlanes {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    scene.add(this.group);
    this.tf = new THREE.DataTexture(new Uint8Array(256 * 4), 256, 1);
    this.tf.needsUpdate = true;
    this.quads = [];
    for (let axis = 0; axis < 3; axis++) {
      const mat = new THREE.ShaderMaterial({
        glslVersion: THREE.GLSL3, vertexShader: VERT, fragmentShader: FRAG, clipping: true, side: THREE.DoubleSide, toneMapped: false,
        uniforms: { tGray: { value: null }, tSoft: { value: null }, tLabels: { value: null }, tTF: { value: this.tf }, boxMin: { value: new THREE.Vector3() }, boxSize: { value: new THREE.Vector3(1, 1, 1) }, mode: { value: 1 } },
      });
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat);
      mesh.visible = false;
      mesh.renderOrder = -10;
      this.group.add(mesh);
      this.quads.push(mesh);
    }
    this.planes = [new THREE.Plane(), new THREE.Plane(), new THREE.Plane()];
  }

  setVolume({ gray, soft, labels, box }) {
    this.box = box;
    const size = new THREE.Vector3().subVectors(box.max, box.min);
    for (let axis = 0; axis < 3; axis++) {
      const q = this.quads[axis];
      const u = q.material.uniforms;
      u.tGray.value = gray; u.tSoft.value = soft; u.tLabels.value = labels;
      u.boxMin.value.copy(box.min); u.boxSize.value.copy(size);
      // orient the unit plane perpendicular to the axis and scale to the box face
      q.rotation.set(0, 0, 0);
      if (axis === 0) { q.rotation.y = Math.PI / 2; q.scale.set(size.z, size.y, 1); }
      if (axis === 1) { q.rotation.x = -Math.PI / 2; q.scale.set(size.x, size.z, 1); }
      if (axis === 2) { q.scale.set(size.x, size.y, 1); }
    }
    this.centre = box.getCenter(new THREE.Vector3());
  }

  setTransfer(tf) {
    const d = this.tf.image.data; d.fill(0);
    const c = new THREE.Color();
    for (const id in tf) {
      c.set(tf[id].color);  // hex is sRGB; Color.set() already converts to the linear working space
      const i = +id * 4;
      d[i] = Math.round(c.r * 255); d[i + 1] = Math.round(c.g * 255); d[i + 2] = Math.round(c.b * 255); d[i + 3] = 255;
    }
    this.tf.needsUpdate = true;
  }

  update(cuts, sliceMode, window) {
    if (!this.box) return;
    const mode = sliceMode === 'labels' ? 2 : (window === 'bone' ? 0 : 1);
    for (const c of cuts) {
      const p = this.planes[c.axis];
      const n = new THREE.Vector3(); n.setComponent(c.axis, c.flip ? 1 : -1);
      p.normal.copy(n); p.constant = c.flip ? -c.pos : c.pos;
    }
    for (const c of cuts) {
      const q = this.quads[c.axis];
      q.visible = c.on && sliceMode !== 'off';
      if (!q.visible) continue;
      q.position.copy(this.centre);
      q.position.setComponent(c.axis, c.pos + (c.flip ? 0.05 : -0.05));
      q.material.uniforms.mode.value = mode;
      // clip this slice by the *other* active cuts
      q.material.clippingPlanes = cuts.filter((o) => o.on && o.axis !== c.axis).map((o) => this.planes[o.axis]);
    }
  }
}
