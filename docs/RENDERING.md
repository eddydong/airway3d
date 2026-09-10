# Surface depth and transparency

The viewer renders each frame directly. It uses no alpha hashing, changing
coverage pattern, framebuffer accumulation, or temporal smoothing. The previous
stochastic renderer produced visible flicker when the camera moved and was removed.

`viewer/surface-material.js` enables depth writing for opaque surfaces. Lower
opacity uses ordinary alpha blending without depth writing. The invisible skin
depth pass stays disabled: it wrote a solid depth mask and hid nearer
translucent objects.

Nested tissues use a fixed inside→outside draw order (airway first, skin last).
That matches the anatomy and does not re-sort hundreds of thousands of triangles
while the camera moves. Clip planes switch those shells to double-sided faces,
and only then does `viewer/surface-order.js` sort their triangles.

CFD and thermal walls still sort faces when they are translucent, because one
mesh holds the near and far wall. Sorting uses a counting pass on view-space
depth and runs only when the viewing direction or geometry changes. It
preserves field attributes; thermal probes resolve the original face ID through
the reordered index buffer.

CFD thermal walls start at 100% opacity, so their visible surface and occlusion
use normal per-pixel depth testing. Their opacity control still supports
translucent inspection. Thermal color bands and physical readings are unchanged.

Conventional alpha blending still approximates intersections between separate
translucent meshes. This is not an exact order-independent transparency renderer.
Opaque walls do not have that limitation.

Run `node --test tests/cfd-{ui,thermal,particles,transient}.test.mjs tests/surface-order.test.mjs`.
With the viewer server running, open `/tests/surface-depth.html` for synthetic GPU
checks, `/tests/viewer-stability.html` for paused-frame and camera-return checks,
and `/tests/viewer-performance.html` for CPU time while rotating the actual scene.
