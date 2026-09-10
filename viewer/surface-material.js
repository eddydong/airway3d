// Stable alpha blending: no stochastic coverage and no previous-frame history.
// Opaque surfaces participate in ordinary depth testing; transparent surfaces
// blend in camera order without writing an invisible occluding depth layer.
export function setSurfaceOpacity(material,opacity){
  const transparent=opacity<1||material.transmission>0;
  if(material.alphaHash||material.transparent!==transparent||material.depthWrite===transparent)material.needsUpdate=true;
  material.opacity=opacity;
  material.alphaHash=false;
  material.transparent=transparent;
  material.depthTest=true;
  material.depthWrite=!transparent;
}
