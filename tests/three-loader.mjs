// Match the viewer's import map when exercising its real particle engine in Node.
export async function resolve(specifier,context,nextResolve) {
  if(specifier==='three')return {url:new URL('../viewer/vendor/three.module.js',import.meta.url).href,shortCircuit:true};
  if(specifier.startsWith('three/addons/'))return {url:new URL('../viewer/vendor/addons/'+specifier.slice('three/addons/'.length),import.meta.url).href,shortCircuit:true};
  return nextResolve(specifier,context);
}
