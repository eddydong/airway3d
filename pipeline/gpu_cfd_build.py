"""Build the optional native macOS GPU recording backend from pinned sources.

FluidX3D is an external dependency, retained under ignored work/gpu-cfd with
its license. The app never downloads or compiles dependencies during a request.
"""
from pathlib import Path
import hashlib
import subprocess
import shutil

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'work/gpu-cfd'
SOURCE = ROOT / 'FluidX3D'
COMMIT = '9f3a995599740082690ace8cdc7a82b0396adddd'
VERSION = 'gpu-lbm-trt-les-resting-breath-6'
BINARY = ROOT / 'airway-gpu'


def build():
    ROOT.mkdir(exist_ok=True, parents=True)
    if not SOURCE.exists():
        subprocess.run(['git', 'clone', 'https://github.com/ProjectPhysX/FluidX3D.git', str(SOURCE)], check=True)
    subprocess.run(['git', '-C', str(SOURCE), 'cat-file', '-e', COMMIT], check=True)
    def original(name):
        return subprocess.check_output(['git', '-C', str(SOURCE), 'show', f'{COMMIT}:src/{name}'], text=True)
    # Restore just our private build's tracked source files from the pin.
    names=subprocess.check_output(['git','-C',str(SOURCE),'ls-tree','-r','--name-only',COMMIT,'src'],text=True).splitlines()
    for name in names:
        path=SOURCE/name;path.parent.mkdir(exist_ok=True,parents=True)
        path.write_bytes(subprocess.check_output(['git','-C',str(SOURCE),'show',f'{COMMIT}:{name}']))
    defines=original('defines.hpp')
    for flag in ['BENCHMARK','FP16S','SRT']:
        defines=defines.replace(f'#define {flag} //',f'//#define {flag} //')
    # SUBGRID: Smagorinsky-Lilly eddy viscosity. Air at breathing speeds on a
    # 0.35-0.7 mm lattice has a relaxation time within 1e-3 of the stability
    # limit and blows up in under 0.1 s without it. Results are LES, not DNS.
    for flag in ['TRT','VOLUME_FORCE','EQUILIBRIUM_BOUNDARIES','SUBGRID']:
        defines=defines.replace(f'//#define {flag} //',f'#define {flag} //')
    # Opening ghost cells copy their fluid neighbour's velocity every step, so
    # rho/u must be written by stream_collide rather than only on demand.
    defines+='\n#define UPDATE_FIELDS // airway openings read neighbour velocities each step\n'
    (SOURCE/'src/defines.hpp').write_text(defines)
    kernel=original('kernel.cpp')
    old='''\tif(flagsn_bo==TYPE_E) {
\t\trhon = rho[               n]; // apply preset velocity/density
\t\tuxn  = u[                 n];
\t\tuyn  = u[    def_N+(ulong)n];
\t\tuzn  = u[2ul*def_N+(ulong)n];
\t} else {
\t\tcalculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn); // calculate density and velocity fields from fi
\t}'''
    # Pressure openings. Each open face of the voxel domain has one ghost
    # cell outside it (TYPE_E, with the D3Q19 direction index of its fluid
    # neighbour in flag bits 2-4 and TYPE_X / TYPE_Y for nostril / throat).
    # Nostrils sit at ambient pressure, the throat at ambient minus
    # A times the resting-beta-1 waveform; the openings are planar (cfd_geometry.flatten_nostrils
    # and the inferior cut). The ghost's rule follows the flow direction read
    # two cells inside the opening:
    # - Inflow: anti-bounce-back, f_i_bar = -f_i + feq_i + feq_i_bar with the
    #   reservoir density and the local velocity in the O(u^2) term. It
    #   imposes the reservoir pressure exactly at the face (the duct test's
    #   30 % opening loss vanished) and prescribes no velocity.
    # - Outflow: equilibrium at the reservoir density with the extrapolated
    #   velocity, the standard open outlet. Anti-bounce-back on the turbulent
    #   throat outflow reflected the exiting eddies and diverged.
    # The equilibrium ghost alone (FluidX3D's stock TYPE_E) was tried first:
    # at rest it acted as a porous plug, and copying the neighbour velocity
    # into an inflow fed back into 3-14 m/s jets on stair corners of the
    # oblique nostril rim, which is why the nostrils are now planar.
    new='''\tbool opening_inflow=false;
\tif(flagsn_bo==TYPE_E) {
\t\tconst float phase=fx>0.0f ? fmod((float)t*fx/(2.0f*3.141592653589793f),1.0f) : 0.0f;
\t\tconst float x=phase<0.4f ? phase/0.4f : (phase-0.4f)/0.6f;
\t\tconst float tail=1.0f-x;
\t\tconst float breath=phase<0.4f ? 16.0f*x*x*tail*tail : -(112.0f/3.0f)*x*x*tail*tail*tail*tail;
\t\tconst float amplitude=fx>0.0f ? breath : fmin((float)t/1000.0f,1.0f);
\t\trhon=(flagsn&TYPE_Y) ? 1.0f-fy*amplitude : 1.0f;
\t\tconst uint d=(flagsn>>2)&7u; const uxx m=j[d];
\t\tconst uxx m2=(uxx)((long)m+((long)m-(long)n)); // second cell inward along the face normal
\t\tconst uxx r=flags[m2]==0u?m2:m;
\t\tuxn=u[r];uyn=u[def_N+(ulong)r];uzn=u[2ul*def_N+(ulong)r];
\t\topening_inflow=(d==1u?uxn:d==2u?-uxn:d==3u?uyn:d==4u?-uyn:d==5u?uzn:-uzn)>0.0f;
\t} else {
\t\tcalculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn);
\t}'''
    if kernel.count(old)!=1: raise RuntimeError('Pinned boundary patch no longer matches')
    kernel=kernel.replace(old,new)
    old_trt='\tfor(uint i=0u; i<def_velocity_set; i++) fhn[i] = flagsn_bo==TYPE_E ? feq[i] : fma(0.5f*wp, feq[i]-fhn[i]+feb[i]-fhb[i], fma(0.5f*wm, feq[i]-feb[i]-fhn[i]+fhb[i], fhn[i]+Fin[i])); // perform collision (TRT)'
    if kernel.count(old_trt)!=1: raise RuntimeError('Pinned TRT collision no longer matches')
    kernel=kernel.replace(old_trt,'\tfor(uint i=0u; i<def_velocity_set; i++) fhn[i] = flagsn_bo==TYPE_E ? (opening_inflow ? feq[i]+feb[i]-fhb[i] : feq[i]) : fma(0.5f*wp, feq[i]-fhn[i]+feb[i]-fhb[i], fma(0.5f*wm, feq[i]-feb[i]-fhn[i]+fhb[i], fhn[i]+Fin[i])); // TRT collision; anti-bounce-back inflow / equilibrium outflow openings')
    # fx carries waveform angular frequency, not an external body force.
    kernel=kernel.replace('float fxn=fx, fyn=fy, fzn=fz;', 'float fxn=0.0f, fyn=0.0f, fzn=0.0f;')
    # Temporal momentum filter. In passages one or two cells wide, halfway
    # bounce-back on both walls returns a cell's wall-normal momentum to the
    # same cell (or its pair) every step: a period-2 mode that collisions
    # conserve and the over-relaxed stress at tau -> 1/2 feeds, so the run
    # diverged 0.25 s into expiration on both lattices. A Guo force
    # F = -alpha*rho*(u - u_previous) damps period-2 momentum oscillations by
    # 2*alpha per step and leaves smooth flow with an unsteady inertia of
    # (1+alpha)*rho; steady and quasi-steady flow are unchanged. alpha rides
    # in fz (fx, fy carry the waveform).
    old_force='''\t\tcalculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn);
\t}
)+"#endif"+R( // EQUILIBRIUM_BOUNDARIES
\tfloat fxn=0.0f, fyn=0.0f, fzn=0.0f;'''
    if kernel.count(old_force)!=1: raise RuntimeError('Pinned forcing patch no longer matches')
    kernel=kernel.replace(old_force,'''\t\tcalculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn);
\t}
)+"#endif"+R( // EQUILIBRIUM_BOUNDARIES
\tfloat fxn=0.0f, fyn=0.0f, fzn=0.0f;
\tif(flagsn_bo!=TYPE_E) { // temporal momentum filter against the period-2 mode of thin passages
\t\tfxn=-fz*rhon*(uxn-u[n]); fyn=-fz*rhon*(uyn-u[def_N+(ulong)n]); fzn=-fz*rhon*(uzn-u[2ul*def_N+(ulong)n]);
\t}''')
    # Never stabilize an invalid solution by silently clipping its velocities.
    for axis in ['x','y','z']:
        v=f'u{axis}n'
        kernel=kernel.replace(f'clamp(fma(f{axis}n, rho2, {v}), -def_c, def_c)',f'fma(f{axis}n, rho2, {v})')
        kernel=kernel.replace(f'clamp({v}, -def_c, def_c)',v)
    # Apple's OpenCL compiler silently produces a no-op stream_collide kernel
    # when the Smagorinsky block indexes the private c(i) table inside its
    # loop (fields never change, no error is raised). Write the D3Q19
    # non-equilibrium stress tensor out explicitly instead.
    old_loop='''\t\tfor(uint i=1u; i<def_velocity_set; i++) {
\t\t\tconst float fneqi = fhn[i]-feq[i];
\t\t\tconst float cxi=c(i), cyi=c(def_velocity_set+i), czi=c(2u*def_velocity_set+i);
\t\t\tHxx += cxi*cxi*fneqi; //Hyx += cyi*cxi*fneqi; Hzx += czi*cxi*fneqi; // symmetric tensor
\t\t\tHxy += cxi*cyi*fneqi; Hyy += cyi*cyi*fneqi; //Hzy += czi*cyi*fneqi;
\t\t\tHxz += cxi*czi*fneqi; Hyz += cyi*czi*fneqi; Hzz += czi*czi*fneqi;
\t\t}'''
    if kernel.count(old_loop)!=1: raise RuntimeError('Pinned subgrid patch no longer matches')
    kernel=kernel.replace(old_loop,'''\t\tfloat fneq[def_velocity_set];
\t\tfor(uint i=0u; i<def_velocity_set; i++) fneq[i]=fhn[i]-feq[i];
\t\tHxx=fneq[1]+fneq[2]+fneq[7]+fneq[8]+fneq[9]+fneq[10]+fneq[13]+fneq[14]+fneq[15]+fneq[16];
\t\tHyy=fneq[3]+fneq[4]+fneq[7]+fneq[8]+fneq[11]+fneq[12]+fneq[13]+fneq[14]+fneq[17]+fneq[18];
\t\tHzz=fneq[5]+fneq[6]+fneq[9]+fneq[10]+fneq[11]+fneq[12]+fneq[15]+fneq[16]+fneq[17]+fneq[18];
\t\tHxy=fneq[7]+fneq[8]-fneq[13]-fneq[14];
\t\tHxz=fneq[9]+fneq[10]-fneq[15]-fneq[16];
\t\tHyz=fneq[11]+fneq[12]-fneq[17]-fneq[18];''')
    (SOURCE/'src/kernel.cpp').write_text(kernel)
    # Flag bits 2-4 (TYPE_T/F/I) carry the opening's neighbour direction. The
    # SURFACE and TEMPERATURE extensions that interpret them are disabled, so
    # relax the host-side checks that would otherwise reject these flags.
    host=original('lbm.cpp')
    for text in ['if(surface_used) print_error(','if(temperature_used) print_error(']:
        if host.count(text)!=1: raise RuntimeError('Pinned flag check no longer matches')
        host=host.replace(text,'if(false) print_error(')
    (SOURCE/'src/lbm.cpp').write_text(host)
    runner=REPO/'pipeline/gpu_cfd_runner.cpp'
    sources=[str(p) for p in (SOURCE/'src').glob('*.cpp') if p.name not in ['main.cpp','setup.cpp']]
    command=['clang++',str(runner),*sources,'-o',str(BINARY),'-std=c++17','-pthread','-O2',
             '-Wno-comment','-Wno-deprecated-declarations',f'-I{SOURCE}/src',f'-I{SOURCE}/src/OpenCL/include','-framework','OpenCL']
    with (ROOT/'build.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    identity=hashlib.sha256((VERSION+COMMIT+kernel+defines+host+runner.read_text()).encode()).hexdigest()
    (ROOT/'build-id.txt').write_text(identity)
    shutil.copy2(SOURCE/'LICENSE.md',ROOT/'FluidX3D-LICENSE.md')
    print(BINARY, identity)


if __name__=='__main__': build()
