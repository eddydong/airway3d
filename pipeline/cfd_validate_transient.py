"""Compare an oscillatory square duct with its independent Stokes-series solution.

For a harmonic pressure gradient G, mean velocity is (G/rho) times a
sine-eigenfunction transfer function. Fit the computed central pressure gradient
over the last cycle, away from the imposed uniform end profiles.
"""
import json
import numpy as np
from cfd_geometry import ROOT
from cfd_solver import RHO, NU, read_internal


def harmonic_gradient(q_ml_s,period,width=.004):
    m=np.arange(1,202,2,dtype=float);m,n=np.meshgrid(m,m,indexing='ij')
    omega=2*np.pi/period
    transfer=np.sum(64/(np.pi**4*m*m*n*n)/(NU*np.pi**2*(m*m+n*n)/width**2+1j*omega))
    return RHO*(-1j*q_ml_s*1e-6)/(width**2*transfer)


def measure(case):
    result=json.loads((case/'result.json').read_text());solver=result['solver']
    with np.load(case/'domain.npz') as z: centres=z['centres']
    y=centres[:,1]/1000;middle=(y>.02)&(y<.04)
    ts=[];grad=[]
    for f in result['field']['frames']:
        t=f['timeS']
        if t<solver['durationS']-solver['periodS']-1e-7:continue
        folder=min((p for p in case.iterdir() if p.is_dir() and p.name.replace('.','').isdigit()),key=lambda p:abs(float(p.name)-t))
        pressure=read_internal(folder/'p',len(y))[:,0]*RHO
        ts.append(t);grad.append(float(np.polyfit(y[middle],pressure[middle],1)[0]))
    phase=np.array(ts)*2*np.pi/solver['periodS']
    fit=np.linalg.lstsq(np.array([np.cos(phase),np.sin(phase),np.ones(len(phase))]).T,grad,rcond=None)[0]
    computed=fit[0]-1j*fit[1];exact=harmonic_gradient(solver['qMlS'],solver['periodS'])
    return dict(case=case.name,numericalChecksPassed=all(result['gates'].values()),
                amplitudePaM=float(abs(computed)),analyticAmplitudePaM=float(abs(exact)),
                relativeComplexError=float(abs(computed-exact)/abs(exact)),
                phaseErrorRad=float(np.angle(computed/exact)),
                massImbalanceFraction=result['massImbalanceFraction'],wallLeakMlS=result['wallLeakMlS'])


def verify():
    coarse=measure(ROOT/'breathing-duct-6-r2');fine=measure(ROOT/'breathing-duct-6-dt')
    dt_change=abs(coarse['amplitudePaM']-fine['amplitudePaM'])/fine['amplitudePaM']
    passed=all(r['numericalChecksPassed'] and r['relativeComplexError']<.12 and abs(r['phaseErrorRad'])<.03 for r in [coarse,fine]) and dt_change<.01
    report=dict(benchmark='Oscillatory square duct, two cycles, Stokes eigenfunction series',passed=passed,
                grid='6 cells across; central pressure-gradient fit excludes end regions',
                timeStepAmplitudeDifferenceFraction=dt_change,results=[coarse,fine],
                scope='Checks transient viscous response and signed reversal. Does not validate patient geometry or airflow.')
    (ROOT/'transient-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not passed:raise SystemExit('Transient duct validation failed')

if __name__=='__main__':verify()
