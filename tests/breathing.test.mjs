import test from 'node:test';
import assert from 'node:assert/strict';
import {restingBreath} from '../viewer/breathing.js';

test('resting breath is smooth at both reversals, asymmetric and volume balanced',()=>{
  const h=1e-6;
  for(const phase of [0,.4,1]){
    assert.ok(Math.abs(restingBreath(phase))<1e-12);
    assert.ok(Math.abs((restingBreath(phase+h)-restingBreath(phase-h))/(2*h))<.001);
  }
  let inspired=0,expired=0,min=0,peakPhase=0;
  for(let i=0;i<10000;i++){
    const p=(i+.5)/10000,q=restingBreath(p);
    if(q>0)inspired+=q/10000;else expired-=q/10000;
    if(q<min){min=q;peakPhase=p;}
  }
  assert.ok(Math.abs(inspired-expired)<1e-8);
  assert.ok(Math.abs(restingBreath(.2)-1)<1e-12);
  assert.ok(peakPhase>.59&&peakPhase<.61,'expiration peaks one third into its longer phase');
  assert.ok(Math.abs(restingBreath(.9))<Math.abs(restingBreath(.7)),'expiration tapers');
});
