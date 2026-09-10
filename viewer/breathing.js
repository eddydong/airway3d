// A resting-breath model, not a measured patient waveform. C1 at reversals:
// symmetric rounded inspiration, earlier expiratory peak and a longer tail.
// Both lobes have equal integrals, so the 1-D preview has zero net tidal drift.
export function restingBreath(phase,inspiratoryFraction=.4){
  const p=((phase%1)+1)%1,fi=inspiratoryFraction;
  const x=p<fi?p/fi:(p-fi)/(1-fi),tail=1-x;
  return p<fi?16*x*x*tail*tail:-56*fi/(1-fi)*x*x*tail**4;
}
