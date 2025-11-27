export function betasForAlphaBar(numDiffusionTimesteps, maxBeta = 0.999, alphaTransformType = 'cosine') {
  const betas = new Float32Array(numDiffusionTimesteps);

  const alphaBar = (timeStep) => {
    if (alphaTransformType === 'cosine') {
      return Math.cos(((timeStep + 0.008) / 1.008) * Math.PI * 0.5) ** 2;
    }
    if (alphaTransformType === 'exp') {
      return Math.exp(timeStep * -12);
    }
    throw new Error(`Unsupported alphaTransformType: ${alphaTransformType}`);
  };

  for (let i = 0; i < numDiffusionTimesteps; i++) {
    const t1 = i / numDiffusionTimesteps;
    const t2 = (i + 1) / numDiffusionTimesteps;
    betas[i] = Math.min(1 - alphaBar(t2) / alphaBar(t1), maxBeta);
  }

  return betas;
}
