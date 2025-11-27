import { Tensor } from '@xenova/transformers';
import { linspace, range } from '../util/Tensor.js';
import { betasForAlphaBar } from './common.js';

const DEFAULT_CONFIG = {
	beta_start: 0.00085,
	beta_end: 0.012,
	beta_schedule: 'scaled_linear',
	clip_sample: false,
	num_train_timesteps: 1000,
	prediction_type: 'epsilon',
	set_alpha_to_one: false,
	steps_offset: 1,
	trained_betas: null,
	final_alpha_cumprod: null,
};

export class SchedulerBase {
	constructor(config = {}) {
		this.config = { ...DEFAULT_CONFIG, ...config };
		this.num_train_timesteps = this.config.num_train_timesteps;

		const trained = this.config.trained_betas;
		if (Array.isArray(trained) || trained instanceof Float32Array) {
			this.betas = this._tensorFromArray(trained);
		} else if (this.config.beta_schedule === 'linear') {
			this.betas = linspace(this.config.beta_start, this.config.beta_end, this.num_train_timesteps);
		} else if (this.config.beta_schedule === 'scaled_linear') {
			this.betas = linspace(this.config.beta_start ** 0.5, this.config.beta_end ** 0.5, this.num_train_timesteps).pow(2);
		} else if (this.config.beta_schedule === 'squaredcos_cap_v2') {
			this.betas = this._tensorFromArray(betasForAlphaBar(this.num_train_timesteps));
		} else {
            throw new Error(`Unsupported beta_schedule: ${this.config.beta_schedule}`);
        }
		this.alphas = linspace(1, 1, this.num_train_timesteps).sub(this.betas);
		this.alphas_cumprod = this.alphas.cumprod();
        this.final_alpha_cumprod = this.config.set_alpha_to_one ? 1.0 : this.alphas_cumprod.data[0].data;
		// const alphaData = this.alphas_cumprod.data;
		// const configuredFinalAlpha = this.config.final_alpha_cumprod;
		// if (typeof configuredFinalAlpha === 'number') {
		// 	this.final_alpha_cumprod = configuredFinalAlpha;
		// } else {
		// 	this.final_alpha_cumprod = this.config.set_alpha_to_one ? 1 : alphaData[alphaData.length - 1];
		// }
		this.timesteps = range(0, this.num_train_timesteps);
	}

	scaleModelInput(sample) {
		return sample;
	}

	addNoise(originalSamples, noise, timestep) {
		// const alphaData = this.alphas_cumprod.data;
		// const sqrtAlphaProd = Math.sqrt(alphaData[timestep]);
		// const sqrtOneMinusAlphaProd = Math.sqrt(1 - alphaData[timestep]);

		// const originalTensor = this._ensureTensor(originalSamples);
		// const noiseTensor = this._ensureTensor(noise);
		// const mixed = originalTensor.mul(sqrtAlphaProd).add(noiseTensor.mul(sqrtOneMinusAlphaProd));
		// return this._toReferenceTensor(mixed, originalSamples);
        const sqrtAlphaProd = this.alphas_cumprod.data[timestep] ** 0.5
        const sqrtOneMinusAlphaProd = (1 - this.alphas_cumprod.data[timestep]) ** 0.5

        return originalSamples.mul(sqrtAlphaProd).add(noise.mul(sqrtOneMinusAlphaProd))
	}

	_tensorFromArray(values) {
		const data = values instanceof Float32Array ? values.slice() : Float32Array.from(values);
		return new Tensor('float32', data, [data.length]);
	}

	_ensureTensor(value) {
		if (value instanceof Tensor) {
			return value;
		}
		if (!value || !value.data || !value.dims) {
			throw new Error('Invalid tensor-like input provided to scheduler.');
		}
		const type = value.type || 'float32';
		const data = value.data instanceof Float32Array ? value.data.slice() : Float32Array.from(value.data);
		return new Tensor(type, data, value.dims.slice());
	}

	_toReferenceTensor(tensor, reference) {
		if (reference instanceof Tensor) {
			return tensor;
		}
		const data = tensor.data instanceof Float32Array ? tensor.data.slice() : Float32Array.from(tensor.data);
		const type = reference?.type || tensor.type || 'float32';
		const dims = reference?.dims ? reference.dims.slice() : tensor.dims.slice();
		const Ctor = reference?.constructor;
		if (Ctor) {
			return new Ctor(type, data, dims);
		}
		return new Tensor(type, data, dims);
	}
}