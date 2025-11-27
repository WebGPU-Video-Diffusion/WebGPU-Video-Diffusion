// PNDMScheduler.js

import ort from 'onnxruntime-web/webgpu';
import { Tensor } from '@xenova/transformers';
import { SchedulerBase } from './SchedulerBase.js';
import { range, cat } from '../util/Tensor.js';

function log(i) { console.log(i); document.getElementById('status').innerText += `\n${i}`; }

export class PNDMScheduler extends SchedulerBase {
  constructor(config = {}) {
    super(config);
    this.skip_prk_steps = config.skip_prk_steps ?? true;
    this.prediction_type = this.config.prediction_type;
    this.initNoiseSigma = 1.0;
    this.pndmOrder = 4;

    this.num_inference_steps = 0;
    this.timesteps = [];
    this.prk_timesteps = [];
    this.plms_timesteps = [];
    this.ets = [];
    this.counter = 0;
    this.cur_model_output = null;
    this._step_ratio = 1;
  }

  setTimesteps(num_inference_steps) {
    this.num_inference_steps = num_inference_steps;
    const stepRatio = ~~(this.config.num_train_timesteps / this.num_inference_steps);
    this.timesteps = range(0, num_inference_steps).mul(stepRatio).round();
    this.timesteps = this.timesteps.add(this.config.steps_offset);
    // const fixed_timesteps = [
    //     958, 925, 925, 892, 859, 826, 793, 760, 727, 694, 
    //     661, 628, 595, 562, 529, 496, 463, 430, 397, 364, 
    //     331, 298, 265, 232, 199, 166, 133, 100, 67,  34
    // ];
    // fixed_timesteps.reverse();
    // log(`fixed timesteps: ${fixed_timesteps}`);
    
    if (this.skip_prk_steps) {
      this.prkTimesteps = new Tensor(new Int32Array());
      const size = this.timesteps.size;
      this.plmsTimesteps = cat([
        this.timesteps.slice([1, size - 1]),
        this.timesteps.slice([size - 2, size - 1]),
        this.timesteps.slice([size - 1, size]),
      ]).reverse().clone();
      this.timesteps = this.plmsTimesteps;
      log(`PNDM timesteps: ${this.timesteps.data}`);
    } else {
      const prkTimesteps = this.timesteps.slice(-this.pndmOrder)
        .tile([2])
        .add(
          // tf.tensor([0, this.config.num_train_timesteps / numInferenceSteps / 2]).tile([this.pndmOrder])
        );
      this.prk_timesteps = prkTimesteps.slice(0, -1).tile([2]).slice(1, -1).reverse().clone();
      this.plms_timesteps = this.timesteps.slice(0, -3).reverse().clone();
      this.timesteps = cat([this.prk_timesteps, this.plms_timesteps]);
    }

    this.ets = [];
    this.counter = 0;
    this.cur_model_output = null;
  }

  step(model_output, timestep, sample) {
    if (!this.skip_prk_steps && this.counter < this.prk_timesteps.dims[0]) {
      return this._step_prk(model_output, timestep, sample);
    } else {
      return this._step_plms(model_output, timestep, sample);
    }
  }

  _step_prk(model_output, timestep, sample) {
    if (this.num_inference_steps == null) {
      throw new Error("num_inference_steps is null. Call setTimesteps() first.");
    }

    const halfStep = Math.floor(this.num_train_timesteps / this.num_inference_steps / 2);
    const diff_to_prev = this.counter % 2 === 0 ? halfStep : 0;
    const prev_timestep = timestep - diff_to_prev;
    const bucketIndex = Math.floor(this.counter / 4) * 4;
    timestep = this.prk_timesteps[bucketIndex];

    if (this.counter % 4 === 0) {
      this.cur_model_output = this._weightedAdd(this.cur_model_output, model_output, 1 / 6);
      this.ets.push(this._cloneTensor(model_output));
      this.cur_sample = this._cloneTensor(sample);
    } else if ((this.counter - 1) % 4 === 0) {
      this.cur_model_output = this._weightedAdd(this.cur_model_output, model_output, 1 / 3);
    } else if ((this.counter - 2) % 4 === 0) {
      this.cur_model_output = this._weightedAdd(this.cur_model_output, model_output, 1 / 3);
    } else if ((this.counter - 3) % 4 === 0) {
      model_output = this._weightedAdd(this.cur_model_output, model_output, 1 / 6);
      this.cur_model_output = null;
    }

    const current_sample = this.cur_sample ?? sample;
    const prev_sample = this._get_prev_sample(current_sample, timestep, prev_timestep, model_output);
    this.counter += 1;
    return prev_sample;
  }

  _step_plms(model_output, timestep, sample) {
    if (this.num_inference_steps == null) {
      throw new Error("num_inference_steps is null. Call setTimesteps() first.");
    }

    const dt = ~~(this.num_train_timesteps / this.num_inference_steps);
    let prev_timestep = timestep - dt;

    if (this.counter !== 1) {
      if (this.ets.length > 3) {
        this.ets = this.ets.slice(-3);
      }
      this.ets.push(this._cloneTensor(model_output));
    } else {
      prev_timestep = timestep;
      timestep = timestep + dt;
    }

    if (this.ets.length === 1 && this.counter === 0) {
      this.cur_sample = sample;
    } else if (this.ets.length === 1 && this.counter === 1) {
      const last = this.ets[this.ets.length - 1];
      model_output = model_output.add(last).div(2);

      if (this.cur_sample == null) {
        throw new Error("cur_sample is null in _step_plms");
      }
      sample = this.cur_sample;
      this.cur_sample = null;
    } else if (this.ets.length === 2) {
      const e1 = this.ets[this.ets.length - 1];
      const e2 = this.ets[this.ets.length - 2];
      model_output = e1.mul(3).sub(e2).div(2);
    } else if (this.ets.length === 3) {
      const e1 = this.ets[this.ets.length - 1];
      const e2 = this.ets[this.ets.length - 2];
      const e3 = this.ets[this.ets.length - 3];
      model_output = e1.mul(23)
        .sub(e2.mul(16))
        .add(e3.mul(5))
        .div(12);
    } else {
      const n = this.ets.length;
      const e1 = this.ets[n - 1];
      const e2 = this.ets[n - 2];
      const e3 = this.ets[n - 3];
      const e4 = this.ets[n - 4];
      model_output = e1.mul(55)
        .sub(e2.mul(59))
        .add(e3.mul(37))
        .sub(e4.mul(9))
        .mul(1 / 24);
    }

    const prev_sample = this._get_prev_sample(sample, timestep, prev_timestep, model_output);
    this.counter += 1;
    return prev_sample;
  }

  _get_prev_sample(sample, timestep, prev_timestep, model_output) {
    const tIdx = Math.round(timestep);
    const tPrevIdx = Math.round(prev_timestep);
    const alphaData = this.alphas_cumprod.data;

    if (tIdx < 0 || tIdx >= alphaData.length) {
      throw new Error(`timestep index out of range: ${tIdx}`);
    }
    const alpha_prod_t = alphaData[tIdx];

    let alpha_prod_t_prev;
    if (prev_timestep >= 0) {
      if (tPrevIdx < 0 || tPrevIdx >= alphaData.length) {
        throw new Error(`prev_timestep index out of range: ${tPrevIdx}`);
      }
      alpha_prod_t_prev = alphaData[tPrevIdx];
    } else {
      alpha_prod_t_prev = this.final_alpha_cumprod;
    }

    const beta_prod_t = 1 - alpha_prod_t;
    const beta_prod_t_prev = 1 - alpha_prod_t_prev;

    let adjusted_model_output = model_output;
    if (this.prediction_type === "v_prediction") {
      const sqrt_alpha_prod_t = Math.sqrt(alpha_prod_t);
      const sqrt_beta_prod_t = Math.sqrt(beta_prod_t);
      adjusted_model_output = model_output.mul(sqrt_alpha_prod_t).add(sample.mul(sqrt_beta_prod_t));
    } else if (this.prediction_type !== "epsilon") {
      throw new Error(
        `prediction_type given as ${this.prediction_type} must be 'epsilon' or 'v_prediction'`
      );
    }

    const eps = 1e-12;
    const sample_coeff = Math.sqrt(alpha_prod_t_prev / Math.max(alpha_prod_t, eps));
    const denom =
      alpha_prod_t * Math.sqrt(beta_prod_t_prev) +
      Math.sqrt(alpha_prod_t * beta_prod_t * alpha_prod_t_prev);

    if (!isFinite(denom) || Math.abs(denom) < eps) {
      throw new Error(`model_output_denom_coeff is invalid: ${denom}`);
    }

    const coeff = (alpha_prod_t_prev - alpha_prod_t) / denom;
    return sample.mul(sample_coeff).sub(adjusted_model_output.mul(coeff));
  }

  _cloneTensor(t) {
    if (t == null) {
      return null;
    }
    const data = t.data instanceof Float32Array ? t.data.slice() : Float32Array.from(t.data);
    return new Tensor(t.type || "float32", data, t.dims.slice());
  }

  _weightedAdd(target, tensor, weight) {
    const contribution = tensor.mul(weight);
    if (target == null) {
      return contribution;
    }
    return target.add(contribution);
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

  _toOrtTensor(tensor) {
    if (!(tensor instanceof Tensor)) {
      return tensor;
    }
    const data = tensor.data instanceof Float32Array ? tensor.data.slice() : Float32Array.from(tensor.data);
    return new ort.Tensor(tensor.type || 'float32', data, tensor.dims.slice());
  }
}