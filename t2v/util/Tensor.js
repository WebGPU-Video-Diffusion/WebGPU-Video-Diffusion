// modified from diffuser.js
import { Tensor } from '@xenova/transformers';
import seedrandom from 'seedrandom';

const _floatView = new Float32Array(1);
const _intView = new Uint32Array(_floatView.buffer);

function float16BitsToFloat32(bits) {
  const sign = (bits & 0x8000) << 16;
  let exponent = bits & 0x7c00;
  let mantissa = bits & 0x03ff;

  if (exponent === 0x7c00) {
    _intView[0] = sign | 0x7f800000 | (mantissa << 13);
    return _floatView[0];
  }

  if (exponent !== 0) {
    exponent = (exponent >> 10) + 112;
    mantissa <<= 13;
    _intView[0] = sign | (exponent << 23) | mantissa;
    return _floatView[0];
  }

  if (mantissa === 0) {
    _intView[0] = sign;
    return _floatView[0];
  }

  exponent = 113;
  while ((mantissa & 0x0400) === 0) {
    mantissa <<= 1;
    exponent -= 1;
  }
  mantissa = (mantissa & 0x03ff) << 13;
  _intView[0] = sign | (exponent << 23) | mantissa;
  return _floatView[0];
}

export function float16ArrayToFloat32Array(data) {
  if (!data) {
    return new Float32Array();
  }
  const src = data instanceof Uint16Array ? data : Uint16Array.from(data);
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    out[i] = float16BitsToFloat32(src[i]);
  }
  return out;
}

export function ensureFloat32Array(data) {
  if (data instanceof Float32Array) {
    return data;
  }
  if (data instanceof Uint16Array) {
    return float16ArrayToFloat32Array(data);
  }
  if (Array.isArray(data)) {
    return Float32Array.from(data);
  }
  if (data && typeof data.length === 'number') {
    return Float32Array.from(data);
  }
  return new Float32Array();
}

Tensor.prototype.reverse = function () {
  return new Tensor(this.type, this.data.reverse(), this.dims.slice());
};

Tensor.prototype.sub = function (value) {
  return this.clone().sub_(value);
};

Tensor.prototype.sub_ = function (value) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] -= value;
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot subtract tensors of different sizes');
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] -= value.data[i];
    }
  } else {
    throw new Error('Invalid argument');
  }
  return this;
};

Tensor.prototype.add = function (value) {
  return this.clone().add_(value);
};

Tensor.prototype.add_ = function (value) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] += value;
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot subtract tensors of different sizes');
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] += value.data[i];
    }
  } else {
    throw new Error('Invalid argument');
  }
  return this;
};

Tensor.prototype.cumprod = function (dim) {
  return this.clone().cumprod_(dim);
};

Tensor.prototype.cumprod_ = function (dim) {
  const newDims = this.dims.slice();
  if (dim === undefined) {
    dim = this.dims.length - 1;
  }
  if (dim < 0 || dim >= this.dims.length) {
    throw new Error(`Invalid dimension: ${dim}`);
  }
  const size = newDims[dim];
  for (let i = 1; i < size; ++i) {
    for (let j = 0; j < this.data.length / size; ++j) {
      const index = j * size + i;
      this.data[index] *= this.data[index - 1];
    }
  }
  return this;
};

Tensor.prototype.mul = function (value) {
  return this.clone().mul_(value);
};

Tensor.prototype.mul_ = function (value) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] *= value;
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot multiply tensors of different sizes');
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] *= value.data[i];
    }
  } else {
    throw new Error('Invalid argument');
  }
  return this;
};

Tensor.prototype.div = function (value) {
  return this.clone().div_(value);
};

Tensor.prototype.div_ = function (value) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] /= value;
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot multiply tensors of different sizes');
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] /= value.data[i];
    }
  } else {
    throw new Error('Invalid argument');
  }
  return this;
};

Tensor.prototype.pow = function (value) {
  return this.clone().pow_(value);
};

Tensor.prototype.pow_ = function (value) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] = Math.pow(this.data[i], value);
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot multiply tensors of different sizes');
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] = Math.pow(this.data[i], value.data[i]);
    }
  } else {
    throw new Error('Invalid argument');
  }
  return this;
};

Tensor.prototype.round = function () {
  return this.clone().round_();
};

Tensor.prototype.round_ = function () {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.round(this.data[i]);
  }
  return this;
};

Tensor.prototype.tile = function (reps) {
  return this.clone().tile_(reps);
};

Tensor.prototype.tile_ = function (reps) {
  if (typeof reps === 'number') {
    reps = [reps];
  }
  if (reps.length < this.dims.length) {
    throw new Error('Invalid number of repetitions');
  }
  const newDims = [];
  const newStrides = [];
  for (let i = 0; i < this.dims.length; ++i) {
    newDims.push(this.dims[i] * reps[i]);
    newStrides.push(this.strides[i]);
  }
  const newData = new this.data.constructor(newDims.reduce((a, b) => a * b));
  for (let i = 0; i < newData.length; ++i) {
    let index = 0;
    for (let j = 0; j < this.dims.length; ++j) {
      index += Math.floor(i / newDims[j]) * this.strides[j];
    }
    newData[i] = this.data[index];
  }
  return new Tensor(this.type, newData, newDims);
};

Tensor.prototype.clipByValue = function (min, max) {
  return this.clone().clipByValue_(min, max);
};

Tensor.prototype.clipByValue_ = function (min, max) {
  if (max < min) {
    throw new Error('Invalid arguments');
  }
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.min(Math.max(this.data[i], min), max);
  }
  return this;
};

Tensor.prototype.exp = function () {
  return this.clone().exp_();
};

Tensor.prototype.exp_ = function () {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.exp(this.data[i]);
  }
  return this;
};

Tensor.prototype.sin = function () {
  return this.clone().sin_();
};

Tensor.prototype.sin_ = function () {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.sin(this.data[i]);
  }
  return this;
};

Tensor.prototype.cos = function () {
  return this.clone().cos_();
};

Tensor.prototype.cos_ = function () {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.cos(this.data[i]);
  }
  return this;
};

Tensor.prototype.location = 'cpu';

export function range(start, end, step = 1, type = 'float32') {
  const data = [];
  for (let i = start; i < end; i += step) {
    data.push(i);
  }
  return new Tensor(type, data, [data.length]);
}

export function linspace(start, end, num, type = 'float32') {
  const arr = [];
  const step = (end - start) / (num - 1);
  for (let i = 0; i < num; i++) {
    arr.push(start + step * i);
  }
  return new Tensor(type, arr, [num]);
}

function randomNormal(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function scalarTensor(num, type = 'float32') {
  return new Tensor(type, new Float32Array([num]), [1]);
}

export function randomNormalTensor(shape, mean = 0, std = 1, type = 'float32', seed = '') {
  const data = [];
  const rng = seed !== '' ? seedrandom(seed) : seedrandom();
  const total = shape.reduce((a, b) => a * b, 1);
  for (let i = 0; i < total; i++) {
    data.push(randomNormal(rng) * std + mean);
  }
  return new Tensor(type, data, shape);
}

export function cat(tensors, axis = 0) {
  if (!Array.isArray(tensors) || tensors.length === 0) {
    throw new Error('No tensors provided.');
  }
  if (axis < 0) {
    axis = tensors[0].dims.length + axis;
  }
  const tensorType = tensors[0].type;
  const tensorShape = [...tensors[0].dims];
  for (const t of tensors) {
    for (let i = 0; i < tensorShape.length; i++) {
      if (i !== axis && tensorShape[i] !== t.dims[i]) {
        throw new Error('Tensor dimensions must match for concatenation, except along the specified axis.');
      }
    }
  }
  tensorShape[axis] = tensors.reduce((sum, t) => sum + t.dims[axis], 0);
  const total = tensorShape.reduce((product, size) => product * size, 1);
  const data = new tensors[0].data.constructor(total);
  let offset = 0;
  for (const t of tensors) {
    const copySize = t.data.length / t.dims[axis];
    for (let i = 0; i < t.dims[axis]; i++) {
      const sourceStart = i * copySize;
      const sourceEnd = sourceStart + copySize;
      data.set(t.data.slice(sourceStart, sourceEnd), offset);
      offset += copySize;
    }
  }
  return new Tensor(tensorType, data, tensorShape);
}

export function replaceTensors(modelRunResult) {
  const result = {};
  for (const prop in modelRunResult) {
    const modelTensor = modelRunResult[prop];
    if (modelTensor && modelTensor.dims) {
      const isFloatTensor = modelTensor.type === 'float32' || modelTensor.type === 'float16';
      const tensorType = isFloatTensor ? 'float32' : modelTensor.type;
      const tensorData = isFloatTensor
        ? ensureFloat32Array(modelTensor.data)
        : modelTensor.data;
      result[prop] = new Tensor(
        tensorType,
        tensorData,
        modelTensor.dims
      );
    }
  }
  return result;
}
