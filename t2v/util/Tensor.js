// modified from diffuser.js
import { Tensor } from '@xenova/transformers';
import seedrandom from 'seedrandom';

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
      result[prop] = new Tensor(
        modelTensor.type,
        modelTensor.data,
        modelTensor.dims
      );
    }
  }
  return result;
}
