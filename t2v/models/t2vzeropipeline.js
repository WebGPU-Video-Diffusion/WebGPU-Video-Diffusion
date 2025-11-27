// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// console.log(ort.env.versions)
import {
  AutoTokenizer,
  cat,
  Tensor,
  stack,
  interpolate,
  permute,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3"
// console.log(cv)

import { PNDMScheduler, DDIMScheduler, meshgrid, repeatND, range } from "https://codepen.io/jdp8/pen/emOrKYz.js";

const PNDM_schedulerConfig = {
  "_class_name": "PNDMScheduler",
  "_diffusers_version": "0.32.1",
  "beta_end": 0.012,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.00085,
  "clip_sample": false,
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "set_alpha_to_one": false,
  "skip_prk_steps": true,
  "steps_offset": 1,
  "timestep_spacing": "leading",
  "trained_betas": null
}

const DDIM_schedulerConfig = {
  "_class_name": "DDIMScheduler",
  "_diffusers_version": "0.32.1",
  "beta_end": 0.012,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.00085,
  "clip_sample": false,
  "clip_sample_range": 1.0,
  "dynamic_thresholding_ratio": 0.995,
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "rescale_betas_zero_snr": false,
  "sample_max_value": 1.0,
  "set_alpha_to_one": false,
  "skip_prk_steps": true,
  "steps_offset": 1,
  "thresholding": false,
  "timestep_spacing": "leading",
  "trained_betas": null
}

const scheduler = new PNDMScheduler(PNDM_schedulerConfig)

function log(i) { console.log(i); document.getElementById('status').innerText += `\n${i}`; }

/*
 * get configuration from url
 */
function getConfig() {
  // const query = window.location.search.substring(1);
  // console.log(query)
  var config = {
    model: "https://huggingface.co/RanaLLC/small-sd-v0-onnx-fp16/resolve/main",
    provider: "webgpu",
    device: "gpu",
    threads: "1",
  }
  // config.threads = parseInt(config.threads)
  return config
}

/*
 * initialize latents with random noise
 */
function randn_latents(shape) {
  function randn() {
    // Use the Box-Muller transform
    let u = Math.random()
    let v = Math.random()
    let z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
    return z
  }
  let size = 1
  shape.forEach((element) => {
    size *= element
  })

  let data = new Float16Array(size)
  // Loop over the shape dimensions
  for (let i = 0; i < size; i++) {
    data[i] = randn()
  }
  return data
}

/*
 * fetch and cache model
 */
async function fetchAndCache(base_url, model_path) {
  const url = `${base_url}/${model_path}`
  try {
    const cache = await caches.open("onnx")
    let cachedResponse = await cache.match(url)
    if (cachedResponse == undefined) {
      await cache.add(url)
      cachedResponse = await cache.match(url)
      log(`${model_path} (network)`)
    } else {
      log(`${model_path} (cached)`)
    }
    const data = await cachedResponse.arrayBuffer()
    return data
  } catch (error) {
      log(`${model_path} (network)`)
      return await fetch(url).then(response => response.arrayBuffer());
  }
}

/*
 * load models used in the pipeline
 */
async function load_models(models) {
  const cache = await caches.open("onnx")
  let missing = 0
  for (const [name, model] of Object.entries(models)) {
    const url = `${config.model}/${model.url}`
    let cachedResponse = await cache.match(url)
    if (cachedResponse === undefined) {
      missing += model.size
    }
  }
  if (missing > 0) {
    log(
      `downloading ${missing} MB from network ... it might take a while`,
    )
  } else {
    log("loading...")
  }
  for (const [name, model] of Object.entries(models)) {
    try {
      const start = performance.now()
      const model_bytes = await fetchAndCache(config.model, model.url)
      
      const sess_opt = { ...opt, ...model.opt }
      models[name].sess = await ort.InferenceSession.create(model_bytes, sess_opt)
      const stop = performance.now()
      log(`${model.url} in ${(stop - start).toFixed(1)}ms`)
    } catch (e) {
      log(`${model.url} failed, ${e}`)
    }
  }
  log("ready.")
}

const config = getConfig()

const models = {
  unet: {
    url: "unet/model.onnx",
    size: 640,
    // should have 'steps: 1' but will fail to create the session
    opt: {
      freeDimensionOverrides: {
        batch_size: 1,
        num_channels: 4,
        height: 64,
        width: 64,
        sequence_length: 77,
      },
    },
  },
  text_encoder: {
    url: "text_encoder/model.onnx",
    size: 1700,
    // should have 'sequence_length: 77' but produces a bad image
    opt: { freeDimensionOverrides: { batch_size: 1 } },
  },
  vae_decoder: {
    url: "vae_decoder/model.onnx",
    size: 95,
    opt: {
      freeDimensionOverrides: {
        batch_size: 1,
        num_channels_latent: 4,
        height_latent: 64,
        width_latent: 64,
      },
    },
  },
  vae_encoder: {
    url: "vae_encoder/model.onnx",
    size: 137,
    opt: {
      freeDimensionOverrides: {
        batch_size: 1,
        num_channels_latent: 3,
        height: 512,
        width: 512,
      },
    },
  },
}

/* ort.env.wasm.wasmPaths = 'dist/'; */
// ort.env.wasm.numThreads = 1
// ort.env.wasm.simd = true

let tokenizer
let loading
const vae_scaling_factor = 0.18215;
const vaeScaleFactor = 8;
const videoLength = 4;
const motion_field_strength_x = 12;
const motion_field_strength_y = 12;
const t0 = 44;
const t1 = 47;
let prompt =
  "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
document.getElementById("user-prompt").value = prompt
let negative_prompt = "bad anatomy, deformed, ugly, disfigured"
document.getElementById("user-negative-prompt").value = negative_prompt
let numInferenceSteps = 20;
document.getElementById("user-steps").value = numInferenceSteps
const guidance_scale = 7.5
const width = 512
const height = 512

const frame_ids = [];
for (let i = 0; i < videoLength; i++) {
  frame_ids.push(i);
}

const opt = {
  executionProviders: [config.provider],
  enableMemPattern: false,
  enableCpuMemArena: false,
  logVerbosityLevel: 4,
  logSeverityLevel: 4,
  extra: {
    session: {
      /* disable_prepacking: "1", */
      use_device_allocator_for_initializers: "1",
      use_ort_model_bytes_directly: "1",
      use_ort_model_bytes_for_initializers: "1",
    },
  },
}

switch (config.provider) {
  case "webgpu":
    if (!("gpu" in navigator)) {
      throw new Error("webgpu is NOT supported")
    }
    // opt.preferredOutputLocation = { last_hidden_state: "gpu-buffer" };
    break
  case "webnn":
    if (!("ml" in navigator)) {
      throw new Error("webnn is NOT supported")
    }
    opt.executionProviders = [
      {
        name: "webnn",
        deviceType: config.device,
        powerPreference: "default",
      },
    ]
    break
}

// Event listener for Ctrl + Enter or CMD + Enter
// document.getElementById('user-input').addEventListener('keydown', function(e) {
//     if (e.ctrlKey && e.key === 'Enter') {
//         generate_inpainting_image();
//     }
// });

document.getElementById('send-button').addEventListener('click', function(e) {
  generate_video();
})

/**
 * draw an image from tensor
 * @param {ort.Tensor} t
 * @param {number} image_nr
 */
function draw_image(t) {
  let pix = t.data;
  for (var i = 0; i < pix.length; i++) {
    let x = pix[i];
    x = x / 2 + 0.5
    if (x < 0.) x = 0.;
    if (x > 1.) x = 1.;
    pix[i] = x;
  }
  const imageData = t.toImageData({ tensorLayout: 'NCWH', format: 'RGB' });
  const canvas = document.getElementById(`img_canvas_0`);
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext('2d').putImageData(imageData, 0, 0);
  const div = document.getElementById(`img_div_0`);
  div.style.opacity = 1.
}

function reshape(tensor, dims) {
  return new Tensor(tensor.type, tensor.data, dims);
}

function forward_loop(x_t0, t0, t1) {
  const eps = randn_latents(x_t0.dims);
  const alpha_vec = scheduler.alphas.slice([t0, t1]).prod();
  const sqrtAlpha = alpha_vec ** 0.5;
  const oneMinusAlphaSqrt = (1 - alpha_vec) ** 0.5;
  const x_t1 = [];
  for (let i = 0; i < x_t0.data.length; i++) {
    x_t1.push(x_t0.data[i] * sqrtAlpha + eps[i] * oneMinusAlphaSqrt);
  }

  return new Tensor("float32", x_t1, x_t0.dims);
}

async function backward_loop(
  timesteps,
  promptEmbeds,
  latents,
  guidanceScale,
  sdV1 = false
) {
  const doClassifierFreeGuidance = guidanceScale > 1;

  for (const step of timesteps) {
    // for some reason v1.4 takes int64 as timestep input. ideally we should get input dtype from the model
    // but currently onnxruntime-node does not give out types, only input names
    const timestep = new Tensor(new Float16Array([step]));

    const latentInput = doClassifierFreeGuidance
    ? cat([latents, latents.clone()])
    : latents;

    // UNET
    const noise = await models.unet.sess.run({
      sample: latentInput,
      timestep,
      encoder_hidden_states: promptEmbeds,
    });

    let noisePred = noise.out_sample;
    noisePred = new Tensor(noisePred.type, noisePred.data, noisePred.dims)
    if (doClassifierFreeGuidance) {
      const [noisePredUncond, noisePredText] = [
        noisePred.slice([0, 1]),
        noisePred.slice([1, 2]),
      ];
      noisePred = noisePredUncond.add(
        noisePredText.sub(noisePredUncond).mul(guidanceScale)
      );
    }

    latents = scheduler.step(noisePred, step, latents);
  }
  return latents;
}

function coords_grid(batch, ht, wd) {
  // Adapted from https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
  const coords = meshgrid(range(0, ht), range(0, wd));
  const stackedCoords = stack(coords.reverse(), 0);
  const repeatedCoords = repeatND(stackedCoords.tolist(), [batch, 1, 1, 1]);

  const newShape = [
    stackedCoords.dims[0],
    stackedCoords.dims[1],
    stackedCoords.dims[2],
  ];
  return new Tensor("float32", repeatedCoords.flat(3), newShape);
}

/**
   * Reflects a coordinate value to handle reflection padding.
   * @param coord - The normalized coordinate value.
   * @param size - The dimension size (e.g., width or height).
   * @returns The reflected coordinate.
   */
function reflect(coord, size) {
  // Normalize coord from [-1, 1] to [0, size-1] pixel space
  let pixelCoord = ((coord + 1) / 2) * (size - 1);

  if (pixelCoord < 0) {
    // Reflect from the left edge (0)
    let reflected = -pixelCoord;
    let cycles = Math.floor(reflected / (size - 1));
    let remainder = reflected % (size - 1);

    if (cycles % 2 === 0) {
      return remainder;
    } else {
      return size - 1 - remainder;
    }
  } else if (pixelCoord > size - 1) {
    // Reflect from the right edge (size - 1)
    let reflected = pixelCoord - (size - 1);
    let cycles = Math.floor(reflected / (size - 1));
    let remainder = reflected % (size - 1);

    if (cycles % 2 === 0) {
      return size - 1 - remainder;
    } else {
      return remainder;
    }
  } else {
    return pixelCoord;
  }
}

/**
   * Performs nearest-neighbor interpolation on an image with reflection padding.
   * @param input - The source image.
   * @param grid - The flow-field grid with normalized [x, y] coordinates.
   * @returns The resampled output image.
   */
function gridSampleNearestReflection(input, grid) {
  const [inputHeight, inputWidth] = input.dims;
  const [outputHeight, outputWidth] = grid.dims;

  const output = new Array(outputHeight)
  .fill(null)
  .map(() => new Array(outputWidth).fill(0));

  for (let y = 0; y < outputHeight; y++) {
    for (let x = 0; x < outputWidth; x++) {
      // Get the normalized sampling coordinates from the grid
      const gridX = grid[y][x][0];
      const gridY = grid[y][x][1];

      // Convert grid coordinates to pixel space and apply reflection
      const sampleX = Math.round(reflect(gridX, inputWidth));
      const sampleY = Math.round(reflect(gridY, inputHeight));

      // Get the pixel value from the input image using nearest-neighbor
      output[y][x] = input[sampleY][sampleX];
    }
  }

  return output;
}

/**
   * Performs 4D grid sampling with nearest-neighbor interpolation and reflection padding.
   * @param input - 4D input array of shape [N, C, H, W].
   * @param grid - 4D grid of shape [N, H_out, W_out, 2] with normalized coordinates [-1, 1].
   * @returns - Output array of shape [N, C, H_out, W_out].
   */
function gridSample4D(input, grid) {
  const N = input.length; // Batch size
  const C = input[0].length; // Channels
  const H = input[0][0].length; // Input height
  const W = input[0][0][0].length; // Input width
  const H_out = grid[0].length; // Output height
  const W_out = grid[0][0].length; // Output width

  // Initialize output array [N, C, H_out, W_out]
  const output = Array(N)
  .fill()
  .map(() =>
       Array(C)
       .fill()
       .map(() =>
            Array(H_out)
            .fill()
            .map(() => Array(W_out).fill(0))
           )
      );

  /**
     * Applies reflection padding to a coordinate.
     * @param coord - Pixel coordinate.
     * @param size - Size of the dimension (width or height).
     * @returns - Reflected coordinate within [0, size-1].
     */
  function reflect(coord, size) {
    // Normalize coordinate to handle multiple reflections
    if (coord < 0 || coord >= size) {
      const period = 2 * size;
      // Map coordinate to [0, period) and reflect within [0, size)
      coord = Math.abs(((coord % period) + period) % period);
      if (coord >= size) {
        coord = 2 * size - coord - 1;
      }
    }
    return Math.max(0, Math.min(size - 1, coord));
  }

  // Iterate over batch and output dimensions
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      for (let y = 0; y < H_out; y++) {
        for (let x = 0; x < W_out; x++) {
          // Get normalized coordinates from grid
          const nx = grid[n][y][x][0]; // Normalized x in [-1, 1]
          const ny = grid[n][y][x][1]; // Normalized y in [-1, 1]

          // Convert to pixel coordinates
          let px = ((nx + 1) / 2) * (W - 1);
          let py = ((ny + 1) / 2) * (H - 1);

          // Apply reflection padding
          px = reflect(px, W);
          py = reflect(py, H);

          // Nearest-neighbor interpolation: round to nearest integer
          const x_nearest = Math.round(px);
          const y_nearest = Math.round(py);

          // Sample the input value
          output[n][c][y][x] = input[n][c][y_nearest][x_nearest] ?? 0;
        }
      }
    }
  }

  return output;
}

/**
   * Warp latent of a single frame with given flow
   * @param latent latent code of a single frame
   * @param reference_flow flow which to warp the latent with
   * @returns warped latent
   */
function warp_single_latent(
  latent,
  reference_flow,
    latentDims,
      referenceFlowDims
) {
  const [Y, H, W] = referenceFlowDims;
  const [y, h, w] = latentDims;
  const coords0 = coords_grid(1, H, W);

  const coords0_list = coords0.tolist();

  // let coords_t0 = coords0 + reference_flow // should be [512, 512] ???
  let coords_t0_array = [];

  for (let i = 0; i < coords0_list.length; i++) {
    coords_t0_array[i] = [];
    for (let j = 0; j < coords0_list[i].length; j++) {
      coords_t0_array[i][j] = [];
      for (let k = 0; k < coords0_list[i][j].length; k++) {
        coords_t0_array[i][j][k] =
          coords0_list[i][j][k] + reference_flow[i][j][k];
        coords_t0_array[i][j][k] /= i === 0 ? W : H;
      }
    }
  }

  let coords_t0 = new Tensor(
    "float32",
    coords_t0_array.flat(3),
    referenceFlowDims
  );

  coords_t0 = coords_t0.mul(2.0).add(-1.0);
  coords_t0 = interpolate(coords_t0, [h, w], "bilinear");
  coords_t0 = permute(coords_t0.unsqueeze(0), [0, 2, 3, 1]);

  const warped = gridSample4D(
    [latent],
    coords_t0.tolist()
    // (mode = "nearest"),
    // (padding_mode = "reflection")
  );
  return warped;
}

/**
   * Create translation motion field
   * @param motion_field_strength_x motion strength along x-axis
   * @param motion_field_strength_y motion strength along y-axis
   * @param frame_ids indexes of the frames the latents of which are being processed.
   * This is needed when we perform chunk-by-chunk inference
   * @returns
   */
function create_motion_field(
  motion_field_strength_x,
  motion_field_strength_y,
  frame_ids
) {
  const seq_length = frame_ids.length;
  let zerosArray = [];

  for (let i = 0; i < seq_length; i++) {
    zerosArray[i] = [];
    for (let j = 0; j < 2; j++) {
      zerosArray[i][j] = [];
      for (let k = 0; k < 512; k++) {
        if (j === 0) {
          zerosArray[i][j][k] = new Array(512).fill(
            motion_field_strength_x * frame_ids[i]
          );
        } else if (j === 1) {
          zerosArray[i][j][k] = new Array(512).fill(
            motion_field_strength_y * frame_ids[i]
          );
        }
      }
    }
  }

  const reference_flow = new Tensor("float32", zerosArray.flat(4), [
    seq_length,
    2,
    512,
    512,
  ]);

  return reference_flow;
}

/**
   * Creates translation motion and warps the latents accordingly
   * @param motion_field_strength_x motion strength along x-axis
   * @param motion_field_strength_y motion strength along y-axis
   * @param latents latent codes of frames
   * @param frame_ids indexes of the frames the latents of which are being processed.
   * This is needed when we perform chunk-by-chunk inference
   * @returns
   */
function create_motion_field_and_warp_latents(
  motion_field_strength_x,
  motion_field_strength_y,
  latents,
  frame_ids
) {
  const motion_field = create_motion_field(
    motion_field_strength_x,
    motion_field_strength_y,
    frame_ids
  );
  let warped_latents = latents.tolist();
  const latentsList = warped_latents;
  const motionFieldList = motion_field.tolist();
  for (let i = 0; i < warped_latents.length; i++) {
    warped_latents[i] = warp_single_latent(
      latentsList[i],
      motionFieldList[i],
      latents.dims.slice(1),
      motion_field.dims.slice(1)
    );
  }
  return new Tensor(latents.type, warped_latents.flat(4), latents.dims);
}

async function encode_prompt(prompt) {
	const tokenized = await tokenizer(prompt, {
      padding: true,
      max_length: 77,
      truncation: true,
      return_tensor: false,
    })
    // console.log(new Tensor("int32", input_ids, [1, input_ids.length]))
    const encoded_text = await models.text_encoder.sess.run({
      input_ids: new ort.Tensor("int32", tokenized.input_ids, [1, tokenized.input_ids.length]),
    })
    return encoded_text.last_hidden_state
}


async function generate_video() {
  log("Running Video Generation...")
  try {
    if (tokenizer === undefined) {
      tokenizer = await AutoTokenizer.from_pretrained(
        "Xenova/clip-vit-base-patch16",
      )
      tokenizer.pad_token_id = 0
    }
    
    await loading
    
    prompt = document.getElementById("user-prompt").value;
    negative_prompt = document.getElementById("user-negative-prompt").value;
    numInferenceSteps = document.getElementById("user-steps").value;
    
    scheduler.setTimesteps(numInferenceSteps);
    let timesteps = scheduler.timesteps.data;

    // text-encoder
    let start = performance.now()

    const prompt_embeds = await encode_prompt(prompt)
		const negative_prompt_embeds = await encode_prompt(negative_prompt)

    let last_hidden_state = cat([negative_prompt_embeds, prompt_embeds])

    let perf_info = [
      `text_encoder: ${(performance.now() - start).toFixed(1)}ms`,
    ]

    const do_cfg = guidance_scale > 1

		const latent_shape = [1, 4, 64, 64]
		let latents = new Tensor(randn_latents(latent_shape), latent_shape)
    
    start = performance.now()
    
    // Perform the first backward process up to time T_1
    const x_1_t1 = await backward_loop(
      timesteps.slice(0, -t1 - 1),
      last_hidden_state,
      latents,
      guidance_scale
    ); // dims should be [1, 4, 64, 64]
    
    perf_info.push(`backward_loop_1: ${(performance.now() - start).toFixed(1)}ms`)

    // console.log("x_1_t1 shape:", x_1_t1.dims);
    // console.log("x_1_t1 min:", Math.min(...x_1_t1.data));
    // console.log("x_1_t1 max:", Math.max(...x_1_t1.data));
    
    start = performance.now()

    // Perform the second backward process up to time T_0
    const x_1_t0 = await backward_loop(
      timesteps.slice(-t1 - 1, -t0 - 1),
      last_hidden_state,
      x_1_t1,
      guidance_scale
    );
    
    perf_info.push(`backward_loop_2: ${(performance.now() - start).toFixed(1)}ms`)

    // console.log("x_1_t0 shape:", x_1_t0.dims);
    // console.log("x_1_t0 min:", Math.min(...x_1_t0.data));
    // console.log("x_1_t0 max:", Math.max(...x_1_t0.data));

    // Propagate first frame latents at time T_0 to remaining frames
    let x_2k_t0 = repeatND(x_1_t0.tolist(), [videoLength - 1, 1, 1, 1]);
    let newShape = [
      x_1_t0.dims[0] * (videoLength - 1),
      x_1_t0.dims[1],
      x_1_t0.dims[2],
      x_1_t0.dims[3],
    ];
    x_2k_t0 = new Tensor("float32", x_2k_t0.flat(4), newShape);

    // console.log("repeated x_2k_t0 shape:", x_2k_t0.dims);
    // console.log("repeated x_2k_t0 min:", Math.min(...x_2k_t0.data));
    // console.log("repeated x_2k_t0 max:", Math.max(...x_2k_t0.data));

    // Add motion in latents at time T_0
    x_2k_t0 = create_motion_field_and_warp_latents(
      motion_field_strength_x,
      motion_field_strength_y,
      x_2k_t0,
      frame_ids.slice(1)
    );

    // console.log("motion x_2k_t0 shape:", x_2k_t0.dims);
    // console.log("motion x_2k_t0 min:", Math.min(...x_2k_t0.data));
    // console.log("motion x_2k_t0 max:", Math.max(...x_2k_t0.data));
    
    start = performance.now()

    // Perform forward process up to time T_1
    const x_2k_t1 = forward_loop(
      x_2k_t0,
      timesteps.slice(-t0 - 1)[0],
      timesteps.slice(-t1 - 1)[0]
    ); // dims should be [7, 4, 64, 64]
    console.log('Last forward loop')
    console.log(x_2k_t1)
    
    perf_info.push(`forward_loop: ${(performance.now() - start).toFixed(1)}ms`)

    // console.log("x_2k_t1 shape:", x_2k_t1.dims);
    // console.log("x_2k_t1 min:", Math.min(...x_2k_t1.data));
    // console.log("x_2k_t1 max:", Math.max(...x_2k_t1.data));

    // Perform backward process from time T_1 to 0
    const x_1k_t1 = cat([x_1_t1, x_2k_t1]);

    // console.log("x_1k_t1 shape:", x_1k_t1.dims);
    // console.log("x_1k_t1 min:", Math.min(...x_1k_t1.data));
    // console.log("x_1k_t1 max:", Math.max(...x_1k_t1.data));

    const [b, l, d] = last_hidden_state.dims;
    const repeatedPromptEmbeds = repeatND(last_hidden_state.tolist(), [
      1,
      videoLength,
      1,
      1,
    ]);
    newShape = [
      last_hidden_state.dims[0] * videoLength,
      last_hidden_state.dims[1],
      last_hidden_state.dims[2],
    ];
    last_hidden_state = new Tensor(
      last_hidden_state.type,
      repeatedPromptEmbeds.flat(3),
      newShape
    );
    last_hidden_state = reshape(last_hidden_state, [b * videoLength, l, d]);
    
    start = performance.now()

    const x_1k_0 = await backward_loop(
      timesteps.slice(-t1 - 1),
      last_hidden_state, // dims should be [16, any, 768]
      x_1k_t1, // should be [8, 4, 64, 64]
      guidance_scale
    );
    console.log('Last backward loop')
    console.log(x_1k_0)
    
    perf_info.push(`backward_loop_3: ${(performance.now() - start).toFixed(1)}ms`)

    // console.log("x_1k_0 shape:", x_1k_0.dims);
    // console.log("x_1k_0 min:", Math.min(...x_1k_0.data));
    // console.log("x_1k_0 max:", Math.max(...x_1k_0.data));

    latents = x_1k_0; // final frames
    
    // vae_decoder
    start = performance.now();
    const latent = latents.slice([0, 1]).div(vae_scaling_factor) // first frame
    const { sample } = await models.vae_decoder.sess.run({ 
      "latent_sample": new ort.Tensor(latent.type, latent.data, latent.dims)
    });
    perf_info.push(`vae_decoder: ${(performance.now() - start).toFixed(1)}ms`);

    draw_image(sample);
    log(perf_info.join(", "))
    perf_info = [];
						
    // this is a gpu-buffer we own, so we need to dispose it
    // last_hidden_state.dispose()
    
    log("done")
  } catch (e) {
    log(e)
  }
}

async function hasFp16() {
  try {
    const adapter = await navigator.gpu.requestAdapter()
    return adapter.features.has("shader-f16")
  } catch (e) {
    return false
  }
}

hasFp16().then((fp16) => {
  if (fp16) {
    loading = load_models(models)
  } else {
    log("Your GPU or Browser doesn't support webgpu/f16")
  }
})
