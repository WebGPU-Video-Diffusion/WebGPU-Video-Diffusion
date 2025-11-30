import * as ort from 'onnxruntime-web/webgpu';
import { toBigInt64Array } from '../utils/common.js';
import { randomNormalTensor, cat } from '../util/Tensor.js';
import { PNDMScheduler } from '../scheduler/PNDMScheduler.js';
import { Tensor } from '@xenova/transformers';
import { Session } from '../backends/index.js';
import { LatentWarper } from '../shaders/latent_warp.js';

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
// Fix WASM path for both root and subdirectory access
const basePath = window.location.origin + window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/') + 1);
ort.env.wasm.wasmPaths = basePath + 'dist/';


function log(i) { console.log(i); document.getElementById('status').innerText += `\n${i}`; }

const sigma = 14.6146;
const gamma = 0;
const vae_scaling_factor = 0.18215;

//
// load file from server or cache
//
async function fetchAndCache(url) {
    // if (url.endsWith('.onnx_data') || url.endsWith('.weights.pb')) {
    //     log(`${url} (network, no cache)`);
    //     const response = await fetch(url);
    //     if (!response.ok) {
    //         throw new Error(`HTTP ${response.status} when fetching onnx data ${url}`);
    //     }
    //     return await response.arrayBuffer();
    // }
    try {
        const cache = await caches.open("onnx");
        let cachedResponse = await cache.match(url);
        if (cachedResponse === undefined) {
            log(`${url} (network)`);
            const buffer = await fetch(url).then(response => response.arrayBuffer());
            try {
                await cache.put(url, new Response(buffer));
            } catch (error) {
                console.error(error);
            }
            return buffer;
        }
        log(`${url} (cached)`);
        const data = await cachedResponse.arrayBuffer();
        return data;
    } catch (error) {
        log(`can't fetch ${url}`);
        throw error;
    }
}

export class SDModel {
    constructor(modelConfig) {
        this.modelConfig = modelConfig;
        this.models = {"unet": {}, "text_encoder": {}, "vae_decoder": {}};
        this.init_tokenizer();
        this.negativePrompt = "blurry, low quality, bad anatomy";
        this.guidance_scale = 7.5;
        this.batch_size = modelConfig.batchSize || 1;
        this.warper = null;
        this.gpuDevice = null;
        // TODO: for now we use fixed config
        this.scheduler = new PNDMScheduler({
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: 'scaled_linear',
            prediction_type: 'epsilon',
            skip_prk_steps: true,
            final_alpha_cumprod: 1e-3,
        });
    }

    async init_tokenizer() {
        this.tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch16');
        this.tokenizer.pad_token_id = 0;
    }

    async load(base_model, options) {
        const models = options.models;
        const base_model_local = options.base_model_local; 
        const provider = options.provider || "webgpu";
        const verbose = options.verbose;
        const isLocal = options.local === true || options.local === 1 || options.local === '1' || options.local === 'true';
        const hasFP16 = (provider === "wasm") ? false : options.hasFP16;
        this.profiler = options.profiler;
        
        if (provider === "webgpu") {
            const adapter = await navigator.gpu.requestAdapter();
            this.gpuDevice = await adapter.requestDevice();
            this.warper = new LatentWarper(this.gpuDevice);
            log("WebGPU latent warper initialized");
        }
        for (const [name, model] of Object.entries(models)) {
            const useLocal = model.local ?? isLocal;
            const remoteBase = model.remoteBase ?? base_model;
            const basePath = useLocal ? `${base_model_local}` : `https://huggingface.co/${remoteBase}/resolve/main`;
            const model_path = `${basePath}/${model.url}`;

            log(`loading... ${name},  ${provider}`);
            const json_bytes = await fetchAndCache(model_path + "/config.json");
            let textDecoder = new TextDecoder();
            //const model_config = JSON.parse(textDecoder.decode(json_bytes));

            let modelSource;
            let externaldata;
            let extpath;

            if (model.externaldata) {
                modelSource = model_path + "/model.onnx";
                externaldata = model.extfilename === 1 ? model_path + "/model.onnx_data" : model_path + "/weights.pb";
                extpath = model.extfilename === 1 ? "model.onnx_data" : "weights.pb";
                log(`model ${name} uses external data; loading directly from path`);
            } else {
                modelSource = await fetchAndCache(model_path + "/model.onnx");
                externaldata = undefined;
                const modelSizeMB = Math.round(modelSource.byteLength / 1024 / 1024);
                log(`model size ${modelSizeMB} MB`);
            }

            const opt = {
                executionProviders: [provider]
            }

            if (externaldata !== undefined) {
                opt.externalData = [
                    {
                        data: externaldata,
                        path: extpath
                    },
                ]
            }
            if (verbose) {
                opt.logSeverityLevel = 0;
                opt.logVerbosityLevel = 0;
                ort.env.logLevel = "verbose";
            }
            log(`creating session for ${name} ...`);
            this.models[name] = await Session.create(
                modelSource,
                externaldata,
                externaldata ? extpath : undefined,
                model,
                opt
            );
            this.dtype = (hasFP16) ? "float16" : "float32";
        }
    }

    async infer(text) {
        try {
            document.getElementById('status').innerText = "generating video...";
            
            const num_frames = this.batch_size;
            let start = performance.now();
            let prompt_embeds = await this.getPromptEmbeds(text.value, this.negativePrompt);
            const doClassifierFreeGuidance = this.guidance_scale > 1.0;
            const targetPromptBatch = 1 * (doClassifierFreeGuidance ? 2 : 1);
            prompt_embeds = ensureBatchSize(prompt_embeds, targetPromptBatch);
            const prompt_embeds_input = toOrtTensor(prompt_embeds);

            let perf_info = [`text_encoder: ${(performance.now() - start).toFixed(1)}ms`];

            const num_inference_steps = 30;
            const allFrames = [];
            const latent_shape = [1, 4, 64, 64];
            const motion_speed_x = 1.5;
            const motion_speed_y = 0.0;
            
            // Generate initial noise ONCE for all frames
            const initial_noise_data = await tensorData(randomNormalTensor(latent_shape, 0, this.scheduler.initNoiseSigma));
            
            for (let frame_idx = 0; frame_idx < num_frames; frame_idx++) {
                log(`Generating frame ${frame_idx + 1}/${num_frames}`);
                
                // CRITICAL: Reset scheduler for each frame
                this.scheduler.setTimesteps(num_inference_steps);
                const timesteps = getSchedulerTimesteps(this.scheduler);
                if (frame_idx === 0) {
                    log(`PNDM timesteps: ${timesteps.join(',')}`);
                }
                
                // Warp initial noise based on frame index
                let current_noise_data;
                if (frame_idx === 0) {
                    current_noise_data = initial_noise_data;
                } else {
                    const dx = motion_speed_x * frame_idx;
                    const dy = motion_speed_y * frame_idx;
                    
                    if (this.warper) {
                        start = performance.now();
                        current_noise_data = await this.warper.warp(initial_noise_data, dx, dy);
                        perf_info.push(`warp: ${(performance.now() - start).toFixed(1)}ms`);
                    } else {
                        current_noise_data = initial_noise_data;
                    }
                }
                
                let latents = new Tensor('float32', current_noise_data, latent_shape);
                
                // Full denoising for each frame
                for (const t of timesteps) {
                    start = performance.now();
                    const tTensor = new ort.Tensor("float32", new Float32Array([t]), []);
                    const latent_input = doClassifierFreeGuidance ? cat([latents, latents.clone()]) : latents;
                    let feed = {
                        "sample": toOrtTensor(latent_input),
                        "timestep": tTensor,
                        "encoder_hidden_states": prompt_embeds_input,
                    };
                    const noise = await this.models["unet"].run(feed);
                    
                    let noise_pred = noise.out_sample;
                    perf_info.push(`unet t=${t}: ${(performance.now() - start).toFixed(1)}ms`);

                    if(doClassifierFreeGuidance) {
                        const split = noise_pred.dims?.[0] ?? 0;
                        const batchChunk = split / 2;
                        if (!Number.isInteger(batchChunk) || batchChunk === 0) {
                            throw new Error(`Unexpected noise prediction batch: ${split}`);
                        }
                        const noisePredUncond = noise_pred.slice([0, batchChunk]);
                        const noisePredText = noise_pred.slice([batchChunk, batchChunk * 2]);
                        noise_pred = noisePredUncond.add(noisePredText.sub(noisePredUncond).mul(this.guidance_scale));
                    }
                    latents = this.scheduler.step(noise_pred, t, latents);
                }

                start = performance.now();
                const images = await this.makeImages(latents);
                perf_info.push(`vae_decoder: ${(performance.now() - start).toFixed(1)}ms`);
                
                allFrames.push(images[0]);
                await this.draw_image(toOrtTensor(images[0]), frame_idx);
            }
            
            log(perf_info.join(", "));
            log("Video generation done");

        } catch (error) {
            log(error);
        }
    }

      /**
       * Tokenizes and encodes the input prompt. Before encoding, we verify if it is necessary
       * to break the prompt into chunks due to the Tokenizer model limit (which is usually 77)
       * by getting the maximum length of the input prompt and comparing it with the Tokenizer
       * model max length. If the prompt exceeds the Tokenizer model limit, then it is
       * necessary to break the prompt into chunks, otherwise, it is not necessary.
       * 
       * @param prompt Input prompt.
       * @param highestTokenLength Highest token length between prompt and negative prompt or Tokenizer model max length.
       * @returns Tensor containing the prompt embeddings.
       */
      async encodePrompt (prompt, highestTokenLength) {
        let tokens, encoded, inputIds;
        const TokenMaxLength = this.tokenizer.model_max_length; // Tokenizer model max length of tokens including the <START> and <END> tokens
    
        if(highestTokenLength > TokenMaxLength) { // Prompt exceeds tokenizer model max length, therefore we need to use chunks
          let embeddingsTensorArray = []; // Will contain all of the prompt token embedding chunks
          const userTokenMaxLength = TokenMaxLength - 2; // Max length of tokens minus the <START> and <END> tokens
    
          tokens = this.tokenizer(
            prompt,
            {
              return_tensor: false,
              padding: false,
              max_length: TokenMaxLength,
              return_tensor_dtype: 'int32',
            },
          );
    
          inputIds = tokens.input_ids; // Tokenized prompt
          const START_token = inputIds.shift(); // Remove <START> token
          const END_token = inputIds.pop(); // Remove <END> token
    
          for(let i = 0; i < highestTokenLength; i += userTokenMaxLength) {
            let tokenChunk = inputIds.slice(i, i + userTokenMaxLength);
            
            for(let j = tokenChunk.length; j < userTokenMaxLength; j++) { // Pad chunk to userTokenMaxLength if necessary. Use the <END> token to pad.
              tokenChunk.push(END_token);
            }
    
            tokenChunk.unshift(START_token); // Add <START> token to each chunk
            tokenChunk.push(END_token); // Add <END> token to each chunk

            encoded = await this.models["text_encoder"].run({ input_ids: new Tensor('int64', toBigInt64Array(tokenChunk.flat()), [1, tokenChunk.length]) });
            embeddingsTensorArray.push(encoded.last_hidden_state);
          }
    
          return cat(embeddingsTensorArray, 1);
        }
        else { // Prompt that does not exceed tokenizer max length. Padding is used.
          tokens = this.tokenizer(
            prompt,
            {
              return_tensor: false,
              padding: true,
              max_length: TokenMaxLength,
              return_tensor_dtype: 'int32',
            },
          );
    
          inputIds = tokens.input_ids; // Tokenized prompt
          encoded = await this.models["text_encoder"].run({ input_ids: new Tensor('int64', toBigInt64Array(inputIds.flat()), [1, inputIds.length]) });
          return encoded.last_hidden_state;
        }
      }

    /**
     * Returns the prompt and negative prompt text embeddings.
     * 
     * @param prompt Input prompt.
     * @param negativePrompt Input negative prompt.
     * @returns Tensor containing the prompt and negative prompt embeddings.
     */
    async getPromptEmbeds (prompt, negativePrompt) {
        // We check which has more tokens between the prompt and negative prompt
        const promptTokens = this.tokenizer(
            prompt,
            {
            return_tensor: false,
            padding: false,
            max_length: this.tokenizer.model_max_length,
            return_tensor_dtype: 'int32',
            },
        );

        const negPromptTokens = this.tokenizer(
            negativePrompt,
            {
            return_tensor: false,
            padding: false,
            max_length: this.tokenizer.model_max_length,
            return_tensor_dtype: 'int32',
            },
        );

        const promptTokensLength = promptTokens.input_ids.length; // Number of tokens in prompt including the <START> and <END> tokens
        const negPromptTokensLength = negPromptTokens.input_ids.length; // Number of tokens in negative prompt including the <START> and <END> tokens
        const highestTokenLength = Math.max(promptTokensLength, negPromptTokensLength);

        const promptEmbeds = await this.encodePrompt(prompt, highestTokenLength);
        const negativePromptEmbeds = await this.encodePrompt(negativePrompt || '', highestTokenLength);

        return cat([negativePromptEmbeds, promptEmbeds]);
    }

    async makeImages (latents) {
        const scaled = latents.div(vae_scaling_factor);
        const dims = getTensorDims(scaled);
        const batch = Math.max(dims[0] ?? 1, 1);
        const images = [];

        for (let i = 0; i < batch; i++) {
            const latentSample = batch === 1 ? scaled : scaled.slice([i, i + 1]);
            const decoded = await this.models["vae_decoder"].run({ "latent_sample": toOrtTensor(latentSample) });
            const image = decoded.sample
                .div(2)
                .add(0.5)
                .clipByValue(0, 1);
            images.push(image);
        }

        return images;
    }

    /**
     * draw an image from tensor
     * @param {ort.Tensor} t
     * @param {number} image_nr
     */
    async draw_image(t, image_nr) {
        const pix = await tensorData(t);
        for (let i = 0; i < pix.length; i++) {
            let x = pix[i];
            x = x / 2 + 0.5;
            if (x < 0.) x = 0.;
            if (x > 1.) x = 1.;
            pix[i] = x;
        }
        const tmpTensor = new ort.Tensor('float32', pix, getTensorDims(t));
        const imageData = tmpTensor.toImageData({ tensorLayout: 'NCWH', format: 'RGB' });
        const canvas = document.getElementById(`img_canvas_${image_nr}`);
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        canvas.getContext('2d').putImageData(imageData, 0, 0);
        const div = document.getElementById(`img_div_${image_nr}`);
        div.style.opacity = 1.;
    }
}

function getTensorDims(tensor) {
    if (tensor?.dims) {
        return tensor.dims.slice();
    }
    if (tensor?.shape) {
        return Array.from(tensor.shape);
    }
    return [];
}

async function tensorData(tensor) {
    if (!tensor) {
        return new Float32Array();
    }
    if (typeof tensor.getData === 'function') {
        return await tensor.getData();
    }
    if (tensor.data instanceof Float32Array) {
        return tensor.data;
    }
    return Float32Array.from(tensor.data || []);
}

function toOrtTensor(tensor) {
    if (tensor instanceof ort.Tensor) {
        return tensor;
    }
    const data = tensor.data instanceof Float32Array ? tensor.data : Float32Array.from(tensor.data);
    return new ort.Tensor(tensor.type || 'float32', data, getTensorDims(tensor));
}

async function toXTensor(tensor) {
    if (tensor instanceof Tensor) {
        return tensor;
    }
    const data = await tensorData(tensor);
    return new Tensor(tensor.type || 'float32', data.slice ? data : Float32Array.from(data), getTensorDims(tensor));
}

function getSchedulerTimesteps(scheduler) {
    if (!scheduler || !scheduler.timesteps) {
        return [];
    }
    if (Array.isArray(scheduler.timesteps)) {
        return scheduler.timesteps.slice();
    }
    if (scheduler.timesteps.data) {
        return Array.from(scheduler.timesteps.data);
    }
    return [];
}

function safeDispose(tensor) {
    if (tensor && typeof tensor.dispose === 'function') {
        tensor.dispose();
    }
}

function ensureBatchSize(tensor, targetBatch) {
    if (!tensor || targetBatch <= 0) {
        return tensor;
    }
    const dims = getTensorDims(tensor);
    if (!dims.length) {
        return tensor;
    }
    const currentBatch = dims[0] ?? 1;
    if (currentBatch === targetBatch) {
        return tensor;
    }
    if (targetBatch % currentBatch !== 0) {
        throw new Error(`Cannot reshape embeddings batch ${currentBatch} to ${targetBatch}`);
    }
    const repeats = targetBatch / currentBatch;
    const copies = [];
    for (let i = 0; i < repeats; i++) {
        if (typeof tensor.clone === 'function') {
            copies.push(tensor.clone());
        } else {
            const data = tensor.data instanceof Float32Array ? tensor.data.slice() : Float32Array.from(tensor.data || []);
            copies.push(new Tensor(tensor.type || 'float32', data, dims.slice()));
        }
    }
    return cat(copies);
}