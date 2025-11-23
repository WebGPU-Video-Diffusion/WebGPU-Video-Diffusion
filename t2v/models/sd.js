import * as ort from 'onnxruntime-web/webgpu';
import { randn_latents, scale_model_inputs, eulera_step, toBigInt64Array, draw_image } from '../utils/common.js';
import { PNDMScheduler } from '../scheduler/PNDMScheduler.js';

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = document.location.pathname.replace('index.html', '') + 'dist/';


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
        // Batch size for parallel image generation
        this.batch_size = modelConfig.batchSize || 1;
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

    /**
     * Repeat tensor along batch dimension
     * @param {ort.Tensor} tensor - Input tensor with shape [1, ...rest]
     * @param {number} repeat_times - Number of times to repeat
     * @returns {ort.Tensor} Output tensor with shape [repeat_times, ...rest]
     * @example
     * // Input: [1, 77, 768] with repeat_times=4
     * // Output: [4, 77, 768] (same data copied 4 times)
     */
    repeat_tensor(tensor, repeat_times) {
        if (repeat_times === 1) {
            return tensor;
        }
        
        const data = tensor.data;
        const dims = tensor.dims;
        const single_size = data.length;
        
        // Create new array with repeated data
        const repeated_data = new Float32Array(single_size * repeat_times);
        
        // Copy data repeat_times times
        for (let i = 0; i < repeat_times; i++) {
            repeated_data.set(data, i * single_size);
        }
        
        // Update dimensions: [1, ...rest] -> [repeat_times, ...rest]
        const new_dims = [repeat_times, ...dims.slice(1)];
        
        return new ort.Tensor(tensor.type, repeated_data, new_dims);
    }

    /**
     * Extract single item from batch tensor
     * @param {ort.Tensor} batch_tensor - Batch tensor with shape [batch_size, ...rest]
     * @param {number} index - Index to extract (0 to batch_size-1)
     * @returns {ort.Tensor} Single item tensor with shape [1, ...rest]
     * @example
     * // Input: [4, 3, 512, 512] with index=0
     * // Output: [1, 3, 512, 512] (first image from batch)
     */
    extract_from_batch(batch_tensor, index) {
        const data = batch_tensor.data;
        const dims = batch_tensor.dims;
        const batch_size = dims[0];
        
        if (index >= batch_size) {
            throw new Error(`Index ${index} out of range for batch size ${batch_size}`);
        }
        
        // Calculate size of single item: product of all dims except batch
        const single_size = data.length / batch_size;
        
        // Extract data slice for the specified index
        const start_idx = index * single_size;
        
        // Create new TypedArray (not just slice which returns wrong type)
        const single_data = new Float32Array(single_size);
        for (let i = 0; i < single_size; i++) {
            single_data[i] = data[start_idx + i];
        }
        
        // Create new dimensions: [1, ...rest]
        const single_dims = [1, ...dims.slice(1)];
        
        return new ort.Tensor(batch_tensor.type, single_data, single_dims);
    }

    // createOrtFloatTensor(data, dims, type = null) {
    //     const dtype = type || this.modelConfig.floatType || 'float32';
    //     return new ort.Tensor(dtype, data, dims);
    // }

    // createTypedArray(size) {
    //     const dtype = this.dtype || 'float32';
    //     return dtype === 'float16' ? new Uint16Array(size) : new Float32Array(size);
    // }

    async load(base_model, options) {
        const models = options.models;
        const provider = options.provider || "webgpu";
        const verbose = options.verbose;
        const local = options.local;
        const hasFP16 = (provider === "wasm") ? false : options.hasFP16;
        this.profiler = options.profiler;
        for (const [name, model] of Object.entries(models)) {
            const model_path = (local) ? "models/" + base_model : "https://huggingface.co/" + base_model + "/resolve/main/" + model.url;

            log(`loading... ${name},  ${provider}`);
            const json_bytes = await fetchAndCache(model_path + "/config.json");
            let textDecoder = new TextDecoder();
            //const model_config = JSON.parse(textDecoder.decode(json_bytes));

            const model_bytes = await fetchAndCache(model_path + "/model.onnx");
            //const externaldata = (model.externaldata) ? await fetchAndCache(model_path + "/model.onnx_data") : false;
            const externaldata = (model.externaldata) ? (model_path + "/model.onnx_data") : false;
            //const externaldata = (model.externaldata) ? await fetchAndCache(model_path + "/weights.pb") : false;
            let modelSize = model_bytes.byteLength;
            if (externaldata) {
                modelSize += externaldata.byteLength;
            }
            log(`model size ${Math.round(modelSize / 1024 / 1024)} MB`);

            const opt = {
                executionProviders: [provider]
            }

            if (externaldata !== undefined) {
                opt.externalData = [
                    {
                        data: externaldata,
                        path: "model.onnx_data"
                    },
                ]
            }
            if (verbose) {
                opt.logSeverityLevel = 0;
                opt.logVerbosityLevel = 0;
                ort.env.logLevel = "verbose";
            }
            log(`creating session for ${name} ...`);
            if (externaldata) {
                this.models[name] = await ort.InferenceSession.create(model_path + "/model.onnx", opt);
            } else {
                this.models[name] = await ort.InferenceSession.create(model_bytes, opt);
            }
            this.dtype = (hasFP16) ? "float16" : "float32";
        }
    }

    async infer(text) {
        try {
            document.getElementById('status').innerText = "generating ...";

            const { input_ids } = await this.tokenizer(text.value, {
                padding: true,
                max_length: 77,
                truncation: true,
                return_tensor: false
            });

            const { input_ids: uncond_ids } = await this.tokenizer(this.negativePrompt, {
                padding: true,
                max_length: 77,
                truncation: true,
                return_tensor: false
            });
            
            const batch_size = this.batch_size;
            let start = performance.now();
            const input_ids_i64 = toBigInt64Array(input_ids);
            const uncond_ids_i64 = toBigInt64Array(uncond_ids);

            const condOut = await this.models["text_encoder"].run({
                "input_ids": new ort.Tensor("int64", input_ids_i64, [1, input_ids.length])
            });
            const uncondOut = await this.models["text_encoder"].run({
                "input_ids": new ort.Tensor("int64", uncond_ids_i64, [1, uncond_ids.length])
            });

            const cond_hidden = condOut.last_hidden_state;   // [1,77,768]
            const uncond_hidden = uncondOut.last_hidden_state; // [1,77,768]

            const batch_cond = this.repeat_tensor(cond_hidden, batch_size);     // [B, 77, 768]
            const batch_uncond = this.repeat_tensor(uncond_hidden, batch_size); // [B, 77, 768]

            let perf_info = [`text_encoder: ${(performance.now() - start).toFixed(1)}ms`];

            const num_inference_steps = 30;
            this.scheduler.setTimesteps(num_inference_steps);
            console.log('timesteps', this.scheduler.timesteps.slice(0, 10));

            for (let j = 0; j < this.modelConfig.images; j++) {
                const latent_shape = [batch_size, 4, 64, 64];
                let latents = new ort.Tensor(
                    randn_latents(latent_shape, this.scheduler.initNoiseSigma),
                    latent_shape
                );

                for (let i = 0; i < this.scheduler.timesteps.length; i++) {
                    const t = this.scheduler.timesteps[i];

                    //const latent_model_input = scale_model_inputs(latents);
                    //TODO: now use scaling
                    const latent_model_input = latents;

                    const latentsCpu = await latents.getData();
                    console.log('before step', t, Math.min(...latentsCpu), Math.max(...latentsCpu));

                    start = performance.now();
                    const tTensor = new ort.Tensor("float32", new Float32Array([t]), []);

                    // 1) UNet with unconditional embedding
                    let feed = {
                        "sample": latent_model_input,
                        "timestep": tTensor,
                        "encoder_hidden_states": batch_uncond,
                    };
                    const { out_sample: out_uncond } = await this.models["unet"].run(feed);

                    // 2) UNet with conditional embedding
                    feed = {
                        "sample": latent_model_input,
                        "timestep": tTensor,
                        "encoder_hidden_states": batch_cond,
                    };
                    const { out_sample: out_cond } = await this.models["unet"].run(feed);

                    perf_info.push(`unet t=${t}: ${(performance.now() - start).toFixed(1)}ms`);

                    // CFG: eps = eps_uncond + s * (eps_text - eps_uncond)
                    const eps_uncond = await out_uncond.getData();
                    const eps_text = await out_cond.getData();
                    const guided = new Float32Array(eps_uncond.length);
                    for (let k = 0; k < eps_uncond.length; k++) {
                        guided[k] = eps_uncond[k] + this.guidance_scale * (eps_text[k] - eps_uncond[k]);
                    }
                    const guidedTensor = new ort.Tensor("float32", guided, [batch_size, 4, 64, 64]);

                    latents = this.scheduler.step(guidedTensor, t, latents);

                    const latentsCpuAfter = await latents.getData();
                    console.log('after step', t, Math.min(...latentsCpuAfter), Math.max(...latentsCpuAfter));
                }

                // vae_decoder
                start = performance.now();
                const latentsCpuFinal = await latents.getData();
                const scaledLatentsData = Float32Array.from(latentsCpuFinal, x => x / vae_scaling_factor);
                const scaled_latents = new ort.Tensor(
                    "float32",
                    scaledLatentsData,
                    latents.dims
                );
                const { sample } = await this.models["vae_decoder"].run({ "latent_sample": scaled_latents });
                perf_info.push(`vae_decoder: ${(performance.now() - start).toFixed(1)}ms`);

                const first_image = this.extract_from_batch(sample, 0);
                await this.draw_image(first_image, j);
                log(perf_info.join(", "));
                perf_info = [];
            }

            cond_hidden.dispose();
            uncond_hidden.dispose();

            log("done");

        } catch (error) {
            log(error);
        }
    }

    /**
     * draw an image from tensor
     * @param {ort.Tensor} t
     * @param {number} image_nr
     */
    async draw_image(t, image_nr) {
        const pix = await t.getData();
        for (let i = 0; i < pix.length; i++) {
            let x = pix[i];
            x = x / 2 + 0.5;
            if (x < 0.) x = 0.;
            if (x > 1.) x = 1.;
            pix[i] = x;
        }
        const tmpTensor = new ort.Tensor('float32', pix, t.dims);
        const imageData = tmpTensor.toImageData({ tensorLayout: 'NCWH', format: 'RGB' });
        const canvas = document.getElementById(`img_canvas_${image_nr}`);
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        canvas.getContext('2d').putImageData(imageData, 0, 0);
        const div = document.getElementById(`img_div_${image_nr}`);
        div.style.opacity = 1.;
    }
}