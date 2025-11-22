// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run sd1.5 with webgpu in onnxruntime-web.
//

import ort from 'onnxruntime-web/webgpu';
import { PNDMScheduler } from './PNDMScheduler.js';

function log(i) { console.log(i); document.getElementById('status').innerText += `\n${i}`; }

/*
 * get configuration from url
*/
function toBigInt64Array(ids) {
    const src = Array.isArray(ids) ? ids : Array.from(ids);
    const out = new BigInt64Array(src.length);
    for (let i = 0; i < src.length; i++) {
        out[i] = BigInt(src[i]);
    }
    return out;
}

function getConfig() {
    const query = window.location.search.substring(1);
    var config = {
        // model: "models/onnx-sd-turbo-fp16",
        model: "sd15-onnx-web",
        provider: "webgpu",
        device: "gpu",
        threads: "1",
        images: "2",
    };
    let vars = query.split("&");
    for (var i = 0; i < vars.length; i++) {
        let pair = vars[i].split("=");
        if (pair[0] in config) {
            config[pair[0]] = decodeURIComponent(pair[1]);
        } else if (pair[0].length > 0) {
            throw new Error("unknown argument: " + pair[0]);
        }
    }
    config.threads = parseInt(config.threads);
    config.images = parseInt(config.images);
    return config;
}

/*
 * initialize latents with random noise
 */
function randn_latents(shape, noise_sigma) {
    function randn() {
        // Use the Box-Muller transform
        let u = Math.random();
        let v = Math.random();
        let z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        return z;
    }
    let size = 1;
    shape.forEach(element => {
        size *= element;
    });

    let data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = randn() * noise_sigma;
    }
    return data;
}

/*
 * fetch and cache model
 */
async function fetchAndCache(base_url, model_path) {
    const url = `${base_url}/${model_path}`;

    if (model_path.endsWith(".onnx_data")) {
        log(`${model_path} (network, no cache)`);
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status} when fetching ${url}`);
        }
        return await response.arrayBuffer();
    }

    try {
        const cache = await caches.open("onnx");
        let cachedResponse = await cache.match(url);
        if (cachedResponse == undefined) {
            await cache.add(url);
            cachedResponse = await cache.match(url);
            log(`${model_path} (network)`);
        } else {
            log(`${model_path} (cached)`);
        }
        const data = await cachedResponse.arrayBuffer();
        return data;
    } catch (error) {
        log(`${model_path} (network, fallback)`);
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status} when fetching ${url}`);
        }
        return await response.arrayBuffer();
    }
}

/*
 * load models used in the pipeline
 */
async function load_models(models) {
    const cache = await caches.open("onnx");
    let missing = 0;
    for (const [name, model] of Object.entries(models)) {
        const url = `${config.model}/${model.url}`;
        let cachedResponse = await cache.match(url);
        if (cachedResponse === undefined) {
            missing += model.size;
        }
    }
    if (missing > 0) {
        log(`downloading ${missing} MB from network ... it might take a while`);
    } else {
        log("loading...");
    }

    for (const [name, model] of Object.entries(models)) {
        try {
            const start = performance.now();

            let sess_opt = { ...opt, ...model.opt };

            if (model.external) {
                // ---- UNet 
                const modelUrl = `${config.model}/${model.url}`;
                const externalUrl = `${config.model}/${model.external}`;

                sess_opt = {
                    ...sess_opt,
                    externalData: [
                        {
                            path: "./model.onnx_data",
                            data: externalUrl,
                        },
                    ],
                };

                models[name].sess = await ort.InferenceSession.create(modelUrl, sess_opt);
            } else {
                const model_bytes = await fetchAndCache(config.model, model.url);
                models[name].sess = await ort.InferenceSession.create(model_bytes, sess_opt);
            }

            const stop = performance.now();
            log(`${model.url} in ${(stop - start).toFixed(1)}ms`);
        } catch (e) {
            log(`${model.url} failed, ${e}`);
        }
    }
    log("ready.");
}

const config = getConfig();

const models = {
    "unet": {
        url: "unet/model.onnx", size: 640,
        external: "unet/model.onnx_data",
        opt: { freeDimensionOverrides: { batch_size: 1, num_channels: 4, height: 64, width: 64, sequence_length: 77 } }
    },
    "text_encoder": {
        url: "text_encoder/model.onnx", size: 1700,
        opt: { freeDimensionOverrides: { batch_size: 1 } },
    },
    "vae_decoder": {
        url: "vae_decoder/model.onnx", size: 95,
        opt: { freeDimensionOverrides: { batch_size: 1, num_channels_latent: 4, height_latent: 64, width_latent: 64 } }
    }
};

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/';
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

let tokenizer;
let loading;
const sigma = 14.6146;
const gamma = 0;
const vae_scaling_factor = 0.18215;

// CFG 设置
const negativePrompt = "blurry, low quality, bad anatomy";
const guidance_scale = 7.5;

const scheduler = new PNDMScheduler({
    num_train_timesteps: 1000,
    beta_start: 0.00085,
    beta_end: 0.012,
    beta_schedule: 'scaled_linear',
    prediction_type: 'epsilon',
    skip_prk_steps: true,
    final_alpha_cumprod: 1e-3,
});

const text = document.getElementById("user-input");
text.value = "Paris with the river in the background";

const opt = {
    executionProviders: [config.provider],
    enableMemPattern: false,
    enableCpuMemArena: false,
    extra: {
        session: {
            disable_prepacking: "1",
            use_device_allocator_for_initializers: "1",
            use_ort_model_bytes_directly: "1",
            use_ort_model_bytes_for_initializers: "1"
        }
    },
};

switch (config.provider) {
    case "webgpu":
        if (!("gpu" in navigator)) {
            throw new Error("webgpu is NOT supported");
        }
        opt.preferredOutputLocation = { last_hidden_state: "gpu-buffer" };
        break;
    case "webnn":
        if (!("ml" in navigator)) {
            throw new Error("webnn is NOT supported");
        }
        opt.executionProviders = [{
            name: "webnn",
            deviceType: config.device,
            powerPreference: 'default'
        }];
        break;
}

// Event listener for Ctrl + Enter or CMD + Enter
document.getElementById('user-input').addEventListener('keydown', function (e) {
    if (e.ctrlKey && e.key === 'Enter') {
        generate_image();
    }
});
document.getElementById('send-button').addEventListener('click', function () {
    generate_image();
});

/*
 * scale the latents (sigma 归一化)
 */
function scale_model_inputs(t) {
    return t;
}

/**
 * draw an image from tensor
 * @param {ort.Tensor} t
 * @param {number} image_nr
*/
async function draw_image(t, image_nr) {
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

async function generate_image() {
    try {
        document.getElementById('status').innerText = "generating ...";

        if (tokenizer === undefined) {
            // AutoTokenizer 依然来自全局（通过 HTML script 引入）
            tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch16');
            tokenizer.pad_token_id = 0;
        }
        await loading;

        for (let j = 0; j < config.images; j++) {
            const div = document.getElementById(`img_div_${j}`);
            div.style.opacity = 0.5;
        }

        // 正向 prompt
        const { input_ids } = await tokenizer(text.value, {
            padding: true,
            max_length: 77,
            truncation: true,
            return_tensor: false
        });

        // 负向 prompt
        const { input_ids: uncond_ids } = await tokenizer(negativePrompt, {
            padding: true,
            max_length: 77,
            truncation: true,
            return_tensor: false
        });

        // text-encoder：分别编码 cond / uncond
        let start = performance.now();
        const input_ids_i64 = toBigInt64Array(input_ids);
        const uncond_ids_i64 = toBigInt64Array(uncond_ids);

        const condOut = await models.text_encoder.sess.run({
            "input_ids": new ort.Tensor("int64", input_ids_i64, [1, input_ids.length])
        });
        const uncondOut = await models.text_encoder.sess.run({
            "input_ids": new ort.Tensor("int64", uncond_ids_i64, [1, uncond_ids.length])
        });

        const cond_hidden = condOut.last_hidden_state;   // [1,77,768]
        const uncond_hidden = uncondOut.last_hidden_state; // [1,77,768]

        let perf_info = [`text_encoder: ${(performance.now() - start).toFixed(1)}ms`];

        const num_inference_steps = 30;
        scheduler.setTimesteps(num_inference_steps);
        console.log('timesteps', scheduler.timesteps.slice(0, 10));

        for (let j = 0; j < config.images; j++) {
            const latent_shape = [1, 4, 64, 64];
            let latents = new ort.Tensor(
                randn_latents(latent_shape, scheduler.initNoiseSigma),
                latent_shape
            );

            for (let i = 0; i < scheduler.timesteps.length; i++) {
                const t = scheduler.timesteps[i];

                const latent_model_input = scale_model_inputs(latents);

                const latentsCpu = await latents.getData();
                console.log('before step', t, Math.min(...latentsCpu), Math.max(...latentsCpu));

                start = performance.now();
                const tTensor = new ort.Tensor("float32", new Float32Array([t]), []);

                // 1) UNet with unconditional embedding
                let feed = {
                    "sample": latent_model_input,
                    "timestep": tTensor,
                    "encoder_hidden_states": uncond_hidden,
                };
                const { out_sample: out_uncond } = await models.unet.sess.run(feed);

                // 2) UNet with conditional embedding
                feed = {
                    "sample": latent_model_input,
                    "timestep": tTensor,
                    "encoder_hidden_states": cond_hidden,
                };
                const { out_sample: out_cond } = await models.unet.sess.run(feed);

                perf_info.push(`unet t=${t}: ${(performance.now() - start).toFixed(1)}ms`);

                // CFG: eps = eps_uncond + s * (eps_text - eps_uncond)
                const eps_uncond = await out_uncond.getData();
                const eps_text = await out_cond.getData();
                const guided = new Float32Array(eps_uncond.length);
                for (let k = 0; k < eps_uncond.length; k++) {
                    guided[k] = eps_uncond[k] + guidance_scale * (eps_text[k] - eps_uncond[k]);
                }
                const guidedTensor = new ort.Tensor("float32", guided, [1, 4, 64, 64]);

                latents = scheduler.step(guidedTensor, t, latents);

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
            const { sample } = await models.vae_decoder.sess.run({ "latent_sample": scaled_latents });
            perf_info.push(`vae_decoder: ${(performance.now() - start).toFixed(1)}ms`);

            await draw_image(sample, j);
            log(perf_info.join(", "));
            perf_info = [];
        }

        cond_hidden.dispose();
        uncond_hidden.dispose();

        log("done");
    } catch (e) {
        log(e);
    }
}

async function hasFp16() {
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter.features.has('shader-f16');
    } catch (e) {
        return false;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    hasFp16().then((fp16) => {
        if (fp16) {
            loading = load_models(models);
        } else {
            log("Your GPU or Browser doesn't support webgpu/f16");
        }
    });
});