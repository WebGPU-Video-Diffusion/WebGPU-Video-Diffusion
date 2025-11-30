
import ort from 'onnxruntime-web/webgpu';
//import { SDModel } from './models/sd.js';
import { SDModel } from './models/sd_t2v.js';
import {draw_image} from './utils/common.js';

function log(i) { console.log(i); document.getElementById('status').innerText += `\n${i}`; }

const text = document.getElementById("user-input");

function toBoolean(value) {
    if (typeof value === 'boolean') {
        return value;
    }
    if (typeof value === 'number') {
        return value !== 0;
    }
    if (typeof value === 'string') {
        const normalized = value.trim().toLowerCase();
        return normalized === '1' || normalized === 'true' || normalized === 'yes';
    }
    return Boolean(value);
}

function getConfig() {
    const query = window.location.search.substring(1);
    var config = {
        //model: "rz610/sd-1.5-ort",
        //model: "subpixel/small-stable-diffusion-v0-onnx-ort-web",
        //model: "onnx-community/stable-diffusion-v1-5-ONNX",
        //model: "tlwu/stable-diffusion-v1-5-onnxruntime",
        model: "ykeee/StableDiffusion1.5-fp32",
        local_model: "sd1.5/t2vzero-fp32",
        provider: "webgpu",
        device: "gpu",
        threads: "1",
        images: "1",
        verbose: 0,
        local: 0,
        intType: "int64",
        floatType: "float32",
        batchSize: "1",
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
    config.batchSize = parseInt(config.batchSize);
    config.local = toBoolean(config.local);
    config.verbose = parseInt(config.verbose) || 0;
    return config;
}

const config = getConfig();

const models = {
    "unet": {
        url: "unet",
        externaldata: true,
        extfilename: 1, // 1 = model.onnx_data, 2 = weights.pb
        local: true,
    },
    "text_encoder": {
        url: "text_encoder",
        externaldata: false,
        local: false
    },
    "vae_decoder": {
        url: "vae_decoder",
        externaldata: false,
        local: false
    }
}

const sd = new SDModel(config);

async function Init(hasFP16) {
  try {

    log("Loading model...");
    await sd.load(config.model, {
        models: models,
        provider: config.provider,
        verbose: config.verbose,
        local: config.local,
        hasFP16: hasFP16,
        base_model_local: config.local_model
    });
    log("Ready.");
  } catch (error) {
    log(error);
  }
}

async function hasFp16() {
    try {
        const adapter = await navigator.gpu.requestAdapter()
        return adapter.features.has('shader-f16')
    } catch (e) {
        return false
    }
}

document.addEventListener("DOMContentLoaded", () => {
    loading = Init(false);
});

document.getElementById('send-button').addEventListener('click', function (e) {
    sd.infer(text);
});

// async function test_tokenizer() {
//     const input_ids = await sd.tokenizer(text);
//     log(`input_ids: ${JSON.stringify(input_ids)}`);
//     log(`input_ids length: ${input_ids?.length || 'undefined'}`);
// }

// async function test_text_encoder() {
//     const last_hidden_state = await sd.encode_text(text);
//     log(`last_hidden_state: ${JSON.stringify(last_hidden_state.data.slice(0,10))}...`);
// }

// async function test_unet() {
//     const last_hidden_state = await sd.encode_text(text);
//     const {out_sample, latent} = await sd.one_step_unet(last_hidden_state);
//     log(`out_sample: ${JSON.stringify(out_sample.data.slice(0,10))}...`);
// }

// async function test_full_pipeline() {
//     const last_hidden_state = await sd.encode_text(text);
//     const {out_sample, latent} = await sd.one_step_unet(last_hidden_state);
//     const sample = await sd.decode_latents(out_sample, latent);
//     log(`sample: ${JSON.stringify(sample.data.slice(0,10))}...`);
//     draw_image(sample, 0);
// }
