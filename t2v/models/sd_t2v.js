import * as ort from 'onnxruntime-web/webgpu';
import { SDModel } from './sd.js';
import { randn_latents, toBigInt64Array } from '../utils/common.js';

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = document.location.pathname.replace('index.html', '') + 'dist/';

function log(i) { 
    console.log(i); 
    document.getElementById('status').innerText += `\n${i}`; 
}

const vae_scaling_factor = 0.18215;

/**
 * Text-to-Video Pipeline
 * Extends SDModel to reuse text_encoder, vae_decoder and other components
 */
export class SDT2VModel extends SDModel {
    constructor(modelConfig) {
        super(modelConfig);
        // T2V specific model components
        this.models = {
            "text_encoder": {},      // Inherited from SDModel
            "vae_decoder": {},       // Inherited from SDModel
            "unet_t2v": {},          // T2V specific temporal UNet
        };
        
        // T2V specific parameters
        this.num_frames = modelConfig.num_frames || 16;  // Number of frames to generate
        this.fps = modelConfig.fps || 8;                 // Frames per second
        this.video_length = modelConfig.video_length || 2; // Video length in seconds
    }

    /**
     * Load T2V models
     * @param {string} base_model - HuggingFace model path
     * @param {object} options - Loading options
     */
    async load(base_model, options) {
        const models = options.models;
        const provider = options.provider || "webgpu";
        const verbose = options.verbose;
        const local = options.local;
        const hasFP16 = (provider === "wasm") ? false : options.hasFP16;
        this.profiler = options.profiler;

        for (const [name, model] of Object.entries(models)) {
            const model_path = (local) 
                ? "models/" + base_model 
                : "https://huggingface.co/" + base_model + "/resolve/main/" + model.url;

            log(`loading... ${name}, ${provider}`);
            
            const model_bytes = await this.fetchAndCache(model_path + "/model.onnx");
            const externaldata = (model.externaldata) ? (model_path + "/model.onnx_data") : false;
            
            let modelSize = model_bytes.byteLength;
            if (externaldata) {
                modelSize += externaldata.byteLength;
            }
            log(`model size ${Math.round(modelSize / 1024 / 1024)} MB`);

            const opt = {
                executionProviders: [provider]
            };

            if (externaldata) {
                opt.externalData = [
                    {
                        data: externaldata,
                        path: "model.onnx_data"
                    },
                ];
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

    /**
     * Helper method: Fetch file from cache or network
     */
    async fetchAndCache(url) {
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

    /**
     * Text-to-Video main inference function
     * @param {HTMLInputElement} text - Input text prompt
     */
    async text_to_video(text) {
        try {
            document.getElementById('status').innerText = "generating video...";

            let perf_info = [];
            let start = performance.now();

            // ============ Step 1: Text Encoding ============
            log("Step 1: Encoding text...");
            const { input_ids } = await this.tokenizer(text.value, {
                padding: true,
                max_length: 77,
                truncation: true,
                return_tensor: false
            });

            // Optional: negative prompt for CFG
            const { input_ids: uncond_ids } = await this.tokenizer(this.negativePrompt, {
                padding: true,
                max_length: 77,
                truncation: true,
                return_tensor: false
            });

            const input_ids_i64 = toBigInt64Array(input_ids);
            const uncond_ids_i64 = toBigInt64Array(uncond_ids);

            const condOut = await this.models["text_encoder"].run({
                "input_ids": new ort.Tensor("int64", input_ids_i64, [1, input_ids.length])
            });
            const uncondOut = await this.models["text_encoder"].run({
                "input_ids": new ort.Tensor("int64", uncond_ids_i64, [1, uncond_ids.length])
            });

            const cond_hidden = condOut.last_hidden_state;     // [1, 77, 768]
            const uncond_hidden = uncondOut.last_hidden_state; // [1, 77, 768]
            
            perf_info.push(`text_encoder: ${(performance.now() - start).toFixed(1)}ms`);

            // ============ Step 2: Initialize Video Latents ============
            log("Step 2: Initializing video latents...");
            start = performance.now();
            
            // Video latents shape: [batch, channels, frames, height, width]
            // For SD, typically [1, 4, num_frames, 64, 64]
            const latent_shape = [1, 4, this.num_frames, 64, 64];
            let video_latents = new ort.Tensor(
                randn_latents(latent_shape, this.scheduler.initNoiseSigma),
                latent_shape
            );
            
            perf_info.push(`init_latents: ${(performance.now() - start).toFixed(1)}ms`);

            // ============ Step 3: Denoising Loop ============
            log("Step 3: Denoising loop...");
            const num_inference_steps = 30;
            this.scheduler.setTimesteps(num_inference_steps);

            for (let i = 0; i < this.scheduler.timesteps.length; i++) {
                const t = this.scheduler.timesteps[i];
                log(`Denoising step ${i+1}/${num_inference_steps}, t=${t}`);
                
                start = performance.now();
                
                // 3.1: Prepare latent model input (may need scaling)
                const latent_model_input = video_latents;
                const tTensor = new ort.Tensor("float32", new Float32Array([t]), []);

                // 3.2: UNet forward pass (unconditional)
                let feed = {
                    "sample": latent_model_input,
                    "timestep": tTensor,
                    "encoder_hidden_states": uncond_hidden,
                };
                const { out_sample: out_uncond } = await this.models["unet_t2v"].run(feed);

                // 3.3: UNet forward pass (conditional)
                feed = {
                    "sample": latent_model_input,
                    "timestep": tTensor,
                    "encoder_hidden_states": cond_hidden,
                };
                const { out_sample: out_cond } = await this.models["unet_t2v"].run(feed);

                // 3.4: Classifier-Free Guidance (CFG)
                const eps_uncond = await out_uncond.getData();
                const eps_text = await out_cond.getData();
                const guided = new Float32Array(eps_uncond.length);
                for (let k = 0; k < eps_uncond.length; k++) {
                    guided[k] = eps_uncond[k] + this.guidance_scale * (eps_text[k] - eps_uncond[k]);
                }
                const guidedTensor = new ort.Tensor("float32", guided, latent_shape);

                // 3.5: Scheduler step (update latents)
                video_latents = this.scheduler.step(guidedTensor, t, video_latents);
                
                perf_info.push(`unet_t2v step ${i+1}: ${(performance.now() - start).toFixed(1)}ms`);
            }

            // ============ Step 4: VAE Decode (frame by frame) ============
            log("Step 4: Decoding video frames...");
            start = performance.now();
            
            const frames = await this.decode_video_latents(video_latents);
            
            perf_info.push(`vae_decoder (${this.num_frames} frames): ${(performance.now() - start).toFixed(1)}ms`);

            // ============ Step 5: Render Video ============
            log("Step 5: Rendering video...");
            await this.render_video(frames);

            // Cleanup
            cond_hidden.dispose();
            uncond_hidden.dispose();

            log("Video generation done!");
            log(perf_info.join(", "));

        } catch (error) {
            log(`Error: ${error.message}`);
            console.error(error);
        }
    }

    /**
     * Decode video latents to pixel frames
     * @param {ort.Tensor} video_latents - shape [1, 4, num_frames, 64, 64]
     * @returns {Array<ort.Tensor>} Array of image tensors for each frame
     */
    async decode_video_latents(video_latents) {
        const latentsData = await video_latents.getData();
        const [batch, channels, num_frames, height, width] = video_latents.dims;
        
        const frames = [];
        const frame_size = channels * height * width;

        // Decode frame by frame
        for (let f = 0; f < num_frames; f++) {
            log(`Decoding frame ${f+1}/${num_frames}...`);
            
            // Extract single frame latent
            const frame_latent_data = new Float32Array(frame_size);
            for (let i = 0; i < frame_size; i++) {
                frame_latent_data[i] = latentsData[f * frame_size + i] / vae_scaling_factor;
            }
            
            const frame_latent = new ort.Tensor(
                "float32",
                frame_latent_data,
                [1, channels, height, width]
            );

            // VAE decode
            const { sample } = await this.models["vae_decoder"].run({ 
                "latent_sample": frame_latent 
            });
            
            frames.push(sample);
        }

        return frames;
    }

    /**
     * Render video to canvas or create video file
     * @param {Array<ort.Tensor>} frames - Array of video frames
     */
    async render_video(frames) {
        // Option 1: Display all frames to multiple canvases (for preview)
        for (let i = 0; i < frames.length; i++) {
            await this.draw_image(frames[i], i);
        }

        // Option 2: Create video file (requires additional video encoding library like ffmpeg.wasm)
        // await this.encode_to_video(frames);
    }

    /**
     * Optional: Encode frames to video file
     * Requires integration with ffmpeg.wasm or similar library
     */
    async encode_to_video(frames) {
        log("Video encoding not yet implemented");
        // TODO: Use ffmpeg.wasm or MediaRecorder API
        // 1. Create canvas stream
        // 2. Use MediaRecorder to record
        // 3. Generate .mp4 or .webm file
    }

    /**
     * Draw a single frame to specified canvas
     * @param {ort.Tensor} frame - Image frame tensor
     * @param {number} frame_nr - Frame number
     */
    async draw_image(frame, frame_nr) {
        const pix = await frame.getData();
        for (let i = 0; i < pix.length; i++) {
            let x = pix[i];
            x = x / 2 + 0.5;
            if (x < 0.) x = 0.;
            if (x > 1.) x = 1.;
            pix[i] = x;
        }
        const tmpTensor = new ort.Tensor('float32', pix, frame.dims);
        const imageData = tmpTensor.toImageData({ tensorLayout: 'NCWH', format: 'RGB' });
        
        const canvas = document.getElementById(`video_frame_${frame_nr}`);
        if (canvas) {
            canvas.width = imageData.width;
            canvas.height = imageData.height;
            canvas.getContext('2d').putImageData(imageData, 0, 0);
            const div = canvas.parentElement;
            if (div) div.style.opacity = 1.;
        } else {
            log(`Warning: canvas video_frame_${frame_nr} not found`);
        }
    }
}
