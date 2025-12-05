import * as ort from 'onnxruntime-web/webgpu';
import { toBigInt64Array } from '../utils/common.js';
import { randomNormalTensor, cat, ensureFloat32Array } from '../util/Tensor.js';
import { PNDMScheduler } from '../scheduler/PNDMScheduler.js';
import { AutoTokenizer, Tensor } from '@xenova/transformers';
import { Session } from '../backends/index.js';

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

// Fix wasmPaths for both local and GitHub Pages
const basePath = document.location.pathname.endsWith('/') 
    ? document.location.pathname 
    : document.location.pathname.replace(/\/[^\/]*$/, '/');
ort.env.wasm.wasmPaths = basePath + 'dist/';
console.log('wasmPaths:', ort.env.wasm.wasmPaths);


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
        this.debugShapes = Boolean(modelConfig.debugShapes);
        // Batch size for parallel image generation
        this.batch_size = modelConfig.batchSize || 1;
        // Text-to-Video Zero config
        this.video_length = 8;  // Number of frames to generate (official default: 8)
        this.t0 = 44;  // Timestep index for T0 (official default: 44)
        this.t1 = 47;  // Timestep index for T1 (official default: 47)
        // Motion field strength controls how much the scene moves between frames
        // Official defaults: 12, 12 - this creates apparent camera/scene motion
        this.motion_field_strength_x = 12;
        this.motion_field_strength_y = 12;
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
        
        // Cache for sequential frame generation
        this.cachedFirstFrameT1 = null;  // x_1_t1: First frame latent at T1
        this.cachedFirstFrameT0 = null;  // x_1_t0: First frame latent at T0
        this.cachedPromptEmbeds = null;  // Prompt embeddings
        this.cachedTimesteps = null;     // Timesteps array
        this.cachedTimestepIndices = null; // Timestep indices
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
        // const requestedType = this.modelConfig.floatType || (hasFP16 ? 'float16' : 'float32');
        if (!this.dtype) {
            this.dtype = hasFP16 ? 'float16' : 'float32';
            log(`Using tensor type: ${this.dtype}`);
        }
        this.profiler = options.profiler;
        for (const [name, model] of Object.entries(models)) {
            if (name === "unet") {
                this.is_local_unet = model.local;
            }
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
        }
    }

    async infer(text) {
        try {
            document.getElementById('status').innerText = "generating ...";
            await this.inferVideoSequential(text.value);
            log("done");

        } catch (error) {
            log(error);
        }
    }


    async inferVideoSequential(prompt) {
        const num_inference_steps = 50;
        const video_length = this.video_length;
        
        log(`Starting Text-to-Video Zero: ${video_length} frames`);
        let totalStart = performance.now();
        
        // ======== Phase 1: Setup (CrossFrameAttnProcessor is built into model) ========
        log("Phase 1: Encoding prompt...");
        let start = performance.now();
        
        // Get prompt embeddings (batch_size = 1 for single prompt)
        const originalBatchSize = this.batch_size;
        this.batch_size = 1;
        let prompt_embeds = await this.getPromptEmbeds(prompt, this.negativePrompt);
        this.batch_size = originalBatchSize;
        
        // prompt_embeds shape: [2, L, D] where 2 = (neg, pos) for CFG
        const prompt_embeds_input = this.dtype === 'float16' ? toOrtTensorFp16(prompt_embeds) : toOrtTensor(prompt_embeds);
        log(`  text_encoder: ${(performance.now() - start).toFixed(1)}ms`);

        // ======== Phase 2: Prepare timesteps ========
        this.scheduler.setTimesteps(num_inference_steps);
        const timesteps = getSchedulerTimesteps(this.scheduler);
        
        const numSteps = timesteps.length;
        const idx_T1_end = numSteps - this.t1 - 1;  
        const idx_T0_end = numSteps - this.t0 - 1;
        
        const ts_0_to_T1 = timesteps.slice(0, idx_T1_end);         // First backward
        const ts_T1_to_T0 = timesteps.slice(idx_T1_end, idx_T0_end); // Second backward
        const ts_T1_to_0 = timesteps.slice(idx_T1_end);            // Final backward
        
        log(`  Timesteps: total=${numSteps}, 0->T1=${ts_0_to_T1.length}, T1->T0=${ts_T1_to_T0.length}, T1->0=${ts_T1_to_0.length}`);

        // ======== Phase 3: First backward loop (0 -> T1) for first frame ========
        log("Phase 2: Backward loop 0 -> T1 (first frame)...");
        start = performance.now();

        const latent_shape = [1, 4, 64, 64];
        const initialSeed = '';  // Empty string = random seed each time
        let latents = randomNormalTensor(latent_shape, 0, this.scheduler.initNoiseSigma, 'float32', initialSeed);
        
        this.resetSchedulerState();
        
        let x_1_t1 = await this.backwardLoop(latents, ts_0_to_T1, prompt_embeds_input);
        
        // ======== Phase 4: Second backward loop (T1 -> T0) for first frame ========
        log("Phase 3: Backward loop T1 -> T0 (first frame)...");

        let x_1_t0 = await this.backwardLoop(x_1_t1, ts_T1_to_T0, prompt_embeds_input);
        
        log(`  First frame states ready: ${(performance.now() - start).toFixed(1)}ms`);
        
        // ======== Phase 5: Propagate and warp to other frames ========
        log("Phase 4: Creating motion field and warping latents...");
        start = performance.now();
        
        const frame_ids = Array.from({length: video_length}, (_, i) => i);
        
        const otherFramesT0 = [];
        for (let i = 1; i < video_length; i++) {
            otherFramesT0.push(cloneTensor(x_1_t0));
        }
        let x_2k_t0 = cat(otherFramesT0);  // [video_length-1, 4, 64, 64]
        
        x_2k_t0 = this.createMotionFieldAndWarpLatents(x_2k_t0, frame_ids.slice(1));
        
        log(`  Motion warp complete: ${(performance.now() - start).toFixed(1)}ms`);
        
        // ======== Phase 6: Forward process T0 -> T1 for other frames ========
        log("Phase 5: Forward process T0 -> T1...");
        start = performance.now();
        
        const t0_timestep = timesteps[idx_T0_end];  // Timestep value at T0
        const t1_timestep = timesteps[idx_T1_end];  // Timestep value at T1
        
        let x_2k_t1 = this.forwardLoop(x_2k_t0, t0_timestep, t1_timestep);
        
        log(`  Forward process complete: ${(performance.now() - start).toFixed(1)}ms`);
        
        // ======== Phase 7: Combine all frames at T1 ========
        log("Phase 6: Combining all frames at T1...");
        let x_1k_t1 = cat([x_1_t1, x_2k_t1]);  // [video_length, 4, 64, 64]
        
        let frames = [];
        for (let i = 0; i < video_length; i++) {
            frames.push(x_1k_t1.slice([i, i + 1]));
        }
        
        const pairPromptEmbeds = this.makePromptBatchForPair(prompt_embeds);  // [4, L, D]
        const pairPromptInput = this.dtype === 'float16' 
            ? toOrtTensorFp16(pairPromptEmbeds) 
            : toOrtTensor(pairPromptEmbeds);
        
        // ======== Phase 8: Final backward T1 -> 0 (timestep-by-timestep with batch=2 limitation) ========

        log("Phase 7: Final denoising T1 -> 0 (timestep-by-timestep)...");
        start = performance.now();
        
        const schedulerForFinalStage = this.createSchedulerCopy();
        schedulerForFinalStage.setTimesteps(num_inference_steps);
        
        const originalScheduler = this.scheduler;
        this.scheduler = schedulerForFinalStage;
        this.resetSchedulerState();
        
        const totalSteps = ts_T1_to_0.length;
        
        for (let stepIdx = 0; stepIdx < totalSteps; stepIdx++) {
            const t = ts_T1_to_0[stepIdx];
            
            if (stepIdx % 5 === 0 || stepIdx === totalSteps - 1) {
                log(`  Step ${stepIdx + 1}/${totalSteps} (t=${t})...`);
            }
            
            // Step 1: Collect noise predictions for ALL frames at this timestep
            const noisePreds = [];
            
            
            const anchor = frames[0];
            
            // Process frame 0 and frame 1 together
            const frame1 = frames[1];
            const combined01 = cat([anchor, frame1]);
            const noisePred01 = await this.unetGetNoisePred(combined01, t, pairPromptInput);
            noisePreds[0] = noisePred01.slice([0, 1]);
            noisePreds[1] = noisePred01.slice([1, 2]);
            
            // Get noise predictions for remaining frames (frames 2 to video_length-1)
            for (let k = 2; k < video_length; k++) {
                const frame_k = frames[k];
                const combined = cat([anchor, frame_k]);
                const noisePredPair = await this.unetGetNoisePred(combined, t, pairPromptInput);

                noisePreds[k] = noisePredPair.slice([1, 2]);
            }
            
            // Step 2: Apply scheduler.step() ONCE for ALL frames together
            const allNoisePreds = cat(noisePreds);  // [video_length, 4, 64, 64]
            const allLatents = cat(frames);         // [video_length, 4, 64, 64]
            
            const allNextLatents = this.scheduler.step(allNoisePreds, t, allLatents);
            
            for (let i = 0; i < video_length; i++) {
                frames[i] = allNextLatents.slice([i, i + 1]);
            }
        }
        

        this.scheduler = originalScheduler;
        
        let denoiseTime = performance.now() - start;
        log(`  Total denoise time: ${(denoiseTime / 1000).toFixed(1)}s`);
        
        // ======== Phase 9: Decode and display all frames ========
        log("Phase 8: Decoding frames...");
        start = performance.now();
        const frameCanvases = [];
        
        for (let i = 0; i < video_length; i++) {
            const frame_images = await this.makeImages(frames[i]);
            await this.draw_image(toOrtTensor(frame_images[0]), i);
            const canvas = document.getElementById(`img_canvas_${i}`);
            if (canvas) frameCanvases.push(canvas);
        }
        
        log(`  Decode time: ${(performance.now() - start).toFixed(0)}ms`);
        
        // ======== Phase 10: Create video animation ========
        log("Phase 9: Creating video animation...");
        start = performance.now();
        this.createCanvasAnimation(frameCanvases, document.getElementById('video_container'), 4);  // Lower FPS for smoother appearance
        log(`  Animation created: ${(performance.now() - start).toFixed(0)}ms`);
        
        log(`Complete! Total: ${((performance.now() - totalStart) / 1000).toFixed(1)}s`);
    }


    /**
     * Expand prompt embeddings for multiple frames
     */
    expandPromptEmbedsForFrames(prompt_embeds, num_frames) {
        const expandedParts = [];
        
        // Add negative embeddings for all frames first
        for (let frame = 0; frame < num_frames; frame++) {
            const negEmbed = prompt_embeds.slice([0, 1]);
            expandedParts.push(negEmbed);
        }
        
        // Then add positive embeddings for all frames
        for (let frame = 0; frame < num_frames; frame++) {
            const posEmbed = prompt_embeds.slice([1, 2]);
            expandedParts.push(posEmbed);
        }
        
        return cat(expandedParts);
    }

    /**
     * Create video using WebCodecs API
     */
    async createVideoWithWebCodecs(canvases, container, fps) {
        const width = canvases[0].width;
        const height = canvases[0].height;
        
        const chunks = [];
        
        const encoder = new VideoEncoder({
            output: (chunk, meta) => {
                const buffer = new ArrayBuffer(chunk.byteLength);
                chunk.copyTo(buffer);
                chunks.push({ buffer, meta, timestamp: chunk.timestamp, type: chunk.type });
            },
            error: (e) => { throw e; }
        });
        
        encoder.configure({
            codec: 'vp8',
            width: width,
            height: height,
            bitrate: 2_000_000,
            framerate: fps,
        });
        
        const frameSequence = [...Array(canvases.length).keys()];
        const reverseSequence = [...frameSequence].reverse().slice(1, -1);
        const loopSequence = [...frameSequence, ...reverseSequence];
        
        for (let i = 0; i < loopSequence.length; i++) {
            const canvasIdx = loopSequence[i];
            const frame = new VideoFrame(canvases[canvasIdx], {
                timestamp: (i * 1_000_000) / fps,
                duration: 1_000_000 / fps,
            });
            encoder.encode(frame);
            frame.close();
        }
        
        await encoder.flush();
        encoder.close();
        
        // Create blob and video element
        const blob = new Blob(chunks.map(c => c.buffer), { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        
        container.innerHTML = `
            <video id="generated_video" width="${width}" height="${height}" autoplay loop muted controls>
                <source src="${url}" type="video/webm">
            </video>
        `;
    }

    /**
     * Fallback: Create smooth canvas-based animation with download support
     */
    createCanvasAnimation(canvases, container, fps) {
        if (!canvases || canvases.length === 0) {
            log("Error: No canvases provided for animation");
            return;
        }
        
        const width = canvases[0].width;
        const height = canvases[0].height;
        
        this.generatedCanvases = canvases;
        this.videoFps = fps;
        
        container.innerHTML = `
            <div style="position: relative; display: inline-block;">
                <canvas id="animation_canvas" width="${width}" height="${height}" style="border: 1px solid #444;"></canvas>
                <div style="margin-top: 5px; text-align: center;">
                    <button id="play_pause_btn" class="btn btn-sm btn-secondary">⏸ Pause</button>
                    <button id="download_gif_btn" class="btn btn-sm btn-success" style="margin-left: 5px;">💾 Download Frames</button>
                    <button id="download_webm_btn" class="btn btn-sm btn-info" style="margin-left: 5px;">💾 Download WebM</button>
                    <span id="frame_counter" style="margin-left: 10px; color: #aaa;">Frame: 1/${canvases.length}</span>
                </div>
            </div>
        `;
        
        const animCanvas = document.getElementById('animation_canvas');
        const ctx = animCanvas.getContext('2d');
        const playPauseBtn = document.getElementById('play_pause_btn');
        const frameCounter = document.getElementById('frame_counter');
        
        const frameSequence = [...Array(canvases.length).keys()];
        const reverseSequence = [...frameSequence].reverse().slice(1, -1);
        const loopSequence = [...frameSequence, ...reverseSequence];
        
        let currentIdx = 0;
        let isPlaying = true;
        let animationId = null;
        let lastFrameTime = 0;
        const frameDuration = 1000 / fps;
        
        ctx.drawImage(canvases[0], 0, 0);
        
        const drawFrame = (timestamp) => {
            if (!isPlaying) return;
            
            if (timestamp - lastFrameTime >= frameDuration) {
                const canvasIdx = loopSequence[currentIdx];
                ctx.drawImage(canvases[canvasIdx], 0, 0);
                frameCounter.textContent = `Frame: ${canvasIdx + 1}/${canvases.length}`;
                
                currentIdx = (currentIdx + 1) % loopSequence.length;
                lastFrameTime = timestamp;
            }
            
            animationId = requestAnimationFrame(drawFrame);
        };
        
        animationId = requestAnimationFrame(drawFrame);
        
        playPauseBtn.addEventListener('click', () => {
            isPlaying = !isPlaying;
            playPauseBtn.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
            if (isPlaying) {
                lastFrameTime = 0;
                animationId = requestAnimationFrame(drawFrame);
            } else if (animationId) {
                cancelAnimationFrame(animationId);
            }
        });
        
        const downloadGifBtn = document.getElementById('download_gif_btn');
        downloadGifBtn.addEventListener('click', () => {
            this.downloadAsGif(canvases, fps);
        });
        
        const downloadWebmBtn = document.getElementById('download_webm_btn');
        downloadWebmBtn.addEventListener('click', () => {
            this.downloadAsVideo(canvases, fps);
        });
        
        log(`Animation created: ${canvases.length} frames at ${fps} fps (ping-pong loop)`);
    }

    /**
     * Download animation as video using MediaRecorder (more reliable than WebCodecs)
     */
    async downloadAsVideo(canvases, fps) {
        log("Creating video... Please wait...");
        
        const width = canvases[0].width;
        const height = canvases[0].height;
        
        const frameSequence = [...Array(canvases.length).keys()];
        const reverseSequence = [...frameSequence].reverse().slice(1, -1);
        const loopSequence = [...frameSequence, ...reverseSequence];
        
        const recordCanvas = document.createElement('canvas');
        recordCanvas.width = width;
        recordCanvas.height = height;
        const ctx = recordCanvas.getContext('2d', { 
            alpha: false,  // Disable alpha for better performance
            desynchronized: true  // Allow async rendering
        });
        ctx.imageSmoothingEnabled = false;
        
        // Get supported mime type - prefer VP9 for better quality
        let mimeType = 'video/webm';
        if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) {
            mimeType = 'video/webm;codecs=vp9';
        } else if (MediaRecorder.isTypeSupported('video/webm;codecs=vp8')) {
            mimeType = 'video/webm;codecs=vp8';
        } else if (MediaRecorder.isTypeSupported('video/webm')) {
            mimeType = 'video/webm';
        } else if (MediaRecorder.isTypeSupported('video/mp4')) {
            mimeType = 'video/mp4';
        }
        
        log(`Using codec: ${mimeType}, resolution: ${width}x${height}`);
        
        const captureRate = Math.max(fps, 30);
        const stream = recordCanvas.captureStream(captureRate);
        const chunks = [];
        
        const bitrate = width * height * 100;
        
        const recorder = new MediaRecorder(stream, {
            mimeType: mimeType,
            videoBitsPerSecond: bitrate
        });
        
        log(`Video bitrate: ${(bitrate / 1_000_000).toFixed(1)} Mbps`);
        
        recorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                chunks.push(e.data);
            }
        };
        
        return new Promise((resolve) => {
            recorder.onstop = () => {
                const blob = new Blob(chunks, { type: mimeType });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
                a.download = `t2v_video_${Date.now()}.${ext}`;
                a.click();
                URL.revokeObjectURL(url);
                log(`Video downloaded! (${ext} format)`);
                resolve();
            };
            
            // Start recording
            recorder.start();
            
            // Draw frames at the specified fps
            const frameDuration = 1000 / fps;
            let frameIndex = 0;
            
            // Play through the sequence 2 times for a decent video length
            const totalFrames = loopSequence.length * 2;
            
            const drawNextFrame = () => {
                if (frameIndex >= totalFrames) {
                    recorder.stop();
                    return;
                }
                
                const canvasIdx = loopSequence[frameIndex % loopSequence.length];
                ctx.drawImage(canvases[canvasIdx], 0, 0);
                frameIndex++;
                
                setTimeout(drawNextFrame, frameDuration);
            };
            
            drawNextFrame();
        });
    }

    /**
     * Download animation as GIF using canvas frames
     */
    async downloadAsGif(canvases, fps) {
        log("Creating GIF... (this may take a moment)");
        
        const width = canvases[0].width;
        const height = canvases[0].height;
        
        const frameSequence = [...Array(canvases.length).keys()];
        const reverseSequence = [...frameSequence].reverse().slice(1, -1);
        const loopSequence = [...frameSequence, ...reverseSequence];
        
        try {
            // Try to use gif.js if available
            if (typeof GIF !== 'undefined') {
                log(`GIF.js found, creating ${loopSequence.length} frame animation...`);
                const gif = new GIF({
                    workers: 2,
                    quality: 10,
                    width: width,
                    height: height,
                    workerScript: 'https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.worker.js'
                });
                
                gif.on('error', (err) => {
                    log(`GIF error: ${err}. Falling back to PNG sequence...`);
                    this.downloadAsPngSequence(canvases);
                });
                
                for (const idx of loopSequence) {
                    gif.addFrame(canvases[idx], { delay: 1000 / fps, copy: true });
                }
                
                gif.on('finished', (blob) => {
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `t2v_animation_${Date.now()}.gif`;
                    a.click();
                    URL.revokeObjectURL(url);
                    log("GIF downloaded!");
                });
                
                gif.render();
            } else {
                // Fallback: Download as PNG sequence
                log("GIF.js not available (typeof GIF = " + typeof GIF + "). Downloading as PNG sequence...");
                await this.downloadAsPngSequence(canvases);
            }
        } catch (error) {
            log(`GIF creation failed: ${error.message}. Downloading as PNG sequence...`);
            await this.downloadAsPngSequence(canvases);
        }
    }

    /**
     * Download frames as individual PNG files
     */
    async downloadAsPngSequence(canvases) {
        log(`Downloading ${canvases.length} frames as PNG files...`);
        
        for (let i = 0; i < canvases.length; i++) {
            const canvas = canvases[i];
            const dataUrl = canvas.toDataURL('image/png');
            const a = document.createElement('a');
            a.href = dataUrl;
            a.download = `frame_${String(i).padStart(3, '0')}.png`;
            a.click();
            
            // Small delay to avoid browser blocking
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        log("PNG sequence downloaded!");
    }

    resetSchedulerState() {
        this.scheduler.ets = [];
        this.scheduler.counter = 0;
        this.scheduler.cur_model_output = null;
        this.scheduler.cur_sample = null;
    }

    /**
     * Create a fresh scheduler with the same config for the final stage.
     */
    createSchedulerCopy() {
        return new PNDMScheduler({
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: 'scaled_linear',
            prediction_type: 'epsilon',
            skip_prk_steps: true,
            final_alpha_cumprod: 1e-3,
        });
    }

    async backwardLoop(latents, timesteps, prompt_embeds_input) {
        const doClassifierFreeGuidance = this.guidance_scale > 1.0;
        const latentDims = getTensorDims(latents);
        const batch_size = latentDims[0] || 1;


        for (const t of timesteps) {
            let tTensor;
            let Bt = doClassifierFreeGuidance ? batch_size * 2 : batch_size;
            if (this.is_local_unet) {
                if (this.dtype === 'float16') {
                    const arr = new Float16Array(Bt);
                    arr.fill(t);
                    const tData = new Float16Array(arr);
                    tTensor = new ort.Tensor("float16", tData, [Bt]);
                } else {
                    const arr = new Float32Array(Bt);
                    arr.fill(t);
                    tTensor = new ort.Tensor("float32", arr, [Bt]);
                }
            } else {
                // Remote UNet also needs batch dimension for timestep
                if (this.dtype === 'float16') {
                    const arr = new Float16Array(Bt);
                    arr.fill(t);
                    tTensor = new ort.Tensor("float16", arr, [Bt]);
                } else {
                    const arr = new Float32Array(Bt);
                    arr.fill(t);
                    tTensor = new ort.Tensor("float32", arr, [Bt]);
                }
            }

            const latent_input = doClassifierFreeGuidance ? cat([latents, latents.clone()]) : latents;
            const latent_input_ort = this.dtype === 'float16' ? toOrtTensorFp16(latent_input) : toOrtTensor(latent_input);

            let feed = {
                "sample": latent_input_ort,
                "timestep": tTensor,
                "encoder_hidden_states": prompt_embeds_input,
            };
            const noise = await this.models["unet"].run(feed);

            let noise_pred = noise.out_sample;
            if (this.guidance_scale > 1.0) {
                const split = noise_pred.dims?.[0] ?? 0;
                const batchChunk = split / 2;
                const noisePredUncond = noise_pred.slice([0, batchChunk]);
                const noisePredText = noise_pred.slice([batchChunk, batchChunk * 2]);
                noise_pred = noisePredUncond.add(noisePredText.sub(noisePredUncond).mul(this.guidance_scale));
            }

            latents = this.scheduler.step(noise_pred, t, latents);
        }

        return latents.clone();
    }

    /**
     * Run UNet to get noise prediction WITHOUT calling scheduler.step().
     */
    async unetGetNoisePred(latents, t, prompt_embeds_input) {
        const doClassifierFreeGuidance = this.guidance_scale > 1.0;
        const latentDims = getTensorDims(latents);
        const batch_size = latentDims[0] || 1;

        let tTensor;
        let Bt = doClassifierFreeGuidance ? batch_size * 2 : batch_size;
        if (this.is_local_unet) {
            if (this.dtype === 'float16') {
                const arr = new Float16Array(Bt);
                arr.fill(t);
                const tData = new Float16Array(arr);
                tTensor = new ort.Tensor("float16", tData, [Bt]);
            } else {
                const arr = new Float32Array(Bt);
                arr.fill(t);
                tTensor = new ort.Tensor("float32", arr, [Bt]);
            }
        } else {
            // Remote UNet also needs batch dimension for timestep
            if (this.dtype === 'float16') {
                const arr = new Float16Array(Bt);
                arr.fill(t);
                tTensor = new ort.Tensor("float16", arr, [Bt]);
            } else {
                const arr = new Float32Array(Bt);
                arr.fill(t);
                tTensor = new ort.Tensor("float32", arr, [Bt]);
            }
        }

        const latent_input = doClassifierFreeGuidance ? cat([latents, latents.clone()]) : latents;
        const latent_input_ort = this.dtype === 'float16' ? toOrtTensorFp16(latent_input) : toOrtTensor(latent_input);

        let feed = {
            "sample": latent_input_ort,
            "timestep": tTensor,
            "encoder_hidden_states": prompt_embeds_input,
        };
        const noise = await this.models["unet"].run(feed);

        let noise_pred = noise.out_sample;
        if (this.guidance_scale > 1.0) {
            const split = noise_pred.dims?.[0] ?? 0;
            const batchChunk = split / 2;
            const noisePredUncond = noise_pred.slice([0, batchChunk]);
            const noisePredText = noise_pred.slice([batchChunk, batchChunk * 2]);
            noise_pred = noisePredUncond.add(noisePredText.sub(noisePredUncond).mul(this.guidance_scale));
        }

        return noise_pred;
    }

    /**
     * Build prompt embeddings for a pair of frames (frame 0 as anchor, frame k).
     */
    makePromptBatchForPair(basePromptEmbeds) {

        const negEmbed = basePromptEmbeds.slice([0, 1]);  // [1, L, D]
        const posEmbed = basePromptEmbeds.slice([1, 2]);  // [1, L, D]
        
        return cat([negEmbed, negEmbed, posEmbed, posEmbed]);  // [4, L, D]
    }


    forwardLoop(x_t0, t0, t1, seed = '') {
        const dims = getTensorDims(x_t0);

        const eps = randomNormalTensor(dims, 0, 1, x_t0.type || 'float32', seed);
        
        const alphasData = this.scheduler.alphas.data;
        
        let alpha_vec = 1.0;
        const startIdx = Math.round(t0);
        const endIdx = Math.round(t1);
        
        const minIdx = Math.min(startIdx, endIdx);
        const maxIdx = Math.max(startIdx, endIdx);
        
        for (let t = minIdx; t < maxIdx; t++) {
            if (t >= 0 && t < alphasData.length) {
                alpha_vec *= alphasData[t];
            }
        }
        
        const sqrt_alpha_vec = Math.sqrt(alpha_vec);
        const sqrt_one_minus_alpha_vec = Math.sqrt(1 - alpha_vec);
        
        const x_t1 = x_t0.mul(sqrt_alpha_vec).add(eps.mul(sqrt_one_minus_alpha_vec));
        return x_t1;
    }

    /**
     * Create translation motion field
     */
    createMotionField(motion_field_strength_x, motion_field_strength_y, frame_ids, height, width) {
        const seq_length = frame_ids.length;
        const data = new Float32Array(seq_length * 2 * height * width);
        
        for (let fr_idx = 0; fr_idx < seq_length; fr_idx++) {
            const frame_id = frame_ids[fr_idx];
            // Channel 0: x motion
            const x_offset = fr_idx * 2 * height * width;
            // Channel 1: y motion
            const y_offset = x_offset + height * width;
            
            for (let i = 0; i < height * width; i++) {
                data[x_offset + i] = motion_field_strength_x * frame_id;
                data[y_offset + i] = motion_field_strength_y * frame_id;
            }
        }
        
        return new Tensor('float32', data, [seq_length, 2, height, width]);
    }

    /**
     * Warp a single latent with given flow
     */
    warpSingleLatent(latent, reference_flow) {
        const latentDims = getTensorDims(latent);
        const flowDims = getTensorDims(reference_flow);
        const [_, C, h, w] = latentDims;
        const [__, ___, H, W] = flowDims;
        
        const flowData = reference_flow.data;
        const latentData = latent.data;
        const warpedData = new Float32Array(C * h * w);
        
        const coords_t0_H_W = new Float32Array(2 * H * W);  // [2, H, W] for [x, y]
        
        for (let Y = 0; Y < H; Y++) {
            for (let X = 0; X < W; X++) {
                const idx = Y * W + X;

                const coord_x = X;  
                const coord_y = Y;  
                

                const flow_x = flowData[idx];        
                const flow_y = flowData[H * W + idx];  
                
                let new_x = coord_x + flow_x;
                let new_y = coord_y + flow_y;
                
                new_x /= W;
                new_y /= H;
                
                new_x = new_x * 2.0 - 1.0;
                new_y = new_y * 2.0 - 1.0;
                
                coords_t0_H_W[idx] = new_x;          
                coords_t0_H_W[H * W + idx] = new_y;  
            }
        }
        
        for (let c = 0; c < C; c++) {
            for (let y_out = 0; y_out < h; y_out++) {
                for (let x_out = 0; x_out < w; x_out++) {

                    const src_Y = y_out * (H - 1) / (h - 1);
                    const src_X = x_out * (W - 1) / (w - 1);
                    

                    const X0 = Math.floor(src_X);
                    const X1 = Math.min(X0 + 1, W - 1);
                    const Y0 = Math.floor(src_Y);
                    const Y1 = Math.min(Y0 + 1, H - 1);
                    
                    const wx = src_X - X0;
                    const wy = src_Y - Y0;
                    
                    // Interpolate normalized x coord
                    const nx00 = coords_t0_H_W[Y0 * W + X0];
                    const nx01 = coords_t0_H_W[Y0 * W + X1];
                    const nx10 = coords_t0_H_W[Y1 * W + X0];
                    const nx11 = coords_t0_H_W[Y1 * W + X1];
                    const norm_x = (1 - wx) * (1 - wy) * nx00 + wx * (1 - wy) * nx01 +
                                   (1 - wx) * wy * nx10 + wx * wy * nx11;
                    
                    // Interpolate normalized y coord
                    const ny00 = coords_t0_H_W[H * W + Y0 * W + X0];
                    const ny01 = coords_t0_H_W[H * W + Y0 * W + X1];
                    const ny10 = coords_t0_H_W[H * W + Y1 * W + X0];
                    const ny11 = coords_t0_H_W[H * W + Y1 * W + X1];
                    const norm_y = (1 - wx) * (1 - wy) * ny00 + wx * (1 - wy) * ny01 +
                                   (1 - wx) * wy * ny10 + wx * wy * ny11;
                    

                    let pixel_x = (norm_x + 1.0) / 2.0 * (w - 1);
                    let pixel_y = (norm_y + 1.0) / 2.0 * (h - 1);
                    

                    pixel_x = this.reflectCoord(pixel_x, w);
                    pixel_y = this.reflectCoord(pixel_y, h);
                    

                    const src_x_int = Math.round(pixel_x);
                    const src_y_int = Math.round(pixel_y);
                    
                    const dstIdx = c * h * w + y_out * w + x_out;
                    const srcIdx = c * h * w + src_y_int * w + src_x_int;
                    warpedData[dstIdx] = latentData[srcIdx];
                }
            }
        }
        
        return new Tensor(latent.type || 'float32', warpedData, latentDims);
    }

    /**
     * Reflect coordinate for reflection padding
     */
    reflectCoord(coord, size) {
        if (coord < 0) {
            coord = -coord;
        }
        if (coord >= size) {
            // Reflect from the boundary
            const overflow = coord - (size - 1);
            coord = (size - 1) - overflow;
        }
        // Clamp to valid range
        return Math.max(0, Math.min(size - 1, coord));
    }

    /**
     * Create motion field and warp latents accordingly
     */
    createMotionFieldAndWarpLatents(latents, frame_ids) {
        const dims = getTensorDims(latents);
        const [N, C, h, w] = dims;
        
        const motionH = 512; 
        const motionW = 512;
        const motion_field = this.createMotionField(
            this.motion_field_strength_x,
            this.motion_field_strength_y,
            frame_ids,
            motionH,
            motionW
        );
        
        // Warp each latent
        const warpedLatents = [];
        for (let i = 0; i < N; i++) {
            const latent_i = latents.slice([i, i + 1]); // [1, C, h, w]
            const flow_i_data = new Float32Array(2 * motionH * motionW);
            const srcOffset = i * 2 * motionH * motionW;
            for (let j = 0; j < 2 * motionH * motionW; j++) {
                flow_i_data[j] = motion_field.data[srcOffset + j];
            }
            const flow_i = new Tensor('float32', flow_i_data, [1, 2, motionH, motionW]);
            const warped = this.warpSingleLatent(latent_i, flow_i);
            warpedLatents.push(warped);
        }
        
        return cat(warpedLatents);
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


    async getPromptEmbeds (prompt, negativePrompt) {
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
        const highestTokenLength = Math.max(
            promptTokens.input_ids.length,
            negPromptTokens.input_ids.length,
        );

        const basePromptEmbeds = await this.encodePrompt(prompt, highestTokenLength);           // [1, L, D]
        const baseNegEmbeds    = await this.encodePrompt(negativePrompt || '', highestTokenLength); // [1, L, D]

        const batchSize = this.batch_size || 1;

        const repeatBatch = (t) => {
            const dims = getTensorDims(t); // [1, L, D]
            const copies = [];
            for (let i = 0; i < batchSize; i++) {
                if (typeof t.clone === 'function') {
                    copies.push(t.clone());
                } else {
                    const data = t.data instanceof Float32Array
                        ? t.data.slice()
                        : Float32Array.from(t.data || []);
                    copies.push(new Tensor(t.type || 'float32', data, dims.slice()));
                }
            }
            return cat(copies);  // [batchSize, L, D]
        };

        const promptEmbeds = repeatBatch(basePromptEmbeds);      // [B, L, D]
        const negEmbeds    = repeatBatch(baseNegEmbeds);         // [B, L, D]

        if (this.guidance_scale > 1.0) {

            return cat([negEmbeds, promptEmbeds]);
        } else {
            return promptEmbeds;
        }
    }

    async makeImages (latents) {
        const scaled = latents.div(vae_scaling_factor);
        const dims = getTensorDims(scaled);
        const batch = Math.max(dims[0] ?? 1, 1);
        const images = [];

        for (let i = 0; i < batch; i++) {
            const latent_sample = batch === 1 ? scaled : scaled.slice([i, i + 1]);
            //const latent_sample_ort = this.dtype === 'float16' ? toOrtTensorFp16(latent_sample) : toOrtTensor(latent_sample);
            const latent_sample_ort = toOrtTensor(latent_sample);
            const decoded = await this.models["vae_decoder"].run({ "latent_sample": latent_sample_ort });
            if (this.debugShapes) {
                logTensorShape('decoded_sample', decoded.sample);
            }
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

function toOrtTensorFp16(tensor) {
    if (tensor instanceof ort.Tensor) {
        return tensor;
    }
    const data = tensor.data instanceof Float16Array ? tensor.data : Float16Array.from(tensor.data);
    return new ort.Tensor("float16", data, getTensorDims(tensor));
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

function castTensorType(tensor, type) {
    if (!tensor || !type || tensor.type === type) {
        return tensor;
    }
    const data = ensureFloat32Array(tensor.data);
    return new Tensor(type, data.slice ? data : Float32Array.from(data), getTensorDims(tensor));
}

function logTensorShape(label, tensor) {
    if (!tensor) {
        console.log(`[shape] ${label}: <empty>`);
        return;
    }
    const dims = getTensorDims(tensor) || [];
    console.log(`[shape] ${label}: [${dims.join(', ')}]`);
}

function cloneTensor(tensor) {
    if (!tensor) return null;
    const dims = getTensorDims(tensor);
    const data = tensor.data instanceof Float32Array 
        ? tensor.data.slice() 
        : Float32Array.from(tensor.data || []);
    return new Tensor(tensor.type || 'float32', data, dims);
}