import * as ort from 'onnxruntime-web/webgpu';
import { Tensor } from '@xenova/transformers';
import { SDModel } from './sd_t2v.js';
import { PNDMScheduler } from '../scheduler/PNDMScheduler.js';
import { randomNormalTensor, cat, ensureFloat32Array,range } from '../util/Tensor.js';

function log(i) { console.log(i); document.getElementById('status').innerText += `\n${i}`; }
/**
 * Text-to-video zero pipeline (scaffold).
 *
 * Converts the PyTorch-based `t2vzero.py` pipeline into a browser-friendly
 * ONNX/WebGPU implementation. This class reuses the Stable Diffusion text-to-image
 * primitives defined in {@link SDModel} and extends them with video-specific logic.
 *
 * NOTE: This is an initial architecture pass. It omits the "two frames at a time"
 * memory optimization that we will add later. For now, all frames are processed in
 * a single batch for clarity.
 */
export class SDT2VZeroModel extends SDModel {
	constructor(modelConfig) {
		super(modelConfig);
		this.video_length = modelConfig.videoLength ?? 2;
		this.height = modelConfig.height ?? 512;
		this.width = modelConfig.width ?? 512;
		this.num_videos_per_prompt = modelConfig.numVideosPerPrompt ?? 1;
		this.guidance_scale = modelConfig.guidance_scale ?? this.guidance_scale;
	}

	/**
	 * Entry point mirroring Diffusers pipeline.__call__.
	 * @param {string|HTMLInputElement} textOrElement - Prompt text or DOM input element
	 * containing the prompt value.
	 * @param {object} options - Optional overrides extracted from UI/query params.
	 */
	async infer(textOrElement, options = {}) {
		const resolvedPrompt = typeof textOrElement === 'string'
			? textOrElement
			: (textOrElement?.value ?? '');

		const opts = this._normalizeVideoOptions(options);
		if (opts.videoLength <= 0) {
			throw new Error('videoLength must be positive');
		}
		//const frameIds = opts.frameIds ?? Array.from({ length: opts.videoLength }, (_, idx) => idx);
        const frameIds = [];
        for (let i = 0; i < opts.videoLength; i++) {
            frameIds.push(i);
        }
        log(`frameids: ${frameIds}`);
		if (opts.numVideosPerPrompt !== 1) {
			throw new Error('numVideosPerPrompt > 1 not supported yet');
		}

		const promptEmbedsBatch = await this.getPromptEmbeds(
			resolvedPrompt,
			opts.negativePrompt ?? this.negativePrompt
		);
        const promptEmbeds = cat([promptEmbedsBatch.slice([0,1]), promptEmbedsBatch.slice([this.batch_size, this.batch_size +1])]);
        logTensorShape('Prompt Embeds', promptEmbeds);
		const promptOrt = this.dtype === 'float16'
			? toOrtTensorFp16(promptEmbeds)
			: toOrtTensor(promptEmbeds);

		const schedulerConfig = { ...this.scheduler.config };
		this.scheduler = new PNDMScheduler(schedulerConfig);
		this.scheduler.setTimesteps(opts.numInferenceSteps);
		const timesteps = getSchedulerTimesteps(this.scheduler);
		if (!timesteps.length) {
			throw new Error('Scheduler did not provide timesteps');
		}
		const stageSlices = computeStageSlices(timesteps, opts.t0, opts.t1);
		log(`[Pipeline] Stage1 steps: ${stageSlices.stage1.length}, Stage2: ${stageSlices.stage2.length}, Stage3: ${stageSlices.stage3.length}`);
		log(`[Pipeline] t0Value=${stageSlices.t0Value}, t1Value=${stageSlices.t1Value}`);

		let latents = randomNormalTensor([1, 4, 64, 64], 0, this.scheduler.initNoiseSigma);
        logTensorShape('Initial Latents', latents);
		let x1_t1 = stageSlices.stage1.length
			? await this.backwardLoop(latents, stageSlices.stage1, promptOrt, opts)
			: latents;
        
        logTensorShape('latents after first backward loop', x1_t1);
		
		// Save scheduler state before second backward pass
		const schedulerTemplate = new PNDMScheduler(schedulerConfig);
		schedulerTemplate.setTimesteps(opts.numInferenceSteps);
		const schedulerSnapshot = snapshotScheduler(this.scheduler);
		log('[Pipeline] Saved scheduler snapshot');

		const x1_t0 = stageSlices.stage2.length
			? await this.backwardLoop(x1_t1, stageSlices.stage2, promptOrt, opts)
			: x1_t1;

        logTensorShape('latents after second backward loop', x1_t0);
		let x2k_t1 = null;
		if (opts.videoLength > 1) {
			log(`[Pipeline] Processing additional frames (videoLength=${opts.videoLength})`);
			let propagated = repeatTensorAlongBatch(x1_t0, opts.videoLength - 1);
			//const repeatedFrameIds = repeatFrameIdsForBatch(frameIds.slice(1), this.batch_size);
            const repeatedFrameIds = frameIds.slice(1);
            logTensorShape('init propagated latents', propagated);
            log(`repeated frame ids: ${repeatedFrameIds}`);
			propagated = createMotionFieldAndWarpLatents(
				opts.motionFieldStrengthX,
				opts.motionFieldStrengthY,
				repeatedFrameIds,
				propagated,
			);
            logTensorShape('propagated latents', propagated);
			x2k_t1 = this.forwardLoop(propagated, stageSlices.t0Value, stageSlices.t1Value);
            logTensorShape('latents after forward loop', x2k_t1);
		}

		const combinedLatents = x2k_t1 ? cat([x1_t1, x2k_t1]) : x1_t1;
		log(`[Pipeline] Combining latents: x1_t1 + ${x2k_t1 ? 'x2k_t1' : 'none'}`);
		logTensorShape('Combined Latents', combinedLatents);
		
		const repeatedPrompt = repeatPromptEmbeds(promptEmbeds, opts.videoLength);
		logTensorShape('Repeated Prompt Embeds', repeatedPrompt);
		
		const repeatedPromptOrt = this.dtype === 'float16'
			? toOrtTensorFp16(repeatedPrompt)
			: toOrtTensor(repeatedPrompt);

		// Restore scheduler state before final backward pass
		this.scheduler = restoreSchedulerSnapshot(schedulerSnapshot, schedulerTemplate);
		// CRITICAL: Clear cur_sample to prevent shape mismatch with multi-frame latents
		this.scheduler.cur_sample = null;
		log('[Pipeline] Restored scheduler snapshot');
		
		log(`[Pipeline] Starting final backward loop with Stage3 (${stageSlices.stage3.length} steps)`);
		const finalLatents = stageSlices.stage3.length
			? await this.backwardLoop(combinedLatents, stageSlices.stage3, repeatedPromptOrt, opts)
			: combinedLatents;
		logTensorShape('Final Latents', finalLatents);

		const frames = await this.decodeVideoFrames(finalLatents, { videoLength: opts.videoLength });
        
        log(`frames length: ${frames.length}`);
        for (let i = 0; i < opts.videoLength; i++) {
            await this.draw_image(toOrtTensor(frames[i]), i);
        }
	}

	_normalizeVideoOptions(options) {
		return {
			numInferenceSteps: options.numInferenceSteps ?? 30,
			guidanceScale: options.guidanceScale ?? this.guidance_scale,
			videoLength: options.videoLength ?? this.video_length,
			numImagesPerPrompt: options.numImagesPerPrompt ?? this.num_videos_per_prompt,
			negativePrompt: options.negativePrompt ?? this.negativePrompt,
			frameRate: options.frameRate ?? 4,
			height: options.height ?? this.height,
			width: options.width ?? this.width,
			motionFieldStrengthX: options.motionFieldStrengthX ?? 12,
			motionFieldStrengthY: options.motionFieldStrengthY ?? 12,
			t0: options.t0 ?? 26,
			t1: options.t1 ?? 28,
			//frameIds: Array.isArray(options.frameIds) ? options.frameIds.slice() : undefined,
            frameIds: undefined,
			numVideosPerPrompt: options.numVideosPerPrompt ?? this.num_videos_per_prompt,
		};
	}

	prepareVideoTimesteps(numSteps) {
		this.scheduler.setTimesteps(numSteps);
		return getSchedulerTimesteps(this.scheduler);
	}

	prepareVideoLatents(opts) {
		const frames = opts.videoLength * opts.numImagesPerPrompt;
		const latentBatch = this.batch_size * frames;
		const latentShape = [latentBatch, 4, 64, 64];
		return randomNormalTensor(latentShape, 0, this.scheduler.initNoiseSigma, this.dtype ?? 'float32');
	}

	async denoiseVideo(latents, timesteps, promptEmbedsOrt, opts) {
		return this.backwardLoop(latents, timesteps, promptEmbedsOrt, opts);
	}

	forwardLoop(latents, t0, t1, seed = '') {
		log(`[ForwardLoop] t0=${t0}, t1=${t1}`);
		logTensorShape('[ForwardLoop] input latents', latents);
		
		const alphaData = ensureFloat32Array(this.scheduler?.alphas?.data);
		if (!alphaData.length) {
			throw new Error('Scheduler alphas unavailable for forwardLoop');
		}
		
		const start = clamp(Math.round(Math.min(t0, t1)), 0, alphaData.length - 1);
		const end = clamp(Math.round(Math.max(t0, t1)), 0, alphaData.length);
		log(`[ForwardLoop] alpha range [${start}, ${end}), total alphas=${alphaData.length}`);
		
		if (end <= start) {
			log('[ForwardLoop] No steps needed, returning input');
			return latents;
		}
		
		let alphaProd = 1;
		for (let i = start; i < end; i++) {
			alphaProd *= alphaData[i];
		}
		alphaProd = clamp(alphaProd, 1e-6, 0.999999);
		log(`[ForwardLoop] alphaProd=${alphaProd.toFixed(6)}`);
		
		const sqrtAlpha = Math.sqrt(alphaProd);
		const sqrtOneMinusAlpha = Math.sqrt(Math.max(1 - alphaProd, 1e-6));
		log(`[ForwardLoop] sqrtAlpha=${sqrtAlpha.toFixed(6)}, sqrtOneMinusAlpha=${sqrtOneMinusAlpha.toFixed(6)}`);
		
		const dims = getTensorDims(latents);
		const noise = randomNormalTensor(dims, 0, 1, latents.type || this.dtype || 'float32', seed);
		logTensorShape('[ForwardLoop] noise', noise);
		
		const result = latents.mul(sqrtAlpha).add(noise.mul(sqrtOneMinusAlpha));
		logTensorShape('[ForwardLoop] output latents', result);
		
		return result;
	}

	async backwardLoop(latents, timesteps, promptEmbedsOrt, opts) {
		log(`[BackwardLoop] Starting with ${timesteps.length} timesteps`);
		logTensorShape('[BackwardLoop] input latents', latents);
		
		let currentLatents = latents;
		const do_cfg = opts.guidanceScale > 1.0;
		log(`[BackwardLoop] do_cfg=${do_cfg}, guidanceScale=${opts.guidanceScale}`);
		
		for (let stepIdx = 0; stepIdx < timesteps.length; stepIdx++) {
			const t = timesteps[stepIdx];
			log(`[BackwardLoop] Step ${stepIdx + 1}/${timesteps.length}, timestep=${t}`);
			
			const latentInput = do_cfg ? cat([currentLatents, currentLatents.clone()]) : currentLatents;
			logTensorShape(`[BackwardLoop] Step ${stepIdx + 1} latentInput`, latentInput);
			
			const sample = this.dtype === 'float16'
				? toOrtTensorFp16(latentInput)
				: toOrtTensor(latentInput);
			const timestepTensor = this.makeTimestepTensor(t, latentInput.dims?.[0] ?? this.batch_size);
			
			const feed = {
				sample,
				timestep: timestepTensor,
				encoder_hidden_states: promptEmbedsOrt,
			};
			const noise = await this.models['unet'].run(feed);
			let noisePred = noise.out_sample;
			logTensorShape(`[BackwardLoop] Step ${stepIdx + 1} noisePred (raw)`, noisePred);
			
			if (do_cfg) {
				const split = noisePred.dims?.[0] ?? 0;
				const half = split / 2;
				const noiseUncond = noisePred.slice([0, half]);
				const noiseCond = noisePred.slice([half, split]);
				noisePred = noiseUncond.add(noiseCond.sub(noiseUncond).mul(opts.guidanceScale));
				logTensorShape(`[BackwardLoop] Step ${stepIdx + 1} noisePred (after CFG)`, noisePred);
			}
			
			logTensorShape(`[BackwardLoop] Step ${stepIdx + 1} currentLatents (before step)`, currentLatents);
			const nextLatents = this.scheduler.step(noisePred, t, currentLatents);
			logTensorShape(`[BackwardLoop] Step ${stepIdx + 1} nextLatents (after step)`, nextLatents);
			currentLatents = nextLatents;
		}
		
		log('[BackwardLoop] Completed');
		return currentLatents;
	}

	async decodeVideoFrames(latents, opts) {
		log(`[Decode] Starting decode with videoLength=${opts.videoLength}`);
		logTensorShape('[Decode] input latents', latents);
		
		const dims = getTensorDims(latents);
		const total = dims[0] ?? 1;
		const framesPerPrompt = opts.videoLength;
		const batch = total / framesPerPrompt;
		
		log(`[Decode] total frames=${total}, framesPerPrompt=${framesPerPrompt}, batch=${batch}`);
		
		const decodedFrames = [];
		for (let i = 0; i < total; i++) {
			log(`[Decode] Decoding frame ${i + 1}/${total}`);
			const frameLatent = latents.slice([i, i + 1]);
			logTensorShape(`[Decode] frame ${i} latent`, frameLatent);
			
			const images = await super.makeImages(frameLatent);
			logTensorShape(`[Decode] frame ${i} decoded image`, images[0]);
			decodedFrames.push(images[0]);
		}
		
		log(`[Decode] Completed, decoded ${decodedFrames.length} frames`);
		return decodedFrames;
	}

	makeTimestepTensor(t, batch) {
		const dtype = this.dtype === 'float16' ? 'float16' : 'float32';
		const ctor = dtype === 'float16' ? Float16Array : Float32Array;
		const data = new ctor([t]);
		const dims = this.is_local_unet ? [batch] : [];
		if (this.is_local_unet && batch > 1) {
			const repeated = new ctor(batch);
			repeated.fill(t);
			return new ort.Tensor(dtype, repeated, [batch]);
		}
		return new ort.Tensor(dtype, data, dims);
	}
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
	return new ort.Tensor('float16', data, getTensorDims(tensor));
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

function clamp(value, min, max) {
	return Math.min(Math.max(value, min), max);
}

export function coordsGrid(batch, height, width, dtype = 'float32') {
	const total = batch * 2 * height * width;
	const data = new Float32Array(total);
	const hw = height * width;
	for (let b = 0; b < batch; b++) {
		const batchOffset = b * 2 * hw;
		for (let y = 0; y < height; y++) {
			for (let x = 0; x < width; x++) {
				const idx = batchOffset + y * width + x;
				data[idx] = x;
				data[idx + hw] = y;
			}
		}
	}
	return new Tensor(dtype, data, [batch, 2, height, width]);
}

export function warpSingleLatent(latent, referenceFlow) {
	const flowDims = getTensorDims(referenceFlow);
	if (flowDims.length < 3) {
		throw new Error('referenceFlow must have at least 3 dimensions [B, 2, H, W]');
	}
	const H = flowDims[flowDims.length - 2];
	const W = flowDims[flowDims.length - 1];
	const coords = coordsGrid(1, H, W);
	const coordsData = ensureFloat32Array(coords.data);
	const flowData = ensureFloat32Array(referenceFlow.data);
	const planeSize = H * W;
	for (let i = 0; i < planeSize; i++) {
		coordsData[i] += flowData[i];
		coordsData[i + planeSize] += flowData[i + planeSize];
	}
	for (let i = 0; i < planeSize; i++) {
		coordsData[i] = coordsData[i] / W;
		coordsData[i + planeSize] = coordsData[i + planeSize] / H;
	}
	for (let i = 0; i < coordsData.length; i++) {
		coordsData[i] = coordsData[i] * 2 - 1;
	}
	const latentDims = getTensorDims(latent);
	const h = latentDims[latentDims.length - 2];
	const w = latentDims[latentDims.length - 1];
	const resizedGrid = resizeFlowToShape(coordsData, H, W, h, w);
	return gridSampleNearest(latent, resizedGrid, h, w);
}

export function createMotionField(motionX, motionY, frameIds, dtype = 'float32', height = 512, width = 512) {
	const seq = frameIds.length;
	const data = new Float32Array(seq * 2 * height * width);
	const hw = height * width;
	for (let f = 0; f < seq; f++) {
		const base = f * 2 * hw;
		const frameId = frameIds[f];
		for (let idx = 0; idx < hw; idx++) {
			data[base + idx] = motionX * frameId;
			data[base + hw + idx] = motionY * frameId;
		}
	}
	return new Tensor(dtype, data, [seq, 2, height, width]);
}

export function createMotionFieldAndWarpLatents(motionX, motionY, frameIds, latents) {
	const dtype = latents.type || 'float32';
	const motionField = createMotionField(motionX, motionY, frameIds, dtype);
	const frames = frameIds.length;
	const warpedFrames = [];
	for (let i = 0; i < frames; i++) {
		const latentFrame = latents.slice([i, i + 1]);
		const flowFrame = motionField.slice([i, i + 1]);
		warpedFrames.push(warpSingleLatent(latentFrame, flowFrame));
	}
	return cat(warpedFrames);
}

function resizeFlowToShape(flowData, inH, inW, outH, outW) {
	const output = new Float32Array(outH * outW * 2);
	const scaleY = inH / outH;
	const scaleX = inW / outW;
	for (let y = 0; y < outH; y++) {
		const inY = (y + 0.5) * scaleY - 0.5;
		const y0 = clamp(Math.floor(inY), 0, inH - 1);
		const y1 = clamp(y0 + 1, 0, inH - 1);
		const ly = clamp(inY - y0, 0, 1);
		for (let x = 0; x < outW; x++) {
			const inX = (x + 0.5) * scaleX - 0.5;
			const x0 = clamp(Math.floor(inX), 0, inW - 1);
			const x1 = clamp(x0 + 1, 0, inW - 1);
			const lx = clamp(inX - x0, 0, 1);
			const idx = (y * outW + x) * 2;
			output[idx] = bilinearSample(flowData, 0, x0, y0, x1, y1, lx, ly, inH, inW);
			output[idx + 1] = bilinearSample(flowData, 1, x0, y0, x1, y1, lx, ly, inH, inW);
		}
	}
	return output;
}

function bilinearSample(data, channel, x0, y0, x1, y1, lx, ly, height, width) {
	const hw = height * width;
	const base = channel * hw;
	const v00 = data[base + y0 * width + x0];
	const v01 = data[base + y0 * width + x1];
	const v10 = data[base + y1 * width + x0];
	const v11 = data[base + y1 * width + x1];
	const top = v00 * (1 - lx) + v01 * lx;
	const bottom = v10 * (1 - lx) + v11 * lx;
	return top * (1 - ly) + bottom * ly;
}

function gridSampleNearest(latent, gridData, outH, outW) {
	const dims = getTensorDims(latent);
	const batch = dims[0] ?? 1;
	if (batch !== 1) {
		throw new Error('warpSingleLatent currently supports batch size 1');
	}
	const channels = dims[1];
	const h = dims[2];
	const w = dims[3];
	const latentData = ensureFloat32Array(latent.data);
	const outData = new Float32Array(batch * channels * outH * outW);
	for (let y = 0; y < outH; y++) {
		for (let x = 0; x < outW; x++) {
			const gridIdx = (y * outW + x) * 2;
			const gx = gridData[gridIdx];
			const gy = gridData[gridIdx + 1];
			const srcX = reflectCoordinate(Math.round(((gx + 1) / 2) * (w - 1)), w);
			const srcY = reflectCoordinate(Math.round(((gy + 1) / 2) * (h - 1)), h);
			for (let c = 0; c < channels; c++) {
				const srcIdx = c * h * w + srcY * w + srcX;
				const dstIdx = c * outH * outW + y * outW + x;
				outData[dstIdx] = latentData[srcIdx];
			}
		}
	}
	return new Tensor(latent.type || 'float32', outData, [batch, channels, outH, outW]);
}

function reflectCoordinate(coord, size) {
	if (size <= 1) {
		return 0;
	}
	const max = size - 1;
	let value = coord;
	while (value < 0 || value > max) {
		if (value < 0) {
			value = -value;
		} else if (value > max) {
			value = 2 * max - value;
		}
	}
	return clamp(value, 0, max);
}

function computeStageSlices(timesteps, t0, t1) {
	if (!timesteps.length) {
		return {
			stage1: [],
			stage2: [],
			stage3: [],
			t0Value: 0,
			t1Value: 0,
		};
	}
	const total = timesteps.length;
	// Python: timesteps[: -t1 - 1]  means [0, total - t1 - 1)
	const stage1End = clamp(total - t1 - 1, 0, total);
	// Python: timesteps[-t1 - 1 : -t0 - 1] means [total - t1 - 1, total - t0 - 1)
	const stage2End = clamp(total - t0 - 1, stage1End, total);
	
	const stage1 = timesteps.slice(0, stage1End);
	const stage2 = timesteps.slice(stage1End, stage2End);
	// Python: timesteps[-t1 - 1 :] means [total - t1 - 1, end)
	const stage3 = timesteps.slice(stage1End);
	
	// Get the actual timestep values at these positions
	const t1Index = clamp(total - t1 - 1, 0, total - 1);
	const t0Index = clamp(total - t0 - 1, 0, total - 1);
	
	log(`[computeStageSlices] total=${total}, t0=${t0}, t1=${t1}`);
	log(`[computeStageSlices] stage1End=${stage1End}, stage2End=${stage2End}`);
	log(`[computeStageSlices] t0Index=${t0Index}, t1Index=${t1Index}`);
	log(`[computeStageSlices] t0Value=${timesteps[t0Index]}, t1Value=${timesteps[t1Index]}`);
	
	return {
		stage1,
		stage2,
		stage3,
		t0Value: timesteps[t0Index] ?? timesteps[total - 1],
		t1Value: timesteps[t1Index] ?? timesteps[total - 1],
	};
}

function repeatPromptEmbeds(promptEmbeds, videoLength) {
	if (!promptEmbeds || videoLength <= 1) {
		return promptEmbeds;
	}
	const dims = getTensorDims(promptEmbeds);
	const batch = Math.max(dims[0] ?? 1, 1);
	const perSampleSize = Math.floor(promptEmbeds.data.length / batch);
	const ctor = promptEmbeds.data.constructor ?? Float32Array;
	const data = new ctor(promptEmbeds.data.length * videoLength);
	for (let b = 0; b < batch; b++) {
		const srcStart = b * perSampleSize;
		const srcEnd = srcStart + perSampleSize;
		const slice = promptEmbeds.data.slice(srcStart, srcEnd);
		for (let v = 0; v < videoLength; v++) {
			const dstIndex = (b * videoLength + v) * perSampleSize;
			data.set(slice, dstIndex);
		}
	}
	const newDims = [batch * videoLength, ...dims.slice(1)];
	return new Tensor(promptEmbeds.type || 'float32', data, newDims);
}

function repeatTensorAlongBatch(tensor, repeats) {
	if (!tensor || repeats <= 0) {
		return null;
	}
	const copies = [];
	for (let i = 0; i < repeats; i++) {
		copies.push(cloneTensor(tensor));
	}
	return cat(copies);
}

function repeatFrameIdsForBatch(frameIds, batchSize) {
	if (!Array.isArray(frameIds)) {
		return [];
	}
	if (batchSize <= 1) {
		return frameIds.slice();
	}
	const repeated = [];
	for (let i = 0; i < batchSize; i++) {
		repeated.push(...frameIds);
	}
	return repeated;
}

function cloneTensor(tensor) {
	if (!tensor) {
		return null;
	}
	if (typeof tensor.clone === 'function') {
		return tensor.clone();
	}
	const dims = getTensorDims(tensor);
	const type = tensor.type || 'float32';
	const src = tensor.data;
	if (!src || src.length === 0) {
		// Return null for empty tensors instead of creating invalid ones
		return null;
	}
	const ctor = src.constructor && typeof src.constructor.from === 'function'
		? src.constructor
		: Float32Array;
	const data = typeof src.slice === 'function'
		? src.slice()
		: ctor.from(src);
	return new Tensor(type, data, dims);
}

function logTensorShape(label, tensor) {
    if (!tensor) {
        console.log(`[shape] ${label}: <empty>`);
        return;
    }
    const dims = getTensorDims(tensor) || [];
    console.log(`[shape] ${label}: [${dims.join(', ')}]`);
}

function snapshotScheduler(scheduler) {
	if (!scheduler) {
		return null;
	}
	
	// Helper to safely clone tensors, returning null for uninitialized ones
	const safeClone = (t) => {
		if (!t || !t.data || t.data.length === 0) {
			return null;
		}
		return cloneTensor(t);
	};
	
	return {
		config: { ...scheduler.config },
		num_inference_steps: scheduler.num_inference_steps,
		skip_prk_steps: scheduler.skip_prk_steps,
		initNoiseSigma: scheduler.initNoiseSigma,
		timesteps: safeClone(scheduler.timesteps),
		prk_timesteps: safeClone(scheduler.prk_timesteps),
		plms_timesteps: safeClone(scheduler.plms_timesteps),
		ets: (scheduler.ets || []).map(safeClone).filter(Boolean),
		cur_model_output: safeClone(scheduler.cur_model_output),
		cur_sample: safeClone(scheduler.cur_sample),
		counter: scheduler.counter || 0,
	};
}

function restoreSchedulerSnapshot(snapshot, templateScheduler) {
	if (!snapshot && !templateScheduler) {
		return null;
	}
	const scheduler = new PNDMScheduler(snapshot?.config || templateScheduler?.config || {});
	const steps = snapshot?.num_inference_steps || templateScheduler?.num_inference_steps || scheduler.num_inference_steps;
	scheduler.skip_prk_steps = snapshot?.skip_prk_steps ?? scheduler.skip_prk_steps;
	scheduler.initNoiseSigma = snapshot?.initNoiseSigma ?? scheduler.initNoiseSigma;
	scheduler.setTimesteps(steps || 0);
	
	// Only restore non-null tensors
	if (snapshot?.timesteps) {
		scheduler.timesteps = cloneTensor(snapshot.timesteps) ?? scheduler.timesteps;
	}
	if (snapshot?.prk_timesteps) {
		scheduler.prk_timesteps = cloneTensor(snapshot.prk_timesteps) ?? scheduler.prk_timesteps;
	}
	if (snapshot?.plms_timesteps) {
		scheduler.plms_timesteps = cloneTensor(snapshot.plms_timesteps) ?? scheduler.plms_timesteps;
	}
	
	scheduler.ets = (snapshot?.ets || []).map(cloneTensor).filter(Boolean);
	scheduler.cur_model_output = snapshot?.cur_model_output ? cloneTensor(snapshot.cur_model_output) : null;
	scheduler.cur_sample = snapshot?.cur_sample ? cloneTensor(snapshot.cur_sample) : null;
	scheduler.counter = snapshot?.counter ?? 0;
	return scheduler;
}
