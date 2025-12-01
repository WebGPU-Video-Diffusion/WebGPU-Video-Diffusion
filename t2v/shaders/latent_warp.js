// Latent Warping Shader for temporal consistency
export class LatentWarper {
    constructor(device) {
        this.device = device;
        this.pipeline = null;
        this.initPipeline();
    }

    initPipeline() {
        const shaderCode = `
            struct Uniforms {
                width: u32,
                height: u32,
                channels: u32,
                alpha: f32,
            }

            @group(0) @binding(0) var<storage, read> prev_latent: array<f32>;
            @group(0) @binding(1) var<storage, read> curr_latent: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;
            @group(0) @binding(3) var<uniform> uniforms: Uniforms;

            @compute @workgroup_size(8, 8, 1)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let x = global_id.x;
                let y = global_id.y;
                
                if (x >= uniforms.width || y >= uniforms.height) {
                    return;
                }

                for (var c = 0u; c < uniforms.channels; c++) {
                    let idx = c * uniforms.width * uniforms.height + y * uniforms.width + x;
                    
                    // Simple motion-aware blending
                    let prev_val = prev_latent[idx];
                    let curr_val = curr_latent[idx];
                    
                    // Compute local gradient for motion estimation
                    var motion_weight = uniforms.alpha;
                    if (x > 0u && x < uniforms.width - 1u && y > 0u && y < uniforms.height - 1u) {
                        let grad_x = abs(prev_latent[idx + 1] - prev_latent[idx - 1]);
                        let grad_y = abs(prev_latent[idx + uniforms.width] - prev_latent[idx - uniforms.width]);
                        let motion = sqrt(grad_x * grad_x + grad_y * grad_y);
                        motion_weight = uniforms.alpha * (1.0 - min(motion * 0.5, 0.5));
                    }
                    
                    output[idx] = curr_val * (1.0 - motion_weight) + prev_val * motion_weight;
                }
            }
        `;

        this.shaderModule = this.device.createShaderModule({ code: shaderCode });
        
        this.bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout]
        });

        this.pipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: this.shaderModule, entryPoint: 'main' }
        });
    }

    async warp(sourceLatent, dx, dy) {
        const [batch, channels, height, width] = [1, 4, 64, 64];
        const size = batch * channels * height * width;

        const sourceData = sourceLatent instanceof Float32Array ? sourceLatent : 
                          (sourceLatent.data || Float32Array.from(sourceLatent));

        const outputData = new Float32Array(size);
        
        for (let c = 0; c < channels; c++) {
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const src_x = x - dx;
                    const src_y = y - dy;
                    
                    // Wrap coordinates
                    const x0 = Math.floor(src_x);
                    const x1 = x0 + 1;
                    const y0 = Math.floor(src_y);
                    const y1 = y0 + 1;
                    
                    const wx = src_x - x0;
                    const wy = src_y - y0;
                    
                    let value = 0;
                    
                    const getIdx = (xx, yy) => {
                        const wrapped_x = ((xx % width) + width) % width;
                        const wrapped_y = ((yy % height) + height) % height;
                        return c * height * width + wrapped_y * width + wrapped_x;
                    };
                    
                    value += sourceData[getIdx(x0, y0)] * (1 - wx) * (1 - wy);
                    value += sourceData[getIdx(x1, y0)] * wx * (1 - wy);
                    value += sourceData[getIdx(x0, y1)] * (1 - wx) * wy;
                    value += sourceData[getIdx(x1, y1)] * wx * wy;
                    
                    const out_idx = c * height * width + y * width + x;
                    outputData[out_idx] = value;
                }
            }
        }
        
        return outputData;
    }
}
