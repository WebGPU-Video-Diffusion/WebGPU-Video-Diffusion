#  WebGPU Video Diffusion

### Text-to-Video Generation Powered by WebGPU

<p align="center">
  <img src="images/skiing.gif" alt="Demo: Skiing Animation" width="512">
  <br>
  <em>Example output: "A person skiing down a snowy mountain"</em>
</p>

[![WebGPU](https://img.shields.io/badge/WebGPU-Enabled-blue)](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange)](https://onnxruntime.ai/)
[![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-1.5-purple)](https://huggingface.co/runwayml/stable-diffusion-v1-5)
[![Made with  at Penn](https://img.shields.io/badge/Made%20with%20%20at-Penn-red)](#)

**CIS 5650: GPU Programming Final Project**

**Contributors:**  
[@Yuntian Ke](https://github.com/kytttt)  [@Ruichi Zhang](https://github.com/Pabloo0610)  [@Muqiao Lei](https://github.com/rmurdock41)  [@Lobi Zhao](https://github.com/lobizhao)

---

##  Live Demo

 **[Try it now!](https://webgpu-video-diffusion.github.io/WebGPU-Video-Diffusion/)**

>  Requires a browser with WebGPU support (Chrome 113+, Edge 113+)

---

##  Overview

This project implements **Text-to-Video Zero** - a zero-shot text-to-video generation pipeline that runs entirely in the browser using WebGPU. Generate smooth video animations from text prompts without any server-side computation!

### Key Features

-  **Pure Browser Execution** - No server required, everything runs on your GPU via WebGPU
-  **8-Frame Video Generation** - Creates smooth, temporally consistent video sequences
-  **GPU Accelerated** - Leverages ONNX Runtime WebGPU for fast inference
-  **Cross-Frame Attention** - Uses modified UNet with cross-frame attention for temporal consistency
-  **Zero-Shot** - No video training data required, uses pre-trained Stable Diffusion 1.5

---

##  Technical Architecture

### Pipeline Overview

```
Text Prompt  Text Encoder  Cross-Frame UNet  VAE Decoder  Video Frames
                    
              CLIP Tokenizer
```

### Text-to-Video Zero Algorithm

Our implementation follows the [Text-to-Video Zero](https://arxiv.org/abs/2303.13439) paper:

1. **First Frame Generation (Backward Loop 0->T1)**
   - Start with random noise
   - Denoise using standard diffusion to timestep T1

2. **First Frame Refinement (Backward Loop T1->T0)**
   - Continue denoising to timestep T0 for the anchor frame

3. **Motion Warping**
   - Apply motion field to create latents for frames 2-N
   - Uses configurable motion strength (default: 12px in both X and Y)

4. **Forward Process (T0->T1)**
   - Add noise back to warped latents to reach T1

5. **Final Denoising (T1->0)**
   - Denoise all frames together using **Cross-Frame Attention**
   - Frame 0 serves as anchor for attention in all other frames

### Cross-Frame Attention

The UNet is modified with `CrossFrameAttnProcessor` that:
- Processes frames in pairs: (anchor, frame_k)
- Uses attention keys/values from the anchor frame
- Ensures temporal consistency across all generated frames

---

##  Getting Started

### Prerequisites

- Node.js 18+
- Browser with WebGPU support (Chrome 113+, Edge 113+)
- GPU with WebGPU capabilities

### Installation

```bash
# Clone the repository
git clone https://github.com/WebGPU-Video-Diffusion/WebGPU-Video-Diffusion.git
cd WebGPU-Video-Diffusion

# Install dependencies
npm install

# Build the project
npm run build

# Start local server
npm run dev
```

### Usage

1. Open `http://localhost:8081` in your browser
2. Enter a text prompt (e.g., "a cat walking on grass")
3. Click "Generate" and wait for the video to be created
4. Download the result as GIF or video

---

##  Configuration

You can customize the generation by modifying URL parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `ykeee/StableDiffusion1.5-fp32` | Hugging Face model path |
| `provider` | `webgpu` | Execution provider |
| `local` | `0` | Use local models (1) or remote (0) |


### Video Parameters (in code)

```javascript
video_length: 8,           // Number of frames
t0: 44,                    // Timestep T0 index
t1: 47,                    // Timestep T1 index
motion_field_strength_x: 12,  // Horizontal motion (pixels)
motion_field_strength_y: 12,  // Vertical motion (pixels)
num_inference_steps: 50,      // Total diffusion steps
```

---

##  Performance

| Component | Time (approx.) |
|-----------|---------------|
| Model Loading | 1-2min (first time, cached after) |
| Text Encoding | ~100ms |
| Video Generation (8 frames) | 8-10 minutes |
| VAE Decoding | ~500ms per frame |

*Tested on NVIDIA RTX 5070Ti, performance varies by GPU*

---


##  Technical Details

### Models Used

- **Text Encoder**: CLIP ViT-L/14 (from Stable Diffusion 1.5)
- **UNet**: Modified SD 1.5 UNet with CrossFrameAttnProcessor
- **VAE Decoder**: SD 1.5 VAE

### Key Algorithms

- **PNDM Scheduler**: Pseudo Numerical methods for Diffusion Models
- **Motion Field Warping**: Grid-sample based latent warping with reflection padding
- **Classifier-Free Guidance**: Scale = 7.5 for enhanced prompt adherence

---

##  References

- [Text-to-Video Zero Paper](https://arxiv.org/abs/2303.13439) - Levon Khachatryan et al.
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - CompVis
- [ONNX Runtime Web](https://onnxruntime.ai/) - Microsoft
- [WebGPU Specification](https://www.w3.org/TR/webgpu/) - W3C
- [Diffuser.js](https://github.com/dakenf/diffusers.js/) - Arthur Islamov


---

##  License
This project is for educational purposes as part of CIS 5650 GPU Programming course at University of Pennsylvania.

---

##  Acknowledgments

- Prof. Patrick Cozzi and the CIS 5650 teaching team
- The Hugging Face community for model hosting
- Microsoft for ONNX Runtime WebGPU support
