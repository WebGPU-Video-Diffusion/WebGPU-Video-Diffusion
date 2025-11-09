# WebGPU-Accelerated Video Diffusion Model  
### CIS 5650: Final Project  
**Contributors:**  
[@Yuntian Ke](https://github.com/) · [@Ruichi Zhang](https://github.com/) · [@Muqiao Lei](https://github.com/) · [@Lobi Zhao](https://github.com/)

---

## 📘 Project Scope and Contributions
[![WebGPU](https://img.shields.io/badge/WebGPU-Enabled-blue)](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange)](https://onnxruntime.ai/)
[![Made with ❤️ at Penn](https://img.shields.io/badge/Made%20with%20❤️%20at-Penn-red)](#)
---

### **Baseline**
The existing **ONNX Runtime WebGPU demo** performs a standard text-to-image generation pipeline:

This baseline generates only a single static image. It demonstrates ONNX inference on WebGPU but **lacks temporal modeling**, multi-frame scheduling, or GPU-level postprocessing.

---

### **Our Work**
We extend this baseline into two progressively advanced pipelines for **video synthesis**:

#### **1. Text-to-Image-to-Video (Multi-frame Synthesis)**
We design a **GPU-accelerated multi-frame scheduler** that repeatedly invokes the ONNX UNet and VAE modules to produce a sequence of latent representations and decoded images.  

To ensure **temporal smoothness**:
- The latents are first denoised halfway.  
- A **Warping Shader** performs latent warping on the GPU using motion fields.  
- The warped latents are re-noised and then passed into the **cross-attention block** to generate smooth, temporally consistent video.

#### **2. Text-to-Video (Direct Video Diffusion)**
In parallel, we implement essential **3D operators** such as:
- `Conv3D`
- `GroupNorm3D`
- `Temporal Attention`

These are integrated within **ONNX Runtime WebGPU** using custom WGSL compute kernels.  
Our goal is to test whether a compact **Video Diffusion Model (VDM)** can be exported to ONNX format and executed directly in the browser.

This involves:
- Extending ONNX Runtime’s operator coverage for spatio-temporal tensor processing.  
- Enabling **end-to-end video generation** entirely on WebGPU.  
- Quantifying runtime behavior and performance feasibility for browser-based video diffusion.

This component provides a **research-style exploration** of ONNX Runtime’s video modeling capabilities and introduces substantial GPU programming challenges in shader design and kernel optimization.

---

### **Fallback Plan (if ONNX Video Model Loading Fails)**
If a full ONNX-based video diffusion model cannot be executed due to **operator or memory limitations**, we pivot to a broader **GPU operator and systems study** on top of the text-to-image baseline.

We will:
- Implement multiple **temporal fusion operators** in WGSL (e.g., latent-space and image-space smoothing, various temporal kernels) and compare visual and runtime behavior.
- Develop a small **3D-convolution / temporal enhancement module** purely in WebGPU to emulate temporal blocks in diffusion models.
- Perform a **systematic performance study** across:
  - Backends (WebGPU vs. WASM)
  - Frame counts
  - Resolutions  
  - Detailed profiling of UNet, VAE, and temporal shaders.

This fallback path still provides a **substantial GPU programming workload** involving:
- Custom compute kernels  
- Workgroup design  
- Buffer management  
- Performance analysis  

It remains consistent with our overall goal of exploring **temporal consistency and GPU acceleration** for video generation.

---

### **Added Components vs. Baseline**
- ✅ Multi-frame scheduling and buffer management for sequential UNet/VAE inference  
- ✅ Temporal Fusion WGSL compute shader (core GPU programming contribution)  
- ✅ WebCodecs-based video assembly pipeline (no server or ffmpeg required)  
- ✅ Optional ONNX Video Diffusion prototype with 3D operator evaluation  
- ✅ WebGPU vs. WASM performance profiling and ablation analysis  

---

### **Expected Workload**
**4-person team**, ~12–20 hours per week each:
| Role | Responsibilities |
|------|------------------|
| Member 1 | ONNX Runtime integration and multi-frame scheduling |
| Member 2–3 | WGSL temporal fusion shader design, optimization, and profiling |
| Member 4 | Video assembly (WebCodecs) and benchmarking visualization |

---

### **Summary**
This project transforms a **static text-to-image demo** into a **fully GPU-accelerated text-to-video synthesis pipeline**. It integrates:
- Real shader programming  
- GPU resource management  
- System-level performance analysis  

making it an ambitious, research-oriented, and technically rich **final project** for CIS 5650.
