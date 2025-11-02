# WebGPU-Accelerated Video Diffusion Model  
*CIS 5650: Final Project*
### Contributors:
@Yuntian Ke @Ruichi Zhang @Muqiao Lei @Lobi Zhao

[![WebGPU](https://img.shields.io/badge/WebGPU-Enabled-blue)](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange)](https://onnxruntime.ai/)
[![Made with ❤️ at Penn](https://img.shields.io/badge/Made%20with%20❤️%20at-Penn-red)](#)

---

## Overview
This project implements a **Video Diffusion Model (VDM)** for **temporally consistent video generation**, powered by **ONNX Runtime WebGPU**.

Unlike existing Stable Diffusion demos that generate static images, our model brings **AI-powered video synthesis directly into the browser** through GPU-accelerated inference.

We explore:
- Native **ONNX 3D operators** (`Conv3D`, `GroupNorm3D`)
- Custom **WGSL compute shaders**
  
Our goal: generate **short, coherent video clips (8–16 frames)** directly on the web.

---

## Motivation
Recent diffusion models — such as **VideoCrafter2**, **Open-Sora**, and **AnimateDiff** — produce impressive results but require:
- Heavy GPU resources  
- Proprietary CUDA runtimes  

We aim to explore whether **WebGPU** and **ONNX Runtime** can deliver **lightweight, open, and portable** video generation directly in the browser — making diffusion-based video **accessible on everyday devices**.

---

## Project Goals
- ✅ Build a **browser-based video generation model** powered by WebGPU  
- ⚙️ Use **ONNX Runtime WebGPU** for inference (UNet + VAE + Text Encoder)  
- 🎬 Generate **8–16 frame videos** from text prompts  
- 📊 Benchmark **WebGPU vs. CPU/WASM** performance  

---

## Milestone Plan

| Milestone | Task | Description |
|------------|------|--------------|
| **1** | Image Generation | Run **Stable Diffusion Turbo (image)** with ONNX Runtime WebGPU and verify results |
| **2** | Temporal Extension | Extend to **multi-frame video generation** using temporal layers or GPU shaders |
| **3** | Benchmarking | Export short video sequences (8–16 frames) and compare **WebGPU vs. CPU** performance |
| **Final Demo** | Interactive Browser Demo | Web-based **text-to-video generation** using WebGPU |

---

## References & Inspiration
- [Microsoft ONNX Runtime WebGPU — SD-Turbo Demo](https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu)  
- [Stable Diffusion Turbo — ONNXRuntime/sd-turbo](https://huggingface.co/onnxruntime/sd-turbo)  
- [Web Stable Diffusion — Stable Diffusion Model on WebGPU](https://websd.mlc.ai)

---

## Third-Party Code
We build upon:
- **ONNX Runtime WebGPU inference** from Microsoft’s SD-Turbo example  
- Temporal modeling modules from open-source video diffusion frameworks  


