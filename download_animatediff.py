import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
import os

print("=" * 60)
print("AnimateDiff ONNX download")
print("=" * 60)

os.makedirs("onnx_models", exist_ok=True)

print("\n Downloading AnimateDiff model...")
print("This will download from HuggingFace (~4-5GB)")

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float32
)
print("Model downloaded successfully!")
