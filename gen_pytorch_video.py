import os
import numpy as np
import torch
import imageio.v2 as imageio

from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


motion_id = "guoyww/animatediff-motion-adapter-v1-5-2"
base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"

print("motion adapter:", motion_id)
print("base model    :", base_model_id)


adapter = MotionAdapter.from_pretrained(
    motion_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

pipe = AnimateDiffPipeline.from_pretrained(
    base_model_id,
    motion_adapter=adapter,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)


scheduler = DDIMScheduler.from_pretrained(
    base_model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler

pipe = pipe.to(device)
pipe.enable_vae_slicing()

print("pipeline & scheduler loaded.")
print("UNet class:", pipe.unet.__class__)


prompt = (
    "a cute orange cat turning around its head in a garden, "
    "full body, side view, centered in frame, "
    "highly detailed, sharp, cinematic lighting"
)
negative_prompt = "bad quality, worse quality, low quality"

num_frames = 16
height = 512
width = 512
steps = 25
guidance = 7.5
fps = 8

generator = torch.Generator(device=device).manual_seed(0)

if device == "cuda":
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
else:
    from contextlib import nullcontext
    ctx = nullcontext()

with ctx:
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=generator,
    )

frames = out.frames[0]
print("frames:", len(frames))

os.makedirs("debug_official", exist_ok=True)
for i, f in enumerate(frames):
    f.save(os.path.join("debug_official", f"frame_{i:03d}.png"))

os.makedirs("results", exist_ok=True)
video_path = os.path.join("results", "animatediff_official_cat.mp4")
writer = imageio.get_writer(video_path, fps=fps)
for f in frames:
    arr = np.array(f.convert("RGB"))
    writer.append_data(arr)
writer.close()
print("video saved to:", video_path)
