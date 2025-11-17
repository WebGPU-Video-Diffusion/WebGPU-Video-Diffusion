import onnxruntime as ort
import numpy as np
import cv2
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer
from diffusers import DDIMScheduler

print("=" * 60)
print("AnimateDiff ONNX Video Generation (CPU UNet)")
print("=" * 60)

models_dir = "onnx_models_0"

print("\n[1/5] Loading models...")


# text_encoder_providers = ['CPUExecutionProvider']
# unet_providers         = ['CPUExecutionProvider']  
# vae_providers          = ['CPUExecutionProvider']
text_encoder_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
#unet_providers         = [ 'CPUExecutionProvider']
vae_providers          = ['CUDAExecutionProvider', 'CPUExecutionProvider']
unet_providers = [
    (
        'CUDAExecutionProvider',
        {
            'device_id': 0,          
            'enable_cuda_graph': 0, 
            # 'gpu_mem_limit': 2 * 1024 * 1024 * 1024, 
        }
    ),
    'CPUExecutionProvider'
]
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

so.log_severity_level = 1

text_encoder_session = ort.InferenceSession(
    f"{models_dir}/text_encoder.onnx",
    providers=text_encoder_providers
)
# unet_session = ort.InferenceSession(
#     f"{models_dir}/unet_video.onnx",
#     providers=unet_providers
# )
unet_session = ort.InferenceSession(
    f"{models_dir}/unet_video.onnx",
    sess_options=so,
    providers=unet_providers
)
vae_decoder_session = ort.InferenceSession(
    f"{models_dir}/vae_decoder.onnx",
    providers=vae_providers
)
print("TextEncoder using:", text_encoder_session.get_providers())
print("UNet using       :", unet_session.get_providers())
print("VAE using        :", vae_decoder_session.get_providers())


print(" ONNX models loaded")


print("\n[2/5] Encoding prompt...")
prompt = (
    "a cute orange cat turning around its head in a garden, "
    "full body, side view, centered in frame, "
    "highly detailed, sharp, cinematic lighting"
)


print(f"  Prompt: {prompt!r}")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

tokens = tokenizer(
    prompt,
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="np"
)
text_embeddings_fp32 = text_encoder_session.run(
    None,
    {"input_ids": tokens["input_ids"].astype(np.int64)},
)[0]  # (1,77,768)


uncond_tokens = tokenizer(
    "",
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="np"
)
uncond_embeddings_fp32 = text_encoder_session.run(
    None,
    {"input_ids": uncond_tokens["input_ids"].astype(np.int64)},
)[0]  # (1,77,768)

text_embeddings = text_embeddings_fp32.astype(np.float32)
uncond_embeddings = uncond_embeddings_fp32.astype(np.float32)
text_embeddings_combined = np.concatenate(
    [uncond_embeddings, text_embeddings], axis=0
).astype(np.float32)  # (2,77,768)
print(f"  Base text embeddings: {text_embeddings_combined.shape}")


batch_size = 1
num_frames = 16
height = 32
width = 32
guidance_scale = 7.5

F = num_frames


encoder_hidden_states_np = np.repeat(
    text_embeddings_combined, F, axis=0
).astype(np.float32)  # (2F,77,768)

print(f"  Encoder hidden states for UNet: {encoder_hidden_states_np.shape}")


print(f"\n[3/5] Denoising ({num_frames} frames, {height*8}x{width*8})...")

scheduler = DDIMScheduler.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
scheduler.set_timesteps(25) 


latents_shape = (batch_size, 4, num_frames, height, width)
latents = torch.randn(latents_shape, dtype=torch.float32)
latents = latents * scheduler.init_noise_sigma

for i, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising")):

    latent_model_input = torch.cat([latents] * 2, dim=0)  # (2,4,F,H,W)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)


    sample_np = latent_model_input.cpu().numpy().astype(np.float32)


    timestep_np = np.array(t.item(), dtype=np.int64)  # scalar


    noise_pred_np = unet_session.run(
        None,
        {
            "sample": sample_np,                  # (2,4,F,H,W)
            "timestep": timestep_np,             # scalar int64
            "encoder_hidden_states": encoder_hidden_states_np,  # (2F,77,768)
        },
    )[0]  #  (2,4,F,H,W)

    noise_pred = torch.from_numpy(noise_pred_np).to(torch.float32)
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = scheduler.step(noise_pred, t, latents).prev_sample

print("Denoising complete")

print("\n[4/5] Decoding frames...")


latents = (1.0 / 0.18215) * latents  # (1,4,F,H,W)

frames = []
for f in tqdm(range(num_frames), desc="Decoding"):

    latent_frame = latents[0, :, f, :, :].unsqueeze(0).cpu().numpy().astype(np.float32)

    image = vae_decoder_session.run(
        None,
        {"latent": latent_frame},
    )[0]


    image = image[0].transpose(1, 2, 0)           # (H,W,3)
    image = np.clip(image / 2 + 0.5, 0, 1)
    image = (image * 255).astype(np.uint8)

    frames.append(image)

print(f"Decoded {len(frames)} frames")

print("\n[5/5] Saving video...")
output_path = "trygpu_fp32.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
h, w = frames[0].shape[:2]
out = cv2.VideoWriter(output_path, fourcc, 8.0, (w, h))

for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

out.release()

print(f" Saved: {output_path}")
print("=" * 60)
