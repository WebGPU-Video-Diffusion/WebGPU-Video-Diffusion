import onnxruntime as ort
import numpy as np
import cv2
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer
from diffusers import DDIMScheduler

print("="*60)
print("Fixed Video Generation")
print("="*60)

models_dir = "onnx_models"

print("\n[1/5] Loading models...")
providers = ['CPUExecutionProvider']
text_encoder_session = ort.InferenceSession(f"{models_dir}/text_encoder.onnx", providers=providers)
unet_session = ort.InferenceSession(f"{models_dir}/unet_video.onnx", providers=providers)
vae_decoder_session = ort.InferenceSession(f"{models_dir}/vae_decoder.onnx", providers=providers)
print("✓ Models loaded")

print("\n[2/5] Encoding prompt...")
prompt = "a cat walking in the garden"
print(f"  Prompt: '{prompt}'")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

tokens = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="np")
text_embeddings = text_encoder_session.run(None, {'input_ids': tokens['input_ids'].astype(np.int64)})[0]

uncond_tokens = tokenizer("", padding="max_length", max_length=77, truncation=True, return_tensors="np")
uncond_embeddings = text_encoder_session.run(None, {'input_ids': uncond_tokens['input_ids'].astype(np.int64)})[0]

text_embeddings_combined = np.concatenate([uncond_embeddings, text_embeddings], axis=0)
print(f"✓ Text embeddings: {text_embeddings_combined.shape}")

batch_size = 1
num_frames = 8
height = 64
width = 64
guidance_scale = 7.5

print(f"\n[3/5] Denoising ({num_frames} frames, {height*8}x{width*8})...")

scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
scheduler.set_timesteps(25)

latents_shape = (batch_size, 4, num_frames, height, width)
latents = torch.randn(latents_shape, dtype=torch.float32)
latents = latents * scheduler.init_noise_sigma

for i, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising")):
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    
    text_embeddings_expanded = np.repeat(text_embeddings_combined, num_frames, axis=0).astype(np.float32)
    
    timestep = np.array([t.item()], dtype=np.int64)
    
    noise_pred = unet_session.run(None, {
        'sample': latent_model_input.numpy().astype(np.float32),
        'timestep': timestep,
        'encoder_hidden_states': text_embeddings_expanded
    })[0]
    
    noise_pred = torch.from_numpy(noise_pred)
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    latents = scheduler.step(noise_pred, t, latents).prev_sample

print("✓ Denoising complete")

print("\n[4/5] Decoding frames...")
latents = 1 / 0.18215 * latents

frames = []
for frame_idx in tqdm(range(num_frames), desc="Decoding"):
    latent_frame = latents[0, :, frame_idx, :, :].unsqueeze(0).numpy().astype(np.float32)
    
    image = vae_decoder_session.run(None, {'latent': latent_frame})[0]
    image = image[0].transpose(1, 2, 0)
    image = np.clip(image / 2 + 0.5, 0, 1)
    image = (image * 255).astype(np.uint8)
    
    frames.append(image)

print(f"✓ Decoded {len(frames)} frames")

print("\n[5/5] Saving video...")
output_path = "output_fixed.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
h, w = frames[0].shape[:2]
out = cv2.VideoWriter(output_path, fourcc, 8.0, (w, h))

for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

out.release()
print(f"✓ Saved: {output_path}")
print("="*60)