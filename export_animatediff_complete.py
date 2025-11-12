import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, AutoencoderKL
import os

print("="*60)
print("AnimateDiff Complete ONNX Export Pipeline")
print("="*60)

os.makedirs("onnx_models", exist_ok=True)

print("\n[Step 1/4] Downloading AnimateDiff models...")
print("This will download from HuggingFace (~4-5GB)")

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float32
)
print("✓ Models downloaded successfully!")

print("\n[Step 2/4] Exporting Text Encoder...")
text_encoder = pipe.text_encoder
text_encoder.eval()

with torch.no_grad():
    dummy_input_ids = torch.randint(0, 1000, (1, 77))
    
    torch.onnx.export(
        text_encoder,
        dummy_input_ids,
        "onnx_models/text_encoder.onnx",
        opset_version=14,
        input_names=['input_ids'],
        output_names=['last_hidden_state'],
        dynamic_axes={'input_ids': {0: 'batch'}},
        do_constant_folding=True
    )
print("✓ Text Encoder exported")

print("\n[Step 3/4] Exporting Video UNet...")
unet = pipe.unet
unet.eval()

batch_size = 2
num_frames = 8
channels = 4
height = 64
width = 64

with torch.no_grad():
    sample = torch.randn(batch_size, channels, num_frames, height, width)
    timestep = torch.tensor([999])
    encoder_hidden_states = torch.randn(batch_size * num_frames, 77, 768)
    
    print(f"  Input shapes:")
    print(f"    sample: {sample.shape}")
    print(f"    timestep: {timestep.shape}")
    print(f"    encoder_hidden_states: {encoder_hidden_states.shape}")
    
    print("  Testing forward pass...")
    output = unet(sample, timestep, encoder_hidden_states, return_dict=False)
    print(f"  ✓ Forward pass successful, output shape: {output[0].shape}")
    
    print("  Exporting to ONNX...")
    torch.onnx.export(
        unet,
        (sample, timestep, encoder_hidden_states),
        "onnx_models/unet_video.onnx",
        opset_version=17,
        input_names=['sample', 'timestep', 'encoder_hidden_states'],
        output_names=['noise_pred'],
        dynamic_axes={
            'sample': {0: 'batch', 2: 'num_frames'},
            'encoder_hidden_states': {0: 'batch_frames'},
            'noise_pred': {0: 'batch', 2: 'num_frames'}
        },
        do_constant_folding=True
    )
print("✓ Video UNet exported")

print("\n[Step 4/4] Exporting Full VAE Decoder...")
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")

class VAEDecodeWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder
    
    def forward(self, latent):
        latent = self.post_quant_conv(latent)
        image = self.decoder(latent)
        return image

vae_wrapper = VAEDecodeWrapper(vae)
vae_wrapper.eval()

with torch.no_grad():
    dummy_latent = torch.randn(1, 4, 64, 64)
    
    print("  Testing VAE forward pass...")
    output_pt = vae_wrapper(dummy_latent)
    print(f"  ✓ Output shape: {output_pt.shape}")
    
    print("  Exporting to ONNX...")
    torch.onnx.export(
        vae_wrapper,
        dummy_latent,
        "onnx_models/vae_decoder.onnx",
        opset_version=14,
        input_names=['latent'],
        output_names=['image'],
        dynamic_axes={
            'latent': {0: 'batch'},
            'image': {0: 'batch'}
        },
        do_constant_folding=True
    )
    
    print("  Verifying ONNX export...")
    import onnxruntime as ort
    import numpy as np
    
    session = ort.InferenceSession("onnx_models/vae_decoder.onnx")
    output_onnx = session.run(None, {'latent': dummy_latent.numpy().astype(np.float32)})[0]
    
    diff = np.abs(output_pt.numpy() - output_onnx)
    print(f"  Verification: max diff = {diff.max():.6f}")
    
    if diff.max() < 0.01:
        print("  ✓ VAE export verified!")
    else:
        print(f"  ⚠ Warning: VAE has differences (max={diff.max():.6f})")

print("\n" + "="*60)
print("Export Complete!")
print("="*60)
print("\nExported files:")
print("  ✓ onnx_models/text_encoder.onnx")
print("  ✓ onnx_models/unet_video.onnx (+ external weight files)")
print("  ✓ onnx_models/vae_decoder.onnx")
print("="*60)