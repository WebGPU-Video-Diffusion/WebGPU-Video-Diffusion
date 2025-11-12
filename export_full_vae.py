import torch
from diffusers import AutoencoderKL
import os

print("Exporting FULL VAE decode pipeline...")

os.makedirs("onnx_models", exist_ok=True)

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

dummy_latent = torch.randn(1, 4, 64, 64)

print("Testing forward pass...")
with torch.no_grad():
    output_pt = vae_wrapper(dummy_latent)
    print(f"  Output shape: {output_pt.shape}")
    print(f"  Output stats: mean={output_pt.mean():.4f}, std={output_pt.std():.4f}")

print("\nExporting to ONNX...")
with torch.no_grad():
    torch.onnx.export(
        vae_wrapper,
        dummy_latent,
        "onnx_models/vae_decode_full.onnx",
        opset_version=14,
        input_names=['latent'],
        output_names=['image'],
        dynamic_axes={
            'latent': {0: 'batch'},
            'image': {0: 'batch'}
        },
        do_constant_folding=True
    )

print("✓ Exported to vae_decode_full.onnx")

print("\nVerifying ONNX model...")
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("onnx_models/vae_decode_full.onnx")
output_onnx = session.run(None, {'latent': dummy_latent.numpy().astype(np.float32)})[0]

print(f"  ONNX output shape: {output_onnx.shape}")
print(f"  ONNX output stats: mean={output_onnx.mean():.4f}, std={output_onnx.std():.4f}")

diff = np.abs(output_pt.numpy() - output_onnx)
print(f"\nDifference: mean={diff.mean():.6f}, max={diff.max():.6f}")

if diff.max() < 0.01:
    print("\n✓✓✓ FULL VAE export successful!")
    print("\nNow replace the old VAE:")
    print("  move onnx_models\\vae_decoder.onnx onnx_models\\vae_decoder_incomplete.onnx")
    print("  move onnx_models\\vae_decode_full.onnx onnx_models\\vae_decoder.onnx")
else:
    print("\n✗ Still has issues")