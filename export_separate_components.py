import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
import os

print("="*60)
print("Exporting AnimateDiff Components Separately")
print("="*60)

os.makedirs("onnx_models", exist_ok=True)

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float32
)

print("\n[1/4] Exporting Text Encoder...")
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

print("\n[2/4] Exporting VAE Decoder...")
vae_decoder = pipe.vae.decoder
vae_decoder.eval()

with torch.no_grad():
    dummy_latent_dec = torch.randn(1, 4, 64, 64)
    
    torch.onnx.export(
        vae_decoder,
        dummy_latent_dec,
        "onnx_models/vae_decoder.onnx",
        opset_version=14,
        input_names=['latent'],
        output_names=['image'],
        do_constant_folding=True
    )
print("✓ VAE Decoder exported")

print("\n[3/4] Saving Motion Module weights...")
motion_modules = {}
for name, module in pipe.unet.named_modules():
    if 'motion_modules' in name:
        motion_modules[name] = module

torch.save(motion_modules, "onnx_models/motion_modules.pt")
print(f"✓ Saved {len(motion_modules)} motion module components")

print("\n[4/4] Attempting to export base UNet (without motion)...")

original_unet = pipe.unet

def remove_motion_modules(unet):
    for name, module in unet.named_children():
        if 'motion_modules' in name:
            continue
        if hasattr(module, 'motion_modules'):
            delattr(module, 'motion_modules')
        if len(list(module.children())) > 0:
            remove_motion_modules(module)

try:
    with torch.no_grad():
        batch_size = 2
        dummy_latent = torch.randn(batch_size, 4, 64, 64)
        dummy_timestep = torch.tensor([999])
        dummy_encoder_hidden = torch.randn(batch_size, 77, 768)
        
        test_output = original_unet(
            dummy_latent.unsqueeze(2),
            dummy_timestep,
            dummy_encoder_hidden,
            return_dict=False
        )
        
        print(f" UNet forward pass works, output shape: {test_output[0].shape}")
        
except Exception as e:
    print(f" Full UNet export skipped due to temporal operations")


print("\n" + "="*60)
print("Export Summary:")
print("="*60)
print(" text_encoder.onnx - ONNX format")
print(" vae_decoder.onnx - ONNX format")
print(" motion_modules.pt - PyTorch weights")
