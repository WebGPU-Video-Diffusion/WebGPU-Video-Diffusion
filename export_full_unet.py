import torch
from diffusers import MotionAdapter, AnimateDiffPipeline
import os

print("Attempting full UNet export with correct shapes...")

os.makedirs("onnx_models", exist_ok=True)

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float32
)

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
    
    print(f"Input shapes:")
    print(f"  sample: {sample.shape}")
    print(f"  timestep: {timestep.shape}")
    print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
    
    print("\nTesting forward pass...")
    try:
        output = unet(sample, timestep, encoder_hidden_states, return_dict=False)
        print(f"Forward pass successful!")
        print(f"  Output shape: {output[0].shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        exit(1)
    
    print("\nAttempting ONNX export...")
    try:
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
        print("\n ONNX export successful! ")
        print("Saved to: onnx_models/unet_video.onnx")
        
    except Exception as e:
        print(f"\n ONNX export failed: {e}")
        import traceback
        traceback.print_exc()