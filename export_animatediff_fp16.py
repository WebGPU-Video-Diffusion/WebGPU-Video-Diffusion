import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, AutoencoderKL
import os

print("=" * 60)
print("AnimateDiff ONNX Export (using real UNet shapes)")
print("=" * 60)

os.makedirs("onnx_models_combined", exist_ok=True)


print("\n[Step 1/4] Loading AnimateDiff pipeline...")
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2"
)
pipe = AnimateDiffPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    motion_adapter=adapter,
    torch_dtype=torch.float32,
)

pipe = pipe.to("cpu")
print(" Pipeline loaded")


text_encoder = pipe.text_encoder
text_encoder.eval()
with torch.no_grad():
    dummy_ids = torch.randint(0, 1000, (1, 77))
    torch.onnx.export(
        text_encoder,
        dummy_ids,
        "onnx_models_combined/text_encoder.onnx",
        opset_version=17,
        input_names=["input_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "batch"},
                      "last_hidden_state": {0: "batch"}},
        do_constant_folding=False,
    )
print(" Text Encoder exported")


print("\n[Step 3/4] Exporting UNetMotionModel (video)...")

unet = pipe.unet
unet.eval()
unet = unet.half()  


batch_images = 1       
batch_cfg    = 2      
num_frames   = 8
channels     = 4
height       = 64
width        = 64


with torch.no_grad():
    sample = torch.randn(
        batch_cfg,          # 2 = uncond + cond
        channels,           # 4
        num_frames,         # F
        height,
        width,
        dtype=torch.float16,
    )


    timestep = torch.tensor(999, dtype=torch.int64)

    encoder_hidden_states = torch.randn(
        batch_cfg * num_frames,  # 2 * F
        77,
        768,
        dtype=torch.float16,
    )

    print("  Dummy UNet inputs:")
    print("    sample:", sample.shape, sample.dtype)
    print("    timestep:", timestep.shape, timestep.dtype)
    print("    encoder_hidden_states:", encoder_hidden_states.shape, encoder_hidden_states.dtype)


    class UNetONNXWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states):
            out = self.unet(
                sample,
                timestep,
                encoder_hidden_states,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )
            return out[0]  # noise_pred: (2, 4, F, H, W)

    unet_wrapped = UNetONNXWrapper(unet).eval()

    print("  Testing forward...")
    out = unet_wrapped(sample, timestep, encoder_hidden_states)
    print("   Forward OK, output shape:", out.shape)

    print("  Exporting to ONNX...")
    torch.onnx.export(
        unet_wrapped,
        (sample, timestep, encoder_hidden_states),
        "onnx_models_combined/unet_video.onnx",
        opset_version=17,
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["noise_pred"],
        dynamic_axes={
            # sample: (batch_cfg, 4, F, H, W)
            "sample": {
                0: "batch_cfg",
                2: "num_frames",
                3: "height",
                4: "width",
            },
            # encoder_hidden_states: (batch_cfg * F, 77, 768)
            "encoder_hidden_states": {0: "batch_frames"},
            # noise_pred: (batch_cfg, 4, F, H, W)
            "noise_pred": {
                0: "batch_cfg",
                2: "num_frames",
                3: "height",
                4: "width",
            },

        },
        do_constant_folding=False,
    )

print(" UNet exported to onnx_models_combined/unet_video.onnx")


print("\n[Step 4/4] Exporting VAE Decoder...")
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

vae_wrapper = VAEDecodeWrapper(vae).eval()
with torch.no_grad():
    dummy_latent = torch.randn(1, 4, 64, 64, dtype=torch.float32)
    out_pt = vae_wrapper(dummy_latent)
    print("  VAE PyTorch output:", out_pt.shape)

    torch.onnx.export(
        vae_wrapper,
        dummy_latent,
        "onnx_models_combined/vae_decoder.onnx",
        opset_version=17,
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes={
            "latent": {0: "batch", 2: "height", 3: "width"},
            "image": {0: "batch", 2: "out_height", 3: "out_width"},
        },
        do_constant_folding=False,
    )

print("\nExport done.")
