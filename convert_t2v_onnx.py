# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import shutil
from pathlib import Path

import onnx
import torch
from packaging import version
from torch.onnx import export

from diffusers import OnnxRuntimeModel, OnnxStableDiffusionPipeline, StableDiffusionPipeline, TextToVideoZeroPipeline


is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")

def rearrange_0(tensor, f):
    F, C, H, W = tensor.size()
    tensor = torch.permute(torch.reshape(tensor, (F // f, f, C, H, W)), (0, 2, 1, 3, 4))
    return tensor


def rearrange_1(tensor):
    B, C, F, H, W = tensor.size()
    return torch.reshape(torch.permute(tensor, (0, 2, 1, 3, 4)), (B * F, C, H, W))


def rearrange_3(tensor, f):
    F, D, C = tensor.size()
    return torch.reshape(tensor, (F // f, f, D, C))


def rearrange_4(tensor):
    B, F, D, C = tensor.size()
    return torch.reshape(tensor, (B * F, D, C))

class CrossFrameAttnProcessor:
    """
    Cross frame attention processor. Each frame attends the first frame.

    Args:
        batch_size: The number that represents actual batch size, other than the frames.
            For example, calling unet with a single prompt and num_images_per_prompt=1, batch_size should be equal to
            2, due to classifier-free guidance.
    """

    def __init__(self, batch_size=2):
        self.batch_size = batch_size

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Cross Frame Attention
        if not is_cross_attention:
            video_length = key.size(0) // self.batch_size

            B = key.size(0) // video_length
            D = key.size(1)
            C = key.size(2)

            # (B * F, D, C) -> (B, F, D, C)
            key = key.reshape(B, video_length, D, C)
            value = value.reshape(B, video_length, D, C)

            key_first = key[:, :1, :, :]
            value_first = value[:, :1, :, :]

            key = key_first.expand(B, video_length, D, C)
            value = value_first.expand(B, video_length, D, C)

            key = key.reshape(B * video_length, D, C)
            value = value.reshape(B * video_length, D, C)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class CrossFrameAttnProcessorv2:
    """
    Cross frame attention processor. Each frame attends the first frame.

    Args:
        batch_size: The number that represents actual batch size, other than the frames.
            For example, calling unet with a single prompt and num_images_per_prompt=1, batch_size should be equal to
            2, due to classifier-free guidance.
    """

    def __init__(self, batch_size=2):
        self.batch_size = batch_size

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Cross Frame Attention
        if not is_cross_attention:
            video_length = key.size()[0] // self.batch_size
            first_frame_index = [0] * video_length

            # rearrange keys to have batch and frames in the 1st and 2nd dims respectively
            key = rearrange_3(key, video_length)
            key = key[:, first_frame_index]
            # rearrange values to have batch and frames in the 1st and 2nd dims respectively
            value = rearrange_3(value, video_length)
            value = value[:, first_frame_index]

            # rearrange back to original shape
            key = rearrange_4(key)
            value = rearrange_4(value)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    
def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset,
    use_external_data_format=False,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # PyTorch deprecated the `enable_onnx_checker` and `use_external_data_format` arguments in v1.11,
    # so we check the torch version for backwards compatibility
    if is_torch_less_than_1_11:
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            use_external_data_format=use_external_data_format,
            enable_onnx_checker=True,
            opset_version=opset,
        )
    else:
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=opset,
        )


@torch.no_grad()
def convert_models(model_path: str, output_path: str, opset: int, fp16: bool = False):
    dtype = torch.float16 if fp16 else torch.float32
    if fp16 and torch.cuda.is_available():
        device = "cuda"
    elif fp16 and not torch.cuda.is_available():
        raise ValueError("`float16` model export is only supported on GPUs with CUDA")
    else:
        device = "cpu"
    # pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    pipeline = TextToVideoZeroPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    output_path = Path(output_path)

    # TEXT ENCODER
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    # onnx_export(
    #     pipeline.text_encoder,
    #     # casting to torch.int32 until the CLIP fix is released: https://github.com/huggingface/transformers/pull/18515/files
    #     model_args=(text_input.input_ids.to(device=device, dtype=torch.int32)),
    #     output_path=output_path / "text_encoder" / "model.onnx",
    #     ordered_input_names=["input_ids"],
    #     output_names=["last_hidden_state", "pooler_output"],
    #     dynamic_axes={
    #         "input_ids": {0: "batch", 1: "sequence"},
    #     },
    #     opset=opset,
    # )
    del pipeline.text_encoder

    # UNET
    # IMPORTANT: For Cross-Frame Attention to work correctly in ONNX:
    # - batch_size=2 means CFG (negative + positive)
    # - We trace with batch=4 (2 frames × 2 CFG) so video_length=2 is captured
    # - This allows the ONNX model to correctly identify frame0 and apply cross-frame attention
    #pipeline.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=1)) #wo_cfg
    pipeline.unet.set_attn_processor(CrossFrameAttnProcessor()) #cfg, batch_size=2
    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    unet_path = output_path / "unet" / "model.onnx"
    
    # Trace with batch=4 (2 frames × 2 CFG) so CrossFrameAttnProcessor sees video_length=2
    trace_batch = 4  # 2 frames × 2 CFG
    onnx_export(
        pipeline.unet,
        model_args=(
            torch.randn(trace_batch, unet_in_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
            torch.randn(trace_batch).to(device=device, dtype=dtype),
            torch.randn(trace_batch, num_tokens, text_hidden_size).to(device=device, dtype=dtype),
            False,
        ),
        output_path=unet_path,
        ordered_input_names=["sample", "timestep", "encoder_hidden_states", "return_dict"],
        output_names=["out_sample"],  # has to be different from "sample" for correct tracing
        # dynamic_axes={
        #     "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        #     "timestep": {0: "batch"},
        #     "encoder_hidden_states": {0: "batch", 1: "sequence"},
        # },
        dynamic_axes={
            "sample": {0: "batch", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
        },
        # dynamic_axes={
        #     "sample": {},
        #     "timestep": {},
        #     "encoder_hidden_states": {1: "sequence"},
        # },
        opset=opset,
        use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
    )
    unet_model_path = str(unet_path.absolute().as_posix())
    unet_dir = os.path.dirname(unet_model_path)
    unet = onnx.load(unet_model_path)
    # clean up existing tensor files
    shutil.rmtree(unet_dir)
    os.mkdir(unet_dir)
    # collate external tensor files into one
    onnx.save_model(
        unet,
        unet_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.pb",
        convert_attribute=False,
    )
    del pipeline.unet

    # VAE ENCODER
    vae_encoder = pipeline.vae
    vae_in_channels = vae_encoder.config.in_channels
    vae_sample_size = vae_encoder.config.sample_size
    # need to get the raw tensor output (sample) from the encoder
    vae_encoder.forward = lambda sample, return_dict: vae_encoder.encode(sample, return_dict)[0].sample()
    # onnx_export(
    #     vae_encoder,
    #     model_args=(
    #         torch.randn(1, vae_in_channels, vae_sample_size, vae_sample_size).to(device=device, dtype=dtype),
    #         False,
    #     ),
    #     output_path=output_path / "vae_encoder" / "model.onnx",
    #     ordered_input_names=["sample", "return_dict"],
    #     output_names=["latent_sample"],
    #     dynamic_axes={
    #         "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
    #     },
    #     opset=opset,
    # )

    # VAE DECODER
    vae_decoder = pipeline.vae
    vae_latent_channels = vae_decoder.config.latent_channels
    vae_out_channels = vae_decoder.config.out_channels
    # forward only through the decoder part
    vae_decoder.forward = vae_encoder.decode
    # onnx_export(
    #     vae_decoder,
    #     model_args=(
    #         torch.randn(1, vae_latent_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
    #         False,
    #     ),
    #     output_path=output_path / "vae_decoder" / "model.onnx",
    #     ordered_input_names=["latent_sample", "return_dict"],
    #     output_names=["sample"],
    #     dynamic_axes={
    #         "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
    #     },
    #     opset=opset,
    # )
    del pipeline.vae

    # SAFETY CHECKER
    if pipeline.safety_checker is not None:
        safety_checker = pipeline.safety_checker
        clip_num_channels = safety_checker.config.vision_config.num_channels
        clip_image_size = safety_checker.config.vision_config.image_size
        safety_checker.forward = safety_checker.forward_onnx
        onnx_export(
            pipeline.safety_checker,
            model_args=(
                torch.randn(
                    1,
                    clip_num_channels,
                    clip_image_size,
                    clip_image_size,
                ).to(device=device, dtype=dtype),
                torch.randn(1, vae_sample_size, vae_sample_size, vae_out_channels).to(device=device, dtype=dtype),
            ),
            output_path=output_path / "safety_checker" / "model.onnx",
            ordered_input_names=["clip_input", "images"],
            output_names=["out_images", "has_nsfw_concepts"],
            dynamic_axes={
                "clip_input": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "images": {0: "batch", 1: "height", 2: "width", 3: "channels"},
            },
            opset=opset,
        )
        del pipeline.safety_checker
        safety_checker = OnnxRuntimeModel.from_pretrained(output_path / "safety_checker")
        feature_extractor = pipeline.feature_extractor
    else:
        safety_checker = None
        feature_extractor = None

    onnx_pipeline = OnnxStableDiffusionPipeline(
        vae_encoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_encoder"),
        vae_decoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_decoder"),
        text_encoder=OnnxRuntimeModel.from_pretrained(output_path / "text_encoder"),
        tokenizer=pipeline.tokenizer,
        unet=OnnxRuntimeModel.from_pretrained(output_path / "unet"),
        scheduler=pipeline.scheduler,
        safety_checker=safety_checker,
        feature_extractor=feature_extractor,
        requires_safety_checker=safety_checker is not None,
    )

    onnx_pipeline.save_pretrained(output_path)
    print("ONNX pipeline saved to", output_path)

    del pipeline
    del onnx_pipeline
    _ = OnnxStableDiffusionPipeline.from_pretrained(output_path, provider="CPUExecutionProvider")
    print("ONNX pipeline is loadable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")

    parser.add_argument(
        "--opset",
        default=18,
        type=int,
        help="The version of the ONNX operator set to use.",
    )
    parser.add_argument("--fp16", action="store_true", default=False, help="Export the models in `float16` mode")

    args = parser.parse_args()

    convert_models(args.model_path, args.output_path, args.opset, args.fp16)