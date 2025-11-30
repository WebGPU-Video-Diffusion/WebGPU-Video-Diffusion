#
# just helper file
# download sd1.5 fp32 locally
#


from huggingface_hub import hf_hub_download, list_repo_files
import os

repo_id = "ykeee/StableDiffusion1.5-fp32"
local_dir = "sd1.5/t2vzero-fp32"

print("Checking repository files...")
try:
    all_files = list_repo_files(repo_id)
    unet_files = [f for f in all_files if f.startswith('unet/')]
    print("\nAvailable UNet files:")
    for f in unet_files:
        print(f"  - {f}")
    print()
except Exception as e:
    print(f"Could not list files: {e}\n")

files = [
    "unet/config.json",
    "unet/model.onnx",
    "unet/model.onnx_data"
]

print(f"Downloading FP32 model from {repo_id}...")
print(f"Target directory: {local_dir}\n")

for file in files:
    print(f"Downloading {file}...")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=file,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"[OK] {file} downloaded\n")
    except Exception as e:
        print(f"[FAIL] Failed to download {file}: {e}\n")

print("Download complete!")
print("\nFile structure:")
print("sd1.5/")
print("└── t2vzero-fp32/")
print("    └── unet/")
print("        ├── config.json")
print("        ├── model.onnx")
print("        └── weights.pb")
