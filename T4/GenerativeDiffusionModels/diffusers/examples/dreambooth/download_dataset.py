from huggingface_hub import snapshot_download

# Choose a local directory name
local_dir = "./yarn-art"

# Download the dataset
snapshot_download(
    "Norod78/Yarn-art-style",
    local_dir=local_dir,
    repo_type="dataset",
    ignore_patterns=".gitattributes",
)

print(f"Dataset downloaded to {local_dir}")
