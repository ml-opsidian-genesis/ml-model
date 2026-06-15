import os
import sys
from huggingface_hub import HfApi

version_tag = sys.argv[1] 
repo_id = "teamfalsepositives/flood-risk-model"

api = HfApi()
print(f"Uploading model version {version_tag} to Hugging Face...")

api.upload_file(
    path_or_fileobj="models/flood_model.onnx",
    path_in_repo="flood_model.onnx",
    repo_id=repo_id,
    revision="main"
)

api.upload_file(
    path_or_fileobj="models/flood_model.pkl",
    path_in_repo="flood_model.pkl",
    repo_id=repo_id,
    revision="main"
)

api.create_tag(repo_id=repo_id, tag=version_tag)
print("Upload complete!")
