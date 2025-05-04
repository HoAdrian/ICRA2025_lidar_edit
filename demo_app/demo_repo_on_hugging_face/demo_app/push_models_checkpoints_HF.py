
from huggingface_hub import HfApi
import os

from huggingface_hub import login
login()


api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="demo_models",
    repo_id="Shing-Hei/Inpainting_model_LiDAR_EDIT",
    repo_type="model",
)
