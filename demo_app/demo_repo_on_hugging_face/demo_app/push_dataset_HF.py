
import pickle
# from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import os

# 1) Load your pickle
with open("./demo_data/train_mini_data.pickle","rb") as f:
    train_dict = pickle.load(f)
with open("./demo_data/val_mini_data.pickle","rb") as f:
    val_dict = pickle.load(f)

# 2) Turn dicts→lists (each element must be a plain dict)
def to_records(d):
    return [
        {
          "scene_points_xyz": sample["scene_points_xyz"].tolist(),
          "bbox_list":          sample["bbox_list"],
          "ground_points":      sample["ground_points"].tolist(),
          "other_background_points": sample["other_background_points"].tolist(),
        }
        for sample in d
    ]

train_records = to_records(train_dict)
val_records   = to_records(val_dict)
print("creating and pushing dtaset...")

# 3) Create the Datasets & group them
# train_ds = Dataset.from_list(train_records)
# val_ds   = Dataset.from_list(val_records)
# ds = DatasetDict({"train": train_ds, "val": val_ds})

# ds.push_to_hub("Shing-Hei/Nuscenes-mini-subset")

from huggingface_hub import login
login()

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="demo_data",
    repo_id="Shing-Hei/Nuscenes-mini-subset",
    repo_type="dataset",
)