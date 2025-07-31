import json
import os

import numpy as np
from tqdm import tqdm

val_path = "CVPR-BiomedSegFM/3D_val_npz"
gt_path = "CVPR-BiomedSegFM/3D_val_gt/3D_val_gt_text"
output_path = "CVPR-BiomedSegFM"

val_list = []
for file in tqdm(os.scandir(val_path)):
    if file.name.endswith(".npz"):
        val_list.append(file.path)

gt_list = []
for file in tqdm(os.scandir(gt_path)):
    if file.name.endswith(".npz"):
        gt_list.append(file.path)

assert len(val_list) == len(
    gt_list
), "Mismatch in number of validation and ground truth files."

val_list = sorted(val_list)
gt_list = sorted(gt_list)

data_list = []
for val_file, gt_file in tqdm(zip(val_list, gt_list), total=len(val_list)):
    val_file_name = os.path.basename(val_file)
    gt_file_name = os.path.basename(gt_file)

    assert (
        val_file_name == gt_file_name
    ), f"File names do not match: {val_file_name} vs {gt_file_name}"

    val_data = np.load(val_file, allow_pickle=True)
    prompts = val_data["text_prompts"].item()
    gt_data = np.load(gt_file, allow_pickle=True)

    class_labels = np.unique(gt_data["gts"])

    if len(prompts) == 2:
        data_list.append(
            {
                "file_path": val_file,
                "gt_path": gt_file,
                "class_id": int(1),
                "text_prompt": prompts["1"],
                "only_one_prompt": True,
            }
        )
        continue

    for class_label in class_labels:
        if class_label == 0:
            continue

        if str(class_label) not in prompts:
            print(
                f"Warning: Class {class_label} not found in prompts. file: {val_file_name}"
            )
            continue

        if class_label not in gt_data["gts"]:
            print(
                f"Warning: Class {class_label} not found in gts. file: {val_file_name}"
            )
            continue

        data_list.append(
            {
                "file_path": val_file,
                "gt_path": gt_file,
                "class_id": int(class_label),
                "text_prompt": prompts[str(class_label)],
                "only_one_prompt": False,
            }
        )

data_list = sorted(data_list, key=lambda x: x["file_path"])

json_path = os.path.join(output_path, "val_data.json")
with open(json_path, "w") as f:
    json.dump(data_list, f, indent=4)
print(f"Saved val_data.json to {json_path}")
