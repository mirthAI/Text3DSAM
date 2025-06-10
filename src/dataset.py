import json
import os
import random

import monai.transforms as mtf
import numpy as np
import torch
from torch.utils.data import Dataset


class BiomedCLIPDataset(Dataset):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        self.image_size = args.image_size
        self.max_length = args.max_length

        self.data_dir = args.data_dir
        self.prompt_dir = args.prompt_dir
        self.mapping_dir = args.mapping_dir

        self.data_info = self._load_and_filter_data_info(self.data_dir)

        with open(self.mapping_dir, "r") as f:
            self.mapping_data = json.load(f)

        self.data_list = []

        for modality, datasets in self.data_info.items():
            for dataset_name, dataset_details in datasets.items():
                instance_label = dataset_details["instance_label"]

                for file_info in dataset_details["files"]:
                    file_path = file_info["file_path"]
                    numeric_classes = file_info["numeric_class"]

                    if instance_label == 1:
                        self.data_list.append(
                            {
                                "modality": modality,
                                "dataset_name": dataset_name,
                                "file_path": file_path,
                                "class_id": 1,
                                "instance_label": 1,
                            }
                        )
                    else:
                        for k in numeric_classes:
                            if k == 0:
                                continue
                            self.data_list.append(
                                {
                                    "modality": modality,
                                    "dataset_name": dataset_name,
                                    "file_path": file_path,
                                    "class_id": k,
                                    "instance_label": 0,
                                }
                            )

        random.seed(42)
        random.shuffle(self.data_list)

        with open(self.prompt_dir, "r") as f:
            self.prompt_data = json.load(f)

        self.transform = mtf.Compose(
            [
                mtf.Resize(
                    spatial_size=self.image_size,
                    mode="trilinear",
                ),
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            ]
        )

    def _load_and_filter_data_info(self, data_dir):
        data_info_path = os.path.join(data_dir, "dataset_info.json")
        with open(data_info_path, "r") as f:
            data_info = json.load(f)

        filtered_data_info = {}
        for modality, datasets in data_info.items():
            if modality == "summary":
                continue

            filtered_datasets = {}
            for dataset_name, dataset_details in datasets.items():
                files = dataset_details.get("files", [])
                if not files:
                    continue

                valid_files = [
                    f
                    for f in files
                    if f.get("numeric_class") and len(f["numeric_class"]) > 0
                ]

                if valid_files:
                    filtered_datasets[dataset_name] = {
                        "files": valid_files,
                        "instance_label": dataset_details.get("instance_label"),
                    }

            if filtered_datasets:
                filtered_data_info[modality] = filtered_datasets

        return filtered_data_info

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        dataset_name = data["dataset_name"]
        image_path = data["file_path"]
        class_id = data["class_id"]
        instance_label = data["instance_label"]

        image_data = np.load(image_path, allow_pickle=True)

        image = torch.from_numpy(image_data["imgs"])
        label = torch.from_numpy(image_data["gts"])

        masked_image = self.generate_masked_image_from_box(
            image, label, instance_label, class_id
        )

        masked_image = self.transform(masked_image)

        dataset_prompts = self.prompt_data.get(dataset_name, {})
        class_prompts = dataset_prompts.get(str(class_id), [])

        prompt = random.choice(class_prompts)

        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = prompt_tokens["input_ids"][0]
        attention_mask = prompt_tokens["attention_mask"][0]

        unique_id = self.mapping_data.get(dataset_name, {}).get(str(class_id), None)
        if unique_id is None:
            raise ValueError(
                f"Unique ID not found for dataset {dataset_name} and class {class_id}."
            )
        unique_id = torch.tensor(unique_id, dtype=torch.long)

        result = {
            "image": masked_image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "unique_id": unique_id,
        }

        return result

    def generate_masked_image_from_box(self, image, label, instance_label, class_id):
        label = label.to(torch.int32)
        image = image.to(torch.float32)

        masked_image = torch.zeros_like(image, device=image.device, dtype=image.dtype)

        classes_to_process = []

        if instance_label == 1:
            unique_classes = torch.unique(label)
            classes_to_process = [
                cls_val.item() for cls_val in unique_classes if cls_val != 0
            ]
        elif instance_label == 0:
            if class_id is None:
                raise ValueError("class_id must be provided when instance_label is 0.")
            if class_id == 0:
                print(
                    f"Warning: class_id is 0 (background). No bounding box will be generated unless label contains 0 as a foreground class."
                )
                return masked_image.unsqueeze(0)
            classes_to_process.append(class_id)
        else:
            raise ValueError("instance_label must be 0 or 1.")

        if not classes_to_process:
            print("No classes to process.")
            return masked_image.unsqueeze(0)

        classes_tensor = torch.tensor(classes_to_process, device=label.device)
        mask = torch.isin(label, classes_tensor)

        masked_image[mask] = image[mask]

        return masked_image.unsqueeze(0)


class BiomedSegDataset(Dataset):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        self.image_size = args.image_size
        self.max_length = args.max_length

        self.data_dir = args.data_dir
        self.prompt_dir = args.prompt_dir
        self.data_info = self._load_and_filter_data_info(self.data_dir)

        self.data_list = []
        for modality, datasets in self.data_info.items():
            for dataset_name, dataset_details in datasets.items():
                instance_label = dataset_details["instance_label"]

                for file_info in dataset_details["files"]:
                    file_path = file_info["file_path"]
                    numeric_classes = file_info["numeric_class"]

                    if instance_label == 1:
                        self.data_list.append(
                            {
                                "modality": modality,
                                "dataset_name": dataset_name,
                                "file_path": file_path,
                                "class": 1,
                                "instance_label": 1,
                            }
                        )
                    else:
                        for k in numeric_classes:
                            if k == 0:
                                continue
                            self.data_list.append(
                                {
                                    "modality": modality,
                                    "dataset_name": dataset_name,
                                    "file_path": file_path,
                                    "class": k,
                                    "instance_label": 0,
                                }
                            )

        random.seed(42)
        random.shuffle(self.data_list)

        with open(self.prompt_dir, "r") as f:
            self.prompt_data = json.load(f)

        self.transform = mtf.Compose(
            [
                mtf.Resized(
                    keys=["image", "label"],
                    spatial_size=self.image_size,
                    mode=("trilinear", "nearest"),
                ),
                mtf.RandRotate90d(
                    keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)
                ),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float32),
                mtf.ToTensord(keys=["label"], dtype=torch.long),
            ]
        )

    def _load_and_filter_data_info(self, data_dir):
        data_info_path = os.path.join(data_dir, "dataset_info.json")
        with open(data_info_path, "r") as f:
            data_info = json.load(f)

        filtered_data_info = {}
        for modality, datasets in data_info.items():
            if modality == "summary":
                continue

            filtered_datasets = {}
            for dataset_name, dataset_details in datasets.items():
                files = dataset_details.get("files", [])
                if not files:
                    continue

                valid_files = [
                    f
                    for f in files
                    if f.get("numeric_class") and len(f["numeric_class"]) > 0
                ]

                if valid_files:
                    filtered_datasets[dataset_name] = {
                        "files": valid_files,
                        "instance_label": dataset_details.get("instance_label"),
                    }

            if filtered_datasets:
                filtered_data_info[modality] = filtered_datasets

        return filtered_data_info

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        dataset_name = data["dataset_name"]
        image_path = data["file_path"]
        class_id = data["class"]
        instance_label = data["instance_label"]

        image_data = np.load(image_path, allow_pickle=True)

        image = np.expand_dims(image_data["imgs"], axis=0)
        label = np.expand_dims(image_data["gts"], axis=0)

        if instance_label == 1:
            current_class_label = (label > 0).astype(label.dtype)
        else:
            current_class_label = (label == class_id).astype(label.dtype)

        transformed = self.transform({"image": image, "label": current_class_label})
        image = transformed["image"]
        label = transformed["label"]

        dataset_prompts = self.prompt_data.get(dataset_name, {})
        class_prompts = dataset_prompts.get(str(class_id), [])

        prompt = random.choice(class_prompts)

        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = prompt_tokens["input_ids"][0]
        attention_mask = prompt_tokens["attention_mask"][0]

        result = {
            "image": image,
            "label": label,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        return result


class BiomedValDataset(Dataset):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        self.image_size = args.image_size
        self.max_length = args.max_length

        self.val_data_path = args.val_data_path

        with open(self.val_data_path, "r") as f:
            val_data = json.load(f)

        random.seed(42)
        random.shuffle(val_data)

        self.data_list = []

        if args.max_samples > 0:
            val_data = val_data[: args.max_samples]
        else:
            val_data = val_data[:]

        for item in val_data:
            file_path = item["file_path"]
            gt_path = item["gt_path"]
            class_id = item["class_id"]
            text_prompt = item["text_prompt"]
            only_one_prompt = item["only_one_prompt"]

            modality = os.path.basename(file_path).split("_")[0]

            self.data_list.append(
                {
                    "file_path": file_path,
                    "gt_path": gt_path,
                    "class_id": class_id,
                    "text_prompt": text_prompt,
                    "only_one_prompt": only_one_prompt,
                    "modality": modality,
                }
            )

        self.transform = mtf.Compose(
            [
                mtf.Resized(
                    keys=["image"],
                    spatial_size=self.image_size,
                    mode=("trilinear"),
                ),
                mtf.ToTensord(keys=["image"], dtype=torch.float32),
                mtf.ToTensord(keys=["label"], dtype=torch.long),
            ]
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        file_path = data["file_path"]
        gt_path = data["gt_path"]
        class_id = data["class_id"]
        text_prompt = data["text_prompt"]
        only_one_prompt = data["only_one_prompt"]

        image_data = np.load(file_path, allow_pickle=True)
        label_data = np.load(gt_path, allow_pickle=True)

        origin_image = np.expand_dims(image_data["imgs"], axis=0)
        origin_label = np.expand_dims(label_data["gts"], axis=0)

        if only_one_prompt:
            current_class_label = (origin_label > 0).astype(origin_label.dtype)
        else:
            current_class_label = (origin_label == class_id).astype(origin_label.dtype)

        transformed = self.transform(
            {"image": origin_image, "label": current_class_label}
        )
        image = transformed["image"]
        label = transformed["label"]
        spacing = image_data["spacing"].tolist()

        prompt_tokens = self.tokenizer(
            text_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = prompt_tokens["input_ids"][0]
        attention_mask = prompt_tokens["attention_mask"][0]

        result = {
            "image": image,
            "label": label,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "file_name": os.path.basename(file_path),
            "class_id": class_id,
            "text_prompt": text_prompt,
            "modality": data["modality"],
            "spacing": spacing,
        }

        return result
