import argparse
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm


def process_npz(path, prompt_valid_keys, instance_label):
    if instance_label == 1:
        numeric_keys = [1]
        return numeric_keys

    npz = np.load(path, allow_pickle=True)
    gts = npz["gts"]
    gts_flat = gts.flatten()
    numeric_keys = np.argwhere(np.bincount(gts_flat) > 0).flatten()
    numeric_keys = np.array([k for k in numeric_keys if str(k) in prompt_valid_keys])

    return numeric_keys.tolist()


def generate_train_path_json(data_dir, output_path, class_info_path, max_workers):
    data_dict = {}

    try:
        with open(class_info_path, "r") as f:
            class_info_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Class info file not found at {class_info_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {class_info_path}")
        return

    modalities = []
    for f in os.scandir(data_dir):
        if f.is_dir():
            modality_name = f.name
            for ff in os.scandir(f.path):
                if ff.is_dir():
                    modalities.append((ff.path, ff.name, modality_name))

    for modality_path, dataset_name, modality_name in tqdm(
        modalities, desc="Processing directories"
    ):
        npz_files = [
            file for file in os.scandir(modality_path) if file.name.endswith(".npz")
        ]
        file_paths = [os.path.join(modality_path, file.name) for file in npz_files]
        file_paths.sort(key=lambda x: x.lower())

        try:
            dataset_class_info = class_info_data.get(dataset_name)
            if dataset_class_info is None:
                print(f"Error: No class info found for dataset {dataset_name}")
                continue

            instance_label = dataset_class_info.get("instance_label", None)
            if instance_label is None:
                print(f"Error: No instance label found for dataset {dataset_name}")
                continue

            prompt_valid_keys = [k for k in dataset_class_info.keys() if k.isdigit()]

            files = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        process_npz, path, prompt_valid_keys, instance_label
                    ): path
                    for path in file_paths
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Processing {dataset_name}",
                ):
                    path = futures[future]
                    try:
                        numeric_keys = future.result()

                        if len(numeric_keys) == 0:
                            print(f"Warning: No valid keys found in {path}")
                            continue

                        files.append({"file_path": path, "numeric_class": numeric_keys})
                    except Exception as e:
                        print(f"Error processing file {path}: {e}")

            if modality_name not in data_dict:
                data_dict[modality_name] = {}

            if dataset_name not in data_dict[modality_name]:
                data_dict[modality_name][dataset_name] = {
                    "files": [],
                    "instance_label": None,
                }

            dataset_entry = data_dict[modality_name][dataset_name]
            dataset_entry["files"].extend(files)
            dataset_entry["instance_label"] = instance_label

        except Exception as e:
            print(
                f"Error processing dataset {dataset_name} in modality {modality_name}, {e}"
            )
            continue

    sorted_data_dict = OrderedDict()
    for modality_name in sorted(data_dict.keys(), key=str.lower):
        sorted_datasets = OrderedDict()
        for dataset_name in sorted(data_dict[modality_name].keys(), key=str.lower):
            dataset_data = data_dict[modality_name][dataset_name]

            dataset_data["files"].sort(key=lambda x: x["file_path"].lower())

            sorted_datasets[dataset_name] = {
                "files": dataset_data["files"],
                "instance_label": dataset_data["instance_label"],
            }
        sorted_data_dict[modality_name] = sorted_datasets

    total_files = sum(
        len(dataset_data["files"])
        for modality in data_dict.values()
        for dataset_data in modality.values()
    )
    total_datasets = sum(len(modality) for modality in data_dict.values())
    total_modalities = len(data_dict)

    output_dict = OrderedDict()
    output_dict["summary"] = {
        "total_files": total_files,
        "total_datasets": total_datasets,
        "total_modalities": total_modalities,
    }
    for modality, datasets in sorted_data_dict.items():
        output_dict[modality] = datasets

    output_json_path = os.path.join(output_path, "dataset_info.json")
    with open(output_json_path, "w") as json_file:
        json.dump(output_dict, json_file, indent=4)
    print(f"JSON file saved to {output_json_path}")
    print(
        f"Total {total_files} files found across {total_datasets} datasets in {total_modalities} modalities in {data_dir}"
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate train path JSON")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
    )
    parser.add_argument(
        "--class_info_path",
        type=str,
        default="CVPR-BiomedSegFM/CVPR25_TextSegFMData_with_class.json",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    output_path = args.data_dir if args.output_path is None else args.output_path
    class_info_path = args.class_info_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    generate_train_path_json(data_dir, output_path, class_info_path, args.max_workers)
