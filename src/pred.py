import argparse
import os

import monai.transforms as mtf
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from model.modeling import Med3DSeg


def pred(args):
    model = Med3DSeg.from_pretrained(args.model_path).to(device)
    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    model.eval()

    npz_files = [f for f in os.listdir(args.input_dir) if f.endswith(".npz")]
    if not npz_files:
        raise FileNotFoundError("No .npz files found in the input directory.")
    image_path = os.path.join(args.input_dir, npz_files[0])

    npz = np.load(image_path, allow_pickle=True)
    image_np = npz["imgs"].astype(np.float32)
    text_prompts = npz["text_prompts"].item()
    instance_label = npz["text_prompts"].item()["instance_label"]

    origin_shape = image_np.shape

    pre_transforms = mtf.Compose(
        [
            mtf.EnsureChannelFirst(channel_dim="no_channel"),
            mtf.Resize(
                spatial_size=model.config.image_size,
                mode="trilinear",
            ),
            mtf.ToTensor(dtype=torch.float32),
        ]
    )

    post_transforms = mtf.Compose(
        [
            mtf.EnsureType(dtype=torch.float32),
            mtf.EnsureChannelFirst(channel_dim="no_channel"),
        ]
    )

    preprocessed_image_cpu = pre_transforms(image_np)
    preprocessed_image = preprocessed_image_cpu.unsqueeze(0).to(device)

    segs_probs = []
    class_ids_in_order = []

    with torch.no_grad():
        for k, v in text_prompts.items():
            if k == "instance_label":
                continue

            class_id = int(k)
            class_ids_in_order.append(class_id)
            prompt_text = v

            prompt_tokens = tokenizer(
                prompt_text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            input_ids = prompt_tokens["input_ids"]
            attention_mask = prompt_tokens["attention_mask"]

            logits = model.generate(
                image=preprocessed_image,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = F.interpolate(
                logits,
                size=origin_shape,
                mode="trilinear",
            )
            pred_prob = torch.sigmoid(logits)

            if instance_label == 0:
                segs_probs.append(pred_prob.squeeze(0).squeeze(0).detach())
            else:
                segs_probs = (pred_prob > 0.5).float().squeeze(0).squeeze(0).detach()
                break

    if instance_label == 0:
        stacked_probs_fg = torch.stack(segs_probs, dim=0)

        max_fg_prob, _ = torch.max(stacked_probs_fg, dim=0)
        prob_background = torch.clamp(1.0 - max_fg_prob, min=0.0, max=1.0)
        prob_background = prob_background.unsqueeze(0)

        stacked_probs_all = torch.cat([prob_background, stacked_probs_fg], dim=0)

        pred_indices = torch.argmax(stacked_probs_all, dim=0)

        pred_labels = torch.zeros_like(pred_indices, dtype=torch.long)
        for index_in_stack, class_id_val in enumerate(class_ids_in_order):
            argmax_index = index_in_stack + 1
            pred_labels[pred_indices == argmax_index] = class_id_val

        final_labels_tensor = post_transforms(pred_labels.cpu()).squeeze(0)
        final_labels_np = final_labels_tensor.numpy().astype(np.uint8)
    else:
        final_labels_tensor = post_transforms(segs_probs.cpu()).squeeze(0)
        final_labels_np = final_labels_tensor.numpy().astype(np.uint8)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    base_filename = os.path.basename(image_path)
    output_filename = os.path.join(args.output_dir, base_filename)

    np.savez_compressed(output_filename, segs=final_labels_np)
    print(f"Saved predictions to {output_filename}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="Med3DSeg",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/eval/preds",
    )
    args = parser.parse_args()

    pred(args)
