import os
from dataclasses import dataclass, field
from typing import List

import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer, Trainer

from src.dataset import BiomedSegDataset
from src.model.modeling import Med3DSeg, Med3DSegConfig


def is_rank_zero():
    if "RANK" in os.environ:
        if int(os.environ["RANK"]) != 0:
            return False
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return False
    return True


def rank0_print(*args):
    if is_rank_zero():
        print(*args)


@dataclass
class ModelArguments:
    wb_project: str = "Text3DSAM"

    text_model: str = "bert-base-uncased"

    image_size: List[int] = field(default_factory=lambda: [128, 256, 256])
    embed_dim: int = 768
    patch_size: List[int] = field(default_factory=lambda: [64, 64, 64])
    pass_num: int = 1
    transformer_depth: int = 2
    mlp_dim: int = 2048
    num_heads: int = 8

    focal_weight: float = 1.0
    dice_weight: float = 1.0


@dataclass
class DataArguments:
    data_dir: str = "CVPR-BiomedSegFM/3D_train_npz_all"
    prompt_dir: str = "CVPR-BiomedSegFM/CVPR25_TextSegFMData_with_class.json"
    max_length: int = 512


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    seed: int = 42
    ddp_backend: str = "nccl"
    ddp_timeout: int = 128000

    label_names: List[str] = field(default_factory=lambda: ["labels"])

    bf16: bool = True
    output_dir: str = "./output/Text3DSAM"
    num_train_epochs: float = 30
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    eval_strategy: str = "no"
    eval_accumulation_steps: int = 1
    # eval_steps: float = 0.1
    save_strategy: str = "steps"
    save_steps: int = 0.05
    save_total_limit: int = 2
    logging_steps: float = 0.001

    optim: str = "adamw_torch"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    gradient_checkpointing: bool = False
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 8

    report_to: str = "wandb"
    run_name: str = "Training"


@dataclass
class DataCollator:
    def __init__(self):
        return

    def __call__(self, batch: List[dict]) -> dict:
        images = torch.stack([item["image"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])

        return {
            "image": images,
            "label": labels,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.environ["WANDB_PROJECT"] = model_args.wb_project

    data_args.image_size = model_args.image_size

    if model_args.pretrained_tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_tokenizer)
        except Exception as e:
            rank0_print(
                f"Failed to load pretrained tokenizer from {model_args.pretrained_tokenizer}: {e}"
            )
            rank0_print("Using default tokenizer from transformers library instead.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.text_model)

    train_dataset = BiomedSegDataset(data_args, tokenizer)
    data_collator = DataCollator()

    config = Med3DSegConfig.from_dict(vars(model_args))
    model = Med3DSeg(config)

    model.initialize_weights_for_training()

    rank0_print(f"Model parameters: {model.num_parameters() / 1e6:.2f}M")
    rank0_print(
        f"Text encoder parameters number: {sum(p.numel() for p in model.text_encoder.parameters()) / 1e6:.2f}M"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    resume_checkpoint = None
    if os.path.exists(training_args.output_dir):
        checkpoints = [
            os.path.join(training_args.output_dir, d)
            for d in os.listdir(training_args.output_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(training_args.output_dir, d))
        ]

        if checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
            resume_checkpoint = checkpoints[-1]
            rank0_print(f"Resuming from checkpoint: {resume_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
