# Text3DSAM: Text-Guided 3D Medical Image Segmentation Using SAM-Inspired Architecture

<p align="center">
    <a href="https://openreview.net/forum?id=egbzGkOWVf&noteId=egbzGkOWVf"><b>Paper</b></a> |
    <a href="#datasets"><b>Datasets</b></a> |
    <a href="#model"><b>Model</b></a> |
    <a href="#training"><b>Training</b></a> |
    <a href="#inference"><b>Inference</b></a>
</p>

Official PyTorch implementation of: 

[Text3DSAM: Text-Guided 3D Medical Image Segmentation Using SAM-Inspired Architecture](https://openreview.net/forum?id=egbzGkOWVf&noteId=egbzGkOWVf)

Existing 3D medical image segmentation methods are often constrained by a fixed set of predefined classes or by reliance on manually defined prompts such as bounding boxes and scribbles, which are often labor-intensive and prone to ambiguity. To address these limitations, we present a framework for 3D medical image segmentation across diverse modalities guided solely by free-text descriptions of target anatomies or diseases. Our solution is built on a multi-component architecture that integrates efficient feature encoding via decomposed 3D convolutions and self-attention, multi-scale text-visual alignment, and a SAM-inspired mask decoder with iterative refinement. The model is further conditioned through a prompt encoder that transforms language and intermediate visual cues into spatially aligned embeddings. To train and evaluate our model, we used a large-scale dataset of over 200,000 3D image-mask pairs spanning CT, MRI, PET, ultrasound, and microscopy. Our method achieved an average Dice of 0.609 and F1 score of 0.113 on the open validation set, outperforming baselines such as CAT (Dice 0.532, F1 0.194) and SAT (Dice 0.557, F1 0.096). It showed strong generalization across modalities, with particularly high performance on ultrasound (Dice 0.829) and CT (Dice 0.672). These results confirm the feasibility of free-text-guided 3D segmentation and establish our approach as a strong foundation model for general-purpose medical image segmentation.

## Requirements
* Python==3.12.8

To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets

In the paper, we train and evaluate our model on [CVPR-BiomedSegFM](https://huggingface.co/datasets/junma/CVPR-BiomedSegFM) dataset.

## Training

To train the model(s) in the paper, run this command:

```bash
sh scripts/train.sh
```

## Inference
To run inference on a single 3D image, use the following command:

```bash
python src/pred.py \
    --model_path <path_to_trained_model> \
    --input_dir <path_to_input_3D_image> \
    --output_dir <path_to_output_directory> \
```

## Model
Our model weights are available at [Hugging Face](https://huggingface.co/MagicXin/Text3DSAM).