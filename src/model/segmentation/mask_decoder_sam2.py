# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from .module import MLP, LayerNorm3d


class MaskDecoder_SAM2(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        feat_shape: Tuple[int, int, int],
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        use_high_res_features: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.feat_shape = feat_shape
        self.num_multimask_outputs = num_multimask_outputs

        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.Upsample(
                size=(
                    feat_shape[0] * 4,
                    feat_shape[1] * 4,
                    feat_shape[2] * 4,
                ),
                mode="trilinear",
                align_corners=False,
            ),
            nn.Conv3d(
                transformer_dim,
                transformer_dim // 4,
                kernel_size=1,
            ),
            LayerNorm3d(transformer_dim // 4),
            activation(),
            nn.Upsample(
                size=(
                    feat_shape[0] * 16,
                    feat_shape[1] * 16,
                    feat_shape[2] * 16,
                ),
                mode="trilinear",
                align_corners=False,
            ),
            nn.Conv3d(
                transformer_dim // 4,
                transformer_dim // 16,
                kernel_size=1,
            ),
            LayerNorm3d(transformer_dim // 16),
            activation(),
        )

        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.feat_conv0 = nn.Conv3d(
                transformer_dim // 16,
                transformer_dim // 16,
                kernel_size=1,
            )
            self.feat_conv1 = nn.Conv3d(
                transformer_dim // 4,
                transformer_dim // 4,
                kernel_size=1,
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim // 4, transformer_dim // 16, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.txt_align_upscaled_embedding = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 4),
            nn.LayerNorm(transformer_dim // 4),
            activation(),
            nn.Linear(transformer_dim // 4, transformer_dim // 16),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          text_embedding (torch.Tensor): text embeddings for the desired segmentation
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          high_res_features (Optional[List[torch.Tensor]]): optional high resolution features

        Returns:
          torch.Tensor: batched predicted masks
        """
        masks, mask_tokens_out = self.predict_masks(
            image_embeddings=image_embeddings,
            text_embedding=text_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            high_res_features=high_res_features,
        )

        masks = masks[:, 0:1, :, :]
        sam_tokens_out = mask_tokens_out[:, 0:1]

        return masks, sam_tokens_out

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        text_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Predicts masks. See 'forward' for more details."""
        # Prepare tokens
        output_tokens = self.mask_tokens.weight
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Process the image data
        src = image_embeddings
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = image_pe.expand(src.shape[0], -1, -1, -1, -1)
        b, c, h, w, d = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs[:, : self.num_mask_tokens, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w, d)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            up1, conv1, ln1, act1, up2, conv2, ln2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features

            upscaled_embedding = conv1(up1(src))
            upscaled_embedding = act1(
                ln1(upscaled_embedding + self.feat_conv1(feat_s1))
            )
            upscaled_embedding = conv2(up2(upscaled_embedding))
            upscaled_embedding = act2(
                ln2(upscaled_embedding) + self.feat_conv0(feat_s0)
            )

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w, d = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w * d)).view(
            b, -1, h, w, d
        )

        # Add text embedding influence
        text_embedding_down = self.txt_align_upscaled_embedding(
            text_embedding
        ).unsqueeze(dim=1)
        upscaled_embedding = upscaled_embedding.view(b, c, h * w * d)
        sim = (text_embedding_down @ upscaled_embedding).view(b, -1, h, w, d)
        sim = sim.repeat(1, masks.shape[1], 1, 1, 1)

        masks = masks + sim

        return masks, mask_tokens_out
