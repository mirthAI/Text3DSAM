from typing import Any, Optional, Tuple, Type

import torch
import torch.nn as nn

from .dcformer import DecompConv3D


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int, int],
        input_image_size: Tuple[int, int, int],
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.mask_downscaling = nn.Sequential(
            DecompConv3D(1, embed_dim // 8, kernel_size=7, stride=4),
            activation(),
            DecompConv3D(embed_dim // 8, embed_dim // 4, kernel_size=5, stride=2),
            activation(),
            DecompConv3D(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2),
            activation(),
            DecompConv3D(embed_dim // 2, embed_dim, kernel_size=3),
        )

        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(
        self,
        text_embedding: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        sam_tokens_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = text_embedding.shape[0]
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim),
            device=text_embedding.device,
            dtype=text_embedding.dtype,
        )

        sparse_embeddings = torch.cat(
            [sparse_embeddings, text_embedding.unsqueeze(dim=1)], dim=1
        )

        if sam_tokens_out is not None:
            sparse_embeddings = torch.cat([sparse_embeddings, sam_tokens_out], dim=1)

        if masks is not None:
            dense_embeddings = self.mask_downscaling(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1, 1).expand(
                bs,
                -1,
                int(self.image_embedding_size[0]),
                int(self.image_embedding_size[1]),
                int(self.image_embedding_size[2]),
            )
        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w, d = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        dtype = self.positional_encoding_gaussian_matrix.dtype
        grid = torch.ones((h, w, d), device=device, dtype=dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        z_embed = grid.cumsum(dim=2) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        z_embed = z_embed / d

        pe = self._pe_encoding(torch.stack([x_embed, y_embed, z_embed], dim=-1))
        return pe.permute(3, 0, 1, 2)  # C x H x W x D

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        coords[:, :, 2] = coords[:, :, 2] / image_size[2]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
