import torch
import torch.nn as nn
from typing import Optional
from .patch_embedding import PatchEmbed
from .transformer_block import TransformerBlock


class IJEPATargetEncoder(nn.Module):
    """
    Standard ViT without classification head.
    Processes full image and outputs patch-level representations.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or nn.LayerNorm

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim, eps=1e-6)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = True,
        patch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          x: input images (B, C, H, W)
          return_all_tokens: If true, return all patch tokens
          patch_indices: If provided, return only specific patch indices

        Returns:
          Patch representations (B, N, D) or (B, len(indices), D)
        """
        # Patch embedding
        x = self.patch_embed(x)   # (B, N, D)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Return specific patches if indices provided
        if patch_indices is not None:
            x = x[:, patch_indices, :]

        return x

    def get_layer_representations(
        self,
        x: torch.Tensor,
        strategy: str = "last",
        specific_indices: Optional[list[int]] = None,
        patch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract semantic representations using different layer strategies.

        Args:
          x: Input images (B, C, H, W)
          strategy: Strategy to extract representations
            - 'last': final layer only (baseline)
            - 'second-last': 2nd-to-last block output
            - 'last_four_concat': concat of last 4 layer (B, N, 4*D)
            - 'specific': layers at specific_indices (eg: [25,27,29,31])
          specific_indices: Block indices to use when strategy='specific'
          patch_indices: If provided, return only these patch posistions

        Returns:
          (B, N, D) for "last"/"second_last", (B, N, 4*D) for concat strategies
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        n_blocks = len(self.blocks)

        # Determnine which block indices to capture
        if strategy == "second_last":
            capture_at = {n_blocks - 2}
        elif strategy == "last_four_concat":
            capture_at = set(range(n_blocks - 4, n_blocks))
        elif strategy == "specific":
            assert specific_indices is not None, "Provide specific_indices when strategy='specific'"
            capture_at = set(specific_indices)
        else:
            capture_at = {n_blocks - 1}

        captured = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in capture_at:
                captured[i] = x.clone()

        # Apply norm and pool
        if strategy in ("last_four_concat", "specific"):
            # Sort by layer order, normalize each, then concat along D
            layers = [self.norm(captured[i]) for i in sorted(captured)]
            out = torch.cat(layers, dim=-1)   # (B, N, num_layers * D)
        elif strategy == "second_last":
            out = self.norm(captured[n_blocks - 2])
        else:
            out = self.norm(x)    # x is already the last block output

        if patch_indices is not None:
            out = out[:, patch_indices, :]

        return out
