import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Splits image into patches and projects to embedding dim."""

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        raise NotImplementedError


class ContextEncoder(nn.Module):
    """Small ViT that encodes unmasked context patches."""

    def __init__(self, embed_dim, depth, num_heads):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, mask=None):
        raise NotImplementedError


class EMATargetEncoder(nn.Module):
    """Maintains an EMA copy of ContextEncoder. Receives no gradients."""

    def __init__(self, online_encoder: ContextEncoder, momentum=0.996):
        super().__init__()
        raise NotImplementedError

    @torch.no_grad()
    def update(self, online_encoder):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class JEPATokenizer(nn.Module):
    """Full JEPA tokenizer: patch embed + context encoder + EMA target encoder."""

    def __init__(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, mask=None):
        raise NotImplementedError

    def jepa_loss(self, predicted, target):
        raise NotImplementedError
