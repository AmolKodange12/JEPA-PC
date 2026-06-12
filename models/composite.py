import torch
import torch.nn as nn

from models.jepa_tokenizer import JEPATokenizer
from models.pc_layer import PCLayer


class PCNetwork(nn.Module):
    """Stack of PCLayers forming the inference hierarchy."""

    def __init__(self, layer_dims: list[int], lr_inference=0.01):
        super().__init__()
        raise NotImplementedError

    def init_states(self, batch_size, device):
        raise NotImplementedError

    def infer(self, x, n_steps: int):
        """Run inference loop for n_steps, keeping weights fixed."""
        raise NotImplementedError

    def update_weights(self, lr_weights=0.001):
        """Update weights after inference has converged."""
        raise NotImplementedError


class JEPAPCComposite(nn.Module):
    """
    Full composite model:
      frozen JEPA tokenizer → PC inference hierarchy in token space.
    """

    def __init__(
        self,
        tokenizer: JEPATokenizer,
        pc_layer_dims: list[int],
        lr_inference=0.01,
    ):
        super().__init__()
        raise NotImplementedError

    @torch.no_grad()
    def tokenize(self, x):
        """Forward pass through frozen tokenizer. No gradients flow."""
        raise NotImplementedError

    def forward(self, x, n_inference_steps: int = 20):
        raise NotImplementedError

    def precision_weights(self):
        """Return per-layer precision (inverse prediction-error variance)."""
        raise NotImplementedError
