import torch
import torch.nn as nn


class PCLayer(nn.Module):
    """Single PC layer.

    W  : top-down prediction weights (in_features, out_features) — frozen during inference
    r  : current representation (B, out_features) — updated every inference step
    e  : prediction error (B, in_features) — propagated upward

    INFERENCE (weights frozen, n_steps iterations):
        prediction = tanh(W @ r)
        e          = r_below - prediction
        r         += lr_infer * ( W.T @ (e * tanh') - e_above )

    LEARNING (only after inference converges):
        dW = e.T @ r / B                        # local Hebbian
        W += lr_weights * dW
    """

    def __init__(self, in_features: int, out_features: int, lr_inference: float = 0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_inference = lr_inference
        # Top-down: r (B, out_features) → prediction (B, in_features)
        # weight.shape = (in_features, out_features)  [PyTorch: Linear(A,B).weight is (B,A)]
        self.W = nn.Linear(out_features, in_features, bias=False)
        # Mutable inference state — set per-batch by PCNetwork.init_states
        self.r: torch.Tensor = None
        self.e: torch.Tensor = None
        self.prediction: torch.Tensor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bottom-up init of r.  x: (B, in_features) → r_init: (B, out_features).
        tanh bounds each layer to [-1,1] so representations don't explode through the chain."""
        return torch.tanh(x @ self.W.weight)  # (B, in) @ (in, out) = (B, out)

    def predict(self) -> torch.Tensor:
        return torch.tanh(self.W(self.r))  # (B, in_features)

    def compute_error(self, r_below: torch.Tensor) -> torch.Tensor:
        self.prediction = self.predict()
        self.e = r_below - self.prediction
        return self.e

    def update_r(self, e_above: torch.Tensor = None) -> None:
        """One inference step. e_above is the raw error from the layer directly above."""
        dtanh = 1.0 - self.prediction ** 2               # Jacobian of tanh, (B, in_features)
        bottom_up = (self.e * dtanh) @ self.W.weight      # (B, out_features)
        delta = bottom_up if e_above is None else bottom_up + e_above
        self.r = self.r + self.lr_inference * delta        # avoid in-place under no_grad

    def update_weights(self, lr_weights: float = 0.001, weight_decay: float = 1e-3) -> None:
        """Hebbian update — call only after inference has converged.
        weight_decay keeps W bounded; without it Hebbian updates grow without limit."""
        with torch.no_grad():
            dW = (self.e.t() @ self.r) / self.e.shape[0]  # (in_features, out_features)
            self.W.weight.add_(lr_weights * (dW - weight_decay * self.W.weight))

    def __repr__(self) -> str:
        return f"PCLayer(in={self.in_features}, out={self.out_features}, lr_i={self.lr_inference})"


class PCNetwork(nn.Module):
    """Stack of PCLayers. Inference runs in-place; weights update only after inference."""

    def __init__(self, layer_dims: list, lr_inference: float = 0.01):
        super().__init__()
        self.layers = nn.ModuleList([
            PCLayer(layer_dims[i], layer_dims[i + 1], lr_inference)
            for i in range(len(layer_dims) - 1)
        ])

    def init_states(self, x: torch.Tensor) -> None:
        """Bottom-up initialisation: feeds x through W.T at each layer.
        Far better than zeros — inference converges in ~20 steps instead of 100+."""
        r = x
        for layer in self.layers:
            r = layer.forward(r).detach()
            layer.r = r

    @torch.no_grad()
    def infer(self, x: torch.Tensor, n_steps: int = 20, clamp_top: torch.Tensor = None) -> None:
        """Run inference loop.  clamp_top: one-hot labels (B, n_classes) for supervised training."""
        for _ in range(n_steps):
            if clamp_top is not None:
                self.layers[-1].r = clamp_top

            # Bottom-up error pass
            self.layers[0].compute_error(x)
            for i in range(1, len(self.layers)):
                self.layers[i].compute_error(self.layers[i - 1].r)

            # Top-down representation update
            if clamp_top is None:
                self.layers[-1].update_r()
            for i in range(len(self.layers) - 2, -1, -1):
                self.layers[i].update_r(e_above=-self.layers[i + 1].e)

    def update_weights(self, lr_weights: float = 0.001) -> None:
        """Hebbian weight update — call only after infer() has converged."""
        for layer in self.layers:
            layer.update_weights(lr_weights)

    def total_energy(self) -> torch.Tensor:
        return sum((layer.e ** 2).sum(dim=-1).mean() for layer in self.layers)
