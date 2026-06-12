import torch
import torch.nn as nn
import torch.nn.functional as F


class PCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, lr_inference=0.01):
        super().__init__()
        self.W = nn.Linear(output_dim, input_dim, bias=True)
        self.lr_inference = lr_inference
        self.r = None
        self.e = None
        self.prediction = None

    def init_r(self, batch_size, device):
        self.r = torch.zeros(batch_size, self.W.in_features,
                             requires_grad=False, device=device)

    def predict(self):
        return torch.tanh(self.W(self.r))

    def compute_error(self, r_below):
        self.prediction = self.predict()
        self.e = r_below - self.prediction
        return self.e

    def update_r(self, e_above=None):
        dtanh = 1 - self.prediction ** 2          # (batch, input_dim)
        bottom_up = (self.e * dtanh) @ self.W.weight  # (batch, output_dim)
        if e_above is None:
            self.r += self.lr_inference * bottom_up
        else:
            self.r += self.lr_inference * (bottom_up + e_above)

    def update_weights(self, lr_weights=0.001, weight_decay=1e-3):
        with torch.no_grad():
            dW = (self.e.t() @ self.r) / self.e.shape[0]
            # Subtract a small portion of the weights to prevent explosion
            self.W.weight.add_(lr_weights * dW - lr_weights * weight_decay * self.W.weight)


class PCNetwork(nn.Module):
    """Stack of PCLayers. Inference runs in-place; weights update only after inference."""

    def __init__(self, layer_dims: list, lr_inference: float = 0.01):
        super().__init__()
        # layers[i] maps from r[i] (dim layer_dims[i+1]) → prediction of layer_dims[i]
        self.layers = nn.ModuleList([
            PCLayer(layer_dims[i], layer_dims[i + 1], lr_inference)
            for i in range(len(layer_dims) - 1)
        ])

    def init_states(self, batch_size: int, device):
        for layer in self.layers:
            layer.init_r(batch_size, device)
    @torch.no_grad()
    def infer(self, x, n_steps: int = 20, clamp_top=None):
        """
        x          : clamped input, shape (B, input_dim)
        clamp_top  : one-hot labels (B, n_classes); if given, top r is fixed (training)
        """
        for _ in range(n_steps):
            if clamp_top is not None:
                self.layers[-1].r.data.copy_(clamp_top)

            # compute all prediction errors bottom → top
            self.layers[0].compute_error(x)
            for i in range(1, len(self.layers)):
                self.layers[i].compute_error(self.layers[i - 1].r)

            # update representations top → bottom
            # correct update: r[i] += lr * (bottom_up − e[i+1])
            # update_r adds e_above, so we pass −e[i+1]
            if clamp_top is None:
                self.layers[-1].update_r()
            for i in range(len(self.layers) - 2, -1, -1):
                self.layers[i].update_r(e_above=-self.layers[i + 1].e)

    def update_weights(self, lr_weights: float = 0.001):
        for layer in self.layers:
            layer.update_weights(lr_weights)

    def total_energy(self) -> torch.Tensor:
        return sum((layer.e ** 2).sum(dim=-1).mean() for layer in self.layers)