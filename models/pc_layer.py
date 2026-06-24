import torch
import torch.nn as nn


class PCLayer(nn.Module):
    """Latent-state holder for predictive coding — no weights.

    During training: stores latent x as nn.Parameter, returns x (not mu) in the
    forward pass, and computes energy E = 0.5 * ||x - mu||^2.
    During eval: identity pass-through (returns mu), behaving like a regular net.

    Matches the Bogacz-Group reference implementation:
    https://github.com/Bogacz-Group/PredictiveCoding
    """

    def __init__(self):
        super().__init__()
        self._x: nn.Parameter | None = None
        self._energy: torch.Tensor | None = None

    def set_x(self, mu: torch.Tensor) -> None:
        """Initialise latent state from bottom-up prediction (called before inference)."""
        self._x = nn.Parameter(mu.detach().clone())

    @property
    def x(self) -> nn.Parameter:
        return self._x

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._energy = 0.5 * (self._x - mu).pow(2).sum()
            return self._x
        return mu

    def energy(self) -> torch.Tensor:
        return self._energy

    def __repr__(self) -> str:
        shape = tuple(self._x.shape) if self._x is not None else "uninitialized"
        return f"PCLayer(x_shape={shape})"


class PCNetwork(nn.Module):
    """Predictive coding network matching the Bogacz-Group reference.

    Architecture:
        Linear(784 → H) → PCLayer → ReLU → Linear(H → H) → PCLayer → ReLU → Linear(H → 10)

    Weights live in the nn.Linear modules.  PCLayers hold only latent states.
    Inference: SGD on latent states (weights frozen).
    Learning:  Adam on Linear weights, applied only at the last inference step.
    """

    def __init__(self, hidden_size: int = 256, output_size: int = 10,
                 lr_inference: float = 0.01):
        super().__init__()
        self.lr_inference = lr_inference
        self.linear1 = nn.Linear(784, hidden_size)
        self.pc1     = PCLayer()
        self.act1    = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.pc2     = PCLayer()
        self.act2    = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.pc_layers = [self.pc1, self.pc2]

    def init_states(self, x: torch.Tensor) -> None:
        """Bottom-up forward pass to initialise latent states before inference."""
        with torch.no_grad():
            mu1 = self.linear1(x)
            self.pc1.set_x(mu1)
            a1  = self.act1(mu1)
            mu2 = self.linear2(a1)
            self.pc2.set_x(mu2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu1 = self.linear1(x)
        x1  = self.pc1(mu1)     # returns pc1._x during train, mu1 during eval
        a1  = self.act1(x1)
        mu2 = self.linear2(a1)
        x2  = self.pc2(mu2)
        a2  = self.act2(x2)
        return self.linear3(a2)

    def total_energy(self) -> torch.Tensor:
        return sum(layer.energy() for layer in self.pc_layers)

    def get_representation(self, x: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """Extract intermediate representation via bottom-up pass (eval mode).

        layer_idx=0: activations after pc1 (H-dim)
        layer_idx=1: activations after pc2 (H-dim)
        """
        with torch.no_grad():
            mu1 = self.linear1(x)
            if layer_idx == 0:
                return self.act1(mu1)
            a1  = self.act1(mu1)
            mu2 = self.linear2(a1)
            if layer_idx == 1:
                return self.act2(mu2)
        raise ValueError(f"layer_idx must be 0 or 1, got {layer_idx}")

    def infer(self, x: torch.Tensor, y_onehot: torch.Tensor = None,
              n_steps: int = 20, return_energy: bool = False):
        """Run inference (update latent states only, weights frozen).

        y_onehot: one-hot labels — supervised inference.
        None     — free inference (minimise PC energy only, no output loss).
        return_energy: if True, return list of total energy recorded before each step.
        """
        was_training = self.training
        self.train()
        self.init_states(x)
        optimizer_x = torch.optim.SGD(
            [layer.x for layer in self.pc_layers], lr=self.lr_inference
        )
        loss_fn = lambda o, t: 0.5 * (o - t).pow(2).sum()
        energies = []

        for _ in range(n_steps):
            optimizer_x.zero_grad()
            output = self(x)
            energy = self.total_energy()
            total  = (loss_fn(output, y_onehot) + energy) if y_onehot is not None else energy
            if return_energy:
                energies.append(total.item())
            total.backward()
            optimizer_x.step()

        if not was_training:
            self.eval()
        return energies if return_energy else None
