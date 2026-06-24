"""Visualisation utilities for Stage 1: convergence curves and latent representations."""

import os
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_energy_curve(energies: list, save_path: str = None):
    """Plot total prediction energy vs. inference step."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(energies)), energies, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Inference step")
    ax.set_ylabel("Total energy  (PC energy + output loss)")
    ax.set_title("PC inference convergence (single image, post-training)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_reconstructions(network, x_samples: torch.Tensor, y_samples: torch.Tensor,
                         n_inference_steps: int, save_path: str = None):
    """Visualise latent representations at each PC level after bottom-up inference.

    The new architecture (Linear → PCLayer → ReLU → ...) has no pixel-level decoder,
    so we show the intermediate representations instead:
      Row 0 — original image
      Row 1 — pc1 latent (256-dim → 16×16 pseudo-image)
      Row 2 — pc2 latent (256-dim → 16×16 pseudo-image)
      Row 3 — output logits (10-dim bar chart)

    x_samples : (N, 784) float32
    y_samples : (N,)     int64
    """
    n = x_samples.shape[0]
    device = x_samples.device

    # Run bottom-up pass to get representations (eval mode)
    network.eval()
    with torch.no_grad():
        r1 = network.get_representation(x_samples, layer_idx=0).cpu().numpy()  # (N, H)
        r2 = network.get_representation(x_samples, layer_idx=1).cpu().numpy()  # (N, H)
        logits = network(x_samples).cpu().numpy()                               # (N, 10)

    fig = plt.figure(figsize=(4 * n, 10))
    gs  = gridspec.GridSpec(4, n, hspace=0.5, wspace=0.3)

    row_labels = [
        "Original (28×28)",
        "pc₁ latent (16×16)",
        "pc₂ latent (16×16)",
        "Output logits (10-dim)",
    ]

    side = int(r1.shape[1] ** 0.5)  # 16 for H=256

    for col in range(n):
        # Row 0 — original
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(x_samples[col].cpu().numpy().reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"digit: {y_samples[col].item()}", fontsize=8)
        ax.axis("off")

        # Row 1 — pc1 latent state
        ax = fig.add_subplot(gs[1, col])
        feat = r1[col]
        ax.imshow(feat[:side * side].reshape(side, side), cmap="viridis")
        ax.axis("off")

        # Row 2 — pc2 latent state
        ax = fig.add_subplot(gs[2, col])
        feat = r2[col]
        ax.imshow(feat[:side * side].reshape(side, side), cmap="viridis")
        ax.axis("off")

        # Row 3 — output logits
        ax = fig.add_subplot(gs[3, col])
        ax.bar(range(10), logits[col], color="steelblue")
        ax.set_xticks(range(10))
        ax.set_xticklabels([str(i) for i in range(10)], fontsize=6)
        ax.tick_params(axis="y", labelsize=6)

    for row, label in enumerate(row_labels):
        fig.add_subplot(gs[row, 0]).set_ylabel(label, fontsize=8, labelpad=4)

    fig.suptitle("PC latent representations at each level (bottom-up pass)", fontsize=10)
    _save_or_show(fig, save_path)
    return fig


def plot_accuracy_history(history: list, save_path: str = None):
    """Plot test accuracy per epoch."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(history) + 1), [a * 100 for a in history], marker="o", linewidth=1.5)
    ax.axhline(95, color="red", linestyle="--", linewidth=1, label="95% target")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("MNIST classification accuracy during training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def _save_or_show(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  [viz] saved → {save_path}")
    plt.close(fig)
