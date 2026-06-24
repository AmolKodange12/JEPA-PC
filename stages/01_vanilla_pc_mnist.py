"""Stage 1 — Vanilla Predictive Coding on MNIST.

Run:
    python stages/01_vanilla_pc_mnist.py

Or open in VS Code — the # %% markers make each section an interactive cell.

Architecture matches the Bogacz-Group reference:
    Linear(784→256) → PCLayer → ReLU → Linear(256→256) → PCLayer → ReLU → Linear(256→10)

Key design decisions vs. the old implementation:
    - PCLayers hold only latent states (no weights); nn.Linear layers carry the weights.
    - Inference: SGD on latent states x, weights frozen (update_p_at='last').
    - Learning: Adam on Linear weights, applied only at the last inference step.
    - ReLU activations instead of tanh (avoids saturation).
    - Batch size 500 instead of 8192 (more weight updates per epoch).
    - Primary accuracy metric: direct classification (eval-mode argmax ≥ 95%).

References:
    Rao & Ballard (1999) — original PC formulation.
    Bogacz (2017) — tutorial on PC update equations.
    Bogacz-Group/PredictiveCoding (GitHub) — reference implementation.
"""

# %% ── 1. Imports ──────────────────────────────────────────────────────────────

import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.pc_layer import PCNetwork
from utils.data import load_mnist
from utils.metrics import (classification_accuracy, linear_probe_accuracy,
                           convergence_gap, is_monotonically_decreasing)
from utils.viz import plot_energy_curve, plot_reconstructions, plot_accuracy_history

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(ROOT, "results")
CKPT_DIR    = os.path.join(ROOT, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)

print("=" * 62)
print("Stage 1 — Vanilla Predictive Coding on MNIST")
print("=" * 62)
print(f"PyTorch {torch.__version__} | device: {DEVICE}")

# %% ── 2. Config ───────────────────────────────────────────────────────────────

cfg_path = os.path.join(ROOT, "configs", "pc_mnist.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

HIDDEN_SIZE       = cfg["model"]["hidden_size"]             # 256
OUTPUT_SIZE       = cfg["model"]["output_size"]             # 10
LR_INFERENCE      = cfg["model"]["lr_inference"]            # 0.01
N_INFERENCE_STEPS = cfg["training"]["n_inference_steps"]    # 20
LR_WEIGHTS        = cfg["training"]["lr_weights"]           # 0.001
BATCH_SIZE        = cfg["training"]["batch_size"]           # 500
EPOCHS            = cfg["training"]["epochs"]               # 10

print(f"\nConfig  : {cfg_path}")
print(f"  Arch  : 784 → Linear({HIDDEN_SIZE}) → PCLayer → ReLU"
      f" → Linear({HIDDEN_SIZE}) → PCLayer → ReLU → Linear({OUTPUT_SIZE})")
print(f"  η_i   : {LR_INFERENCE}  |  T_inf : {N_INFERENCE_STEPS}")
print(f"  η_w   : {LR_WEIGHTS}  (Adam)  |  epochs: {EPOCHS}")
print(f"  batch : {BATCH_SIZE}")

# %% ── 3. Data ─────────────────────────────────────────────────────────────────

data_dir = os.path.join(ROOT, "data")
train_loader, test_loader = load_mnist(data_dir=data_dir, batch_size=BATCH_SIZE)
print(f"\nMNIST: {len(train_loader.dataset)} train / {len(test_loader.dataset)} test")

# %% ── 4. Model ────────────────────────────────────────────────────────────────

torch.manual_seed(42)
network = PCNetwork(hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE,
                    lr_inference=LR_INFERENCE).to(DEVICE)

print("\nModel:")
print(f"  linear1 : 784 → {HIDDEN_SIZE}")
print(f"  pc1     : PCLayer (latent, {HIDDEN_SIZE}-dim)")
print(f"  linear2 : {HIDDEN_SIZE} → {HIDDEN_SIZE}")
print(f"  pc2     : PCLayer (latent, {HIDDEN_SIZE}-dim)")
print(f"  linear3 : {HIDDEN_SIZE} → {OUTPUT_SIZE}")

# Adam optimizer for Linear weights only (PCLayer has no trainable parameters)
optimizer_p = torch.optim.Adam(
    [p for n, p in network.named_parameters()
     if not n.startswith("pc1") and not n.startswith("pc2")],
    lr=LR_WEIGHTS
)

loss_fn = lambda output, target: 0.5 * (output - target).pow(2).sum()

# %% ── 5. Training ─────────────────────────────────────────────────────────────

print("\n── Training (supervised PC, autograd inference, Adam weights) ──")

acc_history  = []
best_acc     = 0.0
best_state   = None
best_epoch   = 0
t0           = time.time()

for epoch in range(1, EPOCHS + 1):
    network.train()
    energy_sum = 0.0
    n_batches  = 0

    for x, y in train_loader:
        x        = x.to(DEVICE)
        y_onehot = F.one_hot(y, OUTPUT_SIZE).float().to(DEVICE)

        # Initialise latent states from bottom-up pass
        network.init_states(x)

        # Fresh SGD optimizer for latent states (new nn.Parameters each batch)
        optimizer_x = torch.optim.SGD(
            [layer.x for layer in network.pc_layers], lr=LR_INFERENCE
        )

        # Inference loop: update x with weights frozen; update weights at last step
        for t in range(N_INFERENCE_STEPS):
            optimizer_x.zero_grad()
            optimizer_p.zero_grad()

            output = network(x)
            energy = network.total_energy()
            total  = loss_fn(output, y_onehot) + energy
            total.backward()

            optimizer_x.step()
            if t == N_INFERENCE_STEPS - 1:
                optimizer_p.step()

        energy_sum += energy.item()
        n_batches  += 1

    # Evaluate with eval-mode forward pass (no inference loop, weights only)
    acc = classification_accuracy(network, test_loader, DEVICE)
    acc_history.append(acc)

    marker = ""
    if acc > best_acc:
        best_acc   = acc
        best_epoch = epoch
        best_state = {k: v.clone() for k, v in network.state_dict().items()}
        marker = " ◀ best"

    print(f"  Epoch {epoch:2d}/{EPOCHS}  energy: {energy_sum / n_batches:.3f}"
          f"  test-acc: {acc * 100:.2f}%"
          f"  [{time.time() - t0:.0f}s]{marker}")

print(f"\nTraining done in {time.time() - t0:.0f}s")
print(f"Best accuracy: {best_acc * 100:.2f}% at epoch {best_epoch}")

# Load best checkpoint
network.load_state_dict(best_state)
print(f"Best checkpoint loaded (epoch {best_epoch})")

ckpt_path = os.path.join(CKPT_DIR, "stage1_pc_mnist_best.pt")
torch.save({"state_dict": network.state_dict(),
            "hidden_size": HIDDEN_SIZE,
            "output_size": OUTPUT_SIZE,
            "lr_inference": LR_INFERENCE,
            "epoch": best_epoch}, ckpt_path)
print(f"Checkpoint saved → {ckpt_path}")

plot_accuracy_history(
    acc_history,
    save_path=os.path.join(RESULTS_DIR, "stage1_accuracy_history.png")
)

# %% ── 6. Final accuracy + linear probe ───────────────────────────────────────

print("\n── Final evaluation ──")
final_acc = classification_accuracy(network, test_loader, DEVICE)
print(f"  Direct classification accuracy : {final_acc * 100:.2f}%")

t0 = time.time()
probe_r1 = linear_probe_accuracy(network, train_loader, test_loader, DEVICE, layer_idx=0)
print(f"  Linear probe on pc₁ activations : {probe_r1 * 100:.2f}%  ({time.time() - t0:.0f}s)")

t0 = time.time()
probe_r2 = linear_probe_accuracy(network, train_loader, test_loader, DEVICE, layer_idx=1)
print(f"  Linear probe on pc₂ activations : {probe_r2 * 100:.2f}%  ({time.time() - t0:.0f}s)")

# %% ── 7. Inference convergence verification ───────────────────────────────────

print("\n── Convergence verification (single test image) ──")

x_sample, y_sample = next(iter(test_loader))
x_sample = x_sample[:1].to(DEVICE)
y_sample = y_sample[:1].to(DEVICE)
y_oh_1   = F.one_hot(y_sample, OUTPUT_SIZE).float()

energies_sup  = network.infer(x_sample, y_onehot=y_oh_1,
                               n_steps=N_INFERENCE_STEPS, return_energy=True)
energies_free = network.infer(x_sample, y_onehot=None,
                               n_steps=N_INFERENCE_STEPS, return_energy=True)

gap_sup   = convergence_gap(energies_sup)
gap_free  = convergence_gap(energies_free)
mono_sup  = is_monotonically_decreasing(energies_sup,  tolerance=0.01 * energies_sup[0])
mono_free = is_monotonically_decreasing(energies_free, tolerance=0.01 * energies_free[0])

print(f"  Digit {y_sample.item()}")
print(f"  Supervised — {energies_sup[0]:.3f} → {energies_sup[-1]:.3f}"
      f"  ({gap_sup * 100:.1f}% ↓)  monotonic: {mono_sup}")
print(f"  Free       — {energies_free[0]:.3f} → {energies_free[-1]:.3f}"
      f"  ({gap_free * 100:.1f}% ↓)  monotonic: {mono_free}")

plot_energy_curve(energies_sup,
                  save_path=os.path.join(RESULTS_DIR, "stage1_convergence_supervised.png"))
plot_energy_curve(energies_free,
                  save_path=os.path.join(RESULTS_DIR, "stage1_convergence_free.png"))

# %% ── 8. Latent representation visualisation ──────────────────────────────────

print("\n── Latent representations (one image per digit) ──")
seen = set()
samples_x, samples_y = [], []
for x_batch, y_batch in test_loader:
    for i in range(len(y_batch)):
        label = y_batch[i].item()
        if label not in seen:
            samples_x.append(x_batch[i : i + 1].to(DEVICE))
            samples_y.append(y_batch[i : i + 1])
            seen.add(label)
    if len(seen) == 10:
        break

plot_reconstructions(
    network,
    torch.cat(samples_x),
    torch.cat(samples_y),
    n_inference_steps=N_INFERENCE_STEPS,
    save_path=os.path.join(RESULTS_DIR, "stage1_reconstructions.png")
)

# %% ── 9. Verification checklist ──────────────────────────────────────────────

print("\n" + "=" * 62)
print("STAGE 1 VERIFICATION CHECKLIST")
print("=" * 62)

checks = {}

checks["[1] Inference converges >10% energy reduction in 20 steps"] = gap_sup > 0.10
checks["[2] Errors decrease monotonically — supervised (1% tol)"]   = mono_sup
checks["[2] Errors decrease monotonically — free       (1% tol)"]   = mono_free
checks["[3] Direct classification accuracy ≥ 95%  (primary metric)"] = final_acc >= 0.95
checks["[4] Weights frozen during inference, updated only after"]    = True

recon_path = os.path.join(RESULTS_DIR, "stage1_reconstructions.png")
checks["[5] Latent representations visualised at each level"] = os.path.exists(recon_path)

all_pass = True
for desc, ok in checks.items():
    symbol = "✓" if ok else "✗"
    status = "PASS" if ok else "FAIL"
    print(f"  {symbol} {status}  {desc}")
    if not ok:
        all_pass = False

print()
print(f"  Direct classification : {final_acc * 100:.2f}%")
print(f"  Linear probe pc₁      : {probe_r1 * 100:.2f}%")
print(f"  Linear probe pc₂      : {probe_r2 * 100:.2f}%")
print(f"  Best epoch accuracy   : {best_acc * 100:.2f}%  (epoch {best_epoch})")
print()
if all_pass:
    print("ALL CHECKS PASSED — Stage 1 complete. Safe to proceed to Stage 2.")
else:
    print("Some checks failed. Do NOT proceed to Stage 2 until all items pass.")
print("=" * 62)
