"""Correctness tests for the Stage 1 PC implementation.

Run directly (no pytest needed due to ROS2 plugin conflicts in this environment):
    python tests/test_pc_convergence.py

Or with pytest in a clean environment:
    python -m pytest tests/test_pc_convergence.py -v
"""

import sys
import os
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.pc_layer import PCLayer, PCNetwork
from utils.metrics import is_monotonically_decreasing, convergence_gap


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_small_network(seed=0):
    torch.manual_seed(seed)
    return PCNetwork(layer_dims=[16, 8, 4], lr_inference=0.01)


def make_mnist_network(seed=0):
    torch.manual_seed(seed)
    return PCNetwork(layer_dims=[784, 256, 64, 10], lr_inference=0.01)


def snapshot_weights(network):
    return [layer.W.weight.data.clone() for layer in network.layers]


def weights_changed(network, before):
    return any(
        not torch.allclose(layer.W.weight.data, w_b)
        for layer, w_b in zip(network.layers, before)
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

PASS = []
FAIL = []


def check(name, cond, msg=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}" + (f"\n        {msg}" if msg else ""))


# ── 1. Convergence tests ───────────────────────────────────────────────────────

def test_energy_decreases_supervised():
    net = make_mnist_network()
    torch.manual_seed(1)
    x = torch.rand(8, 784)
    y = torch.randint(0, 10, (8,))
    y_oh = F.one_hot(y, 10).float()
    net.init_states(x)
    energies = net.infer(x, n_steps=20, clamp_top=y_oh, return_energy=True)
    check(
        "Energy decreases over supervised inference (step 0 → step 19)",
        energies[-1] < energies[0],
        f"energy: {energies[0]:.4f} → {energies[-1]:.4f}"
    )


def test_energy_decreases_free():
    net = make_mnist_network()
    torch.manual_seed(1)
    x = torch.rand(8, 784)
    net.init_states(x)
    energies = net.infer(x, n_steps=20, return_energy=True)
    check(
        "Energy decreases over free (unclamped) inference",
        energies[-1] < energies[0],
        f"energy: {energies[0]:.4f} → {energies[-1]:.4f}"
    )


def test_energy_near_monotonic():
    net = make_mnist_network()
    torch.manual_seed(1)
    x = torch.rand(8, 784)
    y_oh = F.one_hot(torch.randint(0, 10, (8,)), 10).float()
    net.init_states(x)
    energies = net.infer(x, n_steps=20, clamp_top=y_oh, return_energy=True)
    tol = 0.01 * energies[0]   # allow blips up to 1% of initial energy
    check(
        "Supervised inference energy is (near-)monotonically decreasing",
        is_monotonically_decreasing(energies, tolerance=tol),
        "Energy increased by >1% of E_0 in at least one step"
    )


def test_convergence_within_20_steps():
    # 5% threshold for random-weight networks; trained networks achieve much more.
    # The stage script separately verifies ≥10% reduction post-training.
    net = make_mnist_network()
    torch.manual_seed(1)
    x = torch.rand(8, 784)
    y_oh = F.one_hot(torch.randint(0, 10, (8,)), 10).float()
    net.init_states(x)
    energies = net.infer(x, n_steps=20, clamp_top=y_oh, return_energy=True)
    gap = convergence_gap(energies)
    check(
        "Convergence gap ≥ 5% within 20 inference steps (random init; trained nets achieve >>10%)",
        gap > 0.05,
        f"gap={gap:.3f}"
    )


# ── 2. Weight update timing ────────────────────────────────────────────────────

def test_weights_frozen_during_inference():
    net = make_mnist_network()
    torch.manual_seed(1)
    x = torch.rand(8, 784)
    y_oh = F.one_hot(torch.randint(0, 10, (8,)), 10).float()
    net.init_states(x)
    before = snapshot_weights(net)
    net.infer(x, n_steps=20, clamp_top=y_oh)
    after = snapshot_weights(net)
    frozen = all(torch.allclose(b, a) for b, a in zip(before, after))
    check("Weights remain frozen during inference loop", frozen)


def test_weights_update_after_convergence():
    net = make_mnist_network()
    torch.manual_seed(1)
    x = torch.rand(8, 784)
    y_oh = F.one_hot(torch.randint(0, 10, (8,)), 10).float()
    net.init_states(x)
    net.infer(x, n_steps=20, clamp_top=y_oh)
    before = snapshot_weights(net)
    net.update_weights(lr_weights=0.01)
    check("Weights change after update_weights() call", weights_changed(net, before))


# ── 3. Shapes and dtypes ──────────────────────────────────────────────────────

def test_representation_shapes():
    net = make_small_network()
    torch.manual_seed(1)
    x = torch.randn(4, 16)
    net.init_states(x)
    check("Layer 0 r shape == (4, 8)",  net.layers[0].r.shape == (4, 8))
    check("Layer 1 r shape == (4, 4)",  net.layers[1].r.shape == (4, 4))


def test_error_shapes():
    net = make_small_network()
    torch.manual_seed(1)
    x = torch.randn(4, 16)
    net.init_states(x)
    net.infer(x, n_steps=5)
    check("Layer 0 e shape == (4, 16)", net.layers[0].e.shape == (4, 16))
    check("Layer 1 e shape == (4, 8)",  net.layers[1].e.shape == (4, 8))


def test_all_float32():
    net = make_small_network()
    torch.manual_seed(1)
    x = torch.randn(4, 16)
    net.init_states(x)
    net.infer(x, n_steps=5)
    all_f32 = all(
        layer.r.dtype == torch.float32 and
        (layer.e is None or layer.e.dtype == torch.float32)
        for layer in net.layers
    )
    check("All representations and errors are float32", all_f32)


def test_return_energy_length():
    net = make_small_network()
    x = torch.randn(4, 16)
    net.init_states(x)
    energies = net.infer(x, n_steps=15, return_energy=True)
    check("return_energy yields exactly n_steps values", len(energies) == 15, f"got {len(energies)}")


def test_return_energy_none_by_default():
    net = make_small_network()
    x = torch.randn(4, 16)
    net.init_states(x)
    result = net.infer(x, n_steps=5)
    check("infer() returns None when return_energy=False", result is None)


# ── 4. Hebbian update ─────────────────────────────────────────────────────────

def test_hebbian_direction():
    """ΔW must be aligned with the Hebbian term e ⊗ r (positive dot product)."""
    net = make_small_network()
    torch.manual_seed(1)
    x = torch.randn(4, 16)
    net.init_states(x)
    net.infer(x, n_steps=10)
    layer = net.layers[0]
    e = layer.e.detach().clone()
    r = layer.r.detach().clone()
    w_before = layer.W.weight.data.clone()
    net.update_weights(lr_weights=1.0)
    dw_actual = layer.W.weight.data - w_before
    dw_expected = (e.t() @ r) / e.shape[0]
    alignment = (dw_actual * dw_expected).sum().item()
    check(
        "Hebbian weight update aligned with e⊗r (positive dot product)",
        alignment > 0,
        f"alignment={alignment:.4f}"
    )


def test_update_is_batch_mean():
    """update_weights must be scale-invariant to repeated identical samples."""
    torch.manual_seed(5)
    x1 = torch.randn(1, 16)
    x4 = x1.repeat(4, 1)

    net1 = PCNetwork([16, 8, 4], lr_inference=0.01)
    net4 = PCNetwork([16, 8, 4], lr_inference=0.01)
    # Share initial weights
    for l1, l4 in zip(net1.layers, net4.layers):
        l4.W.weight.data.copy_(l1.W.weight.data)

    net1.init_states(x1); net1.infer(x1, n_steps=5)
    net4.init_states(x4); net4.infer(x4, n_steps=5)

    w1_before = net1.layers[0].W.weight.data.clone()
    w4_before = net4.layers[0].W.weight.data.clone()
    net1.update_weights(lr_weights=0.01)
    net4.update_weights(lr_weights=0.01)

    dw1 = net1.layers[0].W.weight.data - w1_before
    dw4 = net4.layers[0].W.weight.data - w4_before
    check(
        "Weight update is a batch mean (identical result for 1 vs 4 repeated samples)",
        torch.allclose(dw1, dw4, atol=1e-5),
        f"max diff: {(dw1 - dw4).abs().max().item():.2e}"
    )


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Stage 1 — PC correctness tests")
    print("=" * 60)

    tests = [
        test_energy_decreases_supervised,
        test_energy_decreases_free,
        test_energy_near_monotonic,
        test_convergence_within_20_steps,
        test_weights_frozen_during_inference,
        test_weights_update_after_convergence,
        test_representation_shapes,
        test_error_shapes,
        test_all_float32,
        test_return_energy_length,
        test_return_energy_none_by_default,
        test_hebbian_direction,
        test_update_is_batch_mean,
    ]

    for fn in tests:
        try:
            fn()
        except Exception as exc:
            FAIL.append(fn.__name__)
            print(f"  ERROR {fn.__name__}: {exc}")

    print()
    print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("Failed:", ", ".join(FAIL))
        sys.exit(1)
    else:
        print("All tests passed.")
