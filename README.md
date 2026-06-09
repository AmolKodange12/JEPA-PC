# JEPA-PC: A Composite World Model Framework

> **Combining JEPA-style representation learning with Predictive Coding inference for robust, adaptive world models**

This project investigates a principled composite of two complementary frameworks for unsupervised world modeling:

- **JEPA (Joint Embedding Predictive Architecture)** — learns abstract, semantically meaningful representations by predicting in latent space, avoiding the capacity waste of pixel-level reconstruction (LeCun, 2022; Assran et al., 2023)
- **Predictive Coding (PC)** — performs hierarchical inference via bidirectional prediction error minimization, enabling inference-time adaptation and uncertainty quantification (Rao & Ballard, 1999; Friston, 2005)

The core thesis: **JEPA solves the representation learning problem; PC solves the inference problem. These are separable, and combining them gives capabilities neither framework provides alone.**

---

## Motivation

### The Gap in Each Framework

JEPA trains a powerful world model but has no runtime mechanism to ask *"how wrong am I right now, given this specific input?"* Once trained, the encoder does a single forward pass and stops. It cannot refine its belief about an ambiguous input, does not quantify uncertainty, and does not adapt to distribution shift without retraining.

Predictive coding provides exactly this inference-time loop — iteratively minimizing prediction errors until internal beliefs converge — but historically operates in pixel or raw sensory space. This forces the model to spend capacity modeling irrelevant variance (texture, lighting, compression artifacts), and makes the architecture difficult to scale.

### The Composite Hypothesis

Replace PC's lowest level — raw pixel inputs — with frozen JEPA token representations. PC then performs hierarchical inference entirely in abstract latent space. The result:

- JEPA's tokenizer absorbs irrelevant pixel-level variance before PC ever sees the input
- PC's inference loop provides online belief refinement that pure JEPA lacks
- Precision-weighting (PC's native uncertainty mechanism) now operates on semantically meaningful tokens rather than noisy pixels
- The system gains inference-time adaptation without retraining

```
Raw Input
    │
    ▼
┌─────────────────────┐
│   JEPA Tokenizer    │  ← Pretrained, frozen. Absorbs pixel-level variance.
│  (patch encoder +   │    Produces stable, semantically meaningful tokens.
│   EMA target enc.)  │
└─────────────────────┘
    │  token representations
    ▼
┌─────────────────────┐
│    PC Level 1       │  ← Predicts token-level features
│  top-down ↕ errors  │    Error = deviation from predicted token repr.
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│    PC Level 2       │  ← Predicts abstract region-level summaries
│  top-down ↕ errors  │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│    PC Level 3       │  ← Highest abstraction: scene/object-level state
└─────────────────────┘
```

---

## Related Work

This project sits at the intersection of several active research threads:

- **I-JEPA** (Assran et al., 2023) — image-based JEPA; predicts masked patch representations using unmasked context
- **V-JEPA** (Bardes et al., 2024) — video extension; predicts future frame representations in latent space
- **Rao & Ballard (1999)** — original hierarchical predictive coding model for visual cortex
- **Whittington & Bogacz (2017)** — demonstrates PC inference approximates backpropagation; establishes PC as a viable learning algorithm
- **Ororbia (2022–2024)** — neural predictive coding on learned representations; closest existing work to this composite
- **data2vec** (Baevski et al., 2022) — predicts latent representations of masked inputs; empirically adjacent to the composite idea
- **Friston (2005, 2010)** — free energy principle; theoretical grounding for PC as a general inference framework
- **Active Inference** (Friston et al., 2023) — extends PC to action via the same error-minimization objective

The specific combination proposed here — JEPA tokenization feeding a PC inference hierarchy — does not have a dominant existing implementation, making this a tractable novel contribution at master's level.

---

## Goals

### Primary Goals

1. **Implement and validate a standalone PC network** on a toy dataset, with well-characterized convergence behavior
2. **Implement and validate a standalone JEPA tokenizer** with confirmed non-collapsed representations
3. **Build the composite**: PC hierarchy operating on frozen JEPA token representations
4. **Empirically test the composite's unique claims** — not just benchmark accuracy, but properties that only the composite should exhibit:
   - Does inference-time adaptation improve representations over a single forward pass?
   - Does the composite forget less under continual learning compared to a JEPA baseline?
   - Does precision-weighting produce interpretable uncertainty estimates?

### Secondary Goals

5. **Characterize the convergence properties** of PC inference in token space vs. pixel space
6. **Ablate the number of PC levels** — does a deeper PC hierarchy improve abstraction quality?
7. **Document failure modes** — where does the composite break, and why?

---

## Development Stages

All stages are designed to run on a laptop (CPU) or free-tier GPU (Kaggle/Colab T4). Do not scale up until each stage is verified correct at small scale.

---

### Stage 1 — Vanilla Predictive Coding on MNIST

**Goal:** Implement PC from scratch. Understand the inference loop, convergence behavior, and failure modes before adding any complexity.

**Dataset:** MNIST (trivially downloadable, fast iteration)

**Architecture:**
```
Input (784,) → PC Level 1 (256,) → PC Level 2 (64,) → PC Level 3 (10,)
```

**What to implement:**

```python
class PCLayer(nn.Module):
    """
    Single PC layer. Maintains:
    - weights W: top-down prediction matrix
    - state r: current representation (updated during inference)
    - prediction e: prediction error (passed upward)
    
    Inference step (weights fixed):
        prediction = W @ r_above
        error = r_below - prediction
        r_above += lr_inference * (error - W.T @ error_above)
    
    Learning step (after inference converges):
        dW = outer(error, r_above)
        W += lr_weights * dW
    """
```

**Verification checklist:**
- [ ] Inference loop converges within ~20 iterations on a single image
- [ ] Prediction errors decrease monotonically during inference (plot this)
- [ ] Classification accuracy ≥ 95% on MNIST test set
- [ ] Weight updates only occur *after* inference converges, not during
- [ ] Visualize top-down reconstructions at each level — they should become coarser at higher levels

**Key reference:** Rao & Ballard (1999), equations 1–4. Whittington & Bogacz (2017) for the learning rule derivation.

**Common failure modes to watch for:**
- Oscillating inference (inference LR too high) → reduce inference step size
- No learning (inference never converges before weight update) → add convergence criterion
- Posterior collapse at higher levels → add L2 regularization on representations

**Deliverable:** `notebooks/01_vanilla_pc_mnist.ipynb`

---

### Stage 2 — JEPA Tokenizer on CIFAR-10

**Goal:** Train a minimal JEPA tokenizer. Understand stop-gradient, EMA updates, and representation collapse.

**Dataset:** CIFAR-10 (32×32, 10 classes, fast to train at small scale)

**Architecture:**
```
Image → patch embed (4×4 patches → 64 tokens of dim 128)
     → Context Encoder (small ViT, 4 layers)
     → Predictor MLP
     → predicted target representations

Target: EMA copy of context encoder, no gradient
Loss: cosine similarity in latent space (target is stop-gradient)
```

**What to implement:**

```python
# EMA update — critical for stability
def update_target_encoder(online_enc, target_enc, momentum=0.996):
    for p_online, p_target in zip(online_enc.parameters(), 
                                   target_enc.parameters()):
        p_target.data = momentum * p_target.data + (1 - momentum) * p_online.data
        # target_enc receives NO gradient — this is the stop-gradient

# Loss — in latent space only, never pixel space
def jepa_loss(predicted, target):
    predicted = F.normalize(predicted, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1 - (predicted * target).sum(dim=-1).mean()
```

**Verification checklist:**
- [ ] Representation variance does not collapse to zero (monitor std of embeddings per epoch)
- [ ] Linear probe accuracy ≥ 60% on CIFAR-10 (reasonable for a small model trained from scratch)
- [ ] Visualize nearest neighbors in embedding space — semantically similar images should cluster
- [ ] Confirm EMA momentum is working: target encoder weights ≠ online encoder weights but track them slowly

**Key reference:** Assran et al., *"Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"* (2023). Pay close attention to Section 3 (architecture) and Appendix A (hyperparameters).

**Deliverable:** `notebooks/02_jepa_cifar10.ipynb`

---

### Stage 3 — PC on Patch Tokens (Partial Composite)

**Goal:** Replace PC's pixel inputs with frozen JEPA patch tokens. Verify that PC inference is better-behaved in token space than pixel space. This is already a novel small contribution.

**Dataset:** CIFAR-10 (same tokenizer from Stage 2, frozen)

**What changes from Stage 1:**
- Input to PC Level 1 is now token representations (dim 128 per patch), not raw pixels
- PC hierarchy has 3 levels operating entirely in latent space
- Compare convergence speed and stability vs. pixel-level PC from Stage 1

**Verification checklist:**
- [ ] PC inference converges in fewer iterations than pixel-space PC (this should happen — tokens are lower-variance)
- [ ] Linear probe accuracy at each PC level — does it increase with level? (it should)
- [ ] Prediction errors in token space are semantically interpretable — high error on occluded or ambiguous patches

**Deliverable:** `notebooks/03_pc_on_patch_tokens.ipynb`

---

### Stage 4 — Full Composite + Evaluation

**Goal:** Build the full composite and run the three evaluations that test its *unique* properties.

**Dataset:** CIFAR-10 / CIFAR-100 (CIFAR-100 for continual learning split)

#### Evaluation 1: Inference-Time Adaptation

Tests whether iterative PC inference improves representations over a single forward pass.

```python
def evaluate_inference_adaptation(model, images, corruption_fn, n_steps=20):
    """
    1. Corrupt images (Gaussian noise, masking, or occlusion)
    2. Single forward pass → representation r_0
    3. Run PC inference loop for n_steps → representation r_n
    4. Linear probe accuracy of r_0 vs r_n
    Expected: r_n > r_0, gap increases with corruption severity
    """
```

Plot: accuracy vs. number of inference steps, for multiple corruption levels. A pure JEPA baseline is flat (no inference loop) — the composite should show a rising curve.

#### Evaluation 2: Continual Learning

Tests whether local PC weight updates reduce catastrophic forgetting vs. backprop-trained JEPA.

```python
# Split CIFAR-100 into 10 tasks of 10 classes each
# Train sequentially: task 1 → task 2 → ... → task 10
# After each task, evaluate on all previous tasks
# Metric: average accuracy on all seen tasks after final task

# Baselines:
# - JEPA fine-tuned with backprop (expected: high forgetting)
# - PC composite with local weight updates (expected: less forgetting)
# - EWC regularization on JEPA (standard continual learning baseline)
```

#### Evaluation 3: Uncertainty via Precision-Weighting

Tests whether PC's precision estimates are meaningful in token space.

```python
# Precision = inverse variance of prediction errors at each token position
# High precision → model is confident about this token
# Low precision → model is uncertain

# Qualitative check: do precision maps highlight informative image regions?
# Quantitative check: do low-precision tokens correspond to occluded/corrupted regions?
```

**Deliverable:** `notebooks/04_composite_evaluation.ipynb`, `models/composite.py`

---

## Repository Structure

```
jepa-pc-composite/
│
├── README.md                        ← this file
│
├── notebooks/
│   ├── 01_vanilla_pc_mnist.ipynb    ← Stage 1
│   ├── 02_jepa_cifar10.ipynb        ← Stage 2
│   ├── 03_pc_on_patch_tokens.ipynb  ← Stage 3
│   └── 04_composite_evaluation.ipynb← Stage 4
│
├── models/
│   ├── pc_layer.py                  ← PCLayer, PCNetwork
│   ├── jepa_tokenizer.py            ← PatchEmbed, ContextEncoder, EMATargetEncoder
│   └── composite.py                 ← JEPAPCComposite: tokenizer + PC hierarchy
│
├── experiments/
│   ├── linear_probe.py              ← evaluate representation quality at each level
│   ├── continual_learning.py        ← sequential task training + forgetting metrics
│   └── inference_adaptation.py      ← corruption + iterative inference evaluation
│
├── configs/
│   ├── pc_mnist.yaml
│   ├── jepa_cifar10.yaml
│   └── composite_cifar100.yaml
│
└── requirements.txt
```

---

## Compute Requirements

All stages are designed to fit within free-tier GPU constraints.

| Stage | Dataset | Model Size | GPU | Est. Time |
|---|---|---|---|---|
| 1 — Vanilla PC | MNIST | ~500K params | CPU only | 10–20 min |
| 2 — JEPA tokenizer | CIFAR-10 | ~5M params | T4 (Kaggle free) | 2–3 hr |
| 3 — PC on tokens | CIFAR-10 | ~6M params | T4 | 1–2 hr |
| 4 — Composite eval | CIFAR-100 | ~7M params | T4 | 3–5 hr |

Do not scale to ImageNet or larger datasets until Stage 4 is verified correct. The architectural claims should hold at CIFAR scale; if they don't, scaling will not fix them.

---

## Key Implementation Notes

**Stop-gradient is non-negotiable in JEPA.** Without EMA and stop-gradient on the target encoder, representations collapse to a constant within a few hundred steps. Monitor embedding variance every epoch.

**PC inference converges before weight updates.** A common bug is updating weights every forward pass rather than after the inference loop converges. This makes the network appear to learn but the representations are not PC representations — they're just backprop with extra steps.

**Precision weights need normalization.** Raw prediction error variances can span several orders of magnitude across layers. Normalize within each layer or training becomes unstable.

**Frozen tokenizer is essential for the composite.** If you fine-tune the JEPA tokenizer jointly with the PC hierarchy, the tokenizer will adapt to minimize PC prediction error rather than learning general representations. Stage 2 and Stage 3/4 should be entirely separate training phases.

---

## References

- Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*, 2(1), 79–87.
- Friston, K. (2005). A theory of cortical responses. *Philosophical Transactions of the Royal Society B*, 360(1456), 815–836.
- Whittington, J. C., & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. *Neural Computation*, 29(5), 1229–1262.
- LeCun, Y. (2022). A path towards autonomous machine intelligence. *OpenReview preprint*.
- Assran, M., et al. (2023). Self-supervised learning from images with a joint-embedding predictive architecture. *CVPR 2023*.
- Bardes, A., et al. (2024). V-JEPA: Revisiting feature prediction for learning visual representations from video. *NeurIPS 2024*.
- Baevski, A., et al. (2022). data2vec: A general framework for self-supervised learning in speech, vision and language. *ICML 2022*.
- Ororbia, A. (2023). The predictive forward-forward algorithm. *arXiv:2301.01452*.
- Friston, K., et al. (2023). Designing ecosystems of intelligence from first principles. *Collective Intelligence*.
