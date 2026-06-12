# JEPA-PC: A Composite World Model Framework

> Combining JEPA-style representation learning with Predictive Coding inference  
> for robust, adaptive, uncertainty-aware world models.

**Type:** Master's research project  
**Scale:** CIFAR-10 / CIFAR-100, 4 GB GPU  
**Status:** Pre-implementation — this document is the specification

---

## Table of Contents

1. [The Core Idea](#1-the-core-idea)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [Why This Combination Works](#3-why-this-combination-works)
4. [Architecture](#4-architecture)
5. [Key Design Decisions](#5-key-design-decisions)
6. [Research Goals and Hypotheses](#6-research-goals-and-hypotheses)
7. [Implementation Plan](#7-implementation-plan)
8. [Evaluation Protocol](#8-evaluation-protocol)
9. [Hardware and Feasibility](#9-hardware-and-feasibility)
10. [Repository Structure](#10-repository-structure)
11. [Setup and Quickstart](#11-setup-and-quickstart)
12. [Future Work](#12-future-work)
13. [Glossary](#13-glossary)
14. [References](#14-references)

---

## 1. The Core Idea

Building a world model requires solving two distinct problems:

| Problem | Description | Best tool |
|---|---|---|
| **Representation** | Turn raw pixels into compact, meaningful features | JEPA |
| **Inference** | Refine beliefs about a specific input, with uncertainty | Predictive Coding |

Most architectures conflate these. JEPA solves representation in one forward pass and stops — it never asks *how confident am I about this particular image?* Predictive Coding (PC) can iterate until its belief settles and produce calibrated uncertainty estimates, but traditionally operates on raw pixels, wasting capacity on texture, lighting, and noise that carry no semantic content.

**The proposal:** train a JEPA encoder, freeze it, and use its output tokens as the input to a PC hierarchy. JEPA strips pixel-level noise; PC then does iterative reasoning in a clean, semantically meaningful space.

**One-line thesis:** JEPA solves representation; PC solves inference. The two are separable, and combining them yields capabilities neither has alone — inference-time adaptation, precision-weighted uncertainty, and (hypothesised) reduced catastrophic forgetting.

**On what is strictly necessary:** PC is the load-bearing component for every primary goal. It is the source of iterative inference, precision-weighted uncertainty, and local Hebbian weight updates. JEPA is a quality multiplier — it gives PC a lower-variance input space, making the inference loop converge faster and precision maps semantically interpretable rather than pixel-level noise. All three evaluations could in principle be run with PC on raw image patches and yield valid results. The composite is the stronger contribution because it adds the token-space vs. pixel-space comparison. If Stage 2 fails or is deprioritised, PC on raw CIFAR patches is an explicit fallback, not a defeat.

**Broader framing:** This two-tier design instantiates a principle from LeCun (2022): intelligent systems benefit from a fixed substrate encoding low-level world regularities (the frozen JEPA tokenizer), on top of which a learned hierarchical world model (the PC hierarchy) reasons at progressively more abstract scales — from object parts, to objects, to scenes.

> **Analogy.** JEPA is a translator who turns a noisy foreign broadcast into clean written sentences. PC is a careful reader who, given those sentences, re-reads and revises until the interpretation is internally consistent — and can mark the words they are unsure about. Previously the reader worked directly from static-filled audio. Now they work from the clean transcript.

---

## 2. Theoretical Foundations

### 2.1 JEPA — Joint Embedding Predictive Architecture

JEPA learns by predicting in feature space rather than pixel space. A portion of an image is masked; the visible portion is encoded; the model predicts the *features* (not the pixels) of the masked portion. Because pixel reconstruction is never required, the encoder is not pressured to represent texture, lighting, or colour — it learns abstract, semantic content.

**Anti-collapse mechanism.** Without intervention, the model can trivially minimise the loss by mapping every image to the same representation (collapse). JEPA prevents this with two devices:
- A **stop-gradient** on the target: the target encoder receives no gradient signal.
- An **EMA (exponentially moving average) target encoder**: a slowly-updating copy of the online encoder, whose weights are never directly optimised.

**Output.** A grid of token vectors, one per image patch. For CIFAR-10 with 4×4 patches: 8×8 = 64 tokens, each of dimension 128.

*Key references: LeCun (2022); Assran et al. (2023) — I-JEPA; Bardes et al. (2024) — V-JEPA.*

### 2.2 Predictive Coding — PC

PC is a computational model of cortical processing. Each layer generates a top-down prediction of the layer below it. The mismatch — the **prediction error** — propagates upward. The network iteratively adjusts its internal representations to minimise total prediction error before any weights are changed. This iterative settling is the **inference loop**.

Two properties make PC distinctive:

1. **Precision-weighting.** Each prediction error is scaled by the inverse of its variance (its *precision*). High precision means "trust this signal"; low precision means "this is noisy — suppress it." This is PC's native uncertainty mechanism and is central to the goals of this project.

2. **Local, Hebbian weight updates.** Weight changes depend only on signals locally available to each synapse — not on a global error signal propagated by backpropagation. This is the basis for the continual-learning hypothesis.

*Key references: Rao & Ballard (1999); Friston (2005); Bogacz (2017).*

### 2.3 Active Inference and Goal-Directed Precision — Friston

Standard PC is purely reactive: the inference loop runs until current-input prediction errors settle, with no goal signal. Friston's **Active Inference** framework introduces a critical extension — two distinct modes of prediction-error minimisation:

**Perceptual inference** — update internal representations to better match the incoming sensory signal. The world is fixed; the model adjusts. This is what standard PC implements.

**Active inference** — direct attention (or action) toward states that would reduce prediction error under a preferred outcome. This requires a **goal prior**: a distribution over preferred future states injected at the top of the hierarchy. Both modes minimise the same quantity — variational free energy — but through different channels.

**Precision-weighting is the bridge.** When a goal prior is engaged, the system actively upweights precision on errors in goal-relevant regions and downweights errors elsewhere. Attention, in this account, is precision-weighting under a goal — not a separate mechanism bolted on, but a direct consequence of the same inference process. This is why precision-weighting is treated as architecturally central in this project, not merely diagnostic.

**Mapping onto JEPA-PC.** The current composite implements perceptual inference. A goal prior can be injected at PC Level 3 as a prior over its representation; precision-weighting then propagates that goal downward. Level 3 saying "I expect a car" automatically amplifies errors at patches corresponding to car-relevant tokens (wheels, windows) and suppresses background. Three levels of goal-directed extension are defined in §8.3 (Evaluation 3) and §12 (Future Work).

> **In plain terms.** Standard PC asks: *what is true right now?* Active inference asks: *what do I want to be true, and how do I direct attention to get there?* Precision-weighting is how the second question shapes the answer to the first.

*Key references: Friston (2005); Friston (2010); Parr & Friston (2019).*

### 2.4 LeCun's Hierarchical World Models and Innate Priors

LeCun (2022) argues that intelligent systems require two components that are largely independent:

**Innate prior substrate.** Before learning object categories, causes, or effects, a system needs grounded models of basic physical regularities — object permanence, approximate gravity, parallax, surface continuity. These are either innate or learned so early from so much data that they are fixed by the time higher-level reasoning begins. They serve as the substrate on which all other learning is built, and critically, they are not overwritten when new concepts are learned.

**Hierarchical world models.** Reasoning does not happen at one level of abstraction. Object parts assemble into objects; objects assemble into scenes; scenes support causal and goal-directed reasoning. Each level models the world at a different spatial and temporal grain, with higher levels predicting over longer horizons.

**Mapping onto JEPA-PC.** The frozen JEPA tokenizer is the analogue of the innate prior substrate. Trained on natural image data and never updated afterward, it encodes basic visual regularities into a fixed representational space — exactly as LeCun's low-level models are not rewritten when new concepts are learned. The PC hierarchy is the analogue of the learned hierarchical world model: Level 1 models patch-level structure, Level 2 regions, Level 3 scenes and categories.

**Honest limitation.** LeCun's hierarchy is also *temporal* — higher levels predict over longer time horizons. The current design processes single images and has no temporal axis. This is appropriate for CIFAR-scale experiments. Extension to video (V-JEPA provides the natural tokenizer) would map PC levels onto prediction horizons as well as spatial scales. See §12.

---

## 3. Why This Combination Works

The central technical objection is: *PC was designed for simple layer-by-layer hierarchies — can it operate on the complex, non-linear feature space produced by a JEPA encoder?*

Two papers answer this directly and form the theoretical backbone of the composite design:

**Millidge, Tschantz & Buckley (2022)** show that PC converges — asymptotically, and quickly in practice — to the same gradients as backpropagation on *any* differentiable computation graph, including CNNs, RNNs, and LSTMs, using only local learning rules. This licenses building a PC hierarchy that is convolutional over the token grid rather than a plain stack of fully-connected layers.

**Salvatori et al. (2021)** show, via zero-divergence inference learning (Z-IL), that PC can *exactly* replicate backprop on convolutional and many-to-one recurrent networks — not just approximate it. This is direct grounding for the convolutional PC Level 1 design over the JEPA token grid.

**Honest caveat on continual learning.** Neither paper addresses forgetting — both are about gradient equivalence. The hypothesis that local PC updates reduce catastrophic forgetting is not established by this literature. It is a hypothesis to be tested in Evaluation 2, not an assumed result.

---

## 4. Architecture

```
                       Raw Input Image  (e.g. 32 × 32 × 3)
                                   │
                                   ▼
               ┌───────────────────────────────────────┐
               │           JEPA Tokenizer               │  PRETRAINED · FROZEN
               │  patch embed (4×4) → ViT context enc   │
               │  + EMA target encoder                  │  Encodes low-level visual
               └───────────────────────────────────────┘  regularities (LeCun's
                                   │                        "innate prior substrate").
                                   ▼
                    Token Grid  (8 × 8 × 128)
                    64 tokens · dim 128 · 2D layout PRESERVED · not flattened
                                   │
                                   ▼
      ┌────────────────────────────────────────────────────────┐
      │  PC Level 1  —  patch-level world model                 │
      │  Convolutional over the token grid                      │
      │  Local receptive fields · spatial priors intact         │
      │               ▲ prediction error                        │
      │               ▼ top-down prediction                     │
      └────────────────────────────────────────────────────────┘
                          │  spatial pooling
                          ▼
      ┌────────────────────────────────────────────────────────┐
      │  PC Level 2  —  region / object-part model              │
      │               ▲ prediction error                        │
      │               ▼ top-down prediction                     │
      └────────────────────────────────────────────────────────┘
                          │  spatial pooling
                          ▼
      ┌────────────────────────────────────────────────────────┐
      │  PC Level 3  —  scene / category model                  │
      │  ← goal prior injected here (active inference ext.)     │
      └────────────────────────────────────────────────────────┘

  At every level:
  · Precision-weighted errors flow UP
  · Top-down predictions flow DOWN
  · Representations iterate (inference loop) until total error settles
  · Precision weights are the mechanism by which goal priors (Level 3)
    shape attention at lower levels: high precision = attend; low = suppress

  JEPA tokenizer + PC hierarchy jointly instantiate LeCun's two-tier structure:
  fixed low-level priors below · learned hierarchical world models above
```

---

## 5. Key Design Decisions

These decisions separate a working project from a silently broken one. Each is here because getting it wrong invalidates results without obvious error messages.

### 5.1 Preserve the 2D Spatial Layout of Tokens

A ViT-based JEPA emits tokens that correspond to a spatial grid of image patches. Flattening them into a single vector before feeding PC Level 1 destroys the geometric priors PC needs for coherent top-down predictions.

**Decision:** keep tokens as an `(8, 8, 128)` grid. PC Level 1 is convolutional over this grid with local receptive fields, building patch → region → scene abstraction via spatial pooling. ViT positional embeddings are retained so each token's grid location is known. This is directly grounded in Millidge (2022) and Salvatori (2021).

### 5.2 Use Occlusion, Not Gaussian Noise (Evaluation 1)

JEPA encoders are trained to be invariant to pixel-level jitter. Gaussian noise is exactly the kind of perturbation they are designed to ignore — testing inference-time adaptation with it would show near-zero effect and prove nothing. Spatial occlusion (blacking out a contiguous block of patches) forces the PC hierarchy to predict the latent representations of hidden tokens from the visible ones — a task the inference loop can meaningfully help with.

### 5.3 The Tokenizer Stays Frozen — Always

`requires_grad=False` on the JEPA encoder from Stage 3 onward. If fine-tuned jointly with PC, the encoder quietly adapts to minimise PC's prediction error rather than learning general visual features, and the entire premise collapses.

**Scope consequence for Evaluation 2:** a frozen tokenizer can only represent visual content within its training distribution. Continual-learning claims are therefore scoped to datasets sharing the same visual distribution (CIFAR-100, all natural images). Claims must not be extended to distribution-shifted domains (e.g. medical scans) where the frozen encoder cannot encode new content. In LeCun's framing, this is natural: the innate substrate is not rewritten; only the higher-level models adapt.

### 5.4 Gate on Non-Collapse Before Stage 3

If the JEPA (Stage 2) collapses to near-identical representations, the PC hierarchy will appear to converge instantly while learning nothing — a silent failure. **Monitor embedding standard deviation every epoch in Stage 2. Do not begin Stage 3 until non-collapse is confirmed.**

### 5.5 Use the Correct PC Inference Update

Hold weights fixed during inference. Update each representation by gradient descent on the free energy:

```
r_i  ←  r_i  +  η · ( e_i  −  W_{i+1}ᵀ · e_{i+1} )
```

where `e_i` is the prediction error from the level below, `e_{i+1}` is the error from the level above back-projected through `W_{i+1}ᵀ`, and `η` is the inference step size. Each representation is pushed to better explain the level below while staying consistent with the level above. Weight updates (Hebbian, local) happen only *after* the inference loop has converged — never during. Follow Rao & Ballard (1999, eqs. 1–4) and Bogacz (2017) for the precise sign conventions.

### 5.6 Normalise Precision Within Each Level

Raw prediction-error variances can differ by orders of magnitude across levels. Normalise precision within each level before applying it as a weight, or training destabilises.

### 5.7 Require the Isolating Baseline in Evaluation 2

A plain MLP trained by backprop on top of the frozen JEPA is a required baseline for continual learning. Without it, any reduced forgetting observed in the PC composite is uninterpretable — it could be due to the local PC updates, or simply because frozen JEPA features are so well-separated that any classifier on top forgets less. The MLP baseline isolates these two effects.

---

## 6. Research Goals and Hypotheses

### 6.1 Primary Goals

1. Implement and validate a standalone PC network with characterised convergence behaviour.
2. Implement and validate a JEPA tokenizer with confirmed non-collapsed representations.
3. Build the composite: convolutional PC hierarchy over frozen JEPA tokens, 2D grid preserved.
4. Empirically test the three properties unique to the composite:
   - Does the PC inference loop improve representations over a single forward pass under occlusion?
   - Do local PC weight updates reduce catastrophic forgetting relative to backprop baselines?
   - Does precision-weighting produce spatially meaningful, interpretable uncertainty estimates?

### 6.2 Secondary Goals

- Characterise PC convergence speed and stability in token space vs. pixel space.
- Ablate the number of PC levels — does depth improve abstraction quality?
- Document failure modes: where does the composite break, and why?

### 6.3 What Would Falsify the Hypotheses

| Hypothesis | Falsifying outcome |
|---|---|
| Inference-time adaptation helps under occlusion | r₀ and rₙ yield identical linear-probe accuracy at all occlusion levels |
| PC reduces forgetting vs. backprop head | PC composite ≤ MLP-on-frozen-JEPA after 10 tasks |
| Precision maps are semantically meaningful | Low-precision tokens do not align with occluded/ambiguous regions |

Null results on any of these are publishable — they constrain the claim space for the composite design.

---

## 7. Implementation Plan

Build strictly in order. Each stage gates the next. Do not advance until every checklist item is confirmed.

---

### Stage 1 — Vanilla Predictive Coding on MNIST

**Goal:** implement PC from scratch and fully characterise the inference loop before any JEPA exists.  
**Dataset:** MNIST — trivially fast, no GPU required.  
**Why first:** the inference loop is subtle; mistakes here propagate silently into later stages.

**Architecture:**
```
Input (784,) → PC Level 1 (256,) → PC Level 2 (64,) → PC Level 3 (10,)
```

**What to implement** — `models/pc_layer.py`:
```python
class PCLayer(nn.Module):
    """
    Single PC layer. Maintains:
      W  — top-down prediction weights (frozen during inference, updated after)
      r  — this layer's current representation (updated every inference step)
      e  — prediction error, passed upward

    INFERENCE (weights frozen, repeat until convergence):
        prediction = W @ r_above
        e          = r_below - prediction
        r_above   += lr_infer * (e - W.T @ e_above)    # energy-minimisation

    LEARNING (only after inference has converged):
        dW = outer(e, r_above)                          # local, Hebbian
        W += lr_weights * dW
    """
```

**Verification checklist** — do not proceed to Stage 2 until all pass:
- [ ] Inference loop converges within ~20 iterations on a single image
- [ ] Prediction errors decrease (near-)monotonically during inference — plot this curve
- [ ] Classification accuracy ≥ 95% on MNIST test set
- [ ] Weight updates fire only after inference converges, never during
- [ ] Top-down reconstructions visualised at each level — coarser at higher levels

**Common failure modes:**

| Symptom | Cause | Fix |
|---|---|---|
| Oscillating inference | Inference lr too high | Reduce η |
| No learning | Inference never converges before weight update fires | Add explicit convergence criterion |
| Posterior collapse at high levels | Unconstrained representations | Add L2 regularisation on r |

**Deliverable:** `notebooks/01_vanilla_pc_mnist.ipynb`  
**References:** Rao & Ballard (1999, eqs. 1–4); Bogacz (2017) for update derivation.

---

### Stage 2 — JEPA Tokenizer on CIFAR-10

**Goal:** train a minimal JEPA tokenizer; master stop-gradient, EMA updates, and collapse detection.  
**Dataset:** CIFAR-10 — 32×32, 10 classes.  
**Why the bottleneck:** online encoder + EMA target encoder + predictor + backward activations must all fit in 4 GB simultaneously.

**Architecture:**
```
Image
  → patch embed (4×4 patches → 8×8 grid = 64 tokens, dim 128)
  → Context Encoder (small ViT, ~4 layers)
  → Predictor MLP
  → predicted target representations

Target encoder: EMA copy of context encoder — receives NO gradient
Loss:           cosine similarity in latent space (target is stop-gradient)
```

**What to implement** — `models/jepa_tokenizer.py` and `utils/ema.py`:
```python
def update_target_encoder(online_enc, target_enc, momentum=0.996):
    """EMA update. Target encoder NEVER receives a gradient."""
    for p_online, p_target in zip(online_enc.parameters(), target_enc.parameters()):
        p_target.data = momentum * p_target.data + (1 - momentum) * p_online.data

def jepa_loss(predicted, target):
    """Loss in latent space only — never pixel space."""
    predicted = F.normalize(predicted, dim=-1)
    target    = F.normalize(target,    dim=-1)
    return 1 - (predicted * target).sum(dim=-1).mean()
```

**Memory discipline (4 GB):**
- Wrap forward/backward in `torch.cuda.amp.autocast`
- Batch size 32–64
- Gradient accumulation to simulate larger effective batch sizes

**Verification checklist** — Stage 3 is gated on all of these:
- [ ] **No collapse:** embedding std stays well above zero — log every epoch; this is the hard gate
- [ ] Linear-probe accuracy ≥ 60% on CIFAR-10 test set
- [ ] Nearest neighbours in embedding space are semantically similar — visualise
- [ ] EMA confirmed working: target weights ≠ online weights but track them slowly

**Deliverable:** `notebooks/02_jepa_cifar10.ipynb`  
**Reference:** Assran et al. (2023), Section 3 and Appendix A.

---

### Stage 3 — PC on Frozen Patch Tokens

**Goal:** connect the PC hierarchy to the frozen JEPA tokenizer, verify behaviour improves vs. pixel-space PC, and confirm the 2D spatial layout is preserved end-to-end.  
**Dataset:** CIFAR-10, using the frozen Stage 2 tokenizer.  
**Novel contribution:** PC operating in a learned token space with a convolutional Level 1 is not previously reported at this scale.

**What changes from Stage 1:**

| | Stage 1 (pixel PC) | Stage 3 (token PC) |
|---|---|---|
| PC Level 1 input | Raw pixels `(784,)` | Token grid `(8, 8, 128)` |
| PC Level 1 type | Fully connected | **Convolutional** over token grid |
| Input variance | High (pixel noise) | Low (semantic tokens) |
| Spatial structure | Implicit | Explicit — grid layout preserved |

**What to implement** — `models/pc_conv.py`:
- `ConvPCLayer`: convolutional PC over the `(8, 8, 128)` token grid with local receptive fields
- Spatial pooling between levels to build patch → region → scene abstraction
- Positional embeddings preserved from the JEPA tokenizer

**Verification checklist:**
- [ ] PC inference converges in fewer iterations than pixel-space PC (expected: lower input variance)
- [ ] Linear-probe accuracy increases with PC level (expected: higher levels = more abstract)
- [ ] Prediction errors are high on occluded or ambiguous patches, low on clear ones
- [ ] Token grid position is recoverable inside PC Level 1 (confirm spatial layout intact)

**Deliverable:** `notebooks/03_pc_on_patch_tokens.ipynb`

---

### Stage 4 — Full Composite and Evaluation

**Goal:** assemble the complete JEPA-PC composite and run the three evaluations that test its unique properties.  
**Dataset:** CIFAR-10 (Evaluations 1, 3) and CIFAR-100 split into 10 sequential tasks of 10 classes each (Evaluation 2).

**What to implement** — `models/composite.py`:
- `JEPAPCComposite`: frozen JEPA tokenizer + 3-level convolutional PC hierarchy
- Evaluation harnesses in `experiments/` (see §8)

**Deliverables:** `notebooks/04_composite_evaluation.ipynb`, `models/composite.py`

---

## 8. Evaluation Protocol

### 8.1 Evaluation 1 — Inference-Time Adaptation Under Occlusion

**Hypothesis:** the iterative PC inference loop improves representations beyond a single forward pass, and the benefit grows with occlusion severity.

**Method:**

```python
def evaluate_inference_adaptation(model, images, occlusion_fn, n_steps=20):
    """
    1. Apply spatial occlusion (contiguous block masking, not Gaussian noise —
       JEPA encoders are invariant to pixel jitter; Gaussian noise proves nothing).
    2. Single forward pass  →  representation r_0
    3. PC inference loop    →  representation r_n  (after n_steps)
    4. Compare linear-probe accuracy: r_0 vs r_n
    Expected: r_n > r_0, gap widens with occlusion level.
    """
```

**Baselines:**
- Pure JEPA (no PC loop): flat across all inference steps — it has no loop
- PC on raw pixels: same protocol, different input space

**Output:** plot of linear-probe accuracy vs. inference steps, for occlusion levels {25%, 50%, 75%}.

---

### 8.2 Evaluation 2 — Continual Learning

**Hypothesis:** local Hebbian PC weight updates reduce catastrophic forgetting relative to backprop-trained heads.

**Protocol:**
- Split CIFAR-100 into 10 tasks × 10 classes
- Train sequentially: task 1 → task 2 → … → task 10
- After each task, evaluate on all previously seen tasks
- **Metric:** average accuracy across all seen tasks after the final task

**Baselines** — all on the same frozen JEPA tokenizer:

| Baseline | Expected behaviour | Purpose |
|---|---|---|
| JEPA + backprop head | High forgetting | Upper bound on forgetting |
| **PC composite (local updates)** | **Hypothesis: less forgetting** | The claim |
| Plain MLP head, backprop | Intermediate | **Isolates PC effect from frozen-feature effect** |
| EWC-regularised head | Less forgetting than backprop | Standard continual-learning comparison |

**Scope:** all tasks are natural images — the frozen tokenizer remains valid. Claims must not extend to distribution-shifted domains where the frozen tokenizer cannot encode new content.

---

### 8.3 Evaluation 3 — Precision-Weighted Uncertainty and Goal-Directed Inference

**Hypothesis:** precision maps are semantically meaningful in token space, and treating uncertainty reduction as an explicit inference goal allocates inferential effort efficiently.

**Part A — Passive precision maps (perceptual inference)**

Precision at each token = inverse variance of prediction error at that position.

- *Qualitative:* do high-uncertainty (low-precision) regions correspond to semantically informative patches?
- *Quantitative:* do low-precision tokens align spatially with occluded or corrupted regions?

This establishes that precision is semantically grounded before testing active use of it.

**Part B — Goal-directed inference (active inference, Level 1 extension)**

Replace fixed-step inference with uncertainty-threshold termination:

```python
def goal_directed_inference(model, tokens, uncertainty_threshold=0.1, max_steps=50):
    """
    Run PC inference until global uncertainty drops below threshold rather than
    for a fixed number of steps. The system has a preferred state (low uncertainty)
    and allocates more iterations to harder or more occluded inputs automatically.
    Step count at termination is itself a diagnostic metric.
    """
    for step in range(max_steps):
        errors, precisions = model.inference_step(tokens)
        uncertainty = (1.0 / precisions).mean()
        if uncertainty < uncertainty_threshold:
            break
    return model.representations, precisions, step
```

- *Does step count at convergence correlate with occlusion level and image ambiguity?*
- *Does threshold-terminated inference match or exceed fixed-step accuracy at lower average cost?*
- *Plot:* precision map evolution over inference steps — do uncertain tokens resolve as the loop runs?

A pure JEPA baseline is flat across both parts. The composite should show structured, spatially meaningful precision that evolves during inference.

---

## 9. Hardware and Feasibility

**Target:** 4 GB VRAM (laptop GPU or free-tier cloud, e.g. Colab T4).

| Stage | Dataset | ~Params | Runs on | Est. time | Notes |
|---|---|---|---|---|---|
| 1 — Vanilla PC | MNIST | 500K | CPU | 10–20 min | No GPU needed |
| 2 — JEPA tokenizer | CIFAR-10 | 5M | 4 GB GPU | 2–3 hr | **Memory bottleneck** |
| 3 — PC on tokens | CIFAR-10 | 6M | 4 GB GPU | 1–2 hr | Cheap: tokenizer frozen |
| 4 — Composite eval | CIFAR-100 | 7M | 4 GB GPU | 3–5 hr | Cheap: tokenizer frozen |

**Stage 2 memory strategy:**
- `torch.cuda.amp.autocast` — roughly halves activation memory
- Batch size 32–64
- Gradient accumulation to simulate larger effective batch without the memory cost

**Stages 3–4:** the frozen tokenizer means no backward graph is stored through it — only forward activations. PC layers are lightweight. Do not scale to ImageNet until Stage 4 is confirmed correct at CIFAR scale.

---

## 10. Repository Structure

```
jepa-pc-composite/
│
├── README.md                           ← this document
├── requirements.txt                    ← pinned dependencies
├── setup.md                            ← environment and dataset instructions
│
├── configs/                            ← all hyperparameters (no magic numbers in code)
│   ├── pc_mnist.yaml                   ← Stage 1
│   ├── jepa_cifar10.yaml               ← Stage 2 (AMP, batch size, grad-accum)
│   ├── pc_tokens_cifar10.yaml          ← Stage 3
│   └── composite_cifar100.yaml         ← Stage 4
│
├── models/                             ← reusable model code, imported by notebooks
│   ├── pc_layer.py                     ← PCLayer, PCNetwork — inference loop + Hebbian learning
│   ├── pc_conv.py                      ← ConvPCLayer — convolutional PC over token grid
│   ├── jepa_tokenizer.py               ← PatchEmbed, ContextEncoder, EMATargetEncoder, Predictor
│   └── composite.py                    ← JEPAPCComposite — frozen tokenizer + conv-PC hierarchy
│
├── experiments/                        ← evaluation logic, separate from model definitions
│   ├── linear_probe.py                 ← representation quality at each PC level
│   ├── inference_adaptation.py         ← occlusion + iterative inference (Eval 1)
│   ├── continual_learning.py           ← sequential tasks + forgetting metrics + baselines (Eval 2)
│   └── precision_uncertainty.py        ← precision maps + goal-directed inference (Eval 3)
│
├── notebooks/                          ← one notebook per stage (narrative + results)
│   ├── 01_vanilla_pc_mnist.ipynb
│   ├── 02_jepa_cifar10.ipynb
│   ├── 03_pc_on_patch_tokens.ipynb
│   └── 04_composite_evaluation.ipynb
│
├── utils/
│   ├── data.py                         ← dataset loading, CIFAR-100 task splits, occlusion fns
│   ├── ema.py                          ← EMA target-encoder update
│   ├── metrics.py                      ← accuracy, forgetting, convergence, variance/collapse
│   └── viz.py                          ← reconstructions, nearest-neighbour, precision-map plots
│
├── tests/                              ← correctness tests — run before trusting any result
│   ├── test_pc_convergence.py          ← errors decrease monotonically; loop converges
│   ├── test_jepa_no_collapse.py        ← embedding variance stays above threshold
│   └── test_spatial_layout.py         ← token grid position preserved into PC Level 1
│
├── checkpoints/
│   └── .gitkeep                        ← frozen Stage 2 tokenizer saved here
│
└── results/
    └── .gitkeep                        ← logged metrics, figures, tables for writeup
```

---

## 11. Setup and Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Verify GPU and AMP (required for Stage 2+; CPU is fine for Stage 1)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 3. Run stages in order — each gates the next
jupyter notebook notebooks/01_vanilla_pc_mnist.ipynb       # Stage 1: verify PC works
jupyter notebook notebooks/02_jepa_cifar10.ipynb           # Stage 2: train tokenizer, confirm no collapse
jupyter notebook notebooks/03_pc_on_patch_tokens.ipynb     # Stage 3: PC on frozen tokens, grid preserved
jupyter notebook notebooks/04_composite_evaluation.ipynb   # Stage 4: three evaluations
```

**Golden rules:**
1. Complete every checklist item before advancing to the next stage.
2. The JEPA tokenizer is frozen from Stage 3 onward — never fine-tune it alongside PC.
3. Confirm non-collapse in Stage 2 before trusting any downstream result.
4. Keep tokens on their 2D grid — PC Level 1 is convolutional, not fully connected.

---

## 12. Future Work

The following extensions follow directly from the composite design and are documented here to ensure current architectural decisions do not foreclose them.

**Level-specific precision priors (Active Inference Level 2).** Add a learned top-down precision pathway alongside the existing prediction pathway at each level boundary. Level 3's representation projects a precision-scaling signal to Levels 2 and 1, amplifying errors in goal-relevant regions. This adds one linear projection per level boundary and is the mechanistic account of spatial attention in Friston's framework.

**Explicit goal states and temporal inference (Active Inference Level 3).** Inject a target representation at PC Level 3 encoding a desired future state. Run inference to minimise the gap between current and desired state rather than to explain the current input. This requires a learned temporal transition model over token grids and is only tractable with sequential data. V-JEPA (Bardes et al., 2024) provides a natural tokenizer: its tokens are already temporal, and PC Level 3's goal state becomes a prediction about a future token grid.

**Temporal hierarchy.** The current PC hierarchy is spatial only (patch → region → scene). LeCun's full hierarchical world model is also temporal — higher levels predict over longer horizons. With video tokens, PC levels would map onto both spatial scales and prediction horizons simultaneously.

---

## 13. Glossary

| Term | Definition |
|---|---|
| **Token** | One vector summarising one image patch; the JEPA's output unit. |
| **Tokenizer** | The frozen JEPA encoder that converts an image into a grid of tokens. |
| **Inference loop** | PC's iterative settling of representations on a single input, weights fixed. |
| **Prediction error** | Mismatch between a top-down prediction and actual lower-level activity. |
| **Precision** | Inverse variance of a prediction error; PC's confidence weight. |
| **Precision-weighting** | Scaling each prediction error by its precision before updating representations. High precision = attend; low = suppress. The mechanism by which uncertainty, attention, and goal-directed inference are all implemented — not a diagnostic add-on. |
| **Perceptual inference** | Minimising prediction error by updating internal representations to match the world. Standard PC. The world is fixed; the model adjusts. |
| **Active inference** | Minimising prediction error by directing attention or action toward preferred states (Friston). Requires a goal prior. |
| **Goal prior** | A distribution over preferred future states injected at PC Level 3. Propagates downward via precision-weighting to shape which errors the inference loop prioritises. |
| **Stop-gradient / EMA target** | JEPA's anti-collapse mechanism. The target encoder is a slowly-updating copy that receives no gradient. |
| **Collapse** | All representations become near-identical; trivially minimises the loss but learns nothing. |
| **Local / Hebbian update** | A weight change computed only from locally available signals — no global backprop. |
| **Catastrophic forgetting** | Losing performance on old tasks after training on new ones. |
| **Innate prior substrate** | LeCun's term for fixed low-level world models (object permanence, gravity, basic geometry). In this project: the frozen JEPA tokenizer. |
| **Hierarchical world model** | A stack of models at different scales of abstraction and (in temporal settings) prediction horizons. In this project: the PC hierarchy. |

---

## 14. References

Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience, 2*(1), 79–87.

Friston, K. (2005). A theory of cortical responses. *Philosophical Transactions of the Royal Society B, 360*(1456), 815–836.

Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience, 11*(2), 127–138. — Canonical statement of Active Inference; source of the perceptual vs. active inference distinction and precision-weighting as the mechanism of attention.

Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference. *Biological Cybernetics, 113*(5–6), 495–513. — Formalises goal priors and precision control; directly relevant to Level 2 and Level 3 extensions.

Whittington, J. C. R., & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. *Neural Computation, 29*(5), 1229–1262.

Bogacz, R. (2017). A tutorial on the free-energy framework for modelling perception and learning. *Journal of Mathematical Psychology, 76*, 198–211. — Recommended for the precise inference and learning update equations.

Millidge, B., Tschantz, A., & Buckley, C. L. (2022). Predictive coding approximates backprop along arbitrary computation graphs. *Neural Computation, 34*(6), 1329–1368. (arXiv:2006.04182.) — PC matches backprop gradients on arbitrary differentiable graphs with local rules; licenses structured/convolutional PC.

Salvatori, T., Song, Y., Lukasiewicz, T., Bogacz, R., & Xu, Z. (2021). Predictive coding can do exact backpropagation on convolutional and recurrent neural networks. arXiv:2103.03725. — PC exactly replicates backprop on convolutional networks; direct grounding for the convolutional PC Level 1 design.

LeCun, Y. (2022). A path towards autonomous machine intelligence. OpenReview preprint. — Source of the two-tier architecture (innate prior substrate + learned hierarchical world models) that this project instantiates.

Assran, M., et al. (2023). Self-supervised learning from images with a joint-embedding predictive architecture (I-JEPA). *CVPR 2023.*

Bardes, A., et al. (2024). V-JEPA: revisiting feature prediction for learning visual representations from video. *NeurIPS 2024.*

Baevski, A., et al. (2022). data2vec: a general framework for self-supervised learning in speech, vision and language. *ICML 2022.*

Ororbia, A. (2023). The predictive forward-forward algorithm. arXiv:2301.01452.

Friston, K., et al. (2023). Designing ecosystems of intelligence from first principles. *Collective Intelligence.*
