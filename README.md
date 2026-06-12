# JEPA-PC: A Composite World Model Framework

    Combining JEPA-style representation learning with Predictive Coding inference for robust, adaptive world models.

This is a master's-level research project. It builds and tests a new architecture that bolts two well-studied ideas together: one that is very good at learning what things mean (JEPA), and one that is very good at reasoning about a specific input in the moment (Predictive Coding). The bet is that each fixes the other's main weakness.

## Read this first: the whole project in one minute

There are two separate problems in building a "world model":

    The representation problem — turning raw pixels into compact, meaningful features. JEPA is excellent at this.
    The inference problem — given those features, refining your belief about this exact input right now, including saying how confident you are. Predictive Coding (PC) is excellent at this.

Today these are usually done by the same network in one shot. JEPA looks at an image once and stops — it never asks "how wrong might I be about this particular image?" PC can ask that question and iterate until its answer settles, but it traditionally does so on raw pixels, which wastes enormous effort modelling texture, lighting, and noise.

The idea of this project: freeze a trained JEPA encoder and use its output tokens as the input to a PC hierarchy. JEPA strips away the irrelevant pixel noise; PC then does its iterative reasoning in a clean, meaningful space.

One-line thesis: JEPA solves representation; PC solves inference; the two are separable, and combining them yields abilities neither has alone — namely inference-time adaptation, calibrated uncertainty estimates via precision-weighting, and (hypothesised) reduced forgetting. Precision-weighting is not an add-on: it is the mechanism by which the system allocates inferential effort, and in goal-directed extensions it is the mechanism by which intent shapes perception.

This two-tier design also instantiates a broader principle from LeCun (2022): intelligent systems benefit from a fixed substrate encoding low-level world regularities (the frozen JEPA tokenizer), on top of which a learned hierarchical world model (the PC hierarchy) reasons at progressively more abstract scales — from object parts, to objects, to scenes.

    Analogy. JEPA is like a translator who turns a noisy foreign broadcast into clean written sentences. PC is like a careful reader who, given those sentences, re-reads and revises their understanding until it is internally consistent — and can point to the words they're unsure about. Previously the reader had to work directly from the static-filled audio. Now they work from the clean transcript.

A note on what is strictly necessary: PC is the load-bearing component for every primary goal — it is the source of iterative inference, precision-weighted uncertainty, and local Hebbian updates. JEPA is a quality multiplier: it gives PC a cleaner, lower-variance input space, making the inference loop converge faster and the precision maps semantically interpretable rather than pixel-level noise. In principle, all three evaluations could be run with PC operating on raw image patches and yield valid results. The composite is the stronger contribution because it adds the token-space vs. pixel-space comparison and makes precision maps meaningful at the level of semantic content — but if Stage 2 fails or is deprioritised, PC on raw CIFAR patches is an explicit fallback, not a defeat.

---

## Background: the two halves, explained simply

### JEPA (Joint Embedding Predictive Architecture)

JEPA learns by predicting in feature space instead of pixel space. You hide part of an image, encode the visible part, and train the model to predict the features (not the pixels) of the hidden part. Because it never has to redraw pixels, it doesn't waste capacity on texture or lighting — it learns abstract, semantic features.

    It uses a stop-gradient and an EMA (exponentially moving average) target encoder to avoid a failure called collapse, where the model cheats by making every feature identical (which trivially minimises the loss but learns nothing).
    Key references: LeCun (2022); I-JEPA (Assran et al., 2023); V-JEPA (Bardes et al., 2024).

    In plain terms: JEPA gives us a frozen "tokenizer" that converts an image into a small grid of meaningful vectors ("tokens"). We never change it after training.

### Predictive Coding (PC)

PC is a model of how the cortex might process information. Each layer predicts the activity of the layer below it, top-down. The mismatch — the prediction error — is sent back up. The network then iteratively adjusts its internal representations to minimise total prediction error before it updates any weights.

    This iterative settling is the inference loop. It is what lets PC refine a belief about one specific input.
    Each error can be weighted by precision (inverse variance) — high precision means "I'm confident about this signal," low precision means "ignore this, it's noisy." This is PC's built-in uncertainty mechanism.
    Crucially, PC's weight updates are local and Hebbian (each weight changes based only on signals it can locally see), not global backprop.
    Key references: Rao & Ballard (1999); Friston (2005).

    In plain terms: PC doesn't answer in one forward pass. It guesses, measures how wrong it is, revises, and repeats until the guess stops changing. That loop is the thing JEPA is missing.

### Active Inference and goal-directed precision (Friston)

Standard PC as described by Rao & Ballard is purely reactive: top-down predictions are driven by whatever the highest level has inferred from the current input, and the inference loop runs until prediction errors settle. Friston's **Active Inference** framework extends this in a critical direction — it distinguishes two modes of minimising prediction error that the brain uses simultaneously:

**Perceptual inference** — update internal representations to better match the incoming sensory signal. This is standard PC. The world is fixed; the model adjusts.

**Active inference** — change the world (or where attention is directed) to match a desired internal state. This requires encoding a *goal prior*: a probability distribution over preferred future states. Prediction error now measures not just "how wrong am I about what's there?" but "how far is the current state from where I want to be?" Both perceptual and active inference minimise the same quantity — variational free energy — but through different channels.

**Precision-weighting is the bridge between these two modes.** In Friston's framework, precision is not just a passive measure of signal reliability — it is an *actively controlled* quantity. When a goal prior is engaged, the system upweights the precision of prediction errors in goal-relevant regions (directing attention there) and downweights errors in irrelevant regions (suppressing them). Attention, in this account, is precision-weighting under a goal. This is why precision is central to this project and not merely a diagnostic: it is the mechanism by which intent shapes what gets processed.

**How this maps onto JEPA-PC:**

The current composite implements perceptual inference only. The PC inference loop runs for a fixed number of steps (or until errors settle) with no goal signal — all prediction errors are weighted by learned, data-driven precision. This is already valuable and is the main object of study.

Goal-directed inference is a natural extension that does not require architectural surgery. The key insight is that a goal prior can be injected at PC Level 3 as a *prior distribution over its representation*, and precision-weighting then propagates that goal downward through the hierarchy. Level 3 saying "I expect this to be a car" automatically upweights precision on the token-level errors corresponding to car-relevant patches (wheels, windows) and downweights errors on background tokens. The inference loop becomes attentional.

Three levels of goal-directed extension are possible in this project, in increasing order of complexity:

**Level 1 — uncertainty reduction as an intrinsic goal (fits current scope).** Treat minimising total weighted prediction error as an explicit termination criterion rather than running for a fixed number of steps. The inference loop runs until global uncertainty (summed low-precision mass across all token positions) drops below a threshold, allocating more iterations to harder or more occluded images automatically. This is active inference in the weak sense: the system has a preferred state (low uncertainty) and stops when it achieves it.

```python
def goal_directed_inference(model, tokens, uncertainty_threshold=0.1, max_steps=50):
    """
    Run PC inference until global uncertainty drops below threshold.
    High-uncertainty tokens are automatically upweighted by low precision —
    the loop allocates inferential effort where it is most needed.
    Replaces fixed-step inference in Evaluation 3.
    """
    for step in range(max_steps):
        errors, precisions = model.inference_step(tokens)
        uncertainty = (1.0 / precisions).mean()
        if uncertainty < uncertainty_threshold:
            break
    return model.representations, precisions, step  # step count is itself a metric
```

This also produces a new evaluation metric: number of inference steps required per image, which should correlate with occlusion level and image ambiguity. Compare against fixed-step inference from Evaluation 1.

**Level 2 — level-specific precision priors (moderate extension).** Each PC level has a different goal: Level 1 resolves patch ambiguity, Level 2 resolves region identity, Level 3 resolves scene category. Rather than a single global uncertainty criterion, Level 3's inferred representation feeds a learned precision-scaling signal down to Levels 2 and 1, amplifying errors in goal-relevant spatial regions. This adds one top-down precision pathway per level boundary — a learned linear projection alongside the existing prediction pathway. This is the mechanistic account of visual spatial attention in Active Inference.

**Level 3 — explicit goal states (future work; requires temporal data).** The full Active Inference picture: inject a target representation at Level 3 encoding a *desired* future state, and run inference to minimise the gap between current and desired. This requires a temporal model (what transitions move the token grid toward the goal state), which is why it is out of scope for static CIFAR images. V-JEPA, operating on video tokens, would make this tractable — Level 3's goal state becomes a prediction about a future token grid, and the inference loop discovers what intermediate states are needed.

Key references: Friston (2005); Friston (2010); Parr & Friston (2019).

    In plain terms: standard PC asks "what is true right now?" Active inference asks "what do I want to be true, and how do I direct my attention to get there?" Precision-weighting is how the second question shapes the answer to the first.

LeCun (2022) argues that intelligent systems need two things that are largely independent:

**1. Low-level innate priors.** Before learning causes, effects, or object categories, a system needs grounded models of basic physical regularities — things like object permanence, approximate gravity, parallax, and surface continuity. These are either innate or learned so early and from so much data that they are effectively fixed by the time higher-level reasoning begins. They serve as the substrate on which everything else is built.

**2. A hierarchy of world models at different scales.** Reasoning doesn't happen at one level of abstraction. Object parts assemble into objects; objects assemble into scenes; scenes support causal and goal-directed reasoning. Each level of this hierarchy models the world at a different spatial (and temporal) grain, with higher levels predicting over longer horizons and more abstract entities.

**How these map onto JEPA-PC:**

The frozen JEPA tokenizer is the analogue of the innate prior substrate. Trained on large amounts of natural image data, it encodes basic visual regularities — patch geometry, local texture invariances, rough depth cues from parallax — into a fixed representational space. It is not updated after Stage 2, exactly as LeCun's low-level models are not rewritten every time a new concept is learned.

The PC hierarchy above it is the analogue of the learned hierarchical world model. PC Level 1 models patch-level features; Level 2 models regions; Level 3 models scenes or object categories. Top-down predictions flow from abstract to concrete; prediction errors flow upward. This is precisely the multi-scale, bidirectional prediction structure LeCun describes.

**One honest limitation of the current design:** LeCun's hierarchy is also *temporal* — higher levels predict over longer time horizons (next frame vs. next second vs. next minute). The current JEPA-PC composite processes single images and has no temporal axis. This is appropriate for CIFAR-scale experiments. If extended to video (V-JEPA provides the natural tokenizer), the PC levels would map onto prediction horizons as well as spatial scales — a straightforward architectural extension, noted here for future work.

---

## Why this is theoretically allowed to work

A reasonable objection is: "PC was designed for simple layer-by-layer hierarchies. Can it really operate on the complex, structured, non-linear feature space that a JEPA produces?"

Two papers say yes, and they are the theoretical backbone of this project:

    Millidge, Tschantz & Buckley (2022), "Predictive Coding Approximates Backprop along Arbitrary Computation Graphs." They show PC converges (asymptotically, and quickly in practice) to the same gradients as backpropagation on any differentiable computation graph — including CNNs, RNNs, and LSTMs with branching and multiplicative structure — using only local learning rules. Why it matters here: it licenses building a PC hierarchy that is structured (e.g. convolutional over a token grid) rather than a plain stack of fully-connected layers. PC is not limited to toy linear hierarchies.

    Salvatori, Song, Lukasiewicz, Bogacz & Xu (2021), "Predictive Coding Can Do Exact Backpropagation on Convolutional and Recurrent Neural Networks." Building on zero-divergence inference learning (Z-IL), they show PC can exactly replicate backprop's behaviour — not just approximate it — on convolutional and (many-to-one) recurrent networks. Why it matters here: the central design fix in this project is to make the lowest PC level convolutional over the 2D grid of JEPA tokens (see "Spatial grounding" below). This paper is direct evidence that a convolutional PC network is a sound, trainable object with local updates.

    Honest caveat on continual learning. The continual-learning hypothesis (that local PC updates forget less than backprop) is not established by these two papers — they are about gradient/accuracy equivalence, not forgetting. Treat reduced forgetting as our hypothesis to test in Evaluation 2, not as a settled result. Evidence that PC-style local updates can reduce interference exists in the broader literature, but we will measure it ourselves rather than assume it.

---

## Key design decisions (read before writing code)

These are the decisions that separate a working project from a broken one. Several come from a careful critique of the original plan; each is here because getting it wrong silently invalidates the results.

### 1. Preserve the 2D spatial layout of tokens — do not flatten or shuffle

A ViT-based JEPA emits tokens as an unordered-looking set, but they correspond to a grid of image patches (for CIFAR-10 with 4×4 patches: an 8×8 grid of 64 tokens, each of dimension 128). PC relies on spatial structure — local features assembling into global features. If you flatten the tokens into one long vector and feed a fully-connected PC layer, you destroy the geometric priors PC needs for sensible top-down predictions.

Decision: keep tokens arranged as an (8, 8, 128) grid. Make PC Level 1 convolutional (local receptive fields over the token grid), and have higher PC levels pool spatially to build patch → region → scene abstraction — exactly the CNN-style hierarchy that Millidge (2022) and Salvatori (2021) prove PC can implement with local updates. Retain the ViT positional information so each token's location is known.

    This is the concrete answer to "how do the flattened tokens connect to PC Level 1": they aren't flattened. They stay on a grid and connect through convolutional/locally-connected PC weights.

### 2. Corrupt with occlusion / masking, not Gaussian noise (Evaluation 1)

JEPA encoders act as low-pass filters: they were trained to be invariant to exactly the kind of pixel jitter that Gaussian noise adds, so they largely ignore it. Testing inference-time adaptation with Gaussian noise would show almost no effect and prove nothing. Instead, black out spatial regions (e.g. occlude half the image, or mask a block of patches). Then the PC hierarchy must use the visible tokens to generate top-down predictions that reconstruct the latents of the occluded tokens — a task the inference loop can actually help with.

### 3. The tokenizer stays frozen — and this scopes the continual-learning claim

The JEPA tokenizer must be frozen (requires_grad=False) for both Stage 3 and Stage 4. If you fine-tune it jointly with PC, it will quietly adapt to minimise PC's prediction error instead of learning general features, and the whole premise collapses.

Consequence for Evaluation 2: because the tokenizer is frozen, it can only represent visual primitives it already learned. If a new continual-learning task introduces genuinely novel visual content (e.g. switching from natural images to medical scans), the frozen tokenizer cannot encode it and PC gets no useful signal. Therefore we scope continual-learning claims to datasets that share the same visual distribution (CIFAR-100 tasks, all natural images). State this limitation explicitly in the writeup; do not over-claim.

In LeCun's framing, this constraint is natural: the innate prior substrate is not rewritten when new concepts are learned. It is only the higher-level world models (the PC hierarchy) that adapt.

### 4. Detect representation collapse before trusting Stage 3

If Stage 2's JEPA collapses to low-variance (near-identical) representations, Stage 3's PC will appear to "converge instantly" while learning nothing. This is a silent failure. Gate progression: monitor embedding standard deviation every epoch in Stage 2, and do not start Stage 3 until non-collapse is confirmed (see Stage 2 checklist).

### 5. Use the precise PC inference update (energy minimisation)

Hold weights fixed during inference and move each representation down the gradient of the free energy (the sum of precision-weighted squared prediction errors). Using the critique's convention, the update for the representation at level i is:

r_i  ←  r_i  +  η · ( e_i  −  W_{i+1}ᵀ · e_{i+1} )

where:

    e_i is the prediction error arriving from the level below (how badly this level's prediction explained the level below),
    e_{i+1} is the prediction error from the level above (how badly the level above predicted this level), back-projected through W_{i+1}ᵀ,
    η is the inference step size (learning rate of the inference loop).

In words: each representation is pushed to better explain the level below it while staying close to what the level above predicts for it. Only after this loop has settled do you update the weights (Hebbian, local). The exact sign/index convention depends on the formulation — follow Rao & Ballard (1999, eqs. 1–4) and the Bogacz (2017) tutorial consistently.

### 6. Normalise precision weights

Raw prediction-error variances can differ by several orders of magnitude across levels. Normalise precision within each level, or training destabilises.

### 7. Add an isolating baseline to Evaluation 2

A simple MLP trained by backprop on top of the frozen JEPA is a required baseline. It tells you whether any reduced forgetting is due to PC specifically, or merely because frozen JEPA features are robust enough that any classifier resists forgetting. Without this control, a positive result is uninterpretable.

---

## Architecture

                         Raw Input image (e.g. 32×32×3)
                                     │
                                     ▼
                 ┌─────────────────────────────────────┐
                 │          JEPA Tokenizer              │  ← Pretrained, FROZEN.
                 │   patch embed (4×4) → ViT context    │    Encodes low-level visual
                 │   encoder + EMA target encoder       │    regularities (the "innate
                 └─────────────────────────────────────┘    prior substrate" in LeCun's
                                     │                       sense). Outputs stable
                                     │                       semantic tokens.
                                     ▼
                      Token grid: (8 × 8 × 128)              ← 2D layout PRESERVED.
                      (64 patch tokens, dim 128)                NOT flattened.
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │  PC Level 1   — CONVOLUTIONAL over the token grid         │  ← patch-level world model
        │  predicts token-level features; local receptive fields    │
        │  top-down prediction ↓ , prediction error ↑               │
        └─────────────────────────────────────────────────────────┘
                                     │  (spatial pooling)
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │  PC Level 2   — region-level summaries                    │  ← object-level world model
        │  top-down prediction ↓ , prediction error ↑               │
        └─────────────────────────────────────────────────────────┘
                                     │  (spatial pooling)
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │  PC Level 3   — scene / object-level state (most abstract)│  ← scene/causal world model
        │                                                            │    goal prior injected here
        │                                                            │    (active inference ext.)
        └─────────────────────────────────────────────────────────┘

   At every level: precision-weighted errors flow UP, predictions flow DOWN,
   and representations iterate (the inference loop) until total error settles.
   Precision weights are the mechanism by which goal priors (injected at Level 3)
   shape attention at lower levels — high precision = attend here; low = suppress.

   The tokenizer + PC hierarchy jointly instantiate LeCun's two-tier structure:
   fixed low-level priors below, learned hierarchical world models above.

---

## Goals

### Primary goals

    Implement and validate a standalone PC network on a toy dataset, with well-characterised convergence behaviour.
    Implement and validate a standalone JEPA tokenizer with confirmed non-collapsed representations.
    Build the composite: a convolutional PC hierarchy operating on frozen JEPA token representations, with the 2D grid preserved.
    Empirically test the composite's unique claims — properties only the composite should exhibit:
        Does inference-time adaptation (the PC loop) improve representations over a single forward pass, especially under occlusion?
        Does the composite forget less under continual learning than a JEPA + backprop baseline and than a plain MLP-on-frozen-JEPA baseline?
        Does precision-weighting produce interpretable uncertainty estimates?

### Secondary goals

    Characterise convergence of PC inference in token space vs. pixel space.
    Ablate the number of PC levels — does a deeper hierarchy improve abstraction quality?
    Document failure modes — where does the composite break, and why?

---

## Hardware and feasibility (target: a 4 GB GPU)

Everything is sized to run on a laptop CPU or a small/free GPU. The binding constraint is 4 GB of VRAM; the plan below respects it.

Stage 	Dataset 	Approx. model size 	Where it runs 	Est. time 	Notes
1 — Vanilla PC 	MNIST 	~500K params 	CPU 	10–20 min 	Trivial.
2 — JEPA tokenizer 	CIFAR-10 	~5M params 	4 GB GPU / T4 	2–3 hr 	The bottleneck.
3 — PC on tokens 	CIFAR-10 	~6M params 	4 GB GPU / T4 	1–2 hr 	Cheap (tokenizer frozen).
4 — Composite eval 	CIFAR-100 	~7M params 	4 GB GPU / T4 	3–5 hr 	Cheap (tokenizer frozen).

Stage 2 is the tight one because you must hold the online encoder, the predictor, and the EMA target encoder in memory at once, plus backward-pass activations. Survival rules:

    Use Automatic Mixed Precision (torch.cuda.amp) — roughly halves activation memory.
    Keep batch size small (32 or 64).
    Use gradient accumulation to simulate the large effective batch sizes self-supervised learning likes, without the memory cost.

Stages 3 and 4 are easy on 4 GB: the JEPA is frozen, so you only store its forward activations (no backward graph), and the PC layers are lightweight. Do not scale to ImageNet until Stage 4 is verified correct — if the architectural claims fail at CIFAR scale, scaling will not rescue them.

---

## Development stages

Build strictly in order. Each stage has a goal, what to implement, a verification checklist (do not advance until every box is ticked), common failure modes, and a deliverable. Verify correctness at small scale before adding any complexity.

### Stage 1 — Vanilla Predictive Coding on MNIST

Goal: implement PC from scratch and understand the inference loop, its convergence, and its failure modes — before any JEPA exists.

Dataset: MNIST (tiny, fast, instantly downloadable).

Architecture:

Input (784,) → PC Level 1 (256,) → PC Level 2 (64,) → PC Level 3 (10,)

What to implement:

class PCLayer(nn.Module):
    """
    A single PC layer. It maintains:
      - W : top-down prediction weights (fixed during inference, updated after)
      - r : this layer's current representation (updated every inference step)
      - e : prediction error, passed upward

    INFERENCE STEP (weights frozen, repeat until r stops changing):
        prediction = W @ r_above              # top-down guess for the layer below
        e          = r_below - prediction     # how wrong the guess was
        r_above   += lr_infer * (e - W.T @ e_above)   # energy-minimisation update

    LEARNING STEP (only AFTER the inference loop has converged):
        dW = outer(e, r_above)                # local, Hebbian
        W += lr_weights * dW
    """

Verification checklist:

    Inference loop converges within ~20 iterations on a single image.
    Prediction errors decrease (near-)monotonically during inference — plot this.
    Classification accuracy ≥ 95% on the MNIST test set.
    Weight updates happen only after inference converges, never during.
    Top-down reconstructions visualised at each level — they should get coarser at higher levels.

Common failure modes:

    Oscillating inference → inference learning rate too high; reduce the step size.
    No learning → inference never converges before the weight update fires; add an explicit convergence criterion.
    Posterior collapse at higher levels → add L2 regularisation on the representations.

Deliverable: notebooks/01_vanilla_pc_mnist.ipynb References: Rao & Ballard (1999, eqs. 1–4); Whittington & Bogacz (2017) for the local learning-rule derivation.

### Stage 2 — JEPA Tokenizer on CIFAR-10

Goal: train a minimal JEPA tokenizer and master stop-gradient, EMA updates, and collapse detection.

Dataset: CIFAR-10 (32×32, 10 classes, fast at small scale).

Architecture:

Image → patch embed (4×4 patches → 8×8 = 64 tokens, dim 128)
      → Context Encoder (small ViT, ~4 layers)
      → Predictor MLP
      → predicted target representations

Target encoder : an EMA copy of the context encoder. Receives NO gradient.
Loss           : cosine similarity in latent space (target is stop-gradient).

What to implement:

# EMA update — the stability mechanism. The target encoder NEVER gets a gradient.
def update_target_encoder(online_enc, target_enc, momentum=0.996):
    for p_online, p_target in zip(online_enc.parameters(), target_enc.parameters()):
        p_target.data = momentum * p_target.data + (1 - momentum) * p_online.data

# Loss — computed in latent space ONLY, never in pixel space.
def jepa_loss(predicted, target):
    predicted = F.normalize(predicted, dim=-1)
    target    = F.normalize(target,    dim=-1)
    return 1 - (predicted * target).sum(dim=-1).mean()

    Memory discipline (4 GB): wrap the forward/backward in torch.cuda.amp.autocast, use batch size 32–64, and accumulate gradients over several steps.

Verification checklist:

    No collapse: embedding standard deviation stays well above zero — log std every epoch. This is the gate for Stage 3; do not proceed if collapsed.
    Linear-probe accuracy ≥ 60% on CIFAR-10 (reasonable for a small from-scratch model).
    Nearest neighbours in embedding space are semantically similar (visualise them).
    EMA is working: target weights ≠ online weights, but track them slowly.

Deliverable: notebooks/02_jepa_cifar10.ipynb Reference: Assran et al. (2023) — read Section 3 (architecture) and Appendix A (hyperparameters) closely.

### Stage 3 — PC on Patch Tokens (partial composite)

Goal: replace PC's pixel inputs with frozen JEPA tokens, keeping the 2D grid, and confirm PC behaves better in token space than in pixel space. This is already a small novel contribution.

Dataset: CIFAR-10, using the frozen Stage-2 tokenizer.

What changes from Stage 1:

    PC Level 1 input is now a grid of token representations (8, 8, 128), not raw pixels.
    PC Level 1 is convolutional over that grid (local receptive fields) so spatial priors survive — grounded in Millidge (2022) and Salvatori (2021).
    The PC hierarchy runs entirely in latent space across 3 levels.
    Compare convergence speed and stability against the pixel-space PC from Stage 1.

Verification checklist:

    PC inference converges in fewer iterations than pixel-space PC (expected — tokens are lower-variance).
    Linear-probe accuracy at each PC level increases with level (expected).
    Prediction errors are semantically interpretable — high error on occluded or ambiguous patches.
    Spatial layout confirmed intact (a token's grid position is recoverable inside PC Level 1).

Deliverable: notebooks/03_pc_on_patch_tokens.ipynb

### Stage 4 — Full Composite + Evaluation

Goal: assemble the full composite and run the three evaluations that test its unique properties.

Dataset: CIFAR-10 / CIFAR-100 (CIFAR-100 split into tasks for continual learning).

#### Evaluation 1 — Inference-time adaptation (use occlusion, not Gaussian noise)

Tests whether the iterative PC loop improves representations beyond a single forward pass.

def evaluate_inference_adaptation(model, images, occlusion_fn, n_steps=20):
    """
    1. OCCLUDE images spatially (block masking / black out half) — NOT Gaussian noise,
       which a JEPA largely ignores.
    2. Single forward pass            -> representation r_0
    3. Run the PC inference loop      -> representation r_n  (after n_steps)
    4. Compare linear-probe accuracy of r_0 vs r_n.
    Expected: r_n > r_0, and the gap widens as occlusion grows.
    The PC hierarchy should predict the latents of the occluded tokens
    from the visible ones.
    """

Plot: accuracy vs. number of inference steps, for several occlusion levels. A pure JEPA baseline is flat (it has no loop); the composite should rise.

#### Evaluation 2 — Continual learning (with the isolating baseline)

Tests whether local PC updates reduce catastrophic forgetting — and whether PC is the reason.

Split CIFAR-100 into 10 tasks of 10 classes each.
Train sequentially: task 1 -> task 2 -> ... -> task 10.
After each task, evaluate on ALL previous tasks.
Metric: average accuracy over all seen tasks after the final task.

Baselines (all on the SAME frozen JEPA tokenizer):
  - JEPA + backprop-trained head        (expected: high forgetting)
  - PC composite, local weight updates   (hypothesis: less forgetting)
  - Plain MLP head, backprop             (CONTROL: isolates whether PC helps,
                                          or frozen JEPA features alone resist forgetting)
  - EWC-regularised head                 (standard continual-learning baseline)

Scope honestly: all tasks are natural images, so the frozen tokenizer stays valid. Do not claim the result transfers to distribution-shifted domains (e.g. medical scans), where a frozen tokenizer would fail. In LeCun's framing: the innate substrate (JEPA) is stable; only the higher-level world models (PC) update across tasks.

#### Evaluation 3 — Uncertainty via precision-weighting and goal-directed inference

Tests whether PC's precision estimates are meaningful in token space, and whether treating uncertainty reduction as an explicit goal improves inference quality over fixed-step inference.

**Part A — Passive precision maps (perceptual inference).**

Precision = inverse variance of prediction errors at each token position.
High precision -> the model is confident about that token.
Low precision  -> the model is uncertain.

Qualitative: do precision maps highlight informative image regions?
Quantitative: do low-precision tokens align with occluded or corrupted regions?
This establishes that precision is semantically grounded before testing active use of it.

**Part B — Goal-directed inference (active inference, Level 1).**

Replace fixed-step inference with uncertainty-threshold termination (see Active Inference section). Metrics:

- Steps to convergence per image: does this correlate with occlusion level and image ambiguity?
- Does threshold-terminated inference match or exceed fixed-step accuracy at lower average step count?
- Plot: precision map evolution over inference steps — do low-precision tokens resolve as the loop runs?

A pure JEPA baseline has no inference loop and therefore no precision dynamics — it is flat across both parts. The composite should show structured, spatially meaningful precision that evolves during inference and responds to goals.

Deliverables: notebooks/04_composite_evaluation.ipynb, models/composite.py

---

## Future work

The following extensions follow naturally from the composite design but are out of scope for the current project. They are documented here so the architecture decisions made now do not accidentally foreclose them.

**Level-specific precision priors (Active Inference Level 2).** Add a learned top-down precision pathway alongside the existing top-down prediction pathway at each level boundary. Level 3's representation projects a precision-scaling signal down to Level 2 and Level 1, amplifying errors in goal-relevant spatial regions and suppressing others. This is the mechanistic account of spatial attention in Friston's framework and requires adding one linear projection per level boundary — a small change with significant implications for how the model allocates inferential effort.

**Explicit goal states and temporal inference (Active Inference Level 3).** Inject a target representation at PC Level 3 encoding a desired future state. Run inference to minimise the gap between current and desired state rather than to explain the current input. This requires a temporal model — a learned transition function over token grids — and is only tractable with sequential data. V-JEPA (Bardes et al., 2024) provides a natural tokenizer for this extension: its tokens are already temporal, and PC Level 3's goal state would become a prediction about a future token grid.

**Temporal hierarchy.** The current PC hierarchy is spatial only (patch → region → scene). LeCun's full hierarchical world model is also temporal: higher levels predict over longer horizons. With video tokens, PC levels would map onto both spatial scales and prediction horizons simultaneously — a direct instantiation of the temporal dimension currently absent from the design.
```text
jepa-pc-composite/
│
├── README.md                      ← this file
├── requirements.txt               ← pinned dependencies
├── setup.md                       ← environment + dataset download instructions
│
├── configs/                       ← all hyperparameters live here (no magic numbers in code)
│   ├── pc_mnist.yaml              ← Stage 1
│   ├── jepa_cifar10.yaml          ← Stage 2 (incl. AMP / batch / grad-accum settings)
│   ├── pc_tokens_cifar10.yaml     ← Stage 3
│   └── composite_cifar100.yaml    ← Stage 4
│
├── notebooks/                     ← one notebook per stage (the narrative / results)
│   ├── 01_vanilla_pc_mnist.ipynb
│   ├── 02_jepa_cifar10.ipynb
│   ├── 03_pc_on_patch_tokens.ipynb
│   └── 04_composite_evaluation.ipynb
│
├── models/                        ← reusable model code (imported by notebooks)
│   ├── pc_layer.py
│   ├── pc_conv.py
│   ├── jepa_tokenizer.py
│   └── composite.py
│
├── experiments/                   ← evaluation logic, kept separate from models
│   ├── linear_probe.py
│   ├── inference_adaptation.py
│   ├── continual_learning.py
│   └── precision_uncertainty.py
│
├── utils/                         ← shared helpers
│   ├── data.py
│   ├── ema.py
│   ├── metrics.py
│   └── viz.py
│
├── checkpoints/
│   └── .gitkeep
│
├── results/
│   └── .gitkeep
│
└── tests/
    ├── test_pc_convergence.py
    ├── test_jepa_no_collapse.py
    └── test_spatial_layout.py
```

## Setup / quickstart

# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Verify GPU + AMP availability (Stage 2+). CPU is fine for Stage 1.
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 3. Run the stages in order. DO NOT skip ahead — each gates the next.
jupyter notebook notebooks/01_vanilla_pc_mnist.ipynb      # verify PC works
jupyter notebook notebooks/02_jepa_cifar10.ipynb          # train + FREEZE tokenizer, check no collapse
jupyter notebook notebooks/03_pc_on_patch_tokens.ipynb    # PC on frozen tokens, grid preserved
jupyter notebook notebooks/04_composite_evaluation.ipynb  # the three evaluations

Golden rules:

    Verify each stage's checklist before starting the next.
    The tokenizer is frozen from Stage 3 onward — never fine-tune it with PC.
    Confirm non-collapse (Stage 2) before trusting anything downstream.
    Keep tokens on their 2D grid; PC Level 1 is convolutional.

---

## Glossary (quick reference)

    Token — one vector that summarises one image patch; the JEPA's output unit.
    Tokenizer — the frozen JEPA encoder that turns an image into a grid of tokens.
    Inference loop — PC's iterative settling of representations on a single input (weights fixed).
    Prediction error — the mismatch between a top-down prediction and the actual lower-level activity.
    Precision — inverse variance of an error; PC's confidence weight (high = trust this signal).
    Stop-gradient / EMA target — JEPA's anti-collapse mechanism; the target encoder is a slow copy that receives no gradient.
    Collapse — the failure where all representations become (nearly) identical; trivially minimises the loss, learns nothing.
    Local / Hebbian update — a weight change computed only from locally available signals, not global backprop.
    Catastrophic forgetting — losing performance on old tasks after training on new ones.
    Innate prior substrate — LeCun's term for fixed low-level world models (object permanence, gravity, basic geometry) that underpin higher-level learning; in this project, the frozen JEPA tokenizer plays this role.
    Hierarchical world model — a stack of models operating at different scales of abstraction and (in temporal settings) different prediction horizons; the PC hierarchy is this project's implementation of that idea.
    Perceptual inference — minimising prediction error by updating internal representations to match the world (standard PC). The world is fixed; the model adjusts.
    Active inference — minimising prediction error by directing attention or action to bring the world closer to a preferred state (Friston). Requires a goal prior encoding desired future states.
    Goal prior — a distribution over preferred future states injected at the top of the PC hierarchy; propagates downward via precision-weighting to shape which prediction errors the inference loop prioritises.
    Precision-weighting — scaling each prediction error by the inverse of its variance before using it to update representations. High precision = trust this signal and attend to it; low precision = suppress it. The central mechanism by which uncertainty, attention, and goal-directed inference are all implemented in this framework — not a diagnostic add-on.

---

## References

    Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature Neuroscience, 2(1), 79–87.
    Friston, K. (2005). A theory of cortical responses. Philosophical Transactions of the Royal Society B, 360(1456), 815–836.
    Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127–138. — canonical statement of Active Inference; the source of perceptual vs. active inference distinction and precision-weighting as the mechanism of attention.
    Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference. Biological Cybernetics, 113(5–6), 495–513. — formalises goal priors and precision control in the active inference framework; directly relevant to Level 2 and Level 3 extensions.
    Whittington, J. C. R., & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. Neural Computation, 29(5), 1229–1262.
    Bogacz, R. (2017). A tutorial on the free-energy framework for modelling perception and learning. Journal of Mathematical Psychology, 76, 198–211. (Recommended for the precise inference/learning update equations.)
    Millidge, B., Tschantz, A., & Buckley, C. L. (2022). Predictive coding approximates backprop along arbitrary computation graphs. Neural Computation, 34(6), 1329–1368. (arXiv:2006.04182, 2020.) — PC matches backprop gradients on arbitrary graphs (CNNs, RNNs, LSTMs) with local rules; licenses structured/convolutional PC.
    Salvatori, T., Song, Y., Lukasiewicz, T., Bogacz, R., & Xu, Z. (2021). Predictive coding can do exact backpropagation on convolutional and recurrent neural networks. arXiv:2103.03725. — PC can exactly replicate backprop on convolutional and (many-to-one) recurrent nets; direct grounding for the convolutional PC design over the token grid.
    LeCun, Y. (2022). A path towards autonomous machine intelligence. OpenReview preprint. — source of the two-tier architecture (innate prior substrate + learned hierarchical world models) that this project instantiates.
    Assran, M., et al. (2023). Self-supervised learning from images with a joint-embedding predictive architecture (I-JEPA). CVPR 2023.
    Bardes, A., et al. (2024). V-JEPA: revisiting feature prediction for learning visual representations from video. NeurIPS 2024.
    Baevski, A., et al. (2022). data2vec: a general framework for self-supervised learning in speech, vision and language. ICML 2022.
    Ororbia, A. (2023). The predictive forward-forward algorithm. arXiv:2301.01452.
    Friston, K., et al. (2023). Designing ecosystems of intelligence from first principles. Collective Intelligence.
