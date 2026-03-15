# Classifier Architecture Research — Autoresearch Program

## Objective

Find the optimal encoder + classification head architecture for agent classification tasks. Iterate on the architecture definition in `harness.py` to maximize the composite F1 score across 4 held-out domains in `benchmark.py`.

This directly parallels Karpathy's autoresearch:
- `harness.py` = `train.py` — architecture definition, training data, model building
- `benchmark.py` = `prepare.py` — held-out evaluation, locked, do not modify

## What You're Researching

**Architecture, not training process.** The training hyperparameters (lr=2e-5, epochs=3, batch=8) are fixed. The experiment varies only the structural choices:

| Knob | Options | What it controls |
|------|---------|-----------------|
| `BASE_MODEL` | `deberta-v3-{xsmall,small,base}`, `bert-base-uncased`, `distilbert-base-uncased` | Encoder capacity and pretraining quality |
| `POOLING` | `cls`, `mean`, `max`, `attention_weighted` | How token representations are aggregated into a single vector |
| `HEAD_TYPE` | `linear`, `mlp`, `mlp_residual`, `bilinear` | Classification head architecture |
| `HEAD_HIDDEN` | 64–1024 | Width of MLP/bilinear hidden layers |
| `HEAD_LAYERS` | 1–4 | Depth of MLP head |
| `HEAD_DROPOUT` | 0.0–0.5 | Regularization in the head |
| `HEAD_ACTIVATION` | `relu`, `gelu`, `silu` | Nonlinearity in the head |
| `NORMALIZE_EMBEDDINGS` | `True`/`False` | L2 normalize pooled embeddings before the head |
| `USE_LAYER_COMBINATION` | `True`/`False` | Combine outputs from multiple encoder layers |
| `LAYER_WEIGHTS` | `last`, `last_4_mean`, `learned_weighted` | How to combine encoder layers |

## The Score

```
score = 0.25 * intent_f1      — actionable vs conversational input
      + 0.25 * safety_f1      — harmful vs safe input
      + 0.25 * routing_f1     — code task vs non-code task
      + 0.25 * tone_f1        — frustrated vs satisfied user
```

Equal weight across all 4 domains. The best architecture generalizes across classification tasks that agent runtimes commonly need.

## Why This Matters

Every agent runtime needs classifiers:
- **Intent detection** — decide whether to act or respond conversationally
- **Safety classification** — block harmful requests before they reach the LLM
- **Routing** — send code tasks to a code agent, research tasks to a search agent
- **Tone analysis** — escalate frustrated users to human support

These are all binary/multi-class classification tasks on short text. Finding the optimal encoder + head architecture benefits ALL of them simultaneously. A single architecture experiment here replaces ad-hoc architecture choices in every downstream classifier.

## Setup

```bash
# With ML dependencies (real model training)
pip install transformers torch

# Run (trains 4 classifiers + evaluates on held-out data)
cd architectures/classifier
python benchmark.py

# Without ML dependencies (deterministic simulation)
python benchmark.py
```

Without `torch`/`transformers`, the benchmark uses a TF-IDF + logistic regression fallback that simulates architecture quality differences. The simulation is deterministic — the same knob settings always produce the same score, so the meta-agent gets consistent feedback.

## What You Can Modify

**`harness.py`** — everything in this file is fair game:

### Architecture knobs
The module-level constants at the top of the file. Change these to explore different configurations.

### Training data
- `INTENT_TRAIN`, `SAFETY_TRAIN`, `ROUTING_TRAIN`, `TONE_TRAIN` — 30 examples each
- You can augment, rebalance, or add hard negatives

### Architecture implementation
- Pooling layer classes (`_CLSPooling`, `_MeanPooling`, `_MaxPooling`, `_AttentionPooling`)
- Head classes (`_ResidualHead`, `_BilinearHead`)
- The `_ClassifierModel` wrapper
- You can add entirely new pooling strategies or head designs

## What You CANNOT Modify

- **`benchmark.py`** — held-out test data and scoring. This is the trust boundary.

## Key Experiments to Run

### 1. Pooling strategy (start here)
```python
POOLING = "attention_weighted"  # vs cls, mean, max
```
Attention-weighted pooling learns which tokens matter for classification. For short texts (typical of agent inputs), this often outperforms CLS or mean pooling because it can focus on the most discriminative tokens.

### 2. MLP head with residual connections
```python
HEAD_TYPE = "mlp_residual"
HEAD_HIDDEN = 256
HEAD_LAYERS = 2
HEAD_ACTIVATION = "gelu"
```
A deeper head with residuals can learn more complex decision boundaries than a linear projection, especially for safety classification where the boundary between safe security discussion and actual harmful requests is subtle.

### 3. Layer combination
```python
USE_LAYER_COMBINATION = True
LAYER_WEIGHTS = "last_4_mean"   # or "learned_weighted"
```
Different encoder layers capture different information — early layers capture syntax, later layers capture semantics. Combining them gives the head richer features. `learned_weighted` lets the model discover which layers matter most for each task.

### 4. Combined architecture
```python
POOLING = "attention_weighted"
HEAD_TYPE = "mlp_residual"
HEAD_HIDDEN = 256
HEAD_LAYERS = 2
HEAD_ACTIVATION = "gelu"
NORMALIZE_EMBEDDINGS = True
USE_LAYER_COMBINATION = True
LAYER_WEIGHTS = "learned_weighted"
```
The full stack: attention pooling + residual MLP + layer combination. This should be the highest-scoring configuration if the architecture choices compose well.

### 5. Base model comparison
```python
BASE_MODEL = "microsoft/deberta-v3-base"    # larger encoder
BASE_MODEL = "microsoft/deberta-v3-xsmall"  # faster iteration
```
Larger encoders have more capacity but may overfit on 30 training examples per domain. The sweet spot depends on the head design.

## Experimentation Loop

LOOP FOREVER:

1. Run `python benchmark.py` to train + evaluate across all 4 domains.
2. Check the output: which domains score lowest? What errors occur?
3. Modify `harness.py` — architecture knobs, pooling, head design, layer combination.
4. Run benchmark again.
5. If score improved, `git commit` and keep.
6. If score is same or worse, `git reset` and try something different.
7. Never stop.

## Logging

```
commit	score	status	description
a1b2c3d	0.350000	keep	baseline — TF-IDF fallback, linear head, CLS pooling
b2c3d4e	0.520000	keep	DeBERTa-v3-small, linear head, CLS pooling
c3d4e5f	0.580000	keep	attention_weighted pooling
d4e5f6g	0.610000	keep	mlp head, hidden=256
e5f6g7h	0.640000	keep	mlp_residual + attention_weighted
f6g7h8i	0.660000	keep	+ layer combination (last_4_mean)
g7h8i9j	0.680000	keep	+ learned_weighted layers + gelu
```
