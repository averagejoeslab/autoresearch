# Prompt Injection Classifier Training — Autoresearch Program

## Objective

Train a prompt injection classifier that generalizes to novel attack techniques it has never seen. Iterate on the training process in `harness.py` to maximize the score on held-out evaluation data in `benchmark.py`.

This directly parallels Karpathy's autoresearch:
- `harness.py` = `train.py` — model, training loop, training data, hyperparameters
- `benchmark.py` = `prepare.py` — held-out evaluation, locked, do not modify

## What You're Researching

**Out-of-distribution generalization.** The training data contains basic attack categories (instruction overrides, role-play, delimiter confusion, social engineering, encoding). The held-out test set contains **novel techniques not in the training data** — homoglyphs, zero-width characters, fake conversations, XML injection, code comment injection, output format manipulation.

The gap between in-distribution and out-of-distribution performance is the research target. Current published models (e.g., ProtectAI DeBERTa-v3) drop from 99.93% F1 in-distribution to 70.93% OOD. Closing that gap is the goal.

## The Score

```
score = 0.40 * detection_f1            — precision/recall on ALL attacks
      + 0.25 * (1 - false_positive_rate) — avoiding false alarms on legitimate text
      + 0.20 * advanced_detection_rate   — catching Tier 3 (novel/OOD) attacks
      + 0.15 * technique_coverage        — detecting diverse attack categories
```

## Setup

```bash
# Install dependencies
pip install transformers torch

# Run (trains model + evaluates on held-out data)
cd security/prompt-injection/defense
python benchmark.py
```

Without transformers/torch installed, falls back to TF-IDF + logistic regression.

## What you can modify

**`harness.py`** — everything in this file is fair game:

### Hyperparameters
| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `MODEL_NAME` | `microsoft/deberta-v3-small` | Base encoder |
| `NUM_EPOCHS` | 3 | Training epochs |
| `LEARNING_RATE` | 2e-5 | Optimizer learning rate |
| `BATCH_SIZE` | 8 | Training batch size |
| `MAX_LENGTH` | 256 | Max input tokens |
| `WARMUP_RATIO` | 0.1 | LR warmup fraction |
| `WEIGHT_DECAY` | 0.01 | AdamW weight decay |
| `FREEZE_LAYERS` | 0 | Encoder layers to freeze (0=full fine-tune) |
| `LABEL_SMOOTHING` | 0.0 | Label smoothing for loss |
| `DROP_RATE` | 0.1 | Classification head dropout |

### Training data
- `TRAINING_DATA` — the inline dataset (37 injections + 30 legitimate)
- `augment_training_data()` — data augmentation function

### Architecture
- `_train_transformer()` — the full training pipeline
- Classification head on top of the base model
- Loss function
- Optimizer and scheduler

## What you CANNOT modify

- **`benchmark.py`** — held-out test data and scoring. This is the trust boundary.

## Experimentation Ideas

### Training data (highest expected impact)
- **Hard negative mining**: Add more legitimate texts that contain injection-like keywords
- **Augmentation**: Paraphrase attacks, add encoding variants, combine techniques
- **Class balance**: Experiment with oversampling rare techniques vs uniform sampling
- **Synthetic attacks**: Generate novel attack patterns to pre-expose the model

### Hyperparameters
- **Learning rate**: Try 1e-5, 3e-5, 5e-5. This is often the single most impactful knob.
- **Epochs**: More epochs can overfit to training distribution. Fewer may underfit. Try 1-10.
- **Freeze layers**: Freezing early layers (FREEZE_LAYERS=4) speeds training and can improve OOD by preserving general representations
- **Label smoothing**: 0.05-0.15 can help generalization by preventing overconfident predictions

### Loss function
- **Focal loss**: Upweights hard examples. Could improve detection of subtle attacks.
- **Contrastive loss**: Learn to separate attack/legitimate embeddings in feature space.
- **Weighted CE**: Upweight injection class if recall is low.

### Model selection
- `microsoft/deberta-v3-xsmall` — fastest (22M), good for rapid iteration
- `microsoft/deberta-v3-small` — best tradeoff (44M), default
- `microsoft/deberta-v3-base` — highest capacity (86M), slower

### Architecture modifications
- Custom classification head (MLP with hidden layer instead of linear)
- Attention pooling instead of [CLS] token
- Multi-task: classify technique category alongside binary injection/safe

## Experimentation Loop

LOOP FOREVER:

1. Run `python benchmark.py` to train + evaluate.
2. Check the output: which attacks are missed? Which cause false positives?
3. Modify `harness.py` — training data, hyperparameters, augmentation, architecture.
4. Run benchmark again.
5. If score improved → `git commit` and keep.
6. If score is same or worse → `git reset` and try something different.
7. Never stop.

## Logging

```
commit	score	status	description
a1b2c3d	0.285000	keep	baseline — TF-IDF fallback (no transformers)
b2c3d4e	0.650000	keep	DeBERTa-v3-small, 3 epochs, lr=2e-5
c3d4e5f	0.710000	keep	added hard negative training examples
d4e5f6g	0.680000	discard	focal loss — hurt more than helped
e5f6g7h	0.750000	keep	label smoothing 0.1 + freeze first 2 layers
```
