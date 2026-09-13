# CEFR Multi-Prefix Tuning

This directory contains the **CEFR-conditioned Multi-Prefix Tuning experiments** used in the master's thesis:

> **Beyond Prompting: Resource-Efficient Explicit Control for CEFR-Aligned Language Generation**

These experiments investigate explicit CEFR control through **level-specific continuous prefix representations** attached to a frozen `meta-llama/Llama-3.1-8B-Instruct` backbone.

Three controller capacities are evaluated:

- **~68M parameters**
- **~286M parameters**
- **~537M parameters**

The experiments address two related questions:

1. **Does explicit CEFR-specific prefix conditioning improve controllability over generic adaptation methods?**
2. **How does CEFR-control performance change as controller capacity increases?**

The **~68M** and **~537M** configurations additionally provide parameter-matched comparisons against **Standard Prefix-Tuning** and **Vanilla PrefixMemory-Tuning (PMT)**, respectively.

---

# Common Architecture

All three models use the same general **CEFR Multi-Prefix** mechanism.

Each CEFR proficiency level has a dedicated set of **30 virtual prefix tokens**:

```text
A1 → 30 virtual tokens
A2 → 30 virtual tokens
B1 → 30 virtual tokens
B2 → 30 virtual tokens
C1 → 30 virtual tokens
C2 → 30 virtual tokens
```

This gives:

```text
6 CEFR levels × 30 tokens
            =
180 CEFR-specific virtual-token embeddings
```

At training and inference time, the requested CEFR class selects the corresponding prefix bank.

The selected prefix representations are mapped through a shared projection network into layer-wise key/value states for the frozen Llama-3.1-8B-Instruct backbone.

Conceptually:

```text
                 Target CEFR
                     │
                     ▼
          CEFR-Specific Prefix Bank
                     │
                     ▼
            30 Virtual Prefix Tokens
                     │
                     ▼
           Shared Prefix Projection
                     │
                     ▼
         Layer-Wise Key/Value States
                     │
                     ▼
         Frozen Llama-3.1-8B-Instruct
```

The specific target CEFR label is **not directly inserted into the textual prompt**.

Instead, the requested proficiency level is communicated through the selected CEFR-specific prefix bank.

---

# Shared Experimental Setup

All three configurations use the same underlying experimental framework:

| Setting | Value |
|---|---|
| Backbone | `meta-llama/Llama-3.1-8B-Instruct` |
| Backbone state | Frozen |
| Training dataset | Balanced CEFR Steering Subset |
| Total examples | `5,568` |
| CEFR levels | A1, A2, B1, B2, C1, C2 |
| Samples per level | `928` |
| Train/validation split | Stratified 90/10 |
| Training examples | `5,011` |
| Validation examples | `557` |
| Random seed | `42` |
| Epochs | `3` |
| Learning rate | `2e-4` |
| Optimizer | AdamW |
| Weight decay | `0.01` |
| LR scheduler | Cosine |
| Warmup ratio | `5%` |
| Maximum sequence length | `512` |
| Precision | BF16 |
| Evaluation benchmark | In-Domain Evaluation Prompt Matrix |
| Evaluation conditions | `117 topics × 6 CEFR levels = 702` |
| Primary CEFR evaluator | `MohammadKhosravi/roberta-large-cefr-classifier-JointLoss` |

---

# Experiments

## 1. CEFR Multi-Prefix — ~68M

### `cefr_prefix_tuning_68m_train_and_eval.ipynb`

This notebook implements the **~68M parameter-matched CEFR Multi-Prefix model**.

It is designed as a controlled comparison against the Standard Prefix-Tuning baseline.

### Architecture

```text
Prefix embedding dimension: 1024
Projection MLP:             1024 → 1024 → 65536
Virtual tokens per CEFR:    30
CEFR prefix banks:          6
```

### Parameter Matching

| Model | Trainable Parameters |
|---|---:|
| Standard Prefix-Tuning | 68,254,720 |
| CEFR Multi-Prefix — 68M | 68,408,320 |
| Difference | +153,600 |
| Relative difference | +0.2250% |

### Training Results

| Epoch | Train Loss | Validation Loss | Validation PPL |
|---:|---:|---:|---:|
| 1 | 2.7607 | 2.5314 | 12.57 |
| 2 | 2.4281 | **2.4825** | **11.97** |
| 3 | **2.3348** | 2.4855 | 12.01 |

**Best checkpoint:** Epoch 2

### Evaluation Results

| Metric | Result |
|---|---:|
| Strict Accuracy | **45.44%** |
| Adjacent Accuracy | **68.80%** |
| MAE | **1.0940** |
| Mean Dependency Distance | **1.78** |
| Flesch Reading Ease | **81.08** |
| Sentence-Level CEFR Drift | **2.38** |

### Hugging Face

[MohammadKhosravi/llama3.1-8b-cefr-prefix-tuning-68m-param-matched](https://huggingface.co/MohammadKhosravi/llama3.1-8b-cefr-prefix-tuning-68m-param-matched)

---

## 2. CEFR Multi-Prefix — ~286M

### `cefr_prefix_tuning_286m_train_and_eval.ipynb`

This notebook implements the **intermediate-capacity ~286M CEFR Multi-Prefix model**.

Unlike the 68M configuration, this variant uses the full **4096-dimensional Llama hidden representation** for the prefix embeddings and shared projection network.

### Architecture

```text
Prefix embedding dimension: 4096
Projection MLP:             4096 → 4096 → 65536
Virtual tokens per CEFR:    30
CEFR prefix banks:          6
Trainable parameters:       286,019,584
```

### Training Results

| Epoch | Train Loss | Validation Loss | Validation PPL |
|---:|---:|---:|---:|
| 1 | 2.6608 | 2.5036 | 12.23 |
| 2 | 2.3967 | **2.4328** | **11.39** |
| 3 | **2.2449** | 2.4366 | 11.43 |

**Best checkpoint:** Epoch 2

### Evaluation Results

| Metric | Result |
|---|---:|
| Strict Accuracy | **52.42%** |
| Adjacent Accuracy | **76.35%** |
| MAE | **0.8533** |
| Mean Dependency Distance | **1.82** |
| Flesch Reading Ease | **79.49** |
| Sentence-Level CEFR Drift | **2.34** |

### Hugging Face

[MohammadKhosravi/llama3.1-8b-cefr-prefix-tuning-6k-286M-Variant](https://huggingface.co/MohammadKhosravi/llama3.1-8b-cefr-prefix-tuning-6k-286M-Variant)

---

## 3. CEFR Multi-Prefix — ~537M

### `cefr_prefix_tuning_537m_train_and_eval.ipynb`

This notebook implements the **~537M PMT-matched CEFR Multi-Prefix model**.

It is the highest-capacity CEFR Prefix-Tuning configuration and is designed as a parameter-matched control against Vanilla PrefixMemory-Tuning.

### Architecture

```text
Prefix embedding dimension: 4096
Projection MLP:             4096 → 7696 → 65536
Virtual tokens per CEFR:    30
CEFR prefix banks:          6
Trainable parameters:       536,698,384
```

### Parameter Matching

| Model | Trainable Parameters |
|---|---:|
| Vanilla PMT | 536,870,912 |
| CEFR Multi-Prefix — 537M | 536,698,384 |
| Difference | −172,528 |
| Relative difference | −0.0321% |

### Training Results

| Epoch | Train Loss | Validation Loss | Validation PPL |
|---:|---:|---:|---:|
| 1 | 2.6481 | 2.6245 | 13.80 |
| 2 | 2.4545 | 2.4497 | 11.58 |
| 3 | **2.3074** | **2.4371** | **11.44** |

**Best checkpoint:** Epoch 3

### Evaluation Results

| Metric | Result |
|---|---:|
| Strict Accuracy | **64.10%** |
| Adjacent Accuracy | **81.48%** |
| MAE | **0.6966** |
| Mean Dependency Distance | **1.80** |
| Flesch Reading Ease | **81.22** |
| Sentence-Level CEFR Drift | **2.27** |

### Hugging Face

[MohammadKhosravi/llama3.1-8b-cefr-pt-537m-param-matched](https://huggingface.co/MohammadKhosravi/llama3.1-8b-cefr-pt-537m-param-matched)

---

# Capacity-Scaling Comparison

The three experiments provide a direct view of how CEFR-control performance changes with controller capacity.

| Variant | Trainable Parameters | Strict Accuracy | Adjacent Accuracy | MAE |
|---|---:|---:|---:|---:|
| CEFR Multi-Prefix — 68M | 68.41M | 45.44% | 68.80% | 1.0940 |
| CEFR Multi-Prefix — 286M | 286.02M | 52.42% | 76.35% | 0.8533 |
| **CEFR Multi-Prefix — 537M** | **536.70M** | **64.10%** | **81.48%** | **0.6966** |

The progression is:

```text
68M
45.44% Strict
    │
    ▼
286M
52.42% Strict
    │
    ▼
537M
64.10% Strict
```

Within this experimental setup, increasing CEFR-controller capacity consistently improves:

- Strict Accuracy
- Adjacent Accuracy
- Mean Absolute Error

The strict-accuracy progression is:

```text
45.44% → 52.42% → 64.10%
```

From 68M to 286M:

**+6.98 percentage points**

From 286M to 537M:

**+11.68 percentage points**

From 68M to 537M:

**+18.66 percentage points**

---

# Parameter-Matched Controls

Two of the three CEFR Multi-Prefix configurations also serve as controlled parameter-budget comparisons.

## Standard Prefix-Tuning vs. CEFR Multi-Prefix — 68M

| Method | Trainable Parameters | Strict Accuracy |
|---|---:|---:|
| Standard Prefix-Tuning | 68.25M | 32.34% |
| **CEFR Multi-Prefix — 68M** | **68.41M** | **45.44%** |

Parameter difference:

**+0.2250%**

Strict Accuracy difference:

```text
32.34% → 45.44%
```

**+13.10 percentage points**

Because the parameter budgets are nearly identical, this comparison isolates the effect of **explicit CEFR-specific prefix selection** from increased trainable capacity.

---

## Vanilla PMT vs. CEFR Multi-Prefix — 537M

| Method | Trainable Parameters | Strict Accuracy | Adjacent Accuracy | MAE |
|---|---:|---:|---:|---:|
| Vanilla PMT | 536.87M | 15.67% | 47.58% | 1.7308 |
| **CEFR Multi-Prefix — 537M** | **536.70M** | **64.10%** | **81.48%** | **0.6966** |

Parameter difference:

**−0.0321%**

Strict Accuracy difference:

```text
15.67% → 64.10%
```

**+48.43 percentage points**

Because the parameter budgets are effectively identical, this comparison provides a strong control for separating the effect of **raw controller capacity** from the effect of **explicit CEFR conditioning**.

---

# Experimental Interpretation

Together, the three experiments support two observations within the evaluated setup.

## 1. Explicit CEFR Conditioning Matters

The parameter-matched comparisons show that CEFR-specific architectural conditioning improves controllability relative to generic or unconditioned alternatives:

```text
Standard Prefix-Tuning
32.34%
      │
      ▼
CEFR Multi-Prefix — 68M
45.44%
```

and:

```text
Vanilla PMT
15.67%
      │
      ▼
CEFR Multi-Prefix — 537M
64.10%
```

In both cases, the parameter budgets are approximately matched.

---

## 2. Controller Capacity Also Matters

Within the same CEFR Multi-Prefix architecture, increasing controller capacity produces progressively stronger CEFR control:

```text
~68M   →   ~286M   →   ~537M
45.44%     52.42%      64.10%
```

This indicates that both:

- **explicit CEFR conditioning**, and
- **controller capacity**

contribute to performance.

The experiments therefore help separate three factors explored throughout the thesis:

```text
Adaptation Architecture
          │
          ▼
Controller Capacity
          │
          ▼
Explicit CEFR Conditioning
```

---

# Evaluation Metrics

## Strict Accuracy

A generation is counted as correct only when the predicted CEFR class exactly matches the requested level.

For example:

```text
Target B1:

B1       → Correct
A2 / B2  → Incorrect
A1/C1/C2 → Incorrect
```

---

## Adjacent Accuracy

Predictions one neighboring CEFR level away from the target are also counted as acceptable.

For example:

```text
Target B1:

B1           → Strict + Adjacent correct
A2 / B2      → Adjacent correct
A1 / C1 / C2 → Incorrect
```

---

## Mean Absolute Error

CEFR classes are represented as ordered indices:

```text
A1 = 0
A2 = 1
B1 = 2
B2 = 3
C1 = 4
C2 = 5
```

Mean Absolute Error measures the average ordinal distance between the requested and predicted CEFR classes:

$$
\operatorname{MAE}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left|
\hat{y}_i-y_i
\right|
$$

Lower MAE indicates stronger proficiency control.

---

## Linguistic Diagnostics

The evaluation notebooks additionally compute:

- **Mean Dependency Distance (MDD)**
- **Flesch Reading Ease**
- **Sentence-Level CEFR Drift**
- Per-level CEFR statistics
- Classification reports
- Confusion matrices

These diagnostics supplement the primary CEFR-control metrics.

---

# Relationship to Other Thesis Experiments

Generic adaptation baselines are available in:

```text
notebooks/04_generic_peft_baselines/
```

The final CEFR-gated PrefixMemory-Tuning model is available in:

```text
notebooks/06_cefr_gated_pmt/
```

Training-data construction and benchmark preparation are available in:

```text
notebooks/01_data_preparation/
```

The primary CEFR evaluator is documented in:

```text
notebooks/02_cefr_evaluator/
```

---

# Repository Structure

```text
05_cefr_prefix_tuning/
├── README.md
├── cefr_prefix_tuning_68m_train_and_eval.ipynb
├── cefr_prefix_tuning_286m_train_and_eval.ipynb
└── cefr_prefix_tuning_537m_train_and_eval.ipynb
```

---

# External Model Artifacts

The trained controllers are hosted on Hugging Face:

- [68M CEFR Multi-Prefix](https://huggingface.co/MohammadKhosravi/llama3.1-8b-cefr-prefix-tuning-68m-param-matched)
- [286M CEFR Multi-Prefix](https://huggingface.co/MohammadKhosravi/llama3.1-8b-cefr-prefix-tuning-6k-286M-Variant)
- [537M CEFR Multi-Prefix](https://huggingface.co/MohammadKhosravi/llama3.1-8b-cefr-pt-537m-param-matched)

Generated outputs, detailed logs, hardware-profiling files, and other large intermediate artifacts are stored separately from the GitHub repository.

---

# Summary

The CEFR Multi-Prefix experiments establish two central findings within the evaluated setup.

First, **explicit CEFR-specific prefix conditioning substantially improves proficiency control under approximately matched parameter budgets**.

Second, **controller capacity has a strong positive effect within the same CEFR-conditioned architecture**.

| Model | Trainable Parameters | Strict Accuracy |
|---|---:|---:|
| Standard Prefix-Tuning | 68.25M | 32.34% |
| CEFR Multi-Prefix — 68M | 68.41M | 45.44% |
| CEFR Multi-Prefix — 286M | 286.02M | 52.42% |
| CEFR Multi-Prefix — 537M | 536.70M | **64.10%** |
| Vanilla PMT | 536.87M | 15.67% |

These results motivate the subsequent investigation of **CEFR-gated PrefixMemory-Tuning**, where proficiency information is introduced through an alternative memory-conditioning mechanism.
