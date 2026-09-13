# Generic PEFT Baselines

This directory contains the generic parameter-efficient and memory-based adaptation baselines used in the master's thesis:

> **Beyond Prompting: Resource-Efficient Explicit Control for CEFR-Aligned Language Generation**

These baselines are trained and evaluated under a common experimental setup using the frozen `meta-llama/Llama-3.1-8B-Instruct` backbone, the same balanced CEFR steering dataset, and the same in-domain evaluation benchmark.

The purpose of this directory is to establish reference methods against which the explicitly CEFR-conditioned architectures developed later in the thesis can be compared.

---

## Experimental Setup

All three baselines use the same core experimental setting:

| Setting | Value |
|---|---|
| Backbone | `meta-llama/Llama-3.1-8B-Instruct` |
| Backbone parameters | Frozen |
| Training dataset | Balanced CEFR Steering Subset |
| Training examples | `5,568` |
| CEFR levels | A1, A2, B1, B2, C1, C2 |
| Samples per CEFR level | `928` |
| Train/validation split | Stratified 90/10 |
| Training samples | `5,011` |
| Validation samples | `557` |
| Split seed | `random_state=42` |
| Evaluation benchmark | In-Domain Evaluation Prompt Matrix |
| Evaluation conditions | `117 topics × 6 CEFR levels = 702` |
| Primary CEFR evaluator | `MohammadKhosravi/roberta-large-cefr-classifier-JointLoss` |

The generic baselines rely on **textual CEFR instructions** rather than a dedicated CEFR-specific architectural control variable.

---

# Baselines

## 1. LoRA

### `lora_train_and_eval.ipynb`

This notebook trains and evaluates the **Low-Rank Adaptation (LoRA)** baseline.

LoRA adapters are attached to the attention and MLP projection layers of the frozen Llama-3.1-8B-Instruct backbone.

### Configuration

| Setting | Value |
|---|---:|
| Rank `r` | `16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.05` |
| Trainable parameters | `41,943,040` |
| Trainable fraction | `0.5196%` |
| Epochs | `3` |
| Learning rate | `2e-4` |
| Effective batch size | `64` |

Target modules:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

### Training Results

| Epoch | Train Loss | Validation Loss | Validation PPL |
|---:|---:|---:|---:|
| 1 | 2.5610 | 2.4400 | 11.47 |
| 2 | 2.2555 | **2.3686** | **10.68** |
| 3 | **2.0510** | 2.3943 | 10.96 |

The best validation checkpoint is selected at **Epoch 2**.

### Evaluation Results

| Metric | Result |
|---|---:|
| Strict Accuracy | **26.21%** |
| Adjacent Accuracy | **59.69%** |
| MAE | **1.4501** |
| Mean Dependency Distance | **1.75** |
| Flesch Reading Ease | **82.64** |
| Sentence-Level CEFR Drift | **2.23** |

### Hugging Face

[MohammadKhosravi/llama3.1-8b-lora-cefr-steering-6k](https://huggingface.co/MohammadKhosravi/llama3.1-8b-lora-cefr-steering-6k)

---

## 2. Standard Prefix-Tuning

### `standard_prefix_tuning_train_and_eval.ipynb`

This notebook trains and evaluates the **Standard Prefix-Tuning** baseline using the Hugging Face PEFT implementation.

The Llama backbone remains frozen while a continuous learnable prefix is optimized and injected into the attention mechanism.

### Configuration

| Setting | Value |
|---|---:|
| Virtual tokens | `30` |
| Prefix projection | `True` |
| Trainable parameters | `68,254,720` |
| Trainable fraction | `0.8428%` |
| Epochs | `3` |
| Learning rate | `2e-4` |
| Effective batch size | `32` |

### Training Results

| Epoch | Train Loss | Validation Loss | Validation PPL |
|---:|---:|---:|---:|
| 1 | 2.6629 | 2.5390 | 12.67 |
| 2 | 2.4541 | 2.4701 | 11.82 |
| 3 | **2.3293** | **2.4610** | **11.72** |

The best validation checkpoint is selected at **Epoch 3**.

### Evaluation Results

| Metric | Result |
|---|---:|
| Strict Accuracy | **32.34%** |
| Adjacent Accuracy | **64.67%** |
| MAE | **1.2422** |
| Mean Dependency Distance | **1.80** |
| Flesch Reading Ease | **81.73** |
| Sentence-Level CEFR Drift | **2.17** |

### Hugging Face

[MohammadKhosravi/llama3.1-8b-standard-prefix-tuning-6k](https://huggingface.co/MohammadKhosravi/llama3.1-8b-standard-prefix-tuning-6k)

---

## 3. Vanilla PrefixMemory-Tuning (PMT)

### `vanilla_pmt_train_and_eval.ipynb`

This notebook trains and evaluates the **Vanilla PrefixMemory-Tuning (PMT)** baseline.

Unlike the CEFR-conditioned PMT variants developed later in the thesis, Vanilla PMT contains:

- No CEFR-specific embedding
- No CEFR-specific gating mechanism
- No CEFR-specific memory selection
- No dedicated architectural conditioning variable

Instead, one full hidden-dimension memory matrix is learned for each transformer layer, while CEFR information is supplied only through the textual instruction.

### Configuration

| Setting | Value |
|---|---:|
| Transformer layers | `32` |
| Hidden dimension | `4096` |
| Memory matrix per layer | `4096 × 4096` |
| Full memory tensor | `[32, 4096, 4096]` |
| Feature map | ELU |
| Trainable parameters | `536,870,912` |
| Trainable parameters (millions) | `536.87M` |
| Epochs | `3` |
| Learning rate | `2e-4` |
| Effective batch size | `16` |

The Vanilla PMT controller can be summarized as:

```text
             Hidden Representation
                      │
                      ▼
                Frozen q_proj
                      │
                      ▼
                   Query Q
                      │
                      ▼
                    ELU(Q)
                      │
                      ▼
          Layer-Specific Memory M_l
                      │
                      ▼
                ELU(Q) @ M_l
                      │
                      ▼
         Additive Memory Contribution
                      │
                      ▼
             Attention Output
```

### Training Results

| Epoch | Train Loss | Validation Loss | Validation PPL |
|---:|---:|---:|---:|
| 1 | 6.1827 | 5.5490 | 256.97 |
| 2 | 5.3515 | 5.2795 | 196.26 |
| 3 | **5.1446** | **5.2255** | **185.96** |

The best validation checkpoint is selected at **Epoch 3**.

### Evaluation Results

| Metric | Result |
|---|---:|
| Strict Accuracy | **15.67%** |
| Adjacent Accuracy | **47.58%** |
| MAE | **1.7308** |
| Mean Dependency Distance | **1.73** |
| Flesch Reading Ease | **98.00** |
| Sentence-Level CEFR Drift | **2.24** |

### Hugging Face

[MohammadKhosravi/llama3.1-8b-pure-pmt-6k](https://huggingface.co/MohammadKhosravi/llama3.1-8b-pure-pmt-6k)

---

# Baseline Comparison

The three generic adaptation methods differ substantially in both parameter count and CEFR-control performance.

| Method | Trainable Parameters | Strict Accuracy | Adjacent Accuracy | MAE |
|---|---:|---:|---:|---:|
| LoRA | 41.94M | 26.21% | 59.69% | 1.4501 |
| **Standard Prefix-Tuning** | **68.25M** | **32.34%** | **64.67%** | **1.2422** |
| Vanilla PMT | 536.87M | 15.67% | 47.58% | 1.7308 |

Among these generic baselines, **Standard Prefix-Tuning achieves the strongest CEFR-control performance**, despite using substantially fewer trainable parameters than Vanilla PMT.

The Vanilla PMT result is particularly informative: increasing adaptation capacity from tens of millions to more than **536 million trainable parameters** does not by itself produce stronger CEFR control.

This provides an important experimental motivation for the later explicitly CEFR-conditioned architectures.

---

# Evaluation Metrics

## Strict Accuracy

A generation is counted as correct only when its predicted CEFR level exactly matches the requested target level.

For example:

```text
Target: B1
Prediction: B1 → Correct
Prediction: A2 → Incorrect
Prediction: B2 → Incorrect
```

---

## Adjacent Accuracy

Adjacent Accuracy also considers a prediction acceptable when it differs from the target by only one neighboring CEFR level.

For example:

```text
Target B1:

B1           → Strict + Adjacent correct
A2 / B2      → Adjacent correct
A1 / C1 / C2 → Incorrect
```

---

## Mean Absolute Error

CEFR levels are mapped to ordered class indices:

```text
A1 = 0
A2 = 1
B1 = 2
B2 = 3
C1 = 4
C2 = 5
```

Mean Absolute Error measures the average ordinal distance between the requested and predicted CEFR levels:

$$
\operatorname{MAE}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left|
\hat{y}_i-y_i
\right|
$$

Lower values indicate smaller CEFR-control errors.

---

## Linguistic Diagnostics

The notebooks additionally report:

- **Mean Dependency Distance (MDD)** for syntactic complexity
- **Flesch Reading Ease** for readability
- **Sentence-Level CEFR Drift** for within-response proficiency consistency
- Classification reports
- Confusion matrices

These diagnostics supplement the primary CEFR-control metrics.

---

# Relationship to Later Experiments

These three methods provide the generic adaptation baselines for the later CEFR-specific experiments.

The experimental progression can be summarized as:

```text
Generic PEFT / Adaptation Baselines
│
├── LoRA
│
├── Standard Prefix-Tuning
│
└── Vanilla PMT
        │
        ▼
Explicit CEFR-Conditioned Prefix-Tuning
        │
        ▼
CEFR-Gated PrefixMemory-Tuning
```

This experimental structure allows the thesis to distinguish the effects of:

1. **Adaptation architecture**
2. **Controller capacity**
3. **Explicit CEFR conditioning**

In particular, the comparison between Vanilla PMT and the later conditioned PMT variants tests whether explicit proficiency conditioning provides benefits beyond simply increasing the number of trainable parameters.

---

# Repository Structure

```text
04_generic_peft_baselines/
├── README.md
├── lora_train_and_eval.ipynb
├── standard_prefix_tuning_train_and_eval.ipynb
└── vanilla_pmt_train_and_eval.ipynb
```

---

# Related Data-Preparation Notebooks

The balanced CEFR training subset is constructed in:

```text
notebooks/01_data_preparation/balanced_cefr_steering_subset.ipynb
```

The in-domain evaluation matrix is produced in:

```text
notebooks/01_data_preparation/efcamdat_preprocessing_and_partitioning.ipynb
```

---

# External Artifacts

Large trained models and adapter/controller weights are hosted on Hugging Face:

- [LoRA — `llama3.1-8b-lora-cefr-steering-6k`](https://huggingface.co/MohammadKhosravi/llama3.1-8b-lora-cefr-steering-6k)
- [Standard Prefix-Tuning — `llama3.1-8b-standard-prefix-tuning-6k`](https://huggingface.co/MohammadKhosravi/llama3.1-8b-standard-prefix-tuning-6k)
- [Vanilla PMT — `llama3.1-8b-pure-pmt-6k`](https://huggingface.co/MohammadKhosravi/llama3.1-8b-pure-pmt-6k)

Generated outputs, detailed logs, profiling files, and other large intermediate artifacts are stored separately from the GitHub repository.

---

# Summary

The generic baselines establish three important reference points:

| Baseline | Primary Role |
|---|---|
| **LoRA** | Generic low-rank parameter-efficient adaptation |
| **Standard Prefix-Tuning** | Generic continuous prefix-based adaptation |
| **Vanilla PMT** | High-capacity external-memory adaptation without explicit CEFR conditioning |

The results show that **parameter count alone is not sufficient to achieve strong CEFR controllability**.

Standard Prefix-Tuning achieves the best performance among the generic baselines with **32.34% Strict Accuracy**, while the substantially larger Vanilla PMT controller achieves only **15.67%**.

These results motivate the subsequent investigation of **explicit CEFR-conditioned control mechanisms**.
