# Model Registry

This document provides a consolidated index of the trained models, controllers, and evaluation components used in the master's thesis:

> **Beyond Prompting: Resource-Efficient Explicit Control for CEFR-Aligned Language Generation**

Unless otherwise noted, the controllable-generation experiments use `meta-llama/Llama-3.1-8B-Instruct` as the frozen language-model backbone.

Detailed implementation, training, and evaluation information is available in the corresponding notebooks and Hugging Face model cards.

---

# Model Index

| Model                                   | Role                                          |   Trainable Parameters | Hugging Face                                                                                             |
| --------------------------------------- | --------------------------------------------- | ---------------------: | -------------------------------------------------------------------------------------------------------- |
| RoBERTa-Large CEFR Joint-Loss Evaluator | Primary automatic CEFR evaluator              | Full-model fine-tuning | [Model](https://huggingface.co/MohammadKhosravi/roberta-large-cefr-classifier-JointLoss)                 |
| RoBERTa-Large CEFR WCE Evaluator        | CEFR evaluator baseline                       | Full-model fine-tuning | See `notebooks/02_cefr_evaluator/`                                                                       |
| Llama-3.1-8B CEFR Latent Classifier     | PPLM steering classifier                      |                 24,582 | [Model](https://huggingface.co/MohammadKhosravi/llama3.1-8b-cefr-steering-1layer-head-ordinal-universal) |
| LoRA                                    | Generic PEFT baseline                         |             41,943,040 | [Model](https://huggingface.co/MohammadKhosravi/llama3.1-8b-lora-cefr-steering-6k)                       |
| Standard Prefix-Tuning                  | Generic PEFT baseline                         |             68,254,720 | [Model](https://huggingface.co/MohammadKhosravi/llama3.1-8b-standard-prefix-tuning-6k)                   |
| Vanilla PrefixMemory-Tuning             | Unconditioned PMT baseline                    |            536,870,912 | [Model](https://huggingface.co/MohammadKhosravi/llama3.1-8b-pure-pmt-6k)                                 |
| CEFR Multi-Prefix Tuning — 68M          | Standard-PT parameter-matched CEFR controller |             68,408,320 | [Model](https://huggingface.co/MohammadKhosravi/llama3.1-8b-cefr-prefix-tuning-68m-param-matched)        |
| CEFR Multi-Prefix Tuning — 286M         | Intermediate capacity-scaling variant         |            286,019,584 | [Model](https://huggingface.co/MohammadKhosravi/llama3.1-8b-cefr-prefix-tuning-6k-286M-Variant)          |
| CEFR Multi-Prefix Tuning — 537M         | PMT parameter-matched CEFR controller         |            536,698,384 | [Model](https://huggingface.co/MohammadKhosravi/llama3.1-8b-cefr-pt-537m-param-matched)                  |
| CEFR-Gated PrefixMemory-Tuning          | Proposed architecture                         |            536,895,489 | [Model](https://huggingface.co/MohammadKhosravi/llama3.1-8b-pure-pmt-cefr-gating)                        |
| CEFR-Gated PMT — Prompt-Cue Ablation    | Prompt-cue ablation of the proposed method    |            536,895,489 | [Model](https://huggingface.co/MohammadKhosravi/llama3.1-8b-pure-pmt-cefr-gating-no-cefr-cues)           |

---

# CEFR Evaluation Models

## Primary Joint-Loss Evaluator

The primary automatic evaluator used throughout the generation experiments is:

```text
MohammadKhosravi/roberta-large-cefr-classifier-JointLoss
```

It is based on `FacebookAI/roberta-large` and is trained using a combination of:

* Weighted cross-entropy classification loss
* An ordinal expected-class MSE term

The final evaluator achieved:

| Metric                   |     Result |
| ------------------------ | ---------: |
| Strict Accuracy          | **98.33%** |
| Adjacent Accuracy        | **99.43%** |
| Macro F1                 | **97.47%** |
| MAE                      | **0.0240** |
| Quadratic Weighted Kappa | **0.9863** |

This evaluator is used as the primary automatic CEFR judge for the generation experiments.

### Hugging Face

[RoBERTa-Large CEFR Joint-Loss Evaluator](https://huggingface.co/MohammadKhosravi/roberta-large-cefr-classifier-JointLoss)

---

## Weighted Cross-Entropy Evaluator

A weighted-cross-entropy-only RoBERTa-Large model was also trained as an evaluator baseline.

Its evaluation results were:

| Metric                   |     Result |
| ------------------------ | ---------: |
| Strict Accuracy          | **98.41%** |
| Adjacent Accuracy        | **99.49%** |
| Macro F1                 | **97.32%** |
| MAE                      | **0.0230** |
| Quadratic Weighted Kappa | **0.9866** |

Although the WCE-only model obtains slightly higher Strict Accuracy, Adjacent Accuracy, MAE, and QWK, the **Joint-Loss evaluator was selected as the primary evaluator because it achieved the higher Macro F1 score**.

Related notebooks:

```text
notebooks/02_cefr_evaluator/
```

---

# PPLM Steering Classifier

PPLM uses a lightweight CEFR classifier operating on frozen Llama-3.1-8B hidden representations.

Architecture:

```text
4096-Dimensional Hidden Representation
                │
                ▼
         Dropout (0.35)
                │
                ▼
         Linear(4096, 6)
                │
                ▼
          CEFR Class Logits
       A1 A2 B1 B2 C1 C2
```

The classifier contains:

**24,582 trainable parameters**

### Hugging Face

[Llama-3.1-8B CEFR Latent Classifier](https://huggingface.co/MohammadKhosravi/llama3.1-8b-cefr-steering-1layer-head-ordinal-universal)

Related notebooks:

```text
notebooks/03_pplm/
```

---

# Generic Adaptation Baselines

Three generic adaptation baselines are evaluated.

| Method                     | Trainable Parameters | Strict Accuracy | Adjacent Accuracy |        MAE |
| -------------------------- | -------------------: | --------------: | ----------------: | ---------: |
| LoRA                       |           41,943,040 |          26.21% |            59.69% |     1.4501 |
| **Standard Prefix-Tuning** |           68,254,720 |      **32.34%** |        **64.67%** | **1.2422** |
| Vanilla PMT                |          536,870,912 |          15.67% |            47.58% |     1.7308 |

These models rely on textual CEFR instructions rather than a dedicated CEFR-specific architectural control variable.

The results demonstrate that larger controller capacity alone does not guarantee stronger CEFR controllability. In particular, Vanilla PMT contains substantially more trainable parameters than LoRA or Standard Prefix-Tuning while achieving lower CEFR-control accuracy.

Related notebooks:

```text
notebooks/04_generic_peft_baselines/
```

---

# CEFR Multi-Prefix Tuning

The CEFR Multi-Prefix experiments introduce **six CEFR-specific prefix banks**, one for each proficiency level:

```text
A1
A2
B1
B2
C1
C2
```

Three controller capacities are evaluated:

| Variant                      | Trainable Parameters | Strict Accuracy | Adjacent Accuracy |        MAE |
| ---------------------------- | -------------------: | --------------: | ----------------: | ---------: |
| CEFR Multi-Prefix — 68M      |           68,408,320 |          45.44% |            68.80% |     1.0940 |
| CEFR Multi-Prefix — 286M     |          286,019,584 |          52.42% |            76.35% |     0.8533 |
| **CEFR Multi-Prefix — 537M** |      **536,698,384** |      **64.10%** |        **81.48%** | **0.6966** |

The capacity-scaling progression is:

```text
~68M
45.44% Strict
    │
    ▼
~286M
52.42% Strict
    │
    ▼
~537M
64.10% Strict
```

Within the evaluated setup, CEFR-control performance improves as controller capacity increases.

---

## 68M Parameter-Matched Comparison

The 68M CEFR Multi-Prefix model is approximately parameter-matched to Standard Prefix-Tuning:

```text
Standard Prefix-Tuning:      68,254,720
CEFR Multi-Prefix — 68M:     68,408,320
Difference:                    +153,600
Relative Difference:           +0.2250%
```

Their Strict Accuracy results are:

```text
Standard Prefix-Tuning      32.34%
CEFR Multi-Prefix — 68M     45.44%
```

This yields an absolute improvement of:

**+13.10 percentage points**

at an approximately matched trainable parameter budget.

---

## 537M Parameter-Matched Comparison

The 537M CEFR Multi-Prefix model is approximately parameter-matched to Vanilla PMT:

```text
Vanilla PMT:                 536,870,912
CEFR Multi-Prefix — 537M:    536,698,384
Difference:                     −172,528
Relative Difference:            −0.0321%
```

Their Strict Accuracy results are:

```text
Vanilla PMT                   15.67%
CEFR Multi-Prefix — 537M      64.10%
```

This provides a controlled comparison between a high-capacity unconditioned memory controller and an explicitly CEFR-conditioned prefix architecture.

Related notebooks:

```text
notebooks/05_cefr_prefix_tuning/
```

---

# Proposed CEFR-Gated PrefixMemory-Tuning

The proposed architecture combines full PrefixMemory-Tuning memory matrices with **explicit CEFR-specific feature gating**.

## Trainable Components

| Component                          |      Parameters |
| ---------------------------------- | --------------: |
| 32 × `4096 × 4096` memory matrices |     536,870,912 |
| 6 × 4096 CEFR embeddings           |          24,576 |
| Learnable `alpha`                  |               1 |
| **Total**                          | **536,895,489** |

The Llama-3.1-8B-Instruct backbone remains frozen.

---

## Controller Computation

The controller computes:

```text
gated_X = ELU(hidden_states) * cefr_embedding

memory_bias = gated_X @ M_l

output = attention_output + alpha * memory_bias
```

Conceptually:

```text
Target CEFR
    │
    ▼
CEFR Embedding
    │
    ▼
Feature-Wise Multiplicative Gating
    │
    ▼
Full-Rank Layer Memory Matrix
    │
    ▼
Memory-Derived Bias
    │
    ▼
Scaled Residual Injection
```

The target CEFR representation therefore modulates the transformed hidden representation **before external memory retrieval**.

---

# CEFR-Gated PMT Evaluation

## In-Domain Evaluation

The full model is evaluated on:

```text
117 topics × 6 CEFR levels = 702 conditions
```

| Metric            |     Result |
| ----------------- | ---------: |
| Strict Accuracy   | **68.80%** |
| Adjacent Accuracy | **81.34%** |
| MAE               | **0.6368** |

## Out-of-Domain IELTS Evaluation

The same controller is evaluated on:

```text
50 IELTS Writing Task 2 prompts × 6 CEFR levels = 300 conditions
```

| Metric            |     Result |
| ----------------- | ---------: |
| Strict Accuracy   | **58.33%** |
| Adjacent Accuracy | **75.67%** |
| MAE               | **0.7700** |

### Hugging Face

[CEFR-Gated PrefixMemory-Tuning](https://huggingface.co/MohammadKhosravi/llama3.1-8b-pure-pmt-cefr-gating)

Related notebooks:

```text
notebooks/06_cefr_gated_pmt/
```

---

# Prompt-Cue Ablation

A separate ablation keeps the **CEFR-Gated PMT architecture unchanged** while removing the explicit lower- and upper-level proficiency cues from the textual prompt.

The controller still receives the target CEFR class through its architectural conditioning mechanism.

| Configuration           | Strict Accuracy | Adjacent Accuracy |         MAE |
| ----------------------- | --------------: | ----------------: | ----------: |
| **Full CEFR-Gated PMT** |      **68.80%** |        **81.34%** |  **0.6368** |
| Prompt-Cue Ablation     |          63.11% |            78.06% |      0.7407 |
| Difference              |    **−5.69 pp** |      **−3.28 pp** | **+0.1039** |

Removing the textual proficiency cues reduces Strict Accuracy by:

**5.69 percentage points**

while the cue-ablated controller still achieves:

**63.11% Strict Accuracy**

This experiment isolates the additional contribution of explicit textual proficiency guidance while preserving the architectural CEFR-conditioning mechanism.

### Hugging Face

[CEFR-Gated PMT — Prompt-Cue Ablation](https://huggingface.co/MohammadKhosravi/llama3.1-8b-pure-pmt-cefr-gating-no-cefr-cues)

---

# Summary of Main Generation Models

| Method                   | Trainable Parameters | Strict Accuracy | Adjacent Accuracy |        MAE |
| ------------------------ | -------------------: | --------------: | ----------------: | ---------: |
| LoRA                     |               41.94M |          26.21% |            59.69% |     1.4501 |
| Standard Prefix-Tuning   |               68.25M |          32.34% |            64.67% |     1.2422 |
| Vanilla PMT              |              536.87M |          15.67% |            47.58% |     1.7308 |
| CEFR Multi-Prefix — 68M  |               68.41M |          45.44% |            68.80% |     1.0940 |
| CEFR Multi-Prefix — 286M |              286.02M |          52.42% |            76.35% |     0.8533 |
| CEFR Multi-Prefix — 537M |              536.70M |          64.10% |        **81.48%** |     0.6966 |
| **CEFR-Gated PMT**       |          **536.90M** |      **68.80%** |            81.34% | **0.6368** |

The main generation experiments can be viewed as a progression from generic parameter-efficient adaptation toward increasingly explicit proficiency-control mechanisms:

```text
Generic Adaptation
│
├── LoRA
├── Standard Prefix-Tuning
└── Vanilla PMT
        │
        ▼
Explicit CEFR Conditioning
│
├── CEFR Multi-Prefix — 68M
├── CEFR Multi-Prefix — 286M
└── CEFR Multi-Prefix — 537M
        │
        ▼
CEFR-Gated PrefixMemory-Tuning
```

---

# Experimental Dimensions

The model registry highlights the principal experimental dimensions investigated in the thesis:

```text
Adaptation Architecture
          │
          ▼
Controller Capacity
          │
          ▼
Explicit CEFR Conditioning
          │
          ▼
Conditioning Architecture
          │
          ▼
Textual Prompt Cues
```

The parameter-matched comparisons are particularly important for distinguishing these factors:

```text
Standard Prefix-Tuning (~68.25M)
              vs.
CEFR Multi-Prefix (~68.41M)

              and

Vanilla PMT (~536.87M)
              vs.
CEFR Multi-Prefix (~536.70M)
              vs.
CEFR-Gated PMT (~536.90M)
```

These comparisons reduce parameter-count differences while changing the mechanism through which CEFR information is introduced.

---

# Notebook Index

```text
notebooks/
│
├── 01_data_preparation/
│
├── 02_cefr_evaluator/
│
├── 03_pplm/
│
├── 04_generic_peft_baselines/
│
├── 05_cefr_prefix_tuning/
│
└── 06_cefr_gated_pmt/
```

Training code, evaluation code, recorded notebook outputs, and additional experimental details are available under the corresponding directories.

---

# External Model Artifacts

The primary trained artifacts are hosted on Hugging Face:

* [RoBERTa-Large CEFR Joint-Loss Evaluator](https://huggingface.co/MohammadKhosravi/roberta-large-cefr-classifier-JointLoss)
* [Llama-3.1-8B CEFR Latent Classifier](https://huggingface.co/MohammadKhosravi/llama3.1-8b-cefr-steering-1layer-head-ordinal-universal)
* [LoRA](https://huggingface.co/MohammadKhosravi/llama3.1-8b-lora-cefr-steering-6k)
* [Standard Prefix-Tuning](https://huggingface.co/MohammadKhosravi/llama3.1-8b-standard-prefix-tuning-6k)
* [Vanilla PMT](https://huggingface.co/MohammadKhosravi/llama3.1-8b-pure-pmt-6k)
* [CEFR Multi-Prefix — 68M](https://huggingface.co/MohammadKhosravi/llama3.1-8b-cefr-prefix-tuning-68m-param-matched)
* [CEFR Multi-Prefix — 286M](https://huggingface.co/MohammadKhosravi/llama3.1-8b-cefr-prefix-tuning-6k-286M-Variant)
* [CEFR Multi-Prefix — 537M](https://huggingface.co/MohammadKhosravi/llama3.1-8b-cefr-pt-537m-param-matched)
* [CEFR-Gated PrefixMemory-Tuning](https://huggingface.co/MohammadKhosravi/llama3.1-8b-pure-pmt-cefr-gating)
* [CEFR-Gated PMT — Prompt-Cue Ablation](https://huggingface.co/MohammadKhosravi/llama3.1-8b-pure-pmt-cefr-gating-no-cefr-cues)
