# Beyond Prompting: Resource-Efficient Explicit Control for CEFR-Aligned Language Generation

Master's thesis repository for research on **explicit architectural control of English text generation across CEFR proficiency levels (A1–C2)**.

The experiments use `meta-llama/Llama-3.1-8B-Instruct` as a frozen language-model backbone and compare:

* Prompt-based control
* PPLM decoding-time steering
* Parameter-efficient fine-tuning
* Standard Prefix-Tuning
* PrefixMemory-Tuning
* Explicitly CEFR-conditioned controllers

The main proposed method is **CEFR-Gated PrefixMemory-Tuning (PMT)**, which introduces trainable CEFR representations and multiplicative feature gating into a full-rank PrefixMemory controller.

---

# Overview

Controlling a language model to reliably generate text at a requested CEFR proficiency level is challenging because adjacent CEFR levels share substantial lexical and syntactic characteristics, particularly at the upper proficiency levels.

This thesis investigates three primary factors affecting controllability:

```text
Adaptation Architecture
          +
Controller Capacity
          +
Explicit CEFR Conditioning
```

The experimental progression is:

```text
Prompt-Only Base LLM
        │
        ▼
       PPLM
        │
        ▼
Generic PEFT / Adaptation Baselines
│
├── LoRA
├── Standard Prefix-Tuning
└── Vanilla PrefixMemory-Tuning
        │
        ▼
CEFR Multi-Prefix Tuning
│
├── ~68M
├── ~286M
└── ~537M
        │
        ▼
CEFR-Gated PrefixMemory-Tuning
```

The experiments are designed not only to compare final CEFR-control accuracy, but also to distinguish improvements caused by:

* Increased controller capacity
* Explicit CEFR conditioning
* Conditioning architecture
* Textual proficiency cues

---

# Proposed CEFR-Gated PMT Architecture

The proposed controller augments the frozen `meta-llama/Llama-3.1-8B-Instruct` backbone with:

* One full `4096 × 4096` trainable memory matrix for each of the 32 transformer layers
* One trainable 4096-dimensional embedding for each CEFR level
* One learnable scaling scalar `alpha`
* Multiplicative CEFR feature gating before memory retrieval

For target CEFR level `c` and transformer layer `l`, the controller computes:

```text
gated_X = ELU(hidden_states) * cefr_embedding[c]

memory_bias = gated_X @ M_l

output = attention_output + alpha * memory_bias
```

Conceptually:

```text
                 Target CEFR Level
                         │
                         ▼
              4096-D CEFR Embedding
                         │
                         ▼
                  Hidden States
                         │
                         ▼
                       ELU
                         │
                         ▼
           Element-Wise CEFR Gating
                         │
                         ▼
       Layer-Specific Full-Rank Memory
                4096 × 4096
                         │
                         ▼
              Memory-Derived Bias
                         │
                         ▼
             Learnable Scaling α
                         │
                         ▼
       Residual Attention Injection
                         │
                         ▼
         Frozen Llama-3.1-8B
```

---

# Trainable Parameters

| Component              | Trainable Parameters |
| ---------------------- | -------------------: |
| 32 PMT memory matrices |          536,870,912 |
| 6 CEFR embeddings      |               24,576 |
| Learnable `alpha`      |                    1 |
| **Total**              |      **536,895,489** |

The Llama-3.1-8B-Instruct backbone remains frozen.

---

# Main Results

## In-Domain CEFR Control

The principal evaluation benchmark contains:

```text
117 topics × 6 CEFR levels = 702 generation conditions
```

| Method                   |     Trainable Parameters | Strict Accuracy |
| ------------------------ | -----------------------: | --------------: |
| Base LLM — Prompt-Only   |                        0 |          24.50% |
| PPLM                     | External latent steering |          23.17% |
| LoRA                     |                   41.94M |          26.21% |
| Standard Prefix-Tuning   |                   68.25M |          32.34% |
| Vanilla PMT              |                  536.87M |          15.67% |
| CEFR Multi-Prefix — 68M  |                   68.41M |          45.44% |
| CEFR Multi-Prefix — 286M |                  286.02M |          52.42% |
| CEFR Multi-Prefix — 537M |                  536.70M |          64.10% |
| **CEFR-Gated PMT**       |              **536.90M** |      **68.80%** |

The proposed CEFR-Gated PMT model obtains:

| Metric                    |     Result |
| ------------------------- | ---------: |
| Strict Accuracy           | **68.80%** |
| Adjacent Accuracy         | **81.34%** |
| Mean Absolute Error       | **0.6368** |
| Mean Dependency Distance  |   **1.86** |
| Flesch Reading Ease       |  **82.44** |
| Sentence-Level CEFR Drift |   **2.05** |

---

# Parameter-Matched Comparisons

Two controlled comparisons are used to separate controller capacity from explicit CEFR conditioning.

## Standard Prefix-Tuning vs. CEFR Multi-Prefix — 68M

| Method                      | Trainable Parameters | Strict Accuracy |
| --------------------------- | -------------------: | --------------: |
| Standard Prefix-Tuning      |           68,254,720 |          32.34% |
| **CEFR Multi-Prefix — 68M** |       **68,408,320** |      **45.44%** |

Parameter-count difference:

**+153,600 parameters (+0.2250%)**

Strict Accuracy improves by:

**+13.10 percentage points**

---

## Vanilla PMT vs. CEFR Multi-Prefix — 537M

| Method                       | Trainable Parameters | Strict Accuracy |
| ---------------------------- | -------------------: | --------------: |
| Vanilla PMT                  |          536,870,912 |          15.67% |
| **CEFR Multi-Prefix — 537M** |      **536,698,384** |      **64.10%** |

Parameter-count difference:

**−172,528 parameters (−0.0321%)**

Strict Accuracy improves by:

**+48.43 percentage points**

These comparisons show that increased controller capacity alone does not explain the improvements obtained by the explicitly CEFR-conditioned models within the evaluated setup.

---

# Controller-Capacity Scaling

CEFR Multi-Prefix Tuning is evaluated at three controller capacities:

| Variant   | Trainable Parameters | Strict Accuracy | Adjacent Accuracy |        MAE |
| --------- | -------------------: | --------------: | ----------------: | ---------: |
| ~68M      |               68.41M |          45.44% |            68.80% |     1.0940 |
| ~286M     |              286.02M |          52.42% |            76.35% |     0.8533 |
| **~537M** |          **536.70M** |      **64.10%** |        **81.48%** | **0.6966** |

The progression is:

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

Within this architecture, increasing controller capacity consistently improves CEFR-control performance.

---

# Out-of-Domain Evaluation

Generalization is evaluated using **50 IELTS Writing Task 2 prompts**, each generated at all six CEFR levels:

```text
50 prompts × 6 CEFR levels = 300 conditions
```

The proposed CEFR-Gated PMT model obtains:

| Metric                    |     Result |
| ------------------------- | ---------: |
| Strict Accuracy           | **58.33%** |
| Adjacent Accuracy         | **75.67%** |
| Mean Absolute Error       | **0.7700** |
| Mean Dependency Distance  |   **1.88** |
| Flesch Reading Ease       |  **78.13** |
| Sentence-Level CEFR Drift |   **2.08** |

This benchmark measures CEFR control under a prompt and topic distribution distinct from the EFCAMDAT-derived in-domain benchmark.

---

# Prompt-Cue Ablation

A separate ablation removes the explicit lower- and upper-level proficiency cues from the textual prompt while preserving the CEFR-Gated PMT architecture.

| Configuration           | Strict Accuracy | Adjacent Accuracy |         MAE |
| ----------------------- | --------------: | ----------------: | ----------: |
| **Full CEFR-Gated PMT** |      **68.80%** |        **81.34%** |  **0.6368** |
| Prompt-Cue Ablation     |          63.11% |            78.06% |      0.7407 |
| Difference              |    **−5.69 pp** |      **−3.28 pp** | **+0.1039** |

Removing the textual proficiency cues reduces Strict Accuracy by **5.69 percentage points**.

However, the cue-ablated model still achieves **63.11% Strict Accuracy**, indicating that substantial CEFR control remains through the architectural conditioning mechanism.

---

# Resource Efficiency

## Training Resource Profiling

A dedicated profiling run for CEFR-Gated PMT reports:

| Metric                          |                Result |
| ------------------------------- | --------------------: |
| GPU                             | NVIDIA A100-SXM4-80GB |
| Trainable controller parameters |           536,895,489 |
| Frozen backbone parameters      |         8,030,261,248 |
| Training + validation time      |              875.83 s |
| Peak allocated GPU memory       |              48.74 GB |
| Average GPU utilization         |                 96.4% |

This profiling experiment is separate from the canonical training run used for model-quality evaluation.

---

## Inference Throughput

Controlled latency experiments were performed on an **NVIDIA L4 GPU**.

| Method                 | Generation Throughput |
| ---------------------- | --------------------: |
| Base LLM — Prompt-Only |           13.54 tok/s |
| CEFR-Gated PMT         |           13.16 tok/s |

The proposed controller retains approximately:

**97.2% of Base LLM generation throughput**

corresponding to an approximately:

**2.8% throughput reduction**

in this benchmark.

---

# Upper-CEFR Diagnostic Analysis

A separate diagnostic analysis investigates why **C1 and C2** remain more difficult to control.

The analysis examines:

* Lexical features
* Syntactic complexity
* Readability
* Word-count distributions
* Topic diversity
* Exact duplicate rates
* Standardized pairwise effect sizes

Many common linguistic features show relatively small standardized differences between C1 and C2.

The balanced subset also contains unequal topic diversity:

```text
A1–C1: 24 distinct topics per CEFR level
C2:      8 distinct topics
```

These findings provide a data-level explanation for some of the remaining difficulty at the highest proficiency levels.

---

# Repository Structure

```text
master-thesis/
│
├── README.md
├── requirements.txt
│
├── docs/
│   ├── models.md
│   └── datasets.md
│
└── notebooks/
    │
    ├── 01_data_preparation/
    │   ├── README.md
    │   └── ...
    │
    ├── 02_cefr_evaluator/
    │   ├── README.md
    │   └── ...
    │
    ├── 03_pplm/
    │   ├── README.md
    │   └── ...
    │
    ├── 04_generic_peft_baselines/
    │   ├── README.md
    │   ├── lora_train_and_eval.ipynb
    │   ├── standard_prefix_tuning_train_and_eval.ipynb
    │   └── vanilla_pmt_train_and_eval.ipynb
    │
    ├── 05_cefr_prefix_tuning/
    │   ├── README.md
    │   ├── cefr_prefix_tuning_68m_train_and_eval.ipynb
    │   ├── cefr_prefix_tuning_286m_train_and_eval.ipynb
    │   └── cefr_prefix_tuning_537m_train_and_eval.ipynb
    │
    ├── 06_cefr_gated_pmt/
    │   ├── README.md
    │   ├── cefr_gated_pmt_train_and_eval.ipynb
    │   └── prompt_cue_ablation.ipynb
    │
    ├── 07_resource_profiling/
    │   ├── README.md
    │   ├── cefr_gated_pmt_training_resource_profiling.ipynb
    │   └── inference_latency_benchmark.ipynb
    │
    └── 08_diagnostics/
        ├── README.md
        └── upper_cefr_diagnostic_analysis.ipynb
```

Each directory contains its own README describing the corresponding experiment and recorded results.

---

# Documentation

Additional consolidated documentation is available in:

* `docs/models.md` — model registry, Hugging Face artifacts, parameter counts, and principal results
* `docs/datasets.md` — datasets, derived subsets, evaluation matrices, and construction procedures

---

# Datasets and Supplementary Artifacts

Publicly shared dataset artifacts and larger supplementary resources are hosted separately from GitHub:

[**svgThesis Datasets and Supplementary Artifacts — Google Drive**](https://drive.google.com/drive/folders/1HdhOiLqOZ-hJIdMEixh53tcWCtHmonJk?usp=sharing)

Dataset-construction code is provided in:

```text
notebooks/01_data_preparation/
```

Large trained model/controller artifacts are hosted on Hugging Face rather than stored directly in this repository.

See `docs/models.md` for the complete model registry.

---

# Primary Model Artifacts

## Proposed CEFR-Gated PMT

[MohammadKhosravi/llama3.1-8b-pure-pmt-cefr-gating](https://huggingface.co/MohammadKhosravi/llama3.1-8b-pure-pmt-cefr-gating)

## Primary Joint-Loss CEFR Evaluator

[MohammadKhosravi/roberta-large-cefr-classifier-JointLoss](https://huggingface.co/MohammadKhosravi/roberta-large-cefr-classifier-JointLoss)

All additional model artifacts are indexed in:

```text
docs/models.md
```

---

# Installation

The experiments were developed primarily in **Google Colab GPU environments**.

Clone the repository:

```bash
git clone https://github.com/Mohammad1374-kh/master-thesis.git
cd master-thesis
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install the English spaCy model used for linguistic diagnostics:

```bash
python -m spacy download en_core_web_sm
```

---

# Hugging Face Authentication

The experiments use the gated:

```text
meta-llama/Llama-3.1-8B-Instruct
```

model.

A Hugging Face account with access to the model is therefore required.

When running the notebooks in Google Colab, add a Hugging Face access token to:

```text
Colab Secrets → HF_TOKEN
```

The notebooks read the token through `google.colab.userdata`.

Credentials should **never be hard-coded** into notebook source.

---

# Google Drive Paths

The public notebooks use placeholder Drive paths such as:

```text
/content/drive/MyDrive/Your_Path/
```

Replace `Your_Path` with the appropriate location in your own Google Drive before running the notebooks.

---

# Reproducibility

The notebooks preserve the original experimental outputs, including:

* Dataset and split statistics
* Parameter counts
* Training and validation losses
* Validation perplexities
* Generation outputs
* CEFR evaluation metrics
* Classification reports
* Confusion matrices
* Resource-profiling measurements

Randomized dataset operations generally use:

```text
random_state = 42
```

or the equivalent framework-specific seed.

Large model checkpoints, datasets, generated outputs, and profiling artifacts are hosted externally to keep the GitHub repository lightweight.

---

# Evaluation

The main automatic evaluator is:

```text
MohammadKhosravi/roberta-large-cefr-classifier-JointLoss
```

The principal CEFR-control metrics are:

* **Strict Accuracy** — predicted CEFR must exactly match the requested level
* **Adjacent Accuracy** — predictions one CEFR level above or below the target are also accepted
* **Mean Absolute Error (MAE)** — ordinal distance between requested and predicted CEFR levels

Additional linguistic diagnostics include:

* Mean Dependency Distance
* Flesch Reading Ease
* Sentence-Level CEFR Drift

---

# Experimental Summary

The central comparison among the high-capacity controllers is:

| Model                    | Trainable Parameters | Strict Accuracy | Adjacent Accuracy |        MAE |
| ------------------------ | -------------------: | --------------: | ----------------: | ---------: |
| Vanilla PMT              |              536.87M |          15.67% |            47.58% |     1.7308 |
| CEFR Multi-Prefix — 537M |              536.70M |          64.10% |        **81.48%** |     0.6966 |
| **CEFR-Gated PMT**       |          **536.90M** |      **68.80%** |            81.34% | **0.6368** |

These models have nearly identical controller capacities but substantially different CEFR-control behavior.

The complete experimental progression therefore provides evidence that performance depends not only on the number of trainable parameters, but also on:

```text
Controller Capacity
        +
Explicit CEFR Conditioning
        +
Conditioning Architecture
        +
Textual Proficiency Guidance
```

---

# Thesis

This repository accompanies the master's thesis:

> **Beyond Prompting: Resource-Efficient Explicit Control for CEFR-Aligned Language Generation**

**University of Padova**
**Master's Degree in Computer Science**

---

# Citation

If you use the code, trained models, datasets, or experimental resources from this repository, please cite the associated master's thesis and this repository.

A formal citation entry can be added after the thesis is officially archived or published.
