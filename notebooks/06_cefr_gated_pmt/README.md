# CEFR-Gated PrefixMemory-Tuning

This directory contains the experiments for **CEFR-Gated PrefixMemory-Tuning (PMT)**, the main proposed control architecture in the master's thesis:

> **Beyond Prompting: Resource-Efficient Explicit Control for CEFR-Aligned Language Generation**

The method combines **PrefixMemory-Tuning** with explicit CEFR conditioning through trainable level embeddings and multiplicative feature gating.

The directory contains two experiments:

1. **Full CEFR-Gated PMT** — the complete proposed architecture with general textual proficiency cues.
2. **Prompt-Cue Ablation** — the same controller architecture with the explicit textual proficiency cues removed.

---

# Proposed Architecture

The backbone is:

```text
meta-llama/Llama-3.1-8B-Instruct
```

and remains completely frozen during controller training.

The CEFR-Gated PMT controller contains:

- One full `4096 × 4096` trainable memory matrix for each of the 32 transformer layers
- One trainable 4096-dimensional embedding for each CEFR level
- Six CEFR classes: `A1`, `A2`, `B1`, `B2`, `C1`, `C2`
- One learnable scaling scalar `alpha`

The controller performs:

```text
gated_X = ELU(hidden_states) * cefr_embedding

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
        Layer-Specific 4096 × 4096
               PMT Memory Matrix
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

# Mathematical Formulation

For transformer layer \(l\), let the hidden representation be:

$$
X^{(l)}
\in
\mathbb{R}^{B \times T \times 4096}
$$

where:

- \(B\) is the batch size
- \(T\) is the sequence length
- \(4096\) is the Llama hidden dimension

The nonlinear feature mapping is:

$$
\phi(X^{(l)})
=
\operatorname{ELU}(X^{(l)})
$$

For target CEFR class \(c\), the corresponding trainable embedding is:

$$
e_c
\in
\mathbb{R}^{4096}
$$

The CEFR-gated representation is:

$$
\widetilde{X}^{(l)}
=
\phi(X^{(l)})
\odot
e_c
$$

where \(\odot\) denotes element-wise multiplication and the CEFR embedding is broadcast across the batch and sequence dimensions.

Each transformer layer has an independent full-rank memory matrix:

$$
M^{(l)}
\in
\mathbb{R}^{4096 \times 4096}
$$

The external memory-derived bias is:

$$
B_{\mathrm{mem}}^{(l)}
=
\widetilde{X}^{(l)}
M^{(l)}
$$

The final attention representation becomes:

$$
H_{\mathrm{out}}^{(l)}
=
H_{\mathrm{attn}}^{(l)}
+
\alpha B_{\mathrm{mem}}^{(l)}
$$

where \(\alpha\) is a learnable scalar controlling the magnitude of the memory intervention.

---

# Trainable Parameters

The controller parameter budget is:

| Component | Trainable Parameters |
|---|---:|
| 32 PMT memory matrices | 536,870,912 |
| 6 CEFR embeddings | 24,576 |
| Learnable `alpha` | 1 |
| **Total** | **536,895,489** |

The Llama-3.1-8B-Instruct backbone remains frozen.

---

# Shared Experimental Setup

Both experiments use the same underlying experimental framework.

| Setting | Value |
|---|---|
| Backbone | `meta-llama/Llama-3.1-8B-Instruct` |
| Backbone state | Frozen |
| Training dataset | Balanced CEFR Steering Subset |
| Total examples | `5,568` |
| Samples per CEFR level | `928` |
| Train/validation split | Stratified 90/10 |
| Training examples | `5,011` |
| Validation examples | `557` |
| Random seed | `42` |
| Primary CEFR evaluator | `MohammadKhosravi/roberta-large-cefr-classifier-JointLoss` |
| In-domain benchmark | `117 topics × 6 CEFR levels = 702` conditions |
| OOD benchmark | `50 IELTS prompts × 6 CEFR levels = 300` conditions |

---

# Experiment 1 — Full CEFR-Gated PMT

## `cefr_gated_pmt_train_and_eval.ipynb`

This notebook contains the complete training and evaluation workflow for the full **CEFR-Gated PMT** model.

The full textual prompt contains general proficiency-oriented guidance for the lower and higher CEFR ranges.

The specific requested CEFR class itself is supplied through the architectural controller rather than being directly specified as the textual target label.

The model therefore combines:

```text
Architectural CEFR Conditioning
              +
General Textual Proficiency Guidance
```

---

## Training Results

| Epoch | Train Loss | Validation Loss | Validation PPL |
|---:|---:|---:|---:|
| 1 | 2.4011 | **2.0971** | **8.14** |
| 2 | 1.6818 | 2.1277 | 8.40 |
| 3 | 1.0301 | 2.4258 | 11.31 |
| 4 | 0.5113 | 2.7793 | 16.11 |
| 5 | **0.2538** | 3.1331 | 22.94 |

**Best validation checkpoint:** Epoch 1

Although training loss continues to decrease throughout training, validation performance deteriorates after the first epoch.

The Epoch 1 checkpoint is therefore retained for evaluation.

---

# In-Domain Evaluation

The full controller is evaluated on the final **In-Domain Evaluation Prompt Matrix**:

```text
117 topics × 6 CEFR levels = 702 generation conditions
```

## Overall Results

| Metric | Result |
|---|---:|
| Strict Accuracy | **68.80%** |
| Adjacent Accuracy | **81.34%** |
| MAE | **0.6368** |
| Mean Dependency Distance | **1.86** |
| Flesch Reading Ease | **82.44** |
| Sentence-Level CEFR Drift | **2.05** |

## Per-Level Strict Accuracy

| CEFR | Strict Accuracy |
|:---:|---:|
| A1 | **85.47%** |
| A2 | **86.32%** |
| B1 | **76.92%** |
| B2 | **68.38%** |
| C1 | **51.28%** |
| C2 | **44.44%** |

---

# Out-of-Domain IELTS Evaluation

The same trained controller is evaluated on an unseen IELTS Writing Task 2 benchmark:

```text
50 IELTS Writing Task 2 prompts
             ×
        6 CEFR levels
             =
   300 generation conditions
```

## Overall OOD Results

| Metric | Result |
|---|---:|
| Strict Accuracy | **58.33%** |
| Adjacent Accuracy | **75.67%** |
| MAE | **0.7700** |
| Mean Dependency Distance | **1.88** |
| Flesch Reading Ease | **78.13** |
| Sentence-Level CEFR Drift | **2.08** |

## OOD Per-Level Strict Accuracy

| CEFR | Strict Accuracy |
|:---:|---:|
| A1 | 60.00% |
| A2 | **86.00%** |
| B1 | 70.00% |
| B2 | 48.00% |
| C1 | 64.00% |
| C2 | 22.00% |

---

# Experiment 2 — Prompt-Cue Ablation

## `prompt_cue_ablation.ipynb`

This notebook contains the **prompt-cue ablation** of CEFR-Gated PMT.

The architecture remains unchanged:

```text
Same 32 full-rank PMT memory matrices
Same 6 CEFR embeddings
Same multiplicative CEFR gating
Same learnable alpha
Same frozen Llama backbone
```

The only experimental intervention is the removal of the explicit textual proficiency cues used by the full configuration.

Specifically, the ablation removes:

- The A1/A2 cue encouraging simple vocabulary, short sentences, and primitive structures
- The C1/C2 cue encouraging advanced vocabulary, idioms, and complex sentence structures

The target CEFR class continues to be supplied through the architectural conditioning mechanism.

The controlled comparison is therefore:

```text
Full CEFR-Gated PMT
        │
        ├── Architectural CEFR Conditioning
        │
        └── Textual Proficiency Cues
        │
        ▼
      Generation


Prompt-Cue Ablation
        │
        └── Architectural CEFR Conditioning
        │
        ▼
      Generation
```

---

# Prompt-Cue Ablation Training Results

| Epoch | Train Loss | Validation Loss | Validation PPL |
|---:|---:|---:|---:|
| 1 | 2.1370 | 1.8798 | 6.55 |
| 2 | **1.5536** | **1.8392** | **6.29** |
| 3 | 1.0606 | 2.0055 | 7.43 |

**Best validation checkpoint:** Epoch 2

---

# Prompt-Cue Ablation Evaluation

The cue-ablated controller is evaluated on the same **702-condition in-domain benchmark**.

| Metric | Result |
|---|---:|
| Strict Accuracy | **63.11%** |
| Adjacent Accuracy | **78.06%** |
| MAE | **0.7407** |
| Mean Dependency Distance | **1.78** |
| Flesch Reading Ease | **80.62** |
| Sentence-Level CEFR Drift | **2.36** |

## Per-Level Strict Accuracy

| CEFR | Strict Accuracy |
|:---:|---:|
| A1 | **84.62%** |
| A2 | **72.65%** |
| B1 | **77.78%** |
| B2 | **70.94%** |
| C1 | **47.86%** |
| C2 | **24.79%** |

---

# Prompt-Cue Ablation Comparison

The direct comparison between the two configurations is:

| Configuration | Strict Accuracy | Adjacent Accuracy | MAE |
|---|---:|---:|---:|
| **Full CEFR-Gated PMT** | **68.80%** | **81.34%** | **0.6368** |
| Prompt-Cue Ablation | 63.11% | 78.06% | 0.7407 |
| Difference | **−5.69 pp** | **−3.28 pp** | **+0.1039** |

Removing the explicit textual proficiency cues reduces strict accuracy from:

```text
68.80% → 63.11%
```

a decrease of **5.69 percentage points**.

However, the cue-ablated model still achieves **63.11% strict accuracy**.

Relative to the full model:

$$
\frac{63.11}{68.80}
\approx
91.7\%
$$

of the full model's strict accuracy is retained.

Within this experimental setup, this suggests that textual proficiency guidance contributes meaningfully to performance, while substantial CEFR controllability remains when the textual cues are removed.

---

# Comparison with Parameter-Matched Controls

The proposed CEFR-Gated PMT controller can also be compared with the approximately parameter-matched **~537M baselines**:

| Model | Trainable Parameters | Strict Accuracy | Adjacent Accuracy | MAE |
|---|---:|---:|---:|---:|
| Vanilla PMT | 536.87M | 15.67% | 47.58% | 1.7308 |
| CEFR Multi-Prefix — 537M | 536.70M | 64.10% | **81.48%** | 0.6966 |
| **CEFR-Gated PMT** | **536.90M** | **68.80%** | 81.34% | **0.6368** |

The parameter budgets are effectively matched:

```text
Vanilla PMT             ≈ 536.87M
CEFR Multi-Prefix       ≈ 536.70M
CEFR-Gated PMT          ≈ 536.90M
```

while strict accuracy differs substantially:

```text
Vanilla PMT
15.67%
   │
   ▼
CEFR Multi-Prefix
64.10%
   │
   ▼
CEFR-Gated PMT
68.80%
```

These controlled experiments help separate three factors:

```text
Controller Capacity
        │
        ▼
Explicit CEFR Conditioning
        │
        ▼
Conditioning Architecture
```

Vanilla PMT provides a high-capacity controller without explicit CEFR architectural conditioning.

CEFR Multi-Prefix introduces CEFR-specific prefix selection at approximately the same parameter budget.

CEFR-Gated PMT instead performs multiplicative CEFR conditioning directly on the transformed hidden representation before memory retrieval.

---

# Relationship to CEFR Multi-Prefix Tuning

Both approaches provide explicit CEFR conditioning but implement it differently.

## CEFR Multi-Prefix Tuning

```text
Target CEFR
     │
     ▼
Select CEFR-Specific
Prefix Bank
     │
     ▼
Project Virtual Tokens
     │
     ▼
Layer-Wise Prefix
Key/Value States
     │
     ▼
Frozen Llama
```

The target level determines **which continuous prefix representation is injected into attention**.

---

## CEFR-Gated PMT

```text
Target CEFR
     │
     ▼
Select CEFR Embedding
     │
     ▼
Gate Transformed
Hidden Representation
     │
     ▼
Query Layer-Specific
Memory Matrix
     │
     ▼
Memory-Derived Bias
     │
     ▼
Frozen Llama
```

Here, the target level determines **how the current hidden representation interacts with the external memory**.

The experiments therefore compare two structurally different mechanisms for introducing explicit proficiency control into a frozen language model.

---

# Evaluation Metrics

## Strict Accuracy

The predicted CEFR class must exactly match the requested target level.

For example:

```text
Target B1:

B1       → Correct
A2 / B2  → Incorrect
A1/C1/C2 → Incorrect
```

---

## Adjacent Accuracy

Predictions one neighboring CEFR level away from the requested target are also counted as acceptable.

For example:

```text
Target B1:

B1           → Strict + Adjacent correct
A2 / B2      → Adjacent correct
A1 / C1 / C2 → Incorrect
```

---

## Mean Absolute Error

CEFR levels are represented as ordered classes:

```text
A1 = 0
A2 = 1
B1 = 2
B2 = 3
C1 = 4
C2 = 5
```

MAE measures the average ordinal distance between requested and predicted proficiency levels:

$$
\operatorname{MAE}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left|
\hat{y}_i-y_i
\right|
$$

Lower values indicate stronger CEFR control.

---

# Linguistic Diagnostics

The evaluation additionally reports:

- **Mean Dependency Distance (MDD)**
- **Flesch Reading Ease**
- **Sentence-Level CEFR Drift**
- Per-level CEFR statistics
- Classification reports
- Confusion matrices

These metrics provide complementary information about the linguistic properties of the generated texts.

---

# Experimental Interpretation

The experiments in this directory address three related questions.

## 1. Does Explicit CEFR Conditioning Matter?

The parameter-matched comparison with Vanilla PMT provides evidence that high controller capacity alone does not explain the observed CEFR controllability.

```text
Vanilla PMT
536.87M parameters
15.67% Strict Accuracy

          vs.

CEFR-Gated PMT
536.90M parameters
68.80% Strict Accuracy
```

The two systems have effectively identical trainable parameter budgets but substantially different CEFR-control performance.

---

## 2. Does the Conditioning Architecture Matter?

The comparison with CEFR Multi-Prefix controls for both parameter scale and explicit CEFR conditioning:

```text
CEFR Multi-Prefix
536.70M
64.10%

        vs.

CEFR-Gated PMT
536.90M
68.80%
```

Both architectures explicitly encode the target CEFR level, but they inject the conditioning information differently.

Within this experimental setup, CEFR-Gated PMT achieves **4.70 percentage points higher strict accuracy** and lower MAE:

```text
0.6966 → 0.6368
```

while CEFR Multi-Prefix has slightly higher adjacent accuracy:

```text
81.48% vs. 81.34%
```

---

## 3. How Much Do the Prompt Cues Contribute?

The prompt-cue ablation keeps the controller architecture fixed while removing the explicit textual proficiency cues.

```text
Full CEFR-Gated PMT
68.80%

        ↓ remove cues

Prompt-Cue Ablation
63.11%
```

The **5.69 percentage-point decrease** indicates that textual guidance contributes to performance.

At the same time, substantial CEFR control remains without those cues, supporting the conclusion that the architectural conditioning mechanism itself contributes materially to controllability.

---

# Repository Structure

```text
06_cefr_gated_pmt/
├── README.md
├── cefr_gated_pmt_train_and_eval.ipynb
└── prompt_cue_ablation.ipynb
```

---

# Related Directories

CEFR Multi-Prefix capacity-scaling experiments:

```text
notebooks/05_cefr_prefix_tuning/
```

Generic PEFT and PMT baselines:

```text
notebooks/04_generic_peft_baselines/
```

Training-data and evaluation-matrix construction:

```text
notebooks/01_data_preparation/
```

Primary CEFR evaluator:

```text
notebooks/02_cefr_evaluator/
```

---

# External Model Artifacts

The trained controllers are hosted on Hugging Face.

## Full CEFR-Gated PMT

```text
MohammadKhosravi/llama3.1-8b-pure-pmt-cefr-gating
```

https://huggingface.co/MohammadKhosravi/llama3.1-8b-pure-pmt-cefr-gating

## Prompt-Cue Ablation

```text
MohammadKhosravi/llama3.1-8b-pure-pmt-cefr-gating-no-cefr-cues
```

https://huggingface.co/MohammadKhosravi/llama3.1-8b-pure-pmt-cefr-gating-no-cefr-cues

Generated outputs, detailed logs, profiling files, and other large intermediate artifacts are stored separately from the GitHub repository.

---

# Summary

CEFR-Gated PMT introduces explicit proficiency control into PrefixMemory-Tuning through **feature-wise multiplicative CEFR conditioning before full-rank memory retrieval**.

The main results are:

| Configuration | In-Domain Strict | OOD Strict | Adjacent | MAE |
|---|---:|---:|---:|---:|
| Vanilla PMT | 15.67% | — | 47.58% | 1.7308 |
| CEFR Multi-Prefix — 537M | 64.10% | — | **81.48%** | 0.6966 |
| Prompt-Cue Ablation | 63.11% | — | 78.06% | 0.7407 |
| **CEFR-Gated PMT** | **68.80%** | **58.33%** | 81.34% | **0.6368** |

The experiments show that:

- High parameter capacity alone is insufficient for strong CEFR control.
- Explicit architectural CEFR conditioning substantially improves controllability.
- The mechanism used to incorporate CEFR information affects performance.
- Textual proficiency cues provide an additional performance benefit.
- The proposed CEFR-Gated PMT controller retains substantial control when those textual cues are removed.
- The full controller transfers from the in-domain benchmark to an unseen IELTS prompt distribution, achieving **58.33% strict accuracy** across 300 OOD conditions.
