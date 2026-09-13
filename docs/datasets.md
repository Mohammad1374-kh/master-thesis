# Dataset Registry

This document summarizes the datasets and derived data resources used in the master's thesis:

> **Beyond Prompting: Resource-Efficient Explicit Control for CEFR-Aligned Language Generation**

Dataset construction and preprocessing code is available under:

```text
notebooks/01_data_preparation/
```

Publicly shared dataset artifacts and supplementary data files are available from the project Google Drive:

[**svgThesis Datasets and Supplementary Artifacts**](https://drive.google.com/drive/folders/1HdhOiLqOZ-hJIdMEixh53tcWCtHmonJk?usp=sharing)

---

# Dataset Overview

| Dataset / Resource                           | Purpose                                                        |                   Size |
| -------------------------------------------- | -------------------------------------------------------------- | ---------------------: |
| EFCAMDAT Source Corpus                       | Source learner-text corpus                                     |          622,146 texts |
| Steering Training Dataset                    | CEFR-conditioned generation training pool                      |           52,657 texts |
| Balanced CEFR Steering Subset                | Training of PEFT, Prefix-Tuning, PMT, and ablation controllers |            5,568 texts |
| Evaluation-Judge Dataset                     | Training of automatic CEFR evaluators                          |          104,125 texts |
| In-Domain Evaluation Prompt Matrix           | Main in-domain generation benchmark                            |         702 conditions |
| IELTS Out-of-Domain Evaluation Prompt Matrix | Out-of-domain generation benchmark                             |         300 conditions |
| PPLM Latent Representation Dataset           | Training of the PPLM latent CEFR classifier                    | 66,494 representations |

---

# EFCAMDAT Source Corpus

The primary source corpus used in this thesis is **EFCAMDAT**.

The thesis preprocessing pipeline operates on:

**622,146 learner texts**

The source corpus is processed to derive the training and evaluation resources used throughout the experiments.

The corresponding preprocessing pipeline is available in:

```text
notebooks/01_data_preparation/efcamdat_preprocessing_and_partitioning.ipynb
```

---

# Steering Training Dataset

The **Steering Training Dataset** is the main CEFR-labelled text pool used to construct the balanced controller-training subset.

It contains:

| CEFR Level |    Samples |
| :--------: | ---------: |
|     A1     |     14,655 |
|     A2     |     12,889 |
|     B1     |     10,850 |
|     B2     |     10,570 |
|     C1     |      2,765 |
|     C2     |        928 |
|  **Total** | **52,657** |

The distribution is strongly imbalanced, particularly at the advanced C1 and C2 levels.

This motivates construction of a balanced subset for the main controllable-generation experiments.

---

# Balanced CEFR Steering Subset

The **Balanced CEFR Steering Subset** is used to train the controllable-generation models.

Because C2 is the smallest class with **928 available examples**, the subset samples exactly **928 examples from each CEFR level**.

| CEFR Level |   Samples |
| :--------: | --------: |
|     A1     |       928 |
|     A2     |       928 |
|     B1     |       928 |
|     B2     |       928 |
|     C1     |       928 |
|     C2     |       928 |
|  **Total** | **5,568** |

Sampling uses:

```text
random_state = 42
```

For model training, the balanced subset is further divided using a stratified 90/10 split:

| Partition  |   Samples |
| ---------- | --------: |
| Training   |     5,011 |
| Validation |       557 |
| **Total**  | **5,568** |

Construction code:

```text
notebooks/01_data_preparation/balanced_cefr_steering_subset.ipynb
```

This subset is used by:

* LoRA
* Standard Prefix-Tuning
* Vanilla PrefixMemory-Tuning
* CEFR Multi-Prefix Tuning
* CEFR-Gated PMT
* Prompt-cue ablation experiments

---

# Evaluation-Judge Dataset

The **Evaluation-Judge Dataset** is used for training the automatic CEFR evaluation models.

It contains:

**104,125 texts**

This dataset is separate from the balanced generation-controller training subset.

The primary evaluator trained from this resource is:

```text
MohammadKhosravi/roberta-large-cefr-classifier-JointLoss
```

Evaluator training code is available under:

```text
notebooks/02_cefr_evaluator/
```

---

# In-Domain Evaluation Prompt Matrix

The main generation benchmark is the **In-Domain Evaluation Prompt Matrix**.

The original candidate set contained:

```text
128 topics × 6 CEFR levels = 768 conditions
```

After manual removal of **11 problematic conversational / role-play topics**, the final benchmark contains:

```text
117 topics × 6 CEFR levels = 702 conditions
```

Each topic is evaluated at all six CEFR proficiency levels:

```text
A1
A2
B1
B2
C1
C2
```

The same **702-condition matrix** is used across the main generation methods to enable direct and consistent comparison.

Construction code:

```text
notebooks/01_data_preparation/efcamdat_preprocessing_and_partitioning.ipynb
```

---

# IELTS Out-of-Domain Evaluation Prompt Matrix

Out-of-domain generalization is evaluated using **IELTS Writing Task 2 prompts** derived from:

```text
nlpatunt/D_Ielts_Writing_Task_2_Dataset
```

A deterministic sample of **50 unique prompts** is selected using seed `42`.

Each prompt is evaluated at all six CEFR levels:

```text
50 prompts × 6 CEFR levels = 300 conditions
```

The resulting benchmark is used to test the proposed CEFR-Gated PMT architecture outside the EFCAMDAT topic distribution.

Construction code:

```text
notebooks/01_data_preparation/ielts_out_of_domain_evaluation_prompt_matrix.ipynb
```

---

# PPLM Latent Representation Dataset

The PPLM steering classifier is trained on frozen Llama-3.1-8B hidden-state representations.

The final latent dataset contains:

**66,494 representations**

Each sample consists of:

* A **4096-dimensional pooled hidden-state vector**
* A corresponding CEFR class label

The dataset is available on Hugging Face as:

```text
MohammadKhosravi/cefr-llama3.1-8b-hidden-states-efcamdat-universal
```

Construction code:

```text
notebooks/01_data_preparation/pplm_latent_dataset_construction_.ipynb
```

---

# Diagnostic Dataset / Derived Feature Resource

The upper-CEFR diagnostic analysis operates on the same **Balanced CEFR Steering Subset**.

It derives linguistic and corpus-level features for all **5,568 balanced examples**.

The diagnostic features include:

* Lexical complexity
* Syntactic complexity
* Readability
* Topic coverage
* Duplicate-text statistics
* Additional CEFR-related diagnostics

These features are treated as **intermediate analysis artifacts**, rather than as a separate model-training dataset.

Related notebook:

```text
notebooks/08_diagnostics/upper_cefr_diagnostic_analysis.ipynb
```

---

# Dataset Relationships

The main data flow can be summarized as:

```text
                     EFCAMDAT Source Corpus
                        622,146 texts
                              │
                              ▼
                 Preprocessing and Partitioning
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
  Steering Training Dataset         Evaluation-Judge Dataset
       52,657 texts                     104,125 texts
             │                                 │
             ▼                                 ▼
 Balanced CEFR Steering Subset        CEFR Evaluators
        5,568 texts
             │
     ┌───────┼─────────────────────┐
     │       │                     │
     ▼       ▼                     ▼
Generic   CEFR Multi-Prefix    CEFR-Gated PMT
Baselines     Models              Models
```

Evaluation resources are constructed separately:

```text
EFCAMDAT Topics
      │
      ▼
In-Domain Prompt Matrix
117 topics × 6 levels
      │
      ▼
702 Generation Conditions


IELTS Writing Task 2
      │
      ▼
50 Sampled Prompts
      │
      ▼
50 prompts × 6 levels
      │
      ▼
300 OOD Conditions
```

---

# Dataset Usage by Experiment

| Dataset / Resource                 | Primary Use                                                 |
| ---------------------------------- | ----------------------------------------------------------- |
| EFCAMDAT Source Corpus             | Source corpus for derived training and evaluation resources |
| Steering Training Dataset          | Source pool for controller-training data                    |
| Balanced CEFR Steering Subset      | Training all main generation controllers                    |
| Evaluation-Judge Dataset           | Training CEFR evaluation classifiers                        |
| In-Domain Evaluation Prompt Matrix | Main generation benchmark                                   |
| IELTS OOD Prompt Matrix            | Generalization evaluation                                   |
| PPLM Latent Representation Dataset | Training the PPLM steering classifier                       |
| Diagnostic Features                | Upper-CEFR and linguistic diagnostic analysis               |

---

# Data and Supplementary Artifacts

Publicly shared dataset files and larger supplementary artifacts are hosted outside GitHub to keep the repository lightweight.

Google Drive:

[**svgThesis Datasets and Supplementary Artifacts**](https://drive.google.com/drive/folders/1HdhOiLqOZ-hJIdMEixh53tcWCtHmonJk?usp=sharing)

The GitHub repository contains the code required to construct, preprocess, partition, and analyze these resources.

Trained model and controller artifacts are hosted separately on Hugging Face.

---

# Reproducibility

Where sampling is required, the thesis generally uses deterministic random seeds, particularly:

```text
random_state = 42
```

This is used for procedures such as:

* Balanced CEFR sampling
* Train/validation splitting
* IELTS prompt sampling

The individual preprocessing notebooks document the exact:

* Filtering procedures
* Sampling strategies
* Class balancing
* Partitioning logic
* Benchmark construction

used to create each derived data resource.

---

# Repository Locations

```text
notebooks/
│
├── 01_data_preparation/
│   ├── efcamdat_preprocessing_and_partitioning.ipynb
│   ├── balanced_cefr_steering_subset.ipynb
│   ├── ielts_out_of_domain_evaluation_prompt_matrix.ipynb
│   └── pplm_latent_dataset_construction_.ipynb
│
├── 02_cefr_evaluator/
│
└── 08_diagnostics/
    └── upper_cefr_diagnostic_analysis.ipynb
```

---

# Summary

The thesis data pipeline separates three major functions:

```text
Training Data
     │
     ├── Generation Controller Training
     │
     └── CEFR Evaluator Training

Evaluation Data
     │
     ├── In-Domain Benchmark
     │
     └── Out-of-Domain IELTS Benchmark

Derived Representations
     │
     ├── PPLM Hidden-State Dataset
     │
     └── Linguistic Diagnostic Features
```

The core controller-training resource is the **5,568-example Balanced CEFR Steering Subset**, while model evaluation is standardized using a **702-condition in-domain benchmark** and a **300-condition IELTS out-of-domain benchmark**.

This organization allows the thesis experiments to maintain consistent training conditions while evaluating both in-domain CEFR controllability and out-of-domain generalization.
