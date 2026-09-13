# Data Preparation

This directory contains the notebooks used to construct and prepare the datasets employed throughout the thesis.

The data-preparation pipeline covers preprocessing and partitioning of the EFCAMDAT corpus, construction of the balanced CEFR steering subset used for model training, preparation of the latent representation dataset used by the PPLM baseline, and construction of the IELTS-based out-of-domain evaluation benchmark.

## Notebooks

### `efcamdat_preprocessing_and_partitioning_.ipynb`

Preprocesses the EFCAMDAT data and constructs the main experimental resources used in the thesis.

The notebook includes:

- extraction and cleaning of the EFCAMDAT source data;
- construction of the processed EFCAMDAT corpus;
- construction of the **Steering Training Dataset**;
- construction of the **Evaluation-Judge Dataset**;
- construction of the initial **In-Domain Evaluation Prompt Matrix**;
- validation of dataset distributions and overlap constraints.

The initial in-domain prompt matrix contains 128 topics crossed with the six CEFR levels. Following manual inspection, 11 prompts that induced unintended role-play or conversational behavior were removed, producing the final matrix used in the thesis:

**117 topics × 6 CEFR levels = 702 generation conditions.**

---

### `balanced_cefr_steering_subset.ipynb`

Constructs the balanced CEFR training subset used for the controlled adaptation experiments.

The notebook:

- maps EFCAMDAT assignment topics to the Steering Training Dataset;
- normalizes CEFR labels;
- identifies C2 as the smallest class with 928 available examples;
- downsamples each CEFR level to 928 samples using `random_state=42`;
- shuffles the resulting dataset reproducibly.

The final balanced configuration contains:

| CEFR level | Samples |
|---|---:|
| A1 | 928 |
| A2 | 928 |
| B1 | 928 |
| B2 | 928 |
| C1 | 928 |
| C2 | 928 |
| **Total** | **5,568** |

This dataset is used by the LoRA, Prefix-Tuning, PrefixMemory-Tuning, and CEFR-conditioned controller experiments.

---

### `pplm_latent_dataset_construction_.ipynb`

Constructs the latent-representation dataset used for training the CEFR attribute classifier employed by the PPLM baseline.

The notebook:

- loads the **Steering Training Dataset**;
- passes the learner texts through the frozen `Llama-3.1-8B-Instruct` backbone;
- extracts hidden-state representations;
- applies attention-mask-aware mean pooling;
- combines the resulting EFCAMDAT representations with the pre-existing CEFR latent dataset used in the experiment;
- publishes the resulting latent dataset to Hugging Face.

The final latent dataset used by the PPLM pipeline contains **66,494 representations**.

---

### `ielts_out_of_domain_evaluation_prompt_matrix.ipynb`

Constructs the out-of-domain benchmark used to evaluate whether the learned CEFR control generalizes beyond the EFCAMDAT prompt distribution.

The notebook:

- loads the public IELTS Writing Task 2 dataset;
- extracts unique prompts;
- reproducibly samples 50 prompts;
- pairs each prompt with all six CEFR target levels.

The resulting **Out-of-Domain Evaluation Prompt Matrix** contains:

**50 IELTS prompts × 6 CEFR levels = 300 generation conditions.**

This benchmark is used exclusively for generation-time evaluation and is not part of the steering training data.

## Dataset Naming

The terminology in these notebooks follows the naming convention used in the thesis:

- **Steering Training Dataset** — dataset used to train the steering mechanisms;
- **Evaluation-Judge Dataset** — independent dataset used to train the CEFR evaluator;
- **In-Domain Evaluation Prompt Matrix** — EFCAMDAT-derived generation benchmark;
- **Out-of-Domain Evaluation Prompt Matrix** — IELTS-derived generation benchmark;
- **Balanced CEFR Steering Subset** — 5,568-example balanced training configuration.

## Large Files and External Resources

Large datasets, trained model checkpoints, generated outputs, and additional experimental artifacts are not stored directly in this GitHub repository.

Relevant resources are hosted separately on:

- **Hugging Face** — trained models and selected datasets;

Links to these resources are provided in the main repository `README.md` and in the thesis appendix.

## Reproducibility Notes

The notebooks were originally developed and executed in Google Colab. Public versions have been cleaned to remove private credentials and personal Google Drive paths.

Where authentication is required, Hugging Face credentials should be supplied through **Colab Secrets** using the key:

```text
HF_TOKEN
