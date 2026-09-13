# PPLM Baseline

This directory contains the complete PPLM baseline pipeline used in the thesis, from training the CEFR latent attribute classifier to hyperparameter selection and final generation-time evaluation.

The PPLM experiments use a frozen `meta-llama/Llama-3.1-8B-Instruct` backbone together with a lightweight CEFR latent classifier. The classifier provides a differentiable CEFR objective that is used during inference-time latent perturbation without updating the parameters of the underlying language model.

The workflow is organized into three stages:

```text
Train CEFR latent classifier
        ↓
PPLM hyperparameter grid search
        ↓
Final Base LLM vs PPLM evaluation
