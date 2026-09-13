# CEFR Evaluator

This directory contains the notebooks used to train, evaluate, and compare the two RoBERTa-large CEFR classification models considered in the thesis.

Both evaluators are trained on the **Evaluation-Judge Dataset**, which is kept disjoint from the Steering Training Dataset used by the generative control methods. The task is six-class CEFR proficiency classification across the ordered levels A1, A2, B1, B2, C1, and C2.

The two variants differ only in their training objective:

1. a **Weighted Cross-Entropy (WCE)** classifier; and
2. a **Joint WCE + Ordinal Loss** classifier.

The Joint-Loss model is ultimately selected as the primary evaluator used throughout the thesis.

## Notebooks

### `roberta_cefr_wce_train_and_eval.ipynb`

Trains and evaluates the **Weighted Cross-Entropy (WCE)** variant.

The model is based on `FacebookAI/roberta-large` and uses inverse-frequency class weights to compensate for the strong imbalance of the Evaluation-Judge Dataset, particularly at the C1 and C2 levels.

The notebook includes:

- loading and preprocessing of the Evaluation-Judge Dataset;
- CEFR label encoding from A1–C2;
- stratified 90/10 training-validation splitting;
- inverse-frequency class-weight computation;
- RoBERTa-large fine-tuning with weighted cross-entropy;
- validation using:
  - Strict Accuracy,
  - Adjacent Accuracy,
  - Macro F1,
  - Mean Absolute Error (MAE),
  - Quadratic Weighted Kappa (QWK);
- per-class validation analysis;
- publication of the trained model and evaluation results to Hugging Face;
- an additional balanced evaluation on a subset derived from the Steering Training Dataset.

This model serves as the first evaluator baseline against which the Joint-Loss formulation is compared.

---

### `roberta_cefr_jointloss_train_and_eval.ipynb`

Trains and evaluates the **Joint WCE + Ordinal Loss** variant.

This model uses the same RoBERTa-large architecture, Evaluation-Judge Dataset, split configuration, and general training procedure as the WCE-only variant, but augments the weighted cross-entropy objective with an ordinal penalty.

The training objective is:

`L_joint = L_WCE + λ L_ordinal`

with:

`λ = 0.5`

The ordinal component is implemented as a Mean Squared Error penalty between the true CEFR class index and the expected class value computed from the predicted probability distribution.

The notebook includes:

- loading and preprocessing of the Evaluation-Judge Dataset;
- CEFR label encoding from A1–C2;
- stratified 90/10 training-validation splitting;
- inverse-frequency class weighting;
- implementation of the joint WCE + ordinal objective;
- RoBERTa-large fine-tuning;
- validation using:
  - Strict Accuracy,
  - Adjacent Accuracy,
  - Macro F1,
  - Mean Absolute Error (MAE),
  - Quadratic Weighted Kappa (QWK);
- per-class validation analysis;
- publication of the trained evaluator and results to Hugging Face;
- an additional balanced evaluation on data disjoint from the evaluator training set.

The resulting model is published as:

`MohammadKhosravi/roberta-large-cefr-classifier-JointLoss`

and is used as the **primary CEFR evaluator** for the generation experiments reported in the thesis.

## Model Selection

The two evaluator variants achieved very similar validation performance.

| Model | Strict Accuracy | Adjacent Accuracy | Macro F1 | MAE | QWK |
|---|---:|---:|---:|---:|---:|
| WCE Only | 98.41% | 99.49% | 97.32% | 0.0230 | 0.9866 |
| Joint WCE + Ordinal | 98.33% | 99.43% | **97.47%** | 0.0240 | 0.9863 |

Although the WCE-only model achieves marginally higher Strict Accuracy and Adjacent Accuracy, the Joint-Loss model achieves the higher **Macro F1** score.

Because Macro F1 gives equal importance to all six CEFR classes and is therefore particularly informative under the strong class imbalance of the dataset, the **Joint WCE + Ordinal model** is selected as the final evaluator.

## Final Joint-Loss Evaluator

The selected Joint-Loss evaluator is trained for four epochs.

Its final internal validation performance is:

- **Strict Accuracy:** 98.33%
- **Adjacent Accuracy:** 99.43%
- **Macro F1:** 97.47%
- **MAE:** 0.0240
- **QWK:** 0.9863

Per-class accuracy remains high across the CEFR scale, although performance decreases toward the minority C1 and C2 classes.

The trained evaluator is kept frozen during all subsequent generation experiments and is used only to assess the CEFR level of generated text.

## Independent Balanced Evaluation

Both notebooks additionally evaluate the trained classifiers on a balanced subset derived from the **Steering Training Dataset**.

This dataset is text-disjoint from the Evaluation-Judge Dataset used for evaluator training and contains equal representation across the six CEFR levels.

This additional experiment is intended as an independent balanced assessment of evaluator behavior and should not be interpreted as a separate out-of-domain benchmark, since both resources are derived from EFCAMDAT.

## Reproducibility Notes

The notebooks were originally developed and executed in Google Colab.

Public versions have been cleaned to remove private Hugging Face credentials and personal Google Drive paths.

Where authentication is required, the Hugging Face access token should be supplied through **Google Colab Secrets** using:

```text
HF_TOKEN
