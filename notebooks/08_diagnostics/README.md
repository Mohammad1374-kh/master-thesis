# Diagnostics

This directory contains diagnostic analyses used to investigate limitations observed in the CEFR-controlled generation experiments.

The current analysis focuses on the **upper CEFR levels (B2, C1, and C2)** in the Balanced CEFR Steering Subset.

## `upper_cefr_diagnostic_analysis.ipynb`

This notebook examines whether reduced performance at the highest CEFR levels can be related to properties of the training data.

The analysis includes:

- lexical and sentence-level features;
- syntactic complexity;
- readability metrics;
- word-count distributions;
- topic coverage;
- exact duplicate analysis;
- pairwise standardized effect sizes between B2, C1, and C2.

A key focus is the degree of separation between **C1 and C2**, where many common linguistic features show relatively small standardized differences.

The notebook also identifies a difference in topic diversity within the balanced subset:

```text
A1–C1: 24 distinct topics per level
C2:     8 distinct topics
