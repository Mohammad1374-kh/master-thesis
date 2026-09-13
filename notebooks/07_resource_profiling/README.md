# Resource Profiling

This directory contains the resource-efficiency experiments used in the thesis:

> **Beyond Prompting: Resource-Efficient Explicit Control for CEFR-Aligned Language Generation**

The directory focuses on the computational cost of the proposed CEFR-Gated PMT architecture and on cross-method inference latency.

Resource statistics for the other training methods are recorded directly inside their respective training notebooks.

## Files

### `cefr_gated_pmt_training_resource_profiling.ipynb`

Dedicated hardware-profiling run for the proposed **CEFR-Gated PrefixMemory-Tuning (PMT)** architecture.

The notebook measures:

- training + validation runtime;
- peak allocated GPU memory;
- average GPU utilization;
- frozen and trainable parameter counts.

The profiling run reports:

| Metric | Result |
|---|---:|
| GPU | NVIDIA A100-SXM4-80GB |
| CEFR-Gated PMT trainable parameters | 536,895,489 |
| Frozen backbone parameters | 8,030,261,248 |
| Training + validation time | 875.83 s |
| Training + validation time | 0.24 h |
| Peak allocated GPU memory | 48.74 GB |
| Average GPU utilization | 96.4% |

This notebook is a dedicated **resource-profiling rerun** and is separate from the canonical training run used for model-quality results.

---

### `inference_latency_benchmark.ipynb`

Benchmarks inference latency and throughput across representative thesis methods using a common 12-condition benchmark on an NVIDIA L4 GPU.

Reported metrics include:

- Time to First Token (TTFT);
- Time to Last Token (TTLT);
- generation throughput;
- decode throughput.

Representative methods include:

- Base LLM (Prompt-Only);
- PPLM;
- LoRA;
- Standard Prefix-Tuning;
- CEFR Multi-Prefix Tuning (~537M);
- CEFR-Gated PMT.

Because generated sequence lengths can differ across methods, token-normalized throughput is used as the primary efficiency comparison alongside raw end-to-end latency.

A key result is:

| Method | Generation Throughput |
|---|---:|
| Base LLM (Prompt-Only) | 13.54 tok/s |
| CEFR-Gated PMT | 13.16 tok/s |

The proposed CEFR-Gated PMT therefore retains approximately **97.2% of Base LLM generation throughput** in this benchmark.

## Related Experiments

Training-time resource statistics for the other methods are available in their corresponding notebooks:

```text
notebooks/04_generic_peft_baselines/
notebooks/05_cefr_prefix_tuning/
notebooks/06_cefr_gated_pmt/
