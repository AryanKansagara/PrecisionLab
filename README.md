# PrecisionLab

A CPU research harness for precision-control and operator-level bottleneck analysis in a small decoder-only Transformer. Compares FP32 training against **E4M3 FP8-format quantize/dequantize emulation**, not native FP8 training or accelerator speed.

## Reproduce

Python 3.12 and PyTorch 2.10.0 (CPU sufficient):

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest -v
.venv/bin/python precisionlab.py --output artifacts
```

The 30,080-parameter decoder has two blocks with causal multi-head attention, RMSNorm, gated SiLU feed-forward layers, residuals, token embeddings and learned position embeddings. These are common decoder components; it does not load Llama/DeepSeek weights and is not a faithful implementation of either architecture (no RoPE, GQA or MoE).

The precision path dynamically scales each linear input/weight tensor by its maximum absolute value, casts to `torch.float8_e4m3fn`, reconstructs FP32 values and uses an identity straight-through gradient estimator. **Matmuls, master parameters, gradients, optimizer states, attention and accumulation stay FP32.** No process-memory reduction, native low-precision backward pass, kernel acceleration or GPU/NPU execution is claimed.

Both runs start from identical weights and receive the same 240 batches. Validation loss selects a checkpoint every 20 steps. Evaluation uses separately seeded synthetic modular arithmetic sequences. The finite pattern space overlaps across seeds; this tests numerical behavior and repeatability, not language understanding or unseen-pattern generalization.

The harness measures training wall time, nine-repetition median batch inference after three warmups, and CPU operator self time with `torch.profiler`. Profiler timings are collected separately from latency. All data is generated in memory: disk I/O, distributed communication and scheduling are not benchmarked.

## Observed result

Both paths reached 95.31% synthetic token accuracy. FP32 test loss was 0.11729 versus 0.11781 for emulation. Emulation was slower: 1.282 ms versus 0.400 ms per 16-sequence batch on this local run. These are not large-model or accelerator results. See [report](docs/REPORT.md) and [raw metrics](results/metrics.json).

Twelve tests verify shape checks, finite inputs, zero/extreme values, quantization error, straight-through gradients, causal attention, seeded sequences and finite emulated backward gradients. Generated checkpoints are ignored; run the experiment to recreate them.

## Scope

No Ascend/CANN, CUDA, NPU operator library, cluster management, RL-driven agents, multimodal training, parallel-strategy tuning or hardware co-design is implemented. This project makes numerical behavior and overhead inspectable before attempting a hardware-backed implementation. Source was developed with AI assistance; review it and represent your contribution accurately. The CI template is inactive until moved to `.github/workflows` with the necessary permissions.

Primary API references: [PyTorch dtypes](https://docs.pytorch.org/docs/stable/tensor_attributes.html), [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html).
