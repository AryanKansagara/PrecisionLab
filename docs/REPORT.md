# Precision-control experiment report

## Design

The experiment asks whether a small decoder can train with FP8-format perturbations on linear operands while preserving useful numerical behavior. It uses a 32-token synthetic vocabulary and 16-token contexts. Targets follow modular arithmetic progressions. Two layers, width 32, four attention heads and gated feed-forward width 96 give 30,080 parameters. Position embeddings are learned, not rotary.

The emulation dynamically quantizes each linear input/weight using E4M3 finite values and a per-tensor scale, then dequantizes before FP32 linear operations. A straight-through estimator passes identity gradients through quantization. Gradient clipping, AdamW (learning rate 0.003, weight decay 0.01) and validation selection are shared with FP32. This is not a hardware-accurate FP8 optimizer, native FP8 GEMM, or full low-precision training implementation.

Both paths receive the same 240 deterministic batches from identical initial parameters. Validation uses seed 202 and evaluation seed 303; generated training batches use seeds 1000 onward. Different seeds are not disjoint patterns in this deliberately small task. Validation checks occur every 20 steps; both selected step 220.

## Results

Observed on macOS ARM64 CPU, PyTorch 2.10.0, 2026-10-27. Twelve unit tests passed.

| Metric | FP32 | FP8 emulated |
| --- | ---: | ---: |
| Test cross-entropy | 0.117292 | 0.117813 |
| Test perplexity on synthetic tokens | 1.12445 | 1.12503 |
| Token accuracy | 95.3125% | 95.3125% |
| Training wall time, including validation | 0.741 s | 1.129 s |
| Median inference, batch of 16 | 0.400 ms | 1.282 ms |

The equal accuracy is not proof of equivalent precision on realistic models. CPU emulation was slower, as expected from format conversions and safety checks. The single profiled FP32 forward pass placed CPU attention among its largest self-time operators; emulation showed additional copy and finite-check overhead. These observations guide profiling questions, not accelerator-kernel conclusions. No profiler-based optimization speedup is claimed.

## Next experiments and boundaries

Repeat seeds and larger representative workloads, isolate safety-check versus conversion cost, compare precision scaling policies, and inspect activation outliers. A real device study would need reviewed native FP8 kernels or an Ascend backend, hardware-specific measurement, memory accounting and distributed-communication analysis. Those steps are not implemented.

All current model and optimizer storage remains FP32; transient casts do not demonstrate reduced training memory. No actual Llama/DeepSeek, MoE, RL, multimodal model, training cluster or NPU runtime is used. Timing is local, single-threaded, short and subject to process warmup/noise. The experiment is a bounded numerical prototype, not a production framework.
