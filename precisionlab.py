"""CPU FP8-format emulation; no native FP8 matmul or accelerator deployment."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F


def fp8_qdq(x):
    if not x.numel() or not torch.isfinite(x).all():
        raise ValueError('expected nonempty finite tensor')
    scale = x.detach().abs().max().clamp_min(1e-12) / 448.0
    reconstructed = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    # Straight-through estimator: gradients do not differentiate the rounding.
    return x + (reconstructed - x).detach()


class Linear(nn.Linear):
    emulate = False

    def forward(self, x):
        if self.emulate:
            return F.linear(fp8_qdq(x), fp8_qdq(self.weight), self.bias)
        return F.linear(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


class Block(nn.Module):
    def __init__(self, width=32, heads=4):
        super().__init__()
        self.width, self.heads = width, heads
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.qkv, self.projection = Linear(width, width * 3), Linear(width, width)
        self.gate, self.up, self.down = Linear(width, width * 3), Linear(width, width * 3), Linear(width * 3, width)

    def forward(self, x):
        b, length, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = [value.reshape(b, length, self.heads, self.width // self.heads).transpose(1, 2) for value in (q, k, v)]
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.projection(attention.transpose(1, 2).reshape(b, length, self.width))
        norm = self.norm2(x)
        return x + self.down(F.silu(self.gate(norm)) * self.up(norm))


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 32)
        self.position = nn.Embedding(16, 32)
        self.blocks = nn.Sequential(Block(), Block())
        self.norm, self.output = RMSNorm(32), Linear(32, 32)

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] > 16 or tokens.shape[1] == 0:
            raise ValueError('expected batch x length tokens, length 1..16')
        x = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        return self.output(self.norm(self.blocks(x)))

    def precision(self, emulate):
        for module in self.modules():
            if isinstance(module, Linear):
                module.emulate = emulate
        return self


def samples(count, seed):
    generator = torch.Generator().manual_seed(seed)
    start = torch.randint(0, 32, (count, 1), generator=generator)
    step = torch.randint(1, 5, (count, 1), generator=generator)
    tokens = (start + step * torch.arange(17)) % 32
    return tokens[:, :-1], tokens[:, 1:]


def evaluate(model, data):
    model.eval()
    with torch.inference_mode():
        logits = model(data[0])
        loss = F.cross_entropy(logits.reshape(-1, 32), data[1].reshape(-1))
        return {'cross_entropy': loss.item(), 'perplexity': math.exp(loss.item()),
                'token_accuracy': (logits.argmax(-1) == data[1]).float().mean().item()}


def train(initial, emulate, steps=240):
    model = copy.deepcopy(initial).precision(emulate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    validation = samples(128, 202)
    best, state, selected = float('inf'), None, 0
    started = time.perf_counter()
    for step in range(steps):
        model.train()
        x, y = samples(32, 1000 + step)
        loss = F.cross_entropy(model(x).reshape(-1, 32), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 20 == 0:
            score = evaluate(model, validation)['cross_entropy']
            if score < best:
                best, selected = score, step + 1
                state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    return model, {'training_wall_seconds': time.perf_counter() - started,
                   'selected_step': selected, 'validation_cross_entropy': best}


def profile(model, tokens):
    with torch.inference_mode():
        for _ in range(3):
            model(tokens)
        times = []
        for _ in range(9):
            start = time.perf_counter()
            model(tokens)
            times.append((time.perf_counter() - start) * 1000)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            model(tokens)
    top = sorted(profiler.key_averages(), key=lambda item: item.self_cpu_time_total, reverse=True)[:5]
    return {'median_batch_inference_ms': statistics.median(times), 'batch_size': len(tokens),
            'top_cpu_operators': [{'operator': item.key, 'self_cpu_us': item.self_cpu_time_total} for item in top]}


def experiment(output):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    initial = TinyDecoder()
    output.mkdir(parents=True, exist_ok=True)
    test = samples(256, 303)
    results = {}
    for name, emulate in [('fp32', False), ('fp8_emulated', True)]:
        model, training = train(initial, emulate)
        results[name] = {**training, 'test': evaluate(model, test), 'profile': profile(model, test[0][:16])}
        torch.save(model.state_dict(), output / (name + '.pt'))
    metrics = {'environment': 'local CPU; E4M3 quantize/dequantize simulation with FP32 matrix operations',
               'parameters': sum(p.numel() for p in initial.parameters()), 'steps_per_run': 240,
               'task': 'synthetic modular arithmetic next-token prediction; not natural language',
               'results': results,
               'limitations': ['Not native FP8 training, Ascend/NPU, CUDA, or an accelerator benchmark.',
                               'Master weights, gradients, optimizer states, attention and accumulation remain FP32.',
                               'Small synthetic task has overlapping finite patterns across seeds; not held-out-pattern generalization.',
                               'One seed; CPU profiling adds overhead. No real Llama/DeepSeek checkpoint, MoE, RL or multimodal training.']}
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    experiment(parser.parse_args().output)
