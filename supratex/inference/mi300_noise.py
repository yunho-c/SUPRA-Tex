"""Reproduce the A100 release's contiguous float32 completion noise on gfx942."""
import torch
import triton
import triton.language as tl


@triton.jit
def _normal_pair(x, y):
    u = x.to(tl.float32) * 2.3283064365386963e-10 + 1.1641532182693481e-10
    angle = y.to(tl.float32) * 1.4629181192671596e-9 + 7.314590596335798e-10
    radius = tl.sqrt(-2. * tl.log(u))
    return radius * tl.sin(angle), radius * tl.cos(angle)


@triton.jit
def _a100_normal(out, count, stride, seed, block: tl.constexpr):
    draw = tl.program_id(0) * block + tl.arange(0, block)
    thread = draw % stride
    round_id = draw // stride
    # Philox advances the low 64-bit counter per group of four values.
    # Each logical CUDA thread occupies its own high 64-bit subsequence.
    x, y, z, w = tl.philox(seed, round_id.to(tl.uint32), tl.full((block,), 0, tl.uint32),
                           thread.to(tl.uint32), tl.full((block,), 0, tl.uint32))
    a, b = _normal_pair(x, y)
    c, d = _normal_pair(z, w)
    base = round_id * 4 * stride + thread
    tl.store(out + base, a, base < count)
    tl.store(out + base + stride, b, base + stride < count)
    tl.store(out + base + 2 * stride, c, base + 2 * stride < count)
    tl.store(out + base + 3 * stride, d, base + 3 * stride < count)


def a100_noise(shape, seed, device):
    """Match the PyTorch 2.6 A100-SXM4-80GB launch geometry and Philox stream."""
    out = torch.empty(shape, dtype=torch.float32, device=device)
    count = out.numel()
    if count:
        # 108 SMs, 2048 resident threads per SM, 256 threads per CUDA block.
        stride = min(triton.cdiv(count, 256), 864) * 256
        groups = triton.cdiv(count, 4 * stride)
        block = 256
        seed = torch.Generator().manual_seed(int(seed)).initial_seed()
        _a100_normal[(triton.cdiv(groups * stride, block),)](out, count, stride, seed, block)
    return out
