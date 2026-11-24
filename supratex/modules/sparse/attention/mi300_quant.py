"""Coalesced per-tensor FP8 packing for the gfx942 attention kernel."""
import torch
import triton
import triton.language as tl


@triton.jit
def _block_maximum(values, maxima, count, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    tile = tl.load(values + offsets, offsets < count, other=0.).to(tl.float32)
    tl.store(maxima + tl.program_id(0), tl.max(tl.abs(tile), axis=0))


@triton.jit
def _pack(values, output, scale, count, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    tile = tl.load(values + offsets, offsets < count, other=0.).to(tl.float32)
    tl.store(output + offsets, tile / tl.load(scale), offsets < count)


def pack_fp8(values):
    values = values.contiguous()
    count = values.numel()
    block = 4096
    grid = (triton.cdiv(count, block),)
    maxima = torch.empty(grid, device=values.device, dtype=torch.float32)
    _block_maximum[grid](values, maxima, count, block)
    dtype = torch.float8_e4m3fnuz
    scale = maxima.amax().clamp_min_(1e-8).div_(torch.finfo(dtype).max).view(1)
    output = torch.empty_like(values, dtype=dtype)
    _pack[grid](values, output, scale, count, block)
    return output, scale
