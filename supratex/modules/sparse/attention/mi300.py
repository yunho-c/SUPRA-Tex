"""MI300X inference kernels for the completion transformer's full attention."""
from functools import lru_cache
from importlib.util import find_spec
import os

import torch
from torch.nn import functional as F


@lru_cache(maxsize=8)
def _is_mi300(device):
    return torch.cuda.get_device_properties(device).gcnArchName.split(':', 1)[0] == 'gfx942'


def attention(q, k, v, mode):
    """One unmasked sequence. Operands and result have shape [tokens, heads, 128]."""
    if mode == "sdpa":
        return F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None]
        )[0].transpose(0, 1)

    from aiter.ops.mha import fmha_v3_fwd
    operands = [t[None] for t in (q, k, v)]
    kwargs = {}
    if mode == "fp8":
        from .mi300_quant import pack_fp8
        for i, name in enumerate(("q_descale", "k_descale", "v_descale")):
            operands[i], kwargs[name] = pack_fp8(operands[i])
    elif mode != "bf16":
        raise ValueError(f"Unknown MI300 attention mode {mode!r}")

    return fmha_v3_fwd(*operands, 0., q.shape[-1] ** -.5, False, -1, -1,
                       False, False, 1, **kwargs)[0][0]


@lru_cache(maxsize=4)
def resolve_mode(requested):
    if requested == "auto":
        return "bf16" if find_spec("aiter") is not None else "sdpa"
    if requested not in ("bf16", "fp8", "sdpa"):
        raise ValueError("SUPRATEX_MI300_ATTENTION accepts auto, bf16, fp8 or sdpa")
    return requested


def selected_mode(q, k, v):
    mode = resolve_mode(os.environ.get("SUPRATEX_MI300_RESOLVED",
                        os.environ.get("SUPRATEX_MI300_ATTENTION", "auto")))
    if (mode == "sdpa" or torch.is_grad_enabled() or q.dtype != torch.bfloat16
            or not len(q) or not len(k) or q.shape[-1] != 128 or k.shape[1:] != q.shape[1:] or v.shape != k.shape
            or k.dtype != q.dtype or v.dtype != q.dtype
            or (q.stride(-1), k.stride(-1), v.stride(-1)) != (1, 1, 1)
            or not _is_mi300(q.device)):
        return "sdpa"
    return mode
