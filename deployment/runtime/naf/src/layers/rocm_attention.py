"""Dilated neighborhood attention for NAF on ROCm, using batched GEMMs."""

import torch
from einops import rearrange


def neighborhood_attention(q, k, v, heads, kernel, scale, return_weights):
    """Evaluate NATTEN's shifted border windows on nearest-exact resized keys."""
    _, _, height, width = q.shape
    k, v = k.to(q.dtype), v.to(q.dtype)
    kh, kw = k.shape[-2:]
    dy, dx = height // kh, width // kw
    ny, nx = kernel
    if dy < 1 or dx < 1 or height < dy * ny or width < dx * nx:
        raise ValueError("NAF feature grid is smaller than the attention neighborhood")

    # An integral nearest-neighbor resize repeats K/V in each coarse cell.
    # Group those queries to share a single neighborhood and one GEMM.
    if (height, width) == (kh * dy, kw * dx):
        query = rearrange(q, "b (n c) (h y) (w x) -> b (h w) n (y x) c",
                          n=heads, y=dy, x=dx)
        key, value = [rearrange(t, "b (n c) h w -> b (h w) n c", n=heads)
                      for t in (k, v)]
        cells = torch.arange(kh * kw, device=q.device)
        y = (cells // kw - ny // 2).clamp(0, kh - ny)
        x = (cells % kw - nx // 2).clamp(0, kw - nx)
        offsets = (torch.arange(ny, device=q.device)[:, None] * kw
                   + torch.arange(nx, device=q.device)).flatten()
        neighborhoods = (y * kw + x)[:, None] + offsets
        output = q.new_empty((*query.shape[:-1], value.shape[-1]))
        scores = q.new_empty((*query.shape[:-1], ny * nx)) if return_weights else None
        for start in range(0, len(cells), 128):
            rows = slice(start, start + 128)
            ix = neighborhoods[rows]
            keys = key[:, ix].transpose(2, 3)
            logits = (query[:, rows] @ keys.transpose(-1, -2)) * scale
            output[:, rows] = logits.softmax(-1) @ value[:, ix].transpose(2, 3)
            if return_weights:
                scores[:, rows] = logits
        image = rearrange(output, "b (h w) n (y x) c -> b (n c) (h y) (w x)",
                          h=kh, w=kw, y=dy, x=dx)
        if return_weights:
            return image, rearrange(scores, "b (h w) n (y x) c -> b n (h y) (w x) c",
                                    h=kh, w=kw, y=dy, x=dx)
        return image

    # For nonintegral scales, nearest-exact assigns different low-resolution
    # keys to different dilation residues. Evaluate the actual per-pixel window.
    def axis_window(axis, size, dilation, count, source_size):
        residue = axis % dilation
        last = (size - 1 - residue) // dilation - count + 1
        start = torch.minimum((axis // dilation - count // 2).clamp_min(0), last)
        pos = residue[:, None] + dilation * (start[:, None] + torch.arange(count, device=q.device))
        return ((2 * pos + 1) * source_size) // (2 * size)

    query, key, value = [rearrange(t, "b (n c) h w -> b (h w) n c", n=heads)
                         for t in (q, k, v)]
    result = q.new_empty((*query.shape[:-1], value.shape[-1]))
    scores = q.new_empty((*query.shape[:-1], nx * ny)) if return_weights else None
    for first in range(0, height * width, 256):
        rows = torch.arange(first, min(first + 256, height * width), device=q.device)
        ys = axis_window(rows // width, height, dy, ny, kh)
        xs = axis_window(rows % width, width, dx, nx, kw)
        ix = (ys[:, :, None] * kw + xs[:, None, :]).flatten(1)
        logits = ((query[:, rows, :, None] @ key[:, ix].permute(0, 1, 3, 4, 2)) * scale)
        result[:, rows] = (logits.softmax(-1) @ value[:, ix].transpose(2, 3)).squeeze(-2)
        if return_weights:
            scores[:, rows] = logits.squeeze(-2)
    image = rearrange(result, "b (h w) n c -> b (n c) h w", h=height)
    if return_weights:
        return image, rearrange(scores, "b (h w) n c -> b n h w c", h=height)
    return image
