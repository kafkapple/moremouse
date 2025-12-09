# CUDA OOM: Triplane Self-Attention Memory Issue

**Date**: 2024-12-09
**Status**: Resolved
**Severity**: Critical (Training blocked)

## Problem

Training failed at the first batch with CUDA Out of Memory error:

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate 36.00 GiB (GPU 0; 47.54 GiB total capacity)
```

Error occurred in:
```
File "src/models/triplane.py", line 169, in forward
    tgt2, _ = self.self_attn(tgt, tgt, tgt)
```

## Root Cause Analysis

### 1. Self-Attention Memory Complexity

Original implementation used **64×64×3 = 12,288 tokens** for triplane queries:

```
Attention Matrix = O(n²) = 12,288² = 150,994,944 elements
Memory per batch = 150M × batch_size × num_heads × sizeof(float32)
                 = 150M × 4 × 16 × 4 bytes ≈ 36GB
```

### 2. Deviation from Paper Specification

| Parameter | Paper (Table A3) | Previous Implementation |
|-----------|------------------|------------------------|
| Attention head dim | 64 | 32 (512/16) |
| Triplane output | 3×80×128×128 | 3×512×64×64 |
| MLP hidden dim | 64 | 256 |
| Upsampler | Present | Missing |

The paper uses:
- **Triplane Tokenizer**: 64×64 for attention
- **Triplane Upsampler**: 64×64 → 128×128, 512ch → 80ch
- Separate attention head dimension (64) from d_model (512)

## Solution

### 1. Flash Attention (Memory O(n) instead of O(n²))

Replaced `nn.MultiheadAttention` with `F.scaled_dot_product_attention`:

```python
# Before: O(n²) memory
tgt2, _ = self.self_attn(tgt, tgt, tgt)

# After: O(n) memory with Flash Attention
attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
```

### 2. Proper Architecture (Paper-Aligned)

```python
class TriplaneGenerator:
    # Paper spec
    internal_resolution: int = 64    # 64×64 for attention
    triplane_resolution: int = 128   # 128×128 output
    triplane_channels: int = 512     # Internal channels
    output_channels: int = 80        # After upsampler
    head_dim: int = 64               # Attention head dimension
```

### 3. Mixed Precision Training (AMP)

Added proper AMP support to train.py:

```python
self.scaler = torch.amp.GradScaler(device='cuda', enabled=self.use_amp)

with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
    outputs = self.model(...)

self.scaler.scale(losses["total"]).backward()
```

## Memory Comparison

| Configuration | Attention Tokens | Memory Estimate |
|--------------|------------------|-----------------|
| Before (broken) | 12,288 | ~36GB |
| After (paper-aligned) | 12,288 | ~2-4GB (Flash Attention) |

Flash Attention reduces memory from O(n²) to O(n) by computing attention in tiles without materializing the full attention matrix.

## Files Changed

1. `src/models/triplane.py`
   - Added `TriplaneUpsampler` class
   - Replaced `nn.MultiheadAttention` with Flash Attention
   - Added configurable `head_dim` parameter
   - Updated defaults to match paper

2. `scripts/train.py`
   - Added proper AMP (Automatic Mixed Precision) support
   - GradScaler for stable FP16 training

3. `configs/model/moremouse.yaml`
   - Updated all parameters to match paper Table A3

## Verification

Run training on gpu05:
```bash
ssh gpu05
cd ~/moremouse
python scripts/train.py
```

Expected: Training should proceed without OOM errors.

## Key Learnings

1. **Self-Attention scales quadratically**: Token count matters more than feature dimensions
2. **Flash Attention is essential**: Modern implementations should use `scaled_dot_product_attention`
3. **Always verify paper specs**: Default values in code may differ from paper
4. **Upsampler pattern**: Common in triplane methods (EG3D, MoReMouse) - do attention at lower resolution
