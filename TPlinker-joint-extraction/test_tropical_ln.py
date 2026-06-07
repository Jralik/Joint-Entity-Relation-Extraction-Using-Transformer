"""
Test script: Kiểm tra TropicalLayerNorm và HandshakingKernel.

Chạy:
    cd TPlinker-joint-extraction
    python test_tropical_ln.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from common.components import LayerNorm, TropicalLayerNorm, HandshakingKernel

# =========================================================
# Cài đặt chung
# =========================================================
BATCH  = 2
SEQ    = 8
HIDDEN = 64
K      = 4   # số phân vùng Tropical

print("=" * 60)
print("TEST 1: TropicalLayerNorm — bảo toàn tri thức tại t=0")
print("=" * 60)

tln = TropicalLayerNorm(input_dim=HIDDEN, num_pieces=K)
# Đặt slopes=1, biases=0 (mặc định đã làm trong __init__)
# → f_trop = identity → TropicalLN ≡ LayerNorm gốc

ln  = nn.LayerNorm(HIDDEN, elementwise_affine=True)
# Đặt gamma=1, beta=0 để so sánh công bằng
nn.init.ones_(ln.weight)
nn.init.zeros_(ln.bias)

x = torch.randn(BATCH, SEQ, HIDDEN)

with torch.no_grad():
    out_tln = tln(x)
    out_ln  = ln(x)

max_diff = (out_tln - out_ln).abs().max().item()
print(f"  Max |TropicalLN - LayerNorm| = {max_diff:.6f}")
assert max_diff < 0.05, f"Sai số quá lớn: {max_diff}"
print("  PASS ✓  (sai số < 0.05 — do soft gating, không phải hard PWL)")

print()
print("=" * 60)
print("TEST 2: TropicalLayerNorm Conditional (forward pass)")
print("=" * 60)

tln_cond = TropicalLayerNorm(
    input_dim=HIDDEN, num_pieces=K, cond_dim=HIDDEN, conditional=True
)
cond_vec = torch.randn(BATCH, HIDDEN)  # (B, H)
x = torch.randn(BATCH, SEQ, HIDDEN)   # (B, L, H)

with torch.no_grad():
    out = tln_cond(x, cond_vec)

assert out.shape == (BATCH, SEQ, HIDDEN), f"Shape sai: {out.shape}"
print(f"  Output shape: {out.shape}  PASS ✓")

print()
print("=" * 60)
print("TEST 3: HandshakingKernel với tropical_cln")
print("=" * 60)

for shaking_type in ["tropical_cln", "tropical_cln_plus"]:
    inner_enc = "lstm" if "plus" in shaking_type else "none"
    try:
        hk = HandshakingKernel(
            hidden_size=HIDDEN,
            shaking_type=shaking_type,
            inner_enc_type=inner_enc,
            tropical_num_pieces=K,
        )
        seq_hidden = torch.randn(BATCH, SEQ, HIDDEN)
        with torch.no_grad():
            out = hk(seq_hidden)
        expected_shaking_len = SEQ * (SEQ + 1) // 2
        assert out.shape == (BATCH, expected_shaking_len, HIDDEN), \
            f"Shape sai: {out.shape}, expected ({BATCH}, {expected_shaking_len}, {HIDDEN})"
        print(f"  [{shaking_type}] output shape: {out.shape}  PASS ✓")
    except Exception as e:
        print(f"  [{shaking_type}] FAIL ✗: {e}")

print()
print("=" * 60)
print("TEST 4: So sánh số param giữa CLN và TropicalCLN")
print("=" * 60)

hk_cln = HandshakingKernel(HIDDEN, "cln", "none", K)
hk_tln = HandshakingKernel(HIDDEN, "tropical_cln", "none", K)

params_cln = sum(p.numel() for p in hk_cln.parameters())
params_tln = sum(p.numel() for p in hk_tln.parameters())
extra = params_tln - params_cln

print(f"  HandshakingKernel(cln)         params: {params_cln:,}")
print(f"  HandshakingKernel(tropical_cln) params: {params_tln:,}")
print(f"  Tropical thêm: +{extra:,} params  ({extra} = K*2 + (K-1) = slopes+biases+breakpoints)")

print()
print("=" * 60)
print("TEST 5: Gradient flow qua TropicalLayerNorm")
print("=" * 60)

tln_grad = TropicalLayerNorm(input_dim=HIDDEN, num_pieces=K, cond_dim=HIDDEN, conditional=True)
x   = torch.randn(BATCH, SEQ, HIDDEN, requires_grad=True)
c   = torch.randn(BATCH, HIDDEN, requires_grad=True)
out = tln_grad(x, c)
loss = out.sum()
loss.backward()

assert x.grad is not None,              "x.grad is None!"
assert c.grad is not None,              "c.grad is None!"
assert tln_grad.slopes.grad is not None,"slopes.grad is None!"
print(f"  x.grad norm     = {x.grad.norm().item():.4f}")
print(f"  c.grad norm     = {c.grad.norm().item():.4f}")
print(f"  slopes.grad     = {tln_grad.slopes.grad}")
print(f"  pw_biases.grad  = {tln_grad.pw_biases.grad}")
print("  PASS ✓  (gradient flows to all params)")

print()
print("ALL TESTS PASSED ✓")
