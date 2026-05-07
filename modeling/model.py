"""Siamese few-shot localiser with transformer cross-attention bridge.

Architecture (Siamese — backbone weights shared between support and query):

  shared MobileNetV3-Large backbone
   ├── support_imgs (B,K,3,H,W) → P5 feat ────► SupportTokenizer (slot-attn, M=6, 2 iters)
   │                                          ─► (B*K, M, DIM) tokens
   │                                          ─► reshape (B, K*M, DIM)
   │                                          ─► SupportFusion (3-layer pre-LN Transformer
   │                                                            encoder, 8 heads, MLP×4)
   │                                          ─► (B, K*M, DIM) fused
   │                                          ─► AttentionPool (CaiT class-attention)
   │                                                          ─► support_summary (B, DIM)
   │
   └── query_img (B,3,H,W) → P3,P4,P5 ──► FPN(P4,P5) + gated P3 residual
                                       ─► query_pe (learnable + sinusoidal)
                                       ─► CrossAttentionHead (2-layer DETR decoder,
                                                             learnable softmax τ,
                                                             DropPath p=0.15)
                                       ─► enriched query feat (B,DIM,H/16,W/16)
                                       ─► P_lat (1×1 conv) lateral for stride-8 path

Detection heads (weight-shared across two FPN scales):
  stride 16 (14×14 at 224 input):  reg(DFL 17 bins × 4 coords) + conf + centerness
  stride  8 (28×28 at 224 input):  same head, weight-shared

Aux head: a second DetectionHead applied to the layer-0 decoder output (DETR-style aux loss).

Presence head:
  [support_summary ⊕ GAP(enriched_query)] → 320 → 160 → 64 → 1, GELU + LN-pre + dropout 0.2.

decode_topk (inference-time):
  Per image, take top-K cells across both scales above τ_conf, NMS at IoU=0.5,
  score = σ(conf) · σ(centerness) · σ(presence_logit).

Outputs (forward):
  reg_p4, conf_p4, ctr_p4   — stride-16 grids (B, *, 14, 14)
  reg_p3, conf_p3, ctr_p3   — stride-8 grids (B, *, 28, 28)
  reg_aux, conf_aux, ctr_aux— stride-16 decoder layer-0 aux (same shape)
  presence_logit            — (B,)
  prototype                 — (B, DIM) bag-level summary
  per_shot_prototype        — (B, K, DIM)
  support_attn              — (B, K, M, 7, 7)
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.ops import FeaturePyramidNetwork, batched_nms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE = 224
GRID_P4 = 14                         # stride 16 grid at IMG_SIZE=224
GRID_P3 = 28                         # stride  8 grid at IMG_SIZE=224
STRIDE_P4 = 16
STRIDE_P3 = 8

DIM = 160
M_TOKENS = 6
N_HEADS = 8
K_SUPPORT_MAX = 8
FUSION_LAYERS = 3
DECODER_LAYERS = 2
SLOT_ATTN_ITERS = 2
DROP_PATH = 0.15

# Distribution Focal Loss bins per coordinate (l, t, r, b in stride units).
# 17 bins covers offsets up to ~stride*16 (256 px) — sufficient at 224 input.
DFL_BINS = 17

P3_IDX = 6
P3_CHANNELS = 40
P4_IDX = 12
P4_CHANNELS = 112
P5_IDX = 15
P5_CHANNELS = 160


# ---------------------------------------------------------------------------
# Backbone (shared between support and query streams — Siamese)
# ---------------------------------------------------------------------------


class MobileNetBackbone(nn.Module):
    """MobileNetV3-Large exposing P3 (28×28×40), P4 (14×14×112), P5 (7×7×160)."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.features = mobilenet_v3_large(weights=weights).features

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3 = p4 = p5 = None
        for i, block in enumerate(self.features):
            x = block(x)
            if i == P3_IDX:
                p3 = x
            elif i == P4_IDX:
                p4 = x
            elif i == P5_IDX:
                p5 = x
                break
        assert p3 is not None and p4 is not None and p5 is not None
        return p3, p4, p5

    def forward_p5_only(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.features):
            x = block(x)
            if i == P5_IDX:
                return x
        raise RuntimeError("unreachable")

    def freeze_lower(self, freeze_idx_exclusive: int = 7) -> None:
        for i, block in enumerate(self.features):
            req = freeze_idx_exclusive <= i <= P5_IDX
            for p in block.parameters():
                p.requires_grad = req

    def unfreeze_all(self) -> None:
        for i, block in enumerate(self.features):
            for p in block.parameters():
                p.requires_grad = i <= P5_IDX

    def freeze_all(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False


# ---------------------------------------------------------------------------
# 2D sinusoidal positional encoding (DETR-style)
# ---------------------------------------------------------------------------


def _sinusoidal_2d_pe(dim: int, h: int, w: int, device, dtype=torch.float32) -> torch.Tensor:
    """Return (1, dim, h, w) DETR-style 2D sinusoidal PE."""
    if dim % 4 != 0:
        # Pad to nearest multiple of 4; we'll slice back to `dim` at the end.
        pad_dim = ((dim + 3) // 4) * 4
    else:
        pad_dim = dim
    half = pad_dim // 2
    y = torch.arange(h, device=device, dtype=dtype).unsqueeze(1).expand(h, w)
    x = torch.arange(w, device=device, dtype=dtype).unsqueeze(0).expand(h, w)
    div = torch.arange(0, half // 2, device=device, dtype=dtype)
    div = 10000.0 ** (2.0 * div / max(half, 1))
    pe_y = torch.zeros(half, h, w, device=device, dtype=dtype)
    pe_x = torch.zeros(half, h, w, device=device, dtype=dtype)
    n = div.numel()
    pe_y[0:n] = torch.sin(y.unsqueeze(0) / div.view(-1, 1, 1))
    pe_y[n:2 * n] = torch.cos(y.unsqueeze(0) / div.view(-1, 1, 1))
    pe_x[0:n] = torch.sin(x.unsqueeze(0) / div.view(-1, 1, 1))
    pe_x[n:2 * n] = torch.cos(x.unsqueeze(0) / div.view(-1, 1, 1))
    pe = torch.cat([pe_y, pe_x], dim=0)[:dim]
    return pe.unsqueeze(0)


# ---------------------------------------------------------------------------
# Slot-attention support tokenizer (2-iter, M=6, learnable temperature)
# ---------------------------------------------------------------------------


class SupportTokenizer(nn.Module):
    """2-iteration slot-attention with M competing slots per support image.

    Iterating slot-attention >1 (Locatello et al. 2020) gives slots time to
    refine their assignments — empirically much cleaner region tokens than 1
    iteration. Slots compete via softmax over the M dimension, forcing them
    to cover disjoint regions of the support feature map.
    """

    def __init__(
        self,
        in_c: int = P5_CHANNELS,
        dim: int = DIM,
        m_tokens: int = M_TOKENS,
        n_iters: int = SLOT_ATTN_ITERS,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.m = m_tokens
        self.n_iters = n_iters
        self.scale = dim ** -0.5
        self.slots_init = nn.Parameter(torch.randn(m_tokens, dim) * 0.02)
        # Learnable softmax temperature (clamped to a stable range).
        self._log_tau = nn.Parameter(torch.zeros(1))
        self.kv_norm = nn.LayerNorm(in_c)
        self.k_proj = nn.Linear(in_c, dim, bias=False)
        self.v_proj = nn.Linear(in_c, dim, bias=False)
        self.q_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.norm_out = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def _tau(self) -> torch.Tensor:
        # Clamp τ to [0.5, 2.0] via softplus, keeps gradients smooth.
        return F.softplus(self._log_tau) + 0.5

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, _, h, w = feat.shape
        kv_in = self.kv_norm(feat.flatten(2).transpose(1, 2))           # (N, HW, C)
        k = self.k_proj(kv_in)                                          # (N, HW, dim)
        v = self.v_proj(kv_in)                                          # (N, HW, dim)
        slots = self.slots_init.unsqueeze(0).expand(n, -1, -1)          # (N, M, dim)
        tau = self._tau()
        attn = None
        for _ in range(self.n_iters):
            q = self.q_proj(self.q_norm(slots))                          # (N, M, dim)
            logits = torch.einsum("nmd,nld->nml", q, k) * (self.scale / tau)
            attn = logits.softmax(dim=1)                                  # softmax over M
            attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)     # (N, M, 1)
            weights = attn / attn_sum                                     # (N, M, HW)
            updates = torch.einsum("nml,nld->nmd", weights, v)            # (N, M, dim)
            slots = self.norm_out(slots + updates)
            slots = slots + self.ffn(slots)
        assert attn is not None
        attn_map = (attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)).view(
            n, self.m, h, w
        )
        return slots, attn_map


# ---------------------------------------------------------------------------
# Pre-LN Transformer encoder block (used by SupportFusion)
# ---------------------------------------------------------------------------


class _DropPath(nn.Module):
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = (torch.rand(shape, device=x.device, dtype=x.dtype) < keep).to(x.dtype)
        return x * (mask / keep)


class _TransformerEncoderBlock(nn.Module):
    """Pre-LN self-attention + GELU FFN with MLP ratio = 4."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.dp1 = _DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.dp2 = _DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dp1(a)
        x = x + self.dp2(self.ffn(self.norm2(x)))
        return x


class SupportFusion(nn.Module):
    """3-layer pre-LN Transformer encoder over the K*M token bag.

    Adds learnable shot+slot positional embeddings so self-attention can
    consult "which support did this token come from" and "which region in
    that support". Critical for cross-shot voting; without this the K shots
    are independent tokens.
    """

    def __init__(
        self,
        dim: int = DIM,
        n_heads: int = N_HEADS,
        n_layers: int = FUSION_LAYERS,
        k_max: int = K_SUPPORT_MAX,
        m_tokens: int = M_TOKENS,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.shot_pe = nn.Parameter(torch.randn(k_max, dim) * 0.02)
        self.slot_pe = nn.Parameter(torch.randn(m_tokens, dim) * 0.02)
        self.k_max = k_max
        self.m = m_tokens
        self.layers = nn.ModuleList(
            [
                _TransformerEncoderBlock(dim, n_heads, mlp_ratio=4, drop_path=drop_path)
                for _ in range(n_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, k: int) -> torch.Tensor:
        b, n, d = tokens.shape
        m = n // k
        if k > self.k_max:
            raise ValueError(f"k={k} exceeds k_max={self.k_max}")
        if m != self.m:
            raise ValueError(f"slot count {m} != configured M={self.m}")
        shot_pe = self.shot_pe[:k].unsqueeze(1).expand(k, m, d)
        slot_pe = self.slot_pe.unsqueeze(0).expand(k, m, d)
        pe = (shot_pe + slot_pe).reshape(1, k * m, d)
        x = tokens + pe
        for layer in self.layers:
            x = layer(x)
        return self.out_norm(x)


# ---------------------------------------------------------------------------
# Class-attention pool (CaiT-style)
# ---------------------------------------------------------------------------


class ClassAttentionPool(nn.Module):
    """Single learnable [CLS] token attends over the support token bag.

    A fresh CLS query (rather than mean-pool) lets the pool weight informative
    tokens (object-relevant slots) over redundant ones (background, repeated
    shots). 1 layer, pre-LN, 8 heads, residual + FFN.
    """

    def __init__(self, dim: int = DIM, n_heads: int = N_HEADS) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        cls = self.cls.expand(b, -1, -1)
        q = self.norm_q(cls)
        kv = self.norm_kv(tokens)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        cls = cls + out
        cls = cls + self.ffn(self.norm_ffn(cls))
        return cls.squeeze(1)


# ---------------------------------------------------------------------------
# Cross-attention bridge — DETR-style 2-layer decoder, learnable softmax τ
# ---------------------------------------------------------------------------


class _LearnableTempMHA(nn.Module):
    """MultiheadAttention with a learnable softmax temperature applied to
    the attention logits. τ is parameterised via softplus to stay positive
    and clamped to [0.5, 2.0] for stability."""

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self._log_tau = nn.Parameter(torch.zeros(1))

    def _tau(self) -> torch.Tensor:
        return torch.clamp(F.softplus(self._log_tau) + 0.5, max=2.0)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        B, Lq, _ = q.shape
        Lk = k.shape[1]
        nh, hd = self.n_heads, self.head_dim
        qh = self.q_proj(q).reshape(B, Lq, nh, hd).transpose(1, 2)        # (B,nh,Lq,hd)
        kh = self.k_proj(k).reshape(B, Lk, nh, hd).transpose(1, 2)
        vh = self.v_proj(v).reshape(B, Lk, nh, hd).transpose(1, 2)
        scale = (hd ** -0.5) / self._tau()
        attn = (qh @ kh.transpose(-2, -1)) * scale                         # (B,nh,Lq,Lk)
        attn = attn.softmax(dim=-1)
        out = attn @ vh                                                    # (B,nh,Lq,hd)
        out = out.transpose(1, 2).reshape(B, Lq, self.dim)
        return self.out_proj(out)


class _DecoderLayer(nn.Module):
    """SelfAttn(query) → CrossAttn(query → support) → FFN, all pre-LN with DropPath."""

    def __init__(
        self,
        dim: int = DIM,
        n_heads: int = N_HEADS,
        drop_path: float = DROP_PATH,
    ) -> None:
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.dp_self = _DropPath(drop_path)

        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = _LearnableTempMHA(dim, n_heads)
        self.dp_cross = _DropPath(drop_path)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.dp_ffn = _DropPath(drop_path)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_n = self.norm_q1(q)
        sa, _ = self.self_attn(q_n, q_n, q_n, need_weights=False)
        q = q + self.dp_self(sa)

        q_n = self.norm_q2(q)
        kv_n = self.norm_kv(kv)
        ca = self.cross_attn(q_n, kv_n, kv_n)
        q = q + self.dp_cross(ca)

        q = q + self.dp_ffn(self.ffn(self.norm_ffn(q)))
        return q


class CrossAttentionHead(nn.Module):
    """2-layer DETR-style decoder. Returns (out, layer0_out)."""

    def __init__(
        self,
        dim: int = DIM,
        n_heads: int = N_HEADS,
        n_layers: int = DECODER_LAYERS,
        drop_path: float = DROP_PATH,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_DecoderLayer(dim, n_heads, drop_path) for _ in range(n_layers)]
        )

    def forward(
        self, q_feat: torch.Tensor, support_kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = q_feat.shape
        q = q_feat.flatten(2).transpose(1, 2)                              # (B, H*W, dim)
        layer0 = None
        for i, layer in enumerate(self.layers):
            q = layer(q, support_kv)
            if i == 0:
                layer0 = q
        out = q.transpose(1, 2).view(b, c, h, w)
        assert layer0 is not None
        l0 = layer0.transpose(1, 2).view(b, c, h, w)
        return out, l0


# ---------------------------------------------------------------------------
# Detection head — DFL-bins reg + conf + centerness, weight-shared across scales
# ---------------------------------------------------------------------------


class _DWSepGN(nn.Module):
    """Depthwise-separable conv with GroupNorm + GELU."""

    def __init__(self, in_c: int, out_c: int, gn_groups: int = 16) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.gn = nn.GroupNorm(min(gn_groups, out_c), out_c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.pw(self.dw(x))))


class DetectionHead(nn.Module):
    """4× DWSep-GN tower, then three 1×1 heads:

      reg : (B, 4*DFL_BINS, H, W) — discrete distribution per (l, t, r, b)
      conf: (B, 1, H, W)
      ctr : (B, 1, H, W) — centerness logit
    """

    PRIOR_PI: float = 0.01

    def __init__(self, dim: int = DIM, dfl_bins: int = DFL_BINS) -> None:
        super().__init__()
        self.dfl_bins = dfl_bins
        self.tower = nn.Sequential(
            _DWSepGN(dim, dim),
            _DWSepGN(dim, dim),
            _DWSepGN(dim, dim),
            _DWSepGN(dim, dim),
        )
        self.reg = nn.Conv2d(dim, 4 * dfl_bins, 1)
        self.conf = nn.Conv2d(dim, 1, 1)
        self.ctr = nn.Conv2d(dim, 1, 1)

        # RetinaNet bias-init for conf: π=0.01 → bias = -log((1-π)/π).
        nn.init.constant_(
            self.conf.bias, -math.log((1.0 - self.PRIOR_PI) / self.PRIOR_PI)
        )
        nn.init.zeros_(self.ctr.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.tower(x)
        return self.reg(f), self.conf(f), self.ctr(f)


# ---------------------------------------------------------------------------
# Presence head — deeper bottleneck MLP, harder to short-circuit
# ---------------------------------------------------------------------------


class PresenceHead(nn.Module):
    """[support_summary ⊕ GAP(enriched)] → 320 → 160 → 64 → 1, GELU + LN-pre."""

    def __init__(self, dim: int = DIM, dropout: float = 0.2) -> None:
        super().__init__()
        in_dim = dim * 2
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2 + dim // 4),                           # 64
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2 + dim // 4, 1),
        )

    def forward(
        self, support_summary: torch.Tensor, q_gap: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([support_summary, q_gap], dim=1)
        x = self.norm(x)
        return self.fc(x).squeeze(1)


# ---------------------------------------------------------------------------
# DFL utilities
# ---------------------------------------------------------------------------


def dfl_expectation(
    reg_logits: torch.Tensor, dfl_bins: int = DFL_BINS
) -> torch.Tensor:
    """(B, 4*K, H, W) discrete logits → (B, 4, H, W) expected scalar offsets.

    Bins represent integer offsets in stride units: 0, 1, ..., DFL_BINS-1.
    The expected value is sum_i softmax(logits)_i · i.
    """
    b, c, h, w = reg_logits.shape
    assert c == 4 * dfl_bins, f"expected channels={4 * dfl_bins}, got {c}"
    logits = reg_logits.view(b, 4, dfl_bins, h, w)
    p = logits.softmax(dim=2)
    bins = torch.arange(dfl_bins, device=reg_logits.device, dtype=p.dtype)
    return (p * bins.view(1, 1, dfl_bins, 1, 1)).sum(dim=2)                 # (B, 4, H, W)


def decode_ltrb_to_xyxy(
    ltrb: torch.Tensor, stride: int
) -> torch.Tensor:
    """(B, 4, H, W) (l, t, r, b) in stride units → (B, 4, H, W) xyxy boxes in image coords."""
    b, _, h, w = ltrb.shape
    device = ltrb.device
    j = torch.arange(w, device=device, dtype=ltrb.dtype).view(1, 1, w).expand(b, h, w)
    i = torch.arange(h, device=device, dtype=ltrb.dtype).view(1, h, 1).expand(b, h, w)
    cx = (j + 0.5) * stride
    cy = (i + 0.5) * stride
    l = ltrb[:, 0] * stride
    t = ltrb[:, 1] * stride
    r = ltrb[:, 2] * stride
    b_ = ltrb[:, 3] * stride
    return torch.stack([cx - l, cy - t, cx + r, cy + b_], dim=1)


# ---------------------------------------------------------------------------
# Full Siamese localiser
# ---------------------------------------------------------------------------


class FewShotLocalizer(nn.Module):
    """Siamese few-shot localiser. Backbone weights are shared between the
    support and query streams (one ``MobileNetBackbone`` instance, called
    once for supports and once for the query)."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        # Shared backbone — Siamese contract.
        self.backbone = MobileNetBackbone(pretrained=pretrained)

        # Support branch.
        self.support_tokenizer = SupportTokenizer()
        self.support_fusion = SupportFusion(drop_path=DROP_PATH)
        self.support_pool = ClassAttentionPool()

        # Query branch.
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[P4_CHANNELS, P5_CHANNELS],
            out_channels=DIM,
        )
        self.p3_lat = nn.Conv2d(P3_CHANNELS, DIM, kernel_size=1, bias=False)
        self.p3_gate = nn.Parameter(torch.zeros(1, DIM, 1, 1))

        # Stride-8 detection feature map: upsample enriched query feat (14×14)
        # to 28×28 and add a 1×1-projected P3 lateral.
        self.p3_det_lat = nn.Conv2d(P3_CHANNELS, DIM, kernel_size=1, bias=False)

        # Learnable 2D PE for the query feature map (14×14). Sinusoidal floor
        # is added in forward.
        self.query_pe_p4 = nn.Parameter(torch.zeros(1, DIM, GRID_P4, GRID_P4))
        nn.init.trunc_normal_(self.query_pe_p4, std=0.02)

        # [ABSENT] sink token for cross-attention; token-dropout p=0.1 during training.
        self.absent_token = nn.Parameter(torch.zeros(1, 1, DIM))
        nn.init.trunc_normal_(self.absent_token, std=0.02)
        self.absent_dropout_p = 0.1

        # Cross-attention bridge.
        self.cross_attn = CrossAttentionHead()

        # Detection heads — weight-shared across stride-8 and stride-16 outputs.
        self.det_head = DetectionHead()
        # Aux head on decoder-layer-0 output, stride-16 only (cheap aux loss).
        self.aux_head = DetectionHead()

        # Presence head.
        self.presence_head = PresenceHead()

        # Score-calibration scalar (init=1.0). Multiplies conf logits at train
        # time — lets the conf distribution expand without rebalancing other
        # losses.
        self.conf_scale = nn.Parameter(torch.ones(1))

    # ---- support branch ----------------------------------------------------

    def encode_support(
        self,
        support_imgs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """support_imgs: (B, K, 3, H, W).

        Returns:
            tokens             — (B, K*M, DIM) fused token bag.
            per_shot_prototype — (B, K, DIM) per-shot pooled summary.
            attn               — (B, K, M, 7, 7) slot attention maps.
        """
        b, k = support_imgs.shape[:2]
        flat = support_imgs.reshape(b * k, 3, support_imgs.shape[3], support_imgs.shape[4])
        feat = self.backbone.forward_p5_only(flat)
        tokens, attn = self.support_tokenizer(feat)
        m, dim = tokens.shape[1], tokens.shape[2]

        # Per-shot pooled summary (BEFORE fusion) for contrastive losses.
        per_shot = self.support_pool(tokens)
        per_shot = per_shot.view(b, k, dim)

        # Fuse the K*M token bag.
        tokens = tokens.view(b, k * m, dim)
        tokens = self.support_fusion(tokens, k=k)

        attn = attn.view(b, k, m, attn.shape[-2], attn.shape[-1])
        return tokens, per_shot, attn

    @torch.no_grad()
    def compute_prototype(self, support_imgs: torch.Tensor) -> torch.Tensor:
        tokens, _, _ = self.encode_support(support_imgs)
        return tokens

    # ---- query branch ------------------------------------------------------

    def encode_query(
        self, query_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (q_p4 stride-16, p3_for_det stride-8 lateral).

        q_p4 : (B, DIM, H/16, W/16)
        p3_l : (B, DIM, H/8,  W/8 )
        """
        p3, p4, p5 = self.backbone(query_img)
        fpn_out = self.fpn(OrderedDict([("p4", p4), ("p5", p5)]))
        feat = fpn_out["p4"]                                              # (B, DIM, H/16, W/16)
        # Stride-8 residual added to P4 (downsampled)
        p3_down = F.avg_pool2d(self.p3_lat(p3), kernel_size=2)
        gate = torch.sigmoid(self.p3_gate)
        feat = feat + gate * p3_down

        # Stride-8 detection lateral (kept at 28×28).
        p3_det = self.p3_det_lat(p3)
        return feat, p3_det

    # ---- core head ---------------------------------------------------------

    def _heads_forward(
        self,
        q_feat_p4: torch.Tensor,
        q_feat_p3: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b = q_feat_p4.shape[0]
        if support_tokens.shape[0] != b:
            support_tokens = support_tokens.expand(b, -1, -1)

        # Add learnable + sinusoidal PE to query at stride 16. The learnable
        # PE is stored at the canonical 14×14 grid; it's bilinearly resized
        # to whatever grid the current input produced (multi-scale training,
        # 2-scale TTA, etc.).
        h, w = q_feat_p4.shape[-2:]
        learn_pe = self.query_pe_p4
        if learn_pe.shape[-2:] != (h, w):
            learn_pe = F.interpolate(
                learn_pe, size=(h, w), mode="bilinear", align_corners=False
            )
        sin_pe = _sinusoidal_2d_pe(
            q_feat_p4.shape[1], h, w,
            device=q_feat_p4.device, dtype=q_feat_p4.dtype,
        )
        q = q_feat_p4 + learn_pe + sin_pe

        # [ABSENT] sink token; token-dropout during training.
        absent = self.absent_token.expand(b, -1, -1)
        if self.training and self.absent_dropout_p > 0.0:
            keep = (
                torch.rand(b, 1, 1, device=q.device) >= self.absent_dropout_p
            ).to(q.dtype)
            absent = absent * keep
        cross_kv = torch.cat([support_tokens, absent], dim=1)

        enriched_p4, enriched_aux = self.cross_attn(q, cross_kv)

        # Stride-8 enriched feature: upsample stride-16 + add P3 lateral.
        enriched_up = F.interpolate(
            enriched_p4, size=q_feat_p3.shape[-2:], mode="bilinear", align_corners=False
        )
        enriched_p3 = enriched_up + q_feat_p3

        # Detection heads.
        reg_p4_logits, conf_p4, ctr_p4 = self.det_head(enriched_p4)
        reg_p3_logits, conf_p3, ctr_p3 = self.det_head(enriched_p3)
        # Aux head on stride-16 layer-0 decoder output.
        reg_aux_logits, conf_aux, ctr_aux = self.aux_head(enriched_aux)

        # Score-calibration scalar on conf logits (kept ≥0 via softplus shift).
        scale = F.softplus(self.conf_scale) + 1e-3
        conf_p4 = conf_p4 * scale
        conf_p3 = conf_p3 * scale
        conf_aux = conf_aux * scale

        # Presence head on bag-level summary + GAP(enriched_p4).
        q_gap = enriched_p4.mean(dim=(2, 3))
        support_summary = self.support_pool(support_tokens)
        presence_logit = self.presence_head(support_summary, q_gap)

        return {
            "reg_p4_logits": reg_p4_logits,
            "conf_p4": conf_p4,
            "ctr_p4": ctr_p4,
            "reg_p3_logits": reg_p3_logits,
            "conf_p3": conf_p3,
            "ctr_p3": ctr_p3,
            "reg_aux_logits": reg_aux_logits,
            "conf_aux": conf_aux,
            "ctr_aux": ctr_aux,
            "presence_logit": presence_logit,
            "support_tokens": support_tokens,
            "prototype": support_summary,
        }

    # ---- inference path ---------------------------------------------------

    def detect(
        self, prototype: torch.Tensor, query_img: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        q_p4, q_p3 = self.encode_query(query_img)
        return self._heads_forward(q_p4, q_p3, prototype)

    # ---- training forward -------------------------------------------------

    def forward(
        self,
        support_imgs: torch.Tensor,
        query_img: torch.Tensor,
        support_bboxes: torch.Tensor | None = None,
        prototype: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        per_shot: torch.Tensor | None
        attn: torch.Tensor | None
        if prototype is None:
            tokens, per_shot, attn = self.encode_support(support_imgs)
        else:
            tokens, per_shot, attn = prototype, None, None
        q_p4, q_p3 = self.encode_query(query_img)
        out = self._heads_forward(q_p4, q_p3, tokens)
        out["per_shot_prototype"] = per_shot                                # type: ignore[assignment]
        out["support_attn"] = attn                                          # type: ignore[assignment]
        return out


# ---------------------------------------------------------------------------
# Decoding — top-K + NMS, presence-gated, multi-scale union
# ---------------------------------------------------------------------------


def _decode_one_scale(
    reg_logits: torch.Tensor,
    conf_logits: torch.Tensor,
    ctr_logits: torch.Tensor,
    stride: int,
    dfl_bins: int = DFL_BINS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns:
      boxes : (B, H*W, 4) xyxy in image coords
      score : (B, H*W) — σ(conf) · σ(centerness) (presence applied later).
    """
    ltrb = dfl_expectation(reg_logits, dfl_bins=dfl_bins)
    boxes = decode_ltrb_to_xyxy(ltrb, stride=stride)                        # (B, 4, H, W)
    b = boxes.shape[0]
    score = (
        torch.sigmoid(conf_logits) * torch.sigmoid(ctr_logits)
    ).view(b, -1)
    boxes = boxes.permute(0, 2, 3, 1).reshape(b, -1, 4)
    return boxes, score


def decode_topk(
    out: dict[str, torch.Tensor],
    img_size: int = IMG_SIZE,
    top_k: int = 100,
    conf_thr: float = 0.05,
    nms_iou: float = 0.5,
    use_aux: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-image top-K with NMS across the union of stride-8 and stride-16 grids.

    Returns:
      boxes_per_image : list of length B, each (N_i, 4) xyxy boxes (clipped to img_size).
      scores_per_image: list of length B, each (N_i,) float scores in [0, 1].

    Score = σ(conf) · σ(centerness) · σ(presence_logit), conf gated by τ.
    Presence is broadcast as a per-image multiplicative gate.
    """
    boxes_p4, score_p4 = _decode_one_scale(
        out["reg_p4_logits"], out["conf_p4"], out["ctr_p4"], stride=STRIDE_P4
    )
    boxes_p3, score_p3 = _decode_one_scale(
        out["reg_p3_logits"], out["conf_p3"], out["ctr_p3"], stride=STRIDE_P3
    )
    boxes = torch.cat([boxes_p4, boxes_p3], dim=1)                          # (B, N, 4)
    score = torch.cat([score_p4, score_p3], dim=1)                          # (B, N)

    if use_aux and "reg_aux_logits" in out:
        boxes_aux, score_aux = _decode_one_scale(
            out["reg_aux_logits"],
            out["conf_aux"],
            out["ctr_aux"],
            stride=STRIDE_P4,
        )
        boxes = torch.cat([boxes, boxes_aux], dim=1)
        score = torch.cat([score, score_aux], dim=1)

    presence = torch.sigmoid(out["presence_logit"]).view(-1, 1)             # (B, 1)
    score = score * presence

    # Clip to image boundaries.
    boxes = boxes.clamp(min=0.0, max=float(img_size))

    boxes_out: list[torch.Tensor] = []
    scores_out: list[torch.Tensor] = []
    B = boxes.shape[0]
    for i in range(B):
        s = score[i]
        keep = s >= conf_thr
        if keep.sum() == 0:
            boxes_out.append(boxes.new_zeros((0, 4)))
            scores_out.append(boxes.new_zeros((0,)))
            continue
        b_i = boxes[i, keep]
        s_i = s[keep]
        # Top-K before NMS for efficiency.
        if s_i.numel() > top_k * 4:
            tk = torch.topk(s_i, k=top_k * 4, largest=True)
            b_i = b_i[tk.indices]
            s_i = tk.values
        # Class-agnostic NMS via single class id 0.
        idx_classes = torch.zeros_like(s_i, dtype=torch.long)
        keep_idx = batched_nms(b_i, s_i, idx_classes, iou_threshold=nms_iou)
        b_i = b_i[keep_idx][:top_k]
        s_i = s_i[keep_idx][:top_k]
        boxes_out.append(b_i)
        scores_out.append(s_i)
    return boxes_out, scores_out


def decode_top1(
    out: dict[str, torch.Tensor],
    img_size: int = IMG_SIZE,
    conf_thr: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience: top-1 per image after `decode_topk`. Used by IoU-style metrics
    (val_iou, contain). Returns (B, 4) and (B,)."""
    boxes_pi, scores_pi = decode_topk(out, img_size=img_size, conf_thr=conf_thr)
    B = len(boxes_pi)
    box_t = out["conf_p4"].new_zeros((B, 4))
    score_t = out["conf_p4"].new_zeros((B,))
    for i, (bxs, scs) in enumerate(zip(boxes_pi, scores_pi)):
        if bxs.numel() > 0:
            box_t[i] = bxs[0]
            score_t[i] = scs[0]
    return box_t, score_t
