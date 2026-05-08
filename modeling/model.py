"""Siamese few-shot localiser (Phase 0 architecture overhaul).

Architecture (Siamese — backbone weights shared between support and query):

  shared MobileNetV3-Large backbone (+ optional 2-block ViT head over P5)
   ├── support_imgs (B,K,3,H,W) → P5 feat ────► SupportTokenizer (slot-attn, M=6, 2 iters)
   │                                          ─► (B*K, M, DIM) tokens
   │                                          ─► reshape (B, K*M, DIM)
   │                                          ─► SupportFusion (3-layer pre-LN Transformer
   │                                                            encoder, 8 heads, MLP×4)
   │                                          ─► (B, K*M, DIM) fused
   │                                          ─► PerShotPool (K independent CaiT [CLS]
   │                                                          queries → (B, K, DIM))
   │
   └── query_img (B,3,H,W) → P3,P4,P5 ──► FPN(P4,P5) + gated P3 residual
                                       ─► query_pe (learnable bilinear-resizable
                                                    + sinusoidal floor)
                                       ─► CrossAttentionHead (3-layer DETR decoder,
                                                              learnable softmax τ,
                                                              DropPath p=0.20)
                                       ─► enriched query feat (B,DIM,H/16,W/16)
                                       ─► gated P3 lateral for stride-8 detection path

Detection heads — DECOUPLED across stride-8 / stride-16 / aux scales (each is a
separate ``DetectionHead`` instance). Each emits:
    reg     : 4 * DFL_BINS=32 channels (DFL distribution per l/t/r/b)
    conf    : 1 channel
    ctr     : 1 channel  (FCOS centerness)
    pred_iou: 1 channel  (IoU-Aware FCOS / VarifocalNet ranking head)

Inference score = σ(conf) · σ(centerness) · σ(pred_iou) · σ(presence_logit).

Presence head:
  Input is per-shot prototypes (B, K, DIM) flattened ⊕ GAP(enriched_query) →
  K·DIM + DIM → 256 → 64 → 1, GELU + LN-pre + dropout 0.2.
  Per-shot input lets the head consult shot-disagreement directly (the prior
  bag-summary version aggregated this away → presence_acc_neg ~ 0.5).

Outputs (forward):
  reg_p4_logits, conf_p4, ctr_p4, pred_iou_p4   (stride-16 grid)
  reg_p3_logits, conf_p3, ctr_p3, pred_iou_p3   (stride-8 grid)
  reg_aux_logits, conf_aux, ctr_aux, pred_iou_aux  (aux head on decoder layer 0)
  presence_logit  (B,)
  prototype       (B, DIM)  — bag-level summary (avg of per-shot prototypes)
  per_shot_prototype (B, K, DIM)
  support_attn    (B, K, M, 7, 7)
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
GRID_P4 = 14
GRID_P3 = 28
STRIDE_P4 = 16
STRIDE_P3 = 8

DIM = 160
M_TOKENS = 6
N_HEADS = 8
K_SUPPORT_MAX = 8
FUSION_LAYERS = 3
DECODER_LAYERS = 3                                # was 2 — Phase 0 deeper decoder
SLOT_ATTN_ITERS = 2
DROP_PATH = 0.20                                  # was 0.15 — compensate for added capacity

# Distribution Focal Loss bins per coordinate. Bumped from 17 to 32 so the
# regressor can express larger objects without bin-clamp truncation.
DFL_BINS = 32

P3_IDX = 6
P3_CHANNELS = 40
P4_IDX = 12
P4_CHANNELS = 112
P5_IDX = 15
P5_CHANNELS = 160


# ---------------------------------------------------------------------------
# Backbone (shared between support and query — Siamese)
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
# 2D sinusoidal positional encoding
# ---------------------------------------------------------------------------


def _sinusoidal_2d_pe(dim: int, h: int, w: int, device, dtype=torch.float32) -> torch.Tensor:
    if dim % 4 != 0:
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
# DropPath + pre-LN transformer block
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
    """Pre-LN self-attention + GELU FFN."""

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


# ---------------------------------------------------------------------------
# ViT head over P5 — adds global context to the backbone (Phase 0 last-resort)
# ---------------------------------------------------------------------------


class ViTHeadP5(nn.Module):
    """2-block ViT applied to the P5 feature map (7×7×160 → 7×7×160).

    Adds global self-attention so the support tokenizer + cross-attention
    bridge get spatially-mixed P5 features instead of the raw MobileNet
    P5 output. ~0.4M params; cheap because it only sees 49 tokens.
    """

    def __init__(
        self,
        dim: int = P5_CHANNELS,
        n_heads: int = N_HEADS,
        n_layers: int = 2,
        drop_path: float = DROP_PATH,
    ) -> None:
        super().__init__()
        # 7×7 grid PE
        self.pe = nn.Parameter(torch.zeros(1, dim, 7, 7))
        nn.init.trunc_normal_(self.pe, std=0.02)
        self.layers = nn.ModuleList(
            [_TransformerEncoderBlock(dim, n_heads, mlp_ratio=4, drop_path=drop_path)
             for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, p5: torch.Tensor) -> torch.Tensor:
        b, c, h, w = p5.shape
        # Resize PE if input grid size != 7×7 (multi-scale TTA)
        pe = self.pe
        if pe.shape[-2:] != (h, w):
            pe = F.interpolate(pe, size=(h, w), mode="bilinear", align_corners=False)
        x = (p5 + pe).flatten(2).transpose(1, 2)                                # (B, HW, C)
        for layer in self.layers:
            x = layer(x)
        x = self.out_norm(x)
        return x.transpose(1, 2).view(b, c, h, w) + p5                          # residual


# ---------------------------------------------------------------------------
# Slot-attention support tokenizer
# ---------------------------------------------------------------------------


class SupportTokenizer(nn.Module):
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
        return F.softplus(self._log_tau) + 0.5

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, _, h, w = feat.shape
        kv_in = self.kv_norm(feat.flatten(2).transpose(1, 2))
        k = self.k_proj(kv_in)
        v = self.v_proj(kv_in)
        slots = self.slots_init.unsqueeze(0).expand(n, -1, -1)
        tau = self._tau()
        attn = None
        for _ in range(self.n_iters):
            q = self.q_proj(self.q_norm(slots))
            logits = torch.einsum("nmd,nld->nml", q, k) * (self.scale / tau)
            attn = logits.softmax(dim=1)
            attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            weights = attn / attn_sum
            updates = torch.einsum("nml,nld->nmd", weights, v)
            slots = self.norm_out(slots + updates)
            slots = slots + self.ffn(slots)
        assert attn is not None
        attn_map = (attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)).view(
            n, self.m, h, w
        )
        return slots, attn_map


# ---------------------------------------------------------------------------
# Inter-shot fusion — 3-layer pre-LN transformer encoder
# ---------------------------------------------------------------------------


class SupportFusion(nn.Module):
    def __init__(
        self,
        dim: int = DIM,
        n_heads: int = N_HEADS,
        n_layers: int = FUSION_LAYERS,
        k_max: int = K_SUPPORT_MAX,
        m_tokens: int = M_TOKENS,
        drop_path: float = DROP_PATH,
    ) -> None:
        super().__init__()
        self.shot_pe = nn.Parameter(torch.randn(k_max, dim) * 0.02)
        self.slot_pe = nn.Parameter(torch.randn(m_tokens, dim) * 0.02)
        self.k_max = k_max
        self.m = m_tokens
        self.layers = nn.ModuleList(
            [_TransformerEncoderBlock(dim, n_heads, mlp_ratio=4, drop_path=drop_path)
             for _ in range(n_layers)]
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
# Per-shot CaiT-style attention pool — K independent [CLS] queries
# ---------------------------------------------------------------------------


class PerShotAttentionPool(nn.Module):
    """Per-shot pool: ``K`` independent CaiT [CLS] queries each attend over
    the full ``K*M`` token bag. Returns ``(B, K, DIM)`` per-shot prototypes.

    The ``forward_bag`` method preserves the previous bag-level interface
    (single CLS over all tokens) for the cross-attention bridge.
    """

    def __init__(
        self,
        dim: int = DIM,
        n_heads: int = N_HEADS,
        k_max: int = K_SUPPORT_MAX,
    ) -> None:
        super().__init__()
        self.k_max = k_max
        self.cls = nn.Parameter(torch.randn(k_max, 1, dim) * 0.02)
        self.bag_cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def _pool(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: (B, Q, D); kv: (B, N, D)
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        out, _ = self.attn(q_n, kv_n, kv_n, need_weights=False)
        x = q + out
        x = x + self.ffn(self.norm_ffn(x))
        return x

    def forward_per_shot(self, tokens: torch.Tensor, k: int) -> torch.Tensor:
        """Returns per-shot prototypes (B, K, DIM)."""
        b = tokens.shape[0]
        if k > self.k_max:
            raise ValueError(f"k={k} exceeds k_max={self.k_max}")
        q = self.cls[:k].unsqueeze(0).expand(b, k, 1, tokens.shape[-1])
        # We need K independent queries each attending over the bag.
        # Stack into (B, K, D) by treating Q=K and reshaping:
        q = q.reshape(b, k, tokens.shape[-1])                                  # (B, K, D)
        return self._pool(q, tokens)

    def forward_bag(self, tokens: torch.Tensor) -> torch.Tensor:
        """Returns bag-level summary (B, DIM) via a single shared [CLS] query."""
        b = tokens.shape[0]
        q = self.bag_cls.expand(b, -1, -1)                                     # (B, 1, D)
        out = self._pool(q, tokens)
        return out.squeeze(1)

    def forward(
        self, tokens: torch.Tensor, k: int | None = None
    ) -> torch.Tensor:
        """Default ``__call__`` path. Delegates to ``forward_bag`` (the most
        common usage: bag-level summary used by ``build_proto_cache`` and
        anywhere that wants a ``(B, DIM)`` prototype). Pass ``k`` to get
        ``forward_per_shot`` semantics instead.
        """
        if k is None:
            return self.forward_bag(tokens)
        return self.forward_per_shot(tokens, k)


# ---------------------------------------------------------------------------
# Cross-attention bridge — DETR-style 3-layer decoder
# ---------------------------------------------------------------------------


class _LearnableTempMHA(nn.Module):
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
        qh = self.q_proj(q).reshape(B, Lq, nh, hd).transpose(1, 2)
        kh = self.k_proj(k).reshape(B, Lk, nh, hd).transpose(1, 2)
        vh = self.v_proj(v).reshape(B, Lk, nh, hd).transpose(1, 2)
        scale = (hd ** -0.5) / self._tau()
        attn = (qh @ kh.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = attn @ vh
        out = out.transpose(1, 2).reshape(B, Lq, self.dim)
        return self.out_proj(out)


class _DecoderLayer(nn.Module):
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
    """3-layer DETR-style decoder. Returns (final, layer0)."""

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
        q = q_feat.flatten(2).transpose(1, 2)
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
# Detection head — DFL reg + conf + centerness + pred_iou
# ---------------------------------------------------------------------------


class _DWSepGN(nn.Module):
    def __init__(self, in_c: int, out_c: int, gn_groups: int = 16) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.gn = nn.GroupNorm(min(gn_groups, out_c), out_c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.pw(self.dw(x))))


class DetectionHead(nn.Module):
    """4× DWSep-GN tower → reg / conf / ctr / pred_iou.

      reg     : (B, 4*DFL_BINS, H, W)
      conf    : (B, 1, H, W)
      ctr     : (B, 1, H, W)
      pred_iou: (B, 1, H, W)
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
        self.pred_iou = nn.Conv2d(dim, 1, 1)

        nn.init.constant_(
            self.conf.bias, -math.log((1.0 - self.PRIOR_PI) / self.PRIOR_PI)
        )
        nn.init.zeros_(self.ctr.bias)
        nn.init.zeros_(self.pred_iou.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.tower(x)
        return self.reg(f), self.conf(f), self.ctr(f), self.pred_iou(f)


# ---------------------------------------------------------------------------
# Presence head — consumes per-shot prototypes
# ---------------------------------------------------------------------------


class PresenceHead(nn.Module):
    """[per_shot_prototypes ⊕ GAP(enriched)] → 256 → 64 → 1.

    Per-shot input preserves shot-disagreement, which the prior bag-summary
    head aggregated away. This directly attacks the ``presence_acc_neg ≈ 0.5``
    bias.
    """

    def __init__(
        self, dim: int = DIM, k_max: int = K_SUPPORT_MAX, dropout: float = 0.2
    ) -> None:
        super().__init__()
        in_dim = dim * k_max + dim                                              # K*D + D
        self.norm = nn.LayerNorm(in_dim)
        hidden1 = 256
        hidden2 = 64
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )
        self.k_max = k_max
        self.dim = dim

    def forward(
        self, per_shot_prototype: torch.Tensor, q_gap: torch.Tensor
    ) -> torch.Tensor:
        b, k, d = per_shot_prototype.shape
        # Pad up to k_max with zeros so input dim is constant.
        if k < self.k_max:
            pad = per_shot_prototype.new_zeros(b, self.k_max - k, d)
            per_shot_prototype = torch.cat([per_shot_prototype, pad], dim=1)
        flat = per_shot_prototype.reshape(b, -1)
        x = torch.cat([flat, q_gap], dim=1)
        x = self.norm(x)
        return self.fc(x).squeeze(1)


# ---------------------------------------------------------------------------
# DFL utilities
# ---------------------------------------------------------------------------


def dfl_expectation(
    reg_logits: torch.Tensor, dfl_bins: int = DFL_BINS
) -> torch.Tensor:
    b, c, h, w = reg_logits.shape
    assert c == 4 * dfl_bins, f"expected channels={4 * dfl_bins}, got {c}"
    logits = reg_logits.view(b, 4, dfl_bins, h, w)
    p = logits.softmax(dim=2)
    bins = torch.arange(dfl_bins, device=reg_logits.device, dtype=p.dtype)
    return (p * bins.view(1, 1, dfl_bins, 1, 1)).sum(dim=2)


def decode_ltrb_to_xyxy(ltrb: torch.Tensor, stride: float | int) -> torch.Tensor:
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
    """Siamese few-shot localiser. Backbone weights are shared between
    support and query streams (one ``MobileNetBackbone`` instance)."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        # Shared backbone — Siamese contract.
        self.backbone = MobileNetBackbone(pretrained=pretrained)

        # ViT head over P5 — adds global self-attention to the backbone
        # output before slot-attention / FPN consume it.
        self.vit_head = ViTHeadP5()

        # Support branch.
        self.support_tokenizer = SupportTokenizer()
        self.support_fusion = SupportFusion(drop_path=DROP_PATH)
        self.support_pool = PerShotAttentionPool()

        # Query branch.
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[P4_CHANNELS, P5_CHANNELS],
            out_channels=DIM,
        )
        self.p3_lat = nn.Conv2d(P3_CHANNELS, DIM, kernel_size=1, bias=False)
        self.p3_gate = nn.Parameter(torch.zeros(1, DIM, 1, 1))

        # Stride-8 detection lateral with its own per-channel gate
        # (mirrors the FPN-side P3 gate so the model can dial the stride-8
        # detection path up or down independently).
        self.p3_det_lat = nn.Conv2d(P3_CHANNELS, DIM, kernel_size=1, bias=False)
        self.p3_det_gate = nn.Parameter(torch.zeros(1, DIM, 1, 1))

        # Learnable + sinusoidal PE for the query feature map at stride 16.
        self.query_pe_p4 = nn.Parameter(torch.zeros(1, DIM, GRID_P4, GRID_P4))
        nn.init.trunc_normal_(self.query_pe_p4, std=0.02)

        # Per-slot [ABSENT] token bank (one per support slot). Token-dropout
        # p=0.2 applied during training.
        self.absent_tokens = nn.Parameter(torch.zeros(1, M_TOKENS, DIM))
        nn.init.trunc_normal_(self.absent_tokens, std=0.02)
        self.absent_dropout_p = 0.2

        # Cross-attention bridge.
        self.cross_attn = CrossAttentionHead()

        # DECOUPLED detection heads: one per scale.
        self.det_head_p4 = DetectionHead()
        self.det_head_p3 = DetectionHead()
        self.det_head_aux = DetectionHead()

        # Presence head consumes per-shot prototypes ⊕ GAP(enriched_p4).
        self.presence_head = PresenceHead()

        # Score-calibration scalar.
        self.conf_scale = nn.Parameter(torch.ones(1))

    # ---- support branch ----------------------------------------------------

    def encode_support(
        self,
        support_imgs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, k = support_imgs.shape[:2]
        flat = support_imgs.reshape(b * k, 3, support_imgs.shape[3], support_imgs.shape[4])
        feat = self.backbone.forward_p5_only(flat)
        feat = self.vit_head(feat)
        tokens, attn = self.support_tokenizer(feat)
        m, dim = tokens.shape[1], tokens.shape[2]

        # Fuse the K*M token bag.
        bag = tokens.view(b, k * m, dim)
        bag_fused = self.support_fusion(bag, k=k)

        # Per-shot prototypes — independent CaiT [CLS] queries.
        per_shot = self.support_pool.forward_per_shot(bag_fused, k=k)            # (B, K, DIM)

        attn = attn.view(b, k, m, attn.shape[-2], attn.shape[-1])
        return bag_fused, per_shot, attn

    @torch.no_grad()
    def compute_prototype(self, support_imgs: torch.Tensor) -> torch.Tensor:
        tokens, _, _ = self.encode_support(support_imgs)
        return tokens

    # ---- query branch ------------------------------------------------------

    def encode_query(
        self, query_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p3, p4, p5 = self.backbone(query_img)
        p5 = self.vit_head(p5)
        fpn_out = self.fpn(OrderedDict([("p4", p4), ("p5", p5)]))
        feat = fpn_out["p4"]
        p3_down = F.avg_pool2d(self.p3_lat(p3), kernel_size=2)
        gate = torch.sigmoid(self.p3_gate)
        feat = feat + gate * p3_down

        # Stride-8 detection lateral, gated independently.
        p3_det = self.p3_det_lat(p3)
        p3_det_gate = torch.sigmoid(self.p3_det_gate)
        return feat, p3_det, p3_det_gate

    # ---- core head ---------------------------------------------------------

    def _heads_forward(
        self,
        q_feat_p4: torch.Tensor,
        q_feat_p3: torch.Tensor,
        p3_det_gate: torch.Tensor,
        support_tokens: torch.Tensor,
        per_shot_prototype: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b = q_feat_p4.shape[0]
        if support_tokens.shape[0] != b:
            support_tokens = support_tokens.expand(b, -1, -1)

        # Add learnable + sinusoidal PE.
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

        # Per-slot absent token bank with token-dropout.
        absent = self.absent_tokens.expand(b, -1, -1)
        if self.training and self.absent_dropout_p > 0.0:
            keep = (
                torch.rand(b, absent.shape[1], 1, device=q.device)
                >= self.absent_dropout_p
            ).to(q.dtype)
            absent = absent * keep
        cross_kv = torch.cat([support_tokens, absent], dim=1)

        enriched_p4, enriched_aux = self.cross_attn(q, cross_kv)

        # Stride-8 enriched feature: upsample stride-16 + gated P3 lateral.
        enriched_up = F.interpolate(
            enriched_p4, size=q_feat_p3.shape[-2:], mode="bilinear", align_corners=False
        )
        enriched_p3 = enriched_up + p3_det_gate * q_feat_p3

        # Decoupled detection heads.
        reg_p4_logits, conf_p4, ctr_p4, iou_p4 = self.det_head_p4(enriched_p4)
        reg_p3_logits, conf_p3, ctr_p3, iou_p3 = self.det_head_p3(enriched_p3)
        reg_aux_logits, conf_aux, ctr_aux, iou_aux = self.det_head_aux(enriched_aux)

        # Score-calibration scalar.
        scale = F.softplus(self.conf_scale) + 1e-3
        conf_p4 = conf_p4 * scale
        conf_p3 = conf_p3 * scale
        conf_aux = conf_aux * scale

        # Presence head: per-shot prototypes ⊕ GAP(enriched_p4).
        q_gap = enriched_p4.mean(dim=(2, 3))
        presence_logit = self.presence_head(per_shot_prototype, q_gap)

        # Bag-level summary for prototype regularisers / hard-neg cache.
        prototype = per_shot_prototype.mean(dim=1)

        return {
            "reg_p4_logits": reg_p4_logits,
            "conf_p4": conf_p4,
            "ctr_p4": ctr_p4,
            "pred_iou_p4": iou_p4,
            "reg_p3_logits": reg_p3_logits,
            "conf_p3": conf_p3,
            "ctr_p3": ctr_p3,
            "pred_iou_p3": iou_p3,
            "reg_aux_logits": reg_aux_logits,
            "conf_aux": conf_aux,
            "ctr_aux": ctr_aux,
            "pred_iou_aux": iou_aux,
            "presence_logit": presence_logit,
            "support_tokens": support_tokens,
            "prototype": prototype,
            # Enriched-query features at stride 16 — exposed for the
            # presence-aware contrastive + feature-spread losses.
            "enriched_p4": enriched_p4,
        }

    # ---- inference path ---------------------------------------------------

    def detect(
        self, support_tokens: torch.Tensor, per_shot_prototype: torch.Tensor,
        query_img: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        q_p4, q_p3, p3g = self.encode_query(query_img)
        return self._heads_forward(q_p4, q_p3, p3g, support_tokens, per_shot_prototype)

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
        if per_shot is None:
            # Inference path through ``detect`` provides prototype directly;
            # for ``forward`` we always re-encode.
            tokens, per_shot, attn = self.encode_support(support_imgs)
        q_p4, q_p3, p3g = self.encode_query(query_img)
        out = self._heads_forward(q_p4, q_p3, p3g, tokens, per_shot)
        out["per_shot_prototype"] = per_shot
        out["support_attn"] = attn
        return out


# ---------------------------------------------------------------------------
# Decoding — top-K + NMS, multi-signal score
# ---------------------------------------------------------------------------


def _decode_one_scale(
    reg_logits: torch.Tensor,
    conf_logits: torch.Tensor,
    ctr_logits: torch.Tensor,
    pred_iou_logits: torch.Tensor,
    stride: float,
    dfl_bins: int = DFL_BINS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (boxes, score) where score = σ(conf)·σ(centerness)·σ(pred_iou)."""
    ltrb = dfl_expectation(reg_logits, dfl_bins=dfl_bins)
    boxes = decode_ltrb_to_xyxy(ltrb, stride=stride)
    b = boxes.shape[0]
    score = (
        torch.sigmoid(conf_logits)
        * torch.sigmoid(ctr_logits)
        * torch.sigmoid(pred_iou_logits)
    ).view(b, -1)
    boxes = boxes.permute(0, 2, 3, 1).reshape(b, -1, 4)
    return boxes, score


def decode_topk(
    out: dict[str, torch.Tensor],
    img_size: int = IMG_SIZE,
    top_k: int = 100,
    conf_thr: float = 0.03,
    nms_iou: float = 0.5,
    use_aux: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-image top-K with NMS across the union of stride-8 / stride-16 grids.

    Final score = σ(conf) · σ(centerness) · σ(pred_iou) · σ(presence_logit).
    ``conf_thr`` defaulted to 0.03 (was 0.05) — extra ranking signals tighten
    score distribution so a lower threshold is safe.
    """
    p4_h = out["conf_p4"].shape[-2]
    p3_h = out["conf_p3"].shape[-2]
    stride_p4 = img_size / float(p4_h)
    stride_p3 = img_size / float(p3_h)

    boxes_p4, score_p4 = _decode_one_scale(
        out["reg_p4_logits"], out["conf_p4"], out["ctr_p4"], out["pred_iou_p4"],
        stride=stride_p4,
    )
    boxes_p3, score_p3 = _decode_one_scale(
        out["reg_p3_logits"], out["conf_p3"], out["ctr_p3"], out["pred_iou_p3"],
        stride=stride_p3,
    )
    boxes = torch.cat([boxes_p4, boxes_p3], dim=1)
    score = torch.cat([score_p4, score_p3], dim=1)

    if use_aux and "reg_aux_logits" in out:
        aux_h = out["conf_aux"].shape[-2]
        boxes_aux, score_aux = _decode_one_scale(
            out["reg_aux_logits"], out["conf_aux"], out["ctr_aux"], out["pred_iou_aux"],
            stride=img_size / float(aux_h),
        )
        boxes = torch.cat([boxes, boxes_aux], dim=1)
        score = torch.cat([score, score_aux], dim=1)

    presence = torch.sigmoid(out["presence_logit"]).view(-1, 1)
    score = score * presence

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
        if s_i.numel() > top_k * 4:
            tk = torch.topk(s_i, k=top_k * 4, largest=True)
            b_i = b_i[tk.indices]
            s_i = tk.values
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
    conf_thr: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    boxes_pi, scores_pi = decode_topk(out, img_size=img_size, conf_thr=conf_thr)
    B = len(boxes_pi)
    box_t = out["conf_p4"].new_zeros((B, 4))
    score_t = out["conf_p4"].new_zeros((B,))
    for i, (bxs, scs) in enumerate(zip(boxes_pi, scores_pi)):
        if bxs.numel() > 0:
            box_t[i] = bxs[0]
            score_t[i] = scs[0]
    return box_t, score_t


# ---------------------------------------------------------------------------
# EMA wrapper
# ---------------------------------------------------------------------------


class ModelEMA:
    """Exponential moving average of model parameters.

    Standard +0.5–1.5 mAP from reduced gradient noise. Update at every
    optimiser step; evaluate / save with EMA weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        # Lazy-init device on first update so the EMA copy lands on the
        # same device as the live model.
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        # Warmup: linearly ramp decay so early steps have stronger pull.
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        msd = model.state_dict()
        for k, v in self.shadow.items():
            mv = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(mv.detach(), alpha=1 - d)
            else:
                v.copy_(mv)

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay,
                "num_updates": self.num_updates}

    def load_state_dict(self, sd: dict) -> None:
        self.shadow = sd["shadow"]
        self.decay = sd.get("decay", self.decay)
        self.num_updates = sd.get("num_updates", 0)

    def apply_to(self, model: nn.Module) -> dict:
        """Swap model weights to the EMA shadow. Returns the original
        state_dict so the caller can restore later."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: dict) -> None:
        model.load_state_dict(backup, strict=False)
