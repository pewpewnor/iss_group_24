"""Few-shot localization model — token-bag cross-attention design.

Backbone: MobileNetV3-Large (pretrained ImageNet).

Pipeline (per episode, batch B, K shots):

  support_imgs (B, K, 3, 224, 224)
    -> backbone P5 (B*K, 160, 7, 7)
    -> SupportTokenizer (slot-attention): M=4 region tokens per shot —
       softmax COMPETES across the M slots per spatial pixel, forcing the
       M tokens to attend to disjoint regions of the support
       → (B*K, M, 128) tokens + (B*K, M, 7, 7) attention maps
    -> reshape to (B, K*M, 128)                   [token bag — 20 tokens by default]
    -> SupportFusion: 2-layer transformer encoder over the K*M tokens with
       learned shot + slot positional embeddings, so the K supports can
       aggregate consensus and complementary views BEFORE the query attends.
       → (B, K*M, 128) fused tokens

  query_img (B, 3, 224, 224)
    -> backbone P3 (28x28x40), P4 (14x14x112), P5 (7x7x160)
    -> FPN(P4, P5) -> 14x14x128
    -> + sigmoid(g) * AvgPool(p3_lat(P3))         [learnable per-channel gate
                                                   on the stride-8 residual]
    -> q_feat (B, 128, 14, 14)

  CrossAttentionHead: every query position (196 tokens) attends to the K*M
  fused support tokens via 4-head attention → enriched (B, 128, 14, 14).

  DetectionHead: 3x DWSep -> reg (B,4,14,14), conf (B,1,14,14).
  PresenceHead: AttentionPool(support_tokens) ⊕ GAP(enriched) -> presence_logit.

Outputs for losses:
  prototype           (B, dim)     — attention-pooled summary across the bag
  per_shot_prototype  (B, K, dim)  — attention-pooled summary within each shot
                                      (drives stronger nt_xent/vicreg signal)
  support_attn        (B, K, M, 7, 7) — bbox-supervised auxiliary attention
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.ops import FeaturePyramidNetwork

IMG_SIZE = 224
GRID = 14
STRIDE = IMG_SIZE // GRID  # 16

DIM = 128                  # model embedding dim — bumped from 64
M_TOKENS = 4               # region tokens per support image
N_HEADS = 4
K_SUPPORT_MAX = 8          # support-fusion shot-pos table size (covers K=5 + headroom)
FUSION_LAYERS = 2          # depth of inter-shot transformer

# Kept for backwards-compat references elsewhere in the codebase
PROTO_DIM = DIM
FPN_DIM = DIM

P3_IDX = 6
P3_CHANNELS = 40
P3_STRIDE = 8

P4_IDX = 12
P4_CHANNELS = 112

P5_IDX = 15
P5_CHANNELS = 160
P5_STRIDE = 32


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class MobileNetBackbone(nn.Module):
    """MobileNetV3-Large exposing P3 (28x28x40), P4 (14x14x112), P5 (7x7x160)."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.features = mobilenet_v3_large(weights=weights).features

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
# Support tokenizer: M learned queries cross-attend to P5 → M tokens per shot
# ---------------------------------------------------------------------------


class SupportTokenizer(nn.Module):
    """Slot-attention tokenizer over a support's P5 feature map.

    Why slot attention vs. plain cross-attention:
      Plain cross-attention softmaxes each query over H*W keys independently,
      so all M queries can — and empirically do — collapse onto the same
      salient peak (token redundancy). Slot attention softmaxes over the M
      slots per spatial pixel so the slots COMPETE: each pixel is routed
      to the slot whose key best matches it, forcing the M tokens to cover
      disjoint regions. This addresses the observed nt_xent/vicreg plateau
      where the M*K-token bag was effectively rank-1.

    Output:
      tokens   (N, M, dim)   — value-mixture per slot, normalised by slot mass
      attn_map (N, M, H, W)  — per-slot soft-assignment over the grid (the
                                aggregate is bbox-supervised by attention_bbox_loss)
    """

    def __init__(
        self,
        in_c: int = P5_CHANNELS,
        dim: int = DIM,
        m_tokens: int = M_TOKENS,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.m = m_tokens
        self.scale = dim ** -0.5
        # M learned slot queries — initialised small so slots start near-equal
        self.slots = nn.Parameter(torch.randn(m_tokens, dim) * 0.02)
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

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, _, h, w = feat.shape
        kv_in = self.kv_norm(feat.flatten(2).transpose(1, 2))         # (N, HW, C)
        k = self.k_proj(kv_in)                                        # (N, HW, dim)
        v = self.v_proj(kv_in)                                        # (N, HW, dim)

        slots = self.q_proj(self.q_norm(self.slots))                  # (M, dim)
        slots = slots.unsqueeze(0).expand(n, -1, -1)                  # (N, M, dim)

        # Logits: (N, M, HW). Softmax over slots (dim=1) — slot competition.
        logits = torch.einsum("nmd,nld->nml", slots, k) * self.scale
        attn = logits.softmax(dim=1)                                  # over M

        # Each pixel contributes to slots in proportion to attn; normalise
        # by total mass per slot so values sum, not average, but stay scale-stable.
        attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)     # (N, M, 1)
        weights = attn / attn_sum                                     # (N, M, HW)
        out = torch.einsum("nml,nld->nmd", weights, v)                # (N, M, dim)

        # Residual + FFN — keeps gradient path to slot params alive
        out = self.norm_out(out + slots)
        out = out + self.ffn(out)

        # Per-slot spatial distribution (each slot row sums to 1 over HW),
        # which is the shape `attention_bbox_loss` expects: the cross-shot
        # aggregator there sums over M then renormalises to a prob map.
        attn_map = weights.view(n, self.m, h, w)
        return out, attn_map  # (N, M, dim), (N, M, H, W)


# ---------------------------------------------------------------------------
# Inter-shot fusion: 2-layer transformer encoder over the K*M token bag
# ---------------------------------------------------------------------------


class _TransformerEncoderBlock(nn.Module):
    """Pre-norm self-attention + FFN block."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attended, _ = self.attn(h, h, h, need_weights=False)
        x = x + attended
        x = x + self.ffn(self.norm2(x))
        return x


class SupportFusion(nn.Module):
    """Inter-shot transformer over the K*M support-token bag.

    Without this stage the K shots are independent — the M tokens of shot 1
    never see the M tokens of shot 2, and the cross-attention head has no
    way to extract a consensus signal across the K=5 views. With a learned
    `shot_pe` (which support did this token come from) and `slot_pe` (which
    region within that support), self-attention can vote: tokens that
    appear in many shots reinforce each other, tokens unique to one shot
    are contextualised against the others.

    This is the most direct fix for the observed K=5 plateau where val_iou
    stalled near 0.4 with K=5 supports — the model previously had ~no
    inter-shot interaction.
    """

    def __init__(
        self,
        dim: int = DIM,
        n_heads: int = N_HEADS,
        n_layers: int = FUSION_LAYERS,
        k_max: int = K_SUPPORT_MAX,
        m_tokens: int = M_TOKENS,
    ) -> None:
        super().__init__()
        # Positional embeddings for shot index and slot index. Tokens are
        # always laid out in (k, m) order so we can broadcast.
        self.shot_pe = nn.Parameter(torch.randn(k_max, dim) * 0.02)
        self.slot_pe = nn.Parameter(torch.randn(m_tokens, dim) * 0.02)
        self.k_max = k_max
        self.m = m_tokens
        self.layers = nn.ModuleList(
            [_TransformerEncoderBlock(dim, n_heads) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, k: int) -> torch.Tensor:
        """tokens: (B, K*M, dim) laid out as (shot0_slot0..shot0_slotM-1, shot1_..., ...)."""
        _, n, d = tokens.shape
        m = n // k
        if k > self.k_max:
            raise ValueError(f"k={k} exceeds k_max={self.k_max}")
        if m != self.m:
            raise ValueError(f"slot count {m} != configured M={self.m}")
        shot_pe = self.shot_pe[:k].unsqueeze(1).expand(k, m, d)        # (K, M, dim)
        slot_pe = self.slot_pe.unsqueeze(0).expand(k, m, d)            # (K, M, dim)
        pe = (shot_pe + slot_pe).reshape(1, k * m, d)
        x = tokens + pe
        for layer in self.layers:
            x = layer(x)
        return self.out_norm(x)


# ---------------------------------------------------------------------------
# Attention pooling — replaces mean over the K*M bag for summary vectors
# ---------------------------------------------------------------------------


class AttentionPool(nn.Module):
    """Single-query attention pooling — softer alternative to mean-pool.

    The mean pool used previously gave equal weight to all 20 tokens,
    including redundant or background slots. A learned query pulls the
    informative ones forward.
    """

    def __init__(self, dim: int = DIM, n_heads: int = N_HEADS) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        q = self.query.expand(b, -1, -1)                               # (B, 1, dim)
        kv = self.norm_kv(tokens)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return self.norm_out(out).squeeze(1)                           # (B, dim)


# ---------------------------------------------------------------------------
# Cross-attention head: query positions attend to support tokens
# ---------------------------------------------------------------------------


class CrossAttentionHead(nn.Module):
    """Every spatial position in the query feature map attends to the K*M
    support tokens via multi-head attention.

    Replaces the single-vector dot-product correlation + channel-gating scheme.
    With 20 support tokens (K=5, M=4) the query gets a far richer matching
    signal than against a single 64-d prototype.
    """

    def __init__(self, dim: int = DIM, n_heads: int = N_HEADS) -> None:
        super().__init__()
        self.dim = dim
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(
        self, q_feat: torch.Tensor, support_tokens: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = q_feat.shape
        q = q_feat.flatten(2).transpose(1, 2)               # (B, H*W, dim)
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(support_tokens)
        attended, _ = self.attn(q_n, kv_n, kv_n)
        q = q + attended
        q = q + self.ffn(self.norm_ffn(q))
        return q.transpose(1, 2).view(b, c, h, w)


# ---------------------------------------------------------------------------
# Detection head + presence head
# ---------------------------------------------------------------------------


class _DWSep(nn.Module):
    """Depthwise-separable conv: dw 3x3 -> pw 1x1 -> BN -> ReLU6."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class DetectionHead(nn.Module):
    """3x DWSep -> reg/conf 1x1 convs."""

    def __init__(self, dim: int = DIM) -> None:
        super().__init__()
        self.block1 = _DWSep(dim, dim)
        self.block2 = _DWSep(dim, dim)
        self.block3 = _DWSep(dim, dim)
        self.reg = nn.Conv2d(dim, 4, 1)
        self.conf = nn.Conv2d(dim, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.block3(self.block2(self.block1(x)))
        return self.reg(feat), self.conf(feat)


class PresenceHead(nn.Module):
    """Global binary classifier: mean(support_tokens) ⊕ GAP(enriched_q) -> presence."""

    def __init__(self, dim: int = DIM) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(
        self, support_summary: torch.Tensor, q_gap: torch.Tensor
    ) -> torch.Tensor:
        return self.fc(torch.cat([support_summary, q_gap], dim=1)).squeeze(1)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class FewShotLocalizer(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = MobileNetBackbone(pretrained=pretrained)
        self.support_tokenizer = SupportTokenizer()
        self.support_fusion = SupportFusion()
        self.support_pool = AttentionPool()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[P4_CHANNELS, P5_CHANNELS],
            out_channels=DIM,
        )
        # Stride-8 (28x28) lateral projection added as residual to P4 output.
        # The contribution is gated per-channel — start at sigmoid(0)=0.5 to
        # preserve the previous fixed weight, but the model can dial it up
        # for small-object instances or down where P3 hurts.
        self.p3_lat = nn.Conv2d(P3_CHANNELS, DIM, kernel_size=1, bias=False)
        self.p3_gate = nn.Parameter(torch.zeros(1, DIM, 1, 1))
        self.cross_attn = CrossAttentionHead()
        self.det_head = DetectionHead()
        self.presence_head = PresenceHead()

    # ---- support branch ----------------------------------------------------

    def encode_support(
        self,
        support_imgs: torch.Tensor,
        support_bboxes: torch.Tensor | None = None,  # kept for API compat — unused
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """support_imgs: (B, K, 3, 224, 224).

        Returns:
            tokens:             (B, K*M, DIM)     — fused token bag (post SupportFusion)
            per_shot_prototype: (B, K, DIM)       — attention-pooled summary per shot
            attn:               (B, K, M, 7, 7)   — slot attention maps over P5
        """
        b, k = support_imgs.shape[:2]
        flat = support_imgs.reshape(b * k, 3, IMG_SIZE, IMG_SIZE)
        feat = self.backbone.forward_p5_only(flat)              # (B*K, 160, 7, 7)
        tokens, attn = self.support_tokenizer(feat)             # (B*K, M, dim), (B*K, M, 7, 7)
        m, dim = tokens.shape[1], tokens.shape[2]

        # Per-shot attention-pool BEFORE fusion — preserves "what each shot
        # individually thinks it saw", which is the right signal for the
        # per-shot contrastive loss (within-episode positives, across-episode
        # negatives). Fusion would bleed information across shots and weaken
        # the discriminative pressure.
        per_shot = self.support_pool(tokens)                    # (B*K, dim)
        per_shot = per_shot.view(b, k, dim)

        # Fuse the bag so the K shots can vote / share before the query attends.
        tokens = tokens.view(b, k * m, dim)                     # (B, K*M, dim)
        tokens = self.support_fusion(tokens, k=k)               # (B, K*M, dim) fused

        attn = attn.view(b, k, m, attn.shape[-2], attn.shape[-1])
        return tokens, per_shot, attn

    @torch.no_grad()
    def compute_prototype(self, support_imgs: torch.Tensor) -> torch.Tensor:
        """Inference helper: encode K supports into a fused token bag for caching.

        The on-device pipeline calls this once per session and caches the
        returned tensor (shape (1, K*M, DIM)); subsequent ``detect`` calls
        skip the entire support branch (backbone + tokenizer + fusion).
        """
        tokens, _, _ = self.encode_support(support_imgs)
        return tokens

    # ---- query branch ------------------------------------------------------

    def encode_query(self, query_img: torch.Tensor) -> torch.Tensor:
        p3, p4, p5 = self.backbone(query_img)
        fpn_out = self.fpn(OrderedDict([("p4", p4), ("p5", p5)]))
        feat = fpn_out["p4"]                                    # (B, dim, 14, 14)
        p3_down = F.avg_pool2d(self.p3_lat(p3), kernel_size=2)  # 28x28 -> 14x14
        gate = torch.sigmoid(self.p3_gate)                       # (1, dim, 1, 1)
        return feat + gate * p3_down

    # ---- shared head -------------------------------------------------------

    def _head(
        self, support_tokens: torch.Tensor, q_feat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        b = q_feat.shape[0]
        if support_tokens.shape[0] != b:
            support_tokens = support_tokens.expand(b, -1, -1)
        enriched = self.cross_attn(q_feat, support_tokens)      # (B, dim, 14, 14)
        reg, conf = self.det_head(enriched)
        q_gap = enriched.mean(dim=(2, 3))                       # (B, dim)
        support_summary = self.support_pool(support_tokens)     # (B, dim)
        presence_logit = self.presence_head(support_summary, q_gap)
        return {
            "reg": reg,
            "conf": conf,
            "presence_logit": presence_logit,
            "support_tokens": support_tokens,
            # `prototype` is the bag-level attention-pooled summary —
            # used by triplet loss and as a (B, dim) fallback for nt_xent.
            "prototype": support_summary,
        }

    # ---- inference path: precomputed token bag ----------------------------

    def detect(
        self, prototype: torch.Tensor, query_img: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Real-time inference: token bag precomputed, only query branch runs.

        ``prototype`` here is the (B, K*M, DIM) FUSED tensor returned by
        ``compute_prototype``.
        """
        return self._head(prototype, self.encode_query(query_img))

    # ---- training path: full forward --------------------------------------

    def forward(
        self,
        support_imgs: torch.Tensor,
        query_img: torch.Tensor,
        support_bboxes: torch.Tensor | None = None,
        prototype: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        per_shot_prototype: torch.Tensor | None
        support_attn: torch.Tensor | None
        if prototype is None:
            tokens, per_shot_prototype, support_attn = self.encode_support(support_imgs)
        else:
            tokens = prototype
            per_shot_prototype = None
            support_attn = None
        out = self._head(tokens, self.encode_query(query_img))
        out["support_attn"] = support_attn                  # type: ignore[assignment]
        out["per_shot_prototype"] = per_shot_prototype      # type: ignore[assignment]
        return out


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def decode(
    reg: torch.Tensor,
    conf: torch.Tensor,
    stride: int = STRIDE,
    presence_logit: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (bbox xyxy in 224 coords (B,4), presence score (B,)).

    conf map -> spatial argmax -> box decode (localization).
    presence_logit sigmoid -> presence score (classification).
    """
    b, _, _, w = reg.shape
    flat_conf = conf.view(b, -1)
    best = flat_conf.argmax(dim=1)
    cy = (best // w).long()
    cx = (best % w).long()
    idx = torch.arange(b, device=reg.device)

    cx_abs = (cx.float() + reg[idx, 0, cy, cx]) * stride
    cy_abs = (cy.float() + reg[idx, 1, cy, cx]) * stride
    w_abs = torch.exp(reg[idx, 2, cy, cx]) * stride
    h_abs = torch.exp(reg[idx, 3, cy, cx]) * stride

    bbox = torch.stack(
        [cx_abs - w_abs / 2, cy_abs - h_abs / 2, cx_abs + w_abs / 2, cy_abs + h_abs / 2],
        dim=1,
    )
    score = (
        torch.sigmoid(presence_logit)
        if presence_logit is not None
        else torch.sigmoid(flat_conf).max(dim=1).values
    )
    return bbox, score
