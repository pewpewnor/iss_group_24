"""OWLv2-based few-shot single-object localizer.

Architecture (PLAN.md §4):

    backbone (frozen): google/owlv2-base-patch16-ensemble
        ├─ vision_model     : ViT-B/16, hidden=768, P = (img/16)^2 patches
        ├─ class_head       : 768→512 + cosine sim with query embeds
        └─ box_head         : 768→4  (cx, cy, w, h) normalized

    aggregator (trainable, ~3M params):
        4 support images → vision_model patches (B, 4, P, 768)
        → per-view foreground top-K=128 attention pool   (B, 4, K, 768)
        → +view position embedding                       (B, 4, K, 768)
        → flatten                                        (B, 4*K, 768)
        → 2× pre-LN inter-view self-attention (8 heads)  (B, 4*K, 768)
        → ISAB (M=4)                                     (B, 4, 768)
        → PMA (single seed)                              (B, 1, 768)
        → Linear(768, 512) + LN                          (B, 1, 512)
                                              ── matches class_head query dim

    existence_head (trainable, ~10K params):
        8 hand-crafted scalar features (max patch logit, top-5 stats,
        cosine(prototype, GAP(query)), prototype norm, query GAP norm,
        patch entropy) → 64 → 64 → 1 → sigmoid

    LoRA (Stage 2.3 only): r=8, q_proj/v_proj on last 4 vision blocks.

Forward pass returns:
    {
      "existence_prob": (B,)
      "existence_logit": (B,)               (pre-sigmoid)
      "pred_logits":    (B, P)              raw class-head similarity
      "pred_boxes":     (B, P, 4)           cx,cy,w,h in [0,1]
      "best_box":       (B, 4)              top-1 by pred_logits
      "best_score":     (B,)                top-1 logit value
      "prototype":      (B, 512)            class-head query embedding
      "image_feats":    (B, P, 768)         post-class-head image features
    }
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Owlv2ForObjectDetection


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------


class _PreLNTransformerBlock(nn.Module):
    """Pre-LN multi-head self-attention + FFN."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class _CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention from `q` to `kv`."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        a, _ = self.attn(q_n, kv_n, kv_n, need_weights=False)
        q = q + a
        q = q + self.ffn(self.norm_ffn(q))
        return q


class _ISAB(nn.Module):
    """Induced Set Attention Block (Lee et al., Set Transformer 2019)."""

    def __init__(self, dim: int, n_heads: int, m_inducing: int = 4):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(m_inducing, dim) * 0.02)
        self.cross1 = _CrossAttentionBlock(dim, n_heads)
        self.cross2 = _CrossAttentionBlock(dim, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        i = self.inducing.unsqueeze(0).expand(b, -1, -1)            # (B, M, D)
        h = self.cross1(i, x)                                       # (B, M, D)
        out = self.cross2(x, h)                                     # (B, N, D)
        return out


class _PMA(nn.Module):
    """Pooling by Multihead Attention with a single learnable seed."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.cross = _CrossAttentionBlock(dim, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        s = self.seed.unsqueeze(0).expand(b, -1, -1)                # (B, 1, D)
        return self.cross(s, x)                                     # (B, 1, D)


# ---------------------------------------------------------------------------
# Multi-view aggregator
# ---------------------------------------------------------------------------


class MultiViewAggregator(nn.Module):
    """4 support patch grids → 1 prototype embedding (in class-head query space).

    Inputs shape: ``(B, 4, P, vision_dim)`` with vision_dim=768 for OWLv2-base.
    Output shape: ``(B, query_dim)`` with query_dim=512 (text hidden size).

    Internally projects from ``vision_dim`` (768) to a smaller ``work_dim``
    (default 256) before the attention stack, keeping the trainable param
    count under ~3M for the small dataset we have.
    """

    def __init__(
        self,
        vision_dim: int = 768,
        query_dim: int = 512,
        topk: int = 128,
        n_views: int = 4,
        n_heads: int = 8,
        inter_view_layers: int = 2,
        m_inducing: int = 4,
        work_dim: int = 256,
        mlp_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.query_dim = query_dim
        self.topk = topk
        self.n_views = n_views
        self.work_dim = work_dim

        # Foreground gate per patch (small Linear scoring patch tokens).
        self.fg_gate = nn.Linear(vision_dim, 1)
        # Init with small positive bias so all patches have non-zero gate at start.
        nn.init.zeros_(self.fg_gate.weight)
        nn.init.constant_(self.fg_gate.bias, 0.0)

        # Project vision_dim → work_dim before attention stack.
        self.in_proj = nn.Linear(vision_dim, work_dim)
        self.in_norm = nn.LayerNorm(work_dim)

        # View positional embedding (in work_dim).
        self.view_embed = nn.Embedding(n_views, work_dim)
        nn.init.trunc_normal_(self.view_embed.weight, std=0.02)

        # Inter-view self-attention (in work_dim).
        self.inter_view = nn.ModuleList(
            [_PreLNTransformerBlock(work_dim, n_heads, mlp_ratio=mlp_ratio)
             for _ in range(inter_view_layers)]
        )
        self.inter_view_norm = nn.LayerNorm(work_dim)

        # ISAB → PMA pooling (in work_dim).
        self.isab = _ISAB(work_dim, n_heads, m_inducing=m_inducing)
        self.pma = _PMA(work_dim, n_heads)

        # Project to class-head query dim.
        self.proj = nn.Linear(work_dim, query_dim)
        self.proj_norm = nn.LayerNorm(query_dim)

    def _topk_patches(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (N, P, D) → (N, K, D) where K = self.topk.

        Selects the top-K patches per item by foreground gate score.
        When P <= K, returns tokens unchanged (padded by repeating the
        last token if necessary so the output dim is always K).
        """
        n, p, d = tokens.shape
        scores = self.fg_gate(tokens).squeeze(-1)                   # (N, P)
        if p <= self.topk:
            # Pad: repeat the highest-scoring patch.
            if p < self.topk:
                pad_idx = scores.argmax(dim=-1, keepdim=True).expand(n, self.topk - p)
                idx = torch.cat([torch.arange(p, device=tokens.device).unsqueeze(0).expand(n, -1), pad_idx], dim=-1)
            else:
                idx = torch.arange(p, device=tokens.device).unsqueeze(0).expand(n, -1)
        else:
            idx = scores.topk(self.topk, dim=-1).indices            # (N, K)
        gather = idx.unsqueeze(-1).expand(-1, -1, d)
        return torch.gather(tokens, dim=1, index=gather)

    def forward(self, support_tokens: torch.Tensor) -> torch.Tensor:
        """support_tokens: (B, V, P, D_v) → (B, query_dim)."""
        b, v, p, d = support_tokens.shape
        if v != self.n_views:
            raise ValueError(f"expected {self.n_views} views, got {v}")
        flat = support_tokens.reshape(b * v, p, d)
        gated = self._topk_patches(flat)                             # (B*V, K, D_v)
        # Project to work_dim.
        gated = self.in_norm(self.in_proj(gated))                    # (B*V, K, D_w)
        wd = gated.size(-1)
        gated = gated.view(b, v, self.topk, wd)
        # Add view embedding.
        view_idx = torch.arange(v, device=support_tokens.device)
        ve = self.view_embed(view_idx).view(1, v, 1, wd)             # (1, V, 1, D_w)
        gated = gated + ve
        # Flatten to (B, V*K, D_w).
        seq = gated.view(b, v * self.topk, wd)
        for blk in self.inter_view:
            seq = blk(seq)
        seq = self.inter_view_norm(seq)
        # ISAB returns (B, V*K, D_w); PMA pools to (B, 1, D_w).
        seq = self.isab(seq)
        pooled = self.pma(seq).squeeze(1)                            # (B, D_w)
        return self.proj_norm(self.proj(pooled))                     # (B, query_dim)


# ---------------------------------------------------------------------------
# Existence head
# ---------------------------------------------------------------------------


class ExistenceHead(nn.Module):
    """8 hand-crafted scalars → 64 → 64 → 1 → sigmoid.

    Features (computed in :meth:`compute_features`):

      0. softmax-temperature differentiable max patch logit
      1. top-5 mean
      2. top-5 std
      3. mean over all patches
      4. cosine(prototype, GAP(query))
      5. ||prototype||
      6. ||GAP(query)||
      7. entropy of softmax over patch logits
    """

    N_FEAT = 8

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(self.N_FEAT)
        self.fc = nn.Sequential(
            nn.Linear(self.N_FEAT, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    @staticmethod
    def compute_features(
        pred_logits: torch.Tensor,
        prototype: torch.Tensor,
        image_feats: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """pred_logits: (B, P), prototype: (B, D_q), image_feats: (B, P, D_v).

        Returns ``(B, N_FEAT)`` feature tensor. Pure tensor ops — fully
        differentiable through the existence head.
        """
        b, p = pred_logits.shape
        # 0. soft-max over patches
        soft_w = (pred_logits / temperature).softmax(dim=-1)
        f0 = (soft_w * pred_logits).sum(dim=-1)                      # (B,)
        # 1, 2. top-5 mean / std
        top5, _ = pred_logits.topk(min(5, p), dim=-1)                # (B, 5)
        f1 = top5.mean(dim=-1)
        f2 = top5.std(dim=-1, unbiased=False)
        # 3. mean over all patches
        f3 = pred_logits.mean(dim=-1)
        # 4. cosine(prototype, GAP(query in vision space))
        gap = image_feats.mean(dim=1)                                # (B, D_v)
        gap_norm = F.normalize(gap, dim=-1)
        proto_norm = F.normalize(prototype, dim=-1)
        # When dims differ (vision 768 vs query 512), project gap onto prototype dim
        # by truncation/zero-padding.  We keep cosine on equal dim:
        d_q = prototype.size(-1)
        d_v = gap.size(-1)
        if d_q == d_v:
            f4 = (gap_norm * proto_norm).sum(dim=-1)
        else:
            # Use the L2-normalised gap and prototype after PCA-equivalent
            # mean-reduction along the larger dim.  Practical proxy: compare
            # magnitudes via dot product of the first min(d_q, d_v) dims.
            d = min(d_q, d_v)
            f4 = (gap_norm[:, :d] * proto_norm[:, :d]).sum(dim=-1)
        # 5, 6. norms
        f5 = prototype.norm(dim=-1)
        f6 = gap.norm(dim=-1)
        # 7. entropy of softmax over patches
        entropy = -(soft_w * (soft_w + 1e-12).log()).sum(dim=-1)     # (B,)
        f7 = entropy
        return torch.stack([f0, f1, f2, f3, f4, f5, f6, f7], dim=-1)  # (B, 8)

    def forward(
        self,
        pred_logits: torch.Tensor,
        prototype: torch.Tensor,
        image_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.compute_features(pred_logits, prototype, image_feats)
        feats = self.norm(feats)
        logit = self.fc(feats).squeeze(-1)                            # (B,)
        prob = torch.sigmoid(logit)
        return prob, logit


# ---------------------------------------------------------------------------
# Full localizer
# ---------------------------------------------------------------------------


OWLV2_MODEL_NAME = "google/owlv2-base-patch16-ensemble"


class OWLv2FewShotLocalizer(nn.Module):
    """OWLv2 + multi-view aggregator + existence head."""

    def __init__(
        self,
        model_name: str = OWLV2_MODEL_NAME,
        n_views: int = 4,
        topk: int = 128,
        inter_view_layers: int = 2,
        attn_heads: int = 8,
        m_inducing: int = 4,
        existence_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.owlv2 = Owlv2ForObjectDetection.from_pretrained(model_name)
        vision_dim = self.owlv2.config.vision_config.hidden_size
        query_dim = self.owlv2.config.text_config.hidden_size
        self.vision_dim = vision_dim
        self.query_dim = query_dim

        self.aggregator = MultiViewAggregator(
            vision_dim=vision_dim,
            query_dim=query_dim,
            topk=topk,
            n_views=n_views,
            n_heads=attn_heads,
            inter_view_layers=inter_view_layers,
            m_inducing=m_inducing,
        )
        self.existence_head = ExistenceHead(dropout=existence_dropout)

        # Default: backbone frozen.
        self.freeze_owlv2_all()

    # ------------------------------------------------------------------
    # Freezing helpers — call these from the optimiser factory
    # ------------------------------------------------------------------

    def freeze_owlv2_all(self) -> None:
        for p in self.owlv2.parameters():
            p.requires_grad = False

    def unfreeze_owlv2_heads(self) -> None:
        """Unfreeze class_head and box_head only — used in Stage 1.2."""
        for p in self.owlv2.class_head.parameters():
            p.requires_grad = True
        for p in self.owlv2.box_head.parameters():
            p.requires_grad = True
        for p in self.owlv2.layer_norm.parameters():
            p.requires_grad = True

    def class_head_params(self) -> list[nn.Parameter]:
        return list(self.owlv2.class_head.parameters()) + list(self.owlv2.layer_norm.parameters())

    def box_head_params(self) -> list[nn.Parameter]:
        return list(self.owlv2.box_head.parameters())

    def vision_model_params(self) -> list[nn.Parameter]:
        return list(self.owlv2.owlv2.vision_model.parameters())

    # ------------------------------------------------------------------
    # LoRA (Stage 2.3)
    # ------------------------------------------------------------------

    def attach_lora(
        self, r: int = 8, alpha: int = 16, dropout: float = 0.1, last_n_layers: int = 4
    ) -> list[nn.Parameter]:
        """Inject LoRA adapters on q_proj/v_proj of the last ``last_n_layers``
        vision transformer blocks.  Returns the list of LoRA parameter tensors
        (everything else stays frozen).
        """
        from peft import LoraConfig, get_peft_model

        # Resolve the names of layers we want to target.
        encoder_layers = self.owlv2.owlv2.vision_model.encoder.layers
        n_layers = len(encoder_layers)
        target_layer_ids = list(range(n_layers - last_n_layers, n_layers))
        target_modules = [
            f"owlv2.vision_model.encoder.layers.{i}.self_attn.{proj}"
            for i in target_layer_ids
            for proj in ("q_proj", "v_proj")
        ]
        cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        # peft wraps the *whole* model when applied to a torch.nn.Module — but
        # we want to wrap only owlv2.  Apply via the lower-level API:
        self.owlv2 = get_peft_model(self.owlv2, cfg)
        # Collect trainable LoRA params:
        lora_params = [p for n, p in self.owlv2.named_parameters() if "lora_" in n and p.requires_grad]
        return lora_params

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode_support(self, support_imgs: torch.Tensor) -> torch.Tensor:
        """support_imgs: (B, V, 3, H, W) → patch tokens (B, V, P, vision_dim).

        Uses ``image_embedder`` so the support tokens go through the same
        post_layernorm + class-token merge as the query path.  This is
        important — without it, the aggregator trains on a feature space
        that doesn't match what class_head sees.
        """
        b, v, c, h, w = support_imgs.shape
        flat = support_imgs.reshape(b * v, c, h, w)
        feature_map, _ = self.owlv2.image_embedder(
            pixel_values=flat, interpolate_pos_encoding=True
        )                                                            # (B*V, gh, gw, D)
        d = feature_map.size(-1)
        tokens = feature_map.reshape(b * v, -1, d)
        p = tokens.size(1)
        return tokens.view(b, v, p, d)

    def encode_query(
        self, query_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """query_img: (B, 3, H, W) → (image_feats (B, P, D), feature_map (B, gh, gw, D)).
        """
        feature_map, _ = self.owlv2.image_embedder(
            pixel_values=query_img, interpolate_pos_encoding=True
        )                                                            # (B, gh, gw, D)
        b, gh, gw, d = feature_map.shape
        image_feats = feature_map.reshape(b, gh * gw, d)
        return image_feats, feature_map

    def forward(
        self,
        support_imgs: torch.Tensor,
        query_img: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Support → prototype.
        support_tokens = self.encode_support(support_imgs)           # (B, V, P, D_v)
        prototype = self.aggregator(support_tokens)                  # (B, D_q)
        query_embeds = prototype.unsqueeze(1)                        # (B, 1, D_q)

        # Query → patch features.
        image_feats, feature_map = self.encode_query(query_img)      # (B, P, D_v), (B, gh, gw, D_v)

        # Class predictor (cosine similarity to prototype).
        pred_logits, _ = self.owlv2.class_predictor(image_feats, query_embeds)  # (B, P, 1)
        pred_logits = pred_logits.squeeze(-1)                        # (B, P)

        # Box predictor.
        pred_boxes = self.owlv2.box_predictor(
            image_feats, feature_map, interpolate_pos_encoding=True
        )                                                            # (B, P, 4) cx,cy,w,h

        # Existence head.
        existence_prob, existence_logit = self.existence_head(
            pred_logits, prototype, image_feats
        )

        # Top-1 box selection.
        best_idx = pred_logits.argmax(dim=-1)                        # (B,)
        ar = torch.arange(pred_logits.size(0), device=pred_logits.device)
        best_box = pred_boxes[ar, best_idx]                          # (B, 4)
        best_score = pred_logits[ar, best_idx]                       # (B,)

        return {
            "existence_prob": existence_prob,
            "existence_logit": existence_logit,
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "best_box": best_box,
            "best_score": best_score,
            "prototype": prototype,
            "image_feats": image_feats,
        }


# ---------------------------------------------------------------------------
# Box format conversion helpers
# ---------------------------------------------------------------------------


def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = box.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)
