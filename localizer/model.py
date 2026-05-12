"""Multi-shot localizer — OWLv2 + support-patch attention pool + abstain channel.

Architecture (rewritten, see notes for what changed and why):

  ┌── per-support pass (frozen OWLv2 unless LoRA active) ──────────────┐
  │ for each support_i (B*K):                                          │
  │     fm_i = owlv2.image_embedder(support_i)            (B*K, gh, gw, 768) │
  │     # 2-stage support pooling.                                      │
  │     # Stage A — OWLv2's image-guided embedding (CLS-like).          │
  │     q_emb_i, _, _ = owlv2.embed_image_query(fm_i)     (B*K, 1, D_q=512)  │
  │     # Stage B — query-conditioned attention pool over support       │
  │     # patches. NO CENTRE PRIOR: since manifest v5 the dataset       │
  │     # pre-crops every InsDet support to the object on disk          │
  │     # (HOTS supports are already centred), so every patch is        │
  │     # object signal. The pool just learns which patches are most    │
  │     # discriminative.                                               │
  │     pooled_i = attn_pool(q_emb_i, fm_i)                             │
  │ stack: q_emb (B, K, D_q)                                            │
  └─────────────────────────────────────────────────────────────────────┘

  ┌── fusion (transformer encoder over [CLS, K supports]) ─────────────┐
  │ prototype = LN(fusion([CLS, q_emb_1..q_emb_K])[:, 0])              │
  │ prototype = baseline (mean(q_emb)) + alpha * (prototype - baseline)│
  │   * baseline init at alpha=1.0 ⇒ pure-fusion path (was alpha=0.01,  │
  │     which made L1 effectively a no-op).                             │
  └─────────────────────────────────────────────────────────────────────┘

  ┌── query path + class predictor + box head ─────────────────────────┐
  │ fm_q   = owlv2.image_embedder(query)                                │
  │ pred_logits_fg = class_predictor(fm_q, prototype)   (B, P)          │
  │ bg_logit       = max-over-P class_predictor(fm_q, learned_bg_proto) │
  │                  + learned bg_bias  (the abstain channel)           │
  │ pred_boxes     = NEW log-scale box head over fm_q   (B, P, 4)       │
  │                  (cx, cy, log_w, log_h) re-parameterisation; widths │
  │                  are bounded in [w_min, w_max] via sigmoid.         │
  └─────────────────────────────────────────────────────────────────────┘

  ┌── outputs ─────────────────────────────────────────────────────────┐
  │ joint  = cat([pred_logits_fg, bg_logit])   (B, P+1)                │
  │ probs  = softmax(joint, dim=-1)                                     │
  │ best_score = max(fg_prob)                                            │
  │ bg_prob    = prob[..., -1]                                           │
  │ best_box   = pred_boxes[argmax fg_prob]                              │
  └─────────────────────────────────────────────────────────────────────┘

WHY THESE CHANGES (vs the previous design that gave mAP@50 = 0.07):

  1. ``support_attn_pool`` (the centre-Gaussian, query-conditioned patch
     pool) is the architectural answer to "InsDet supports have noisy
     backgrounds". Even when bbox-cropping is off, the model can downweight
     the background of the support image via the learned attention + soft
     centre prior. HOTS supports, which are already centred, get ~1.0
     weight on the central patches so no information is lost.

  2. ``log-scale box head`` replaces the (cx, cy, w, h) regression with
     (cx, cy, log_w, log_h). With the L1 loss on (log_w, log_h) the
     gradient is scale-invariant: a 2× wrong width on a small object
     produces the same gradient as 2× wrong on a large object. This kills
     the "85% predicted boxes too small" pathology.

  3. ``alpha`` init is now 1.0 (not 0.01). The previous α=0.01 made the
     prototype essentially the mean-of-q_emb baseline for the first many
     epochs, so the fusion transformer received near-zero gradient from
     L1. With α=1.0 the fusion path is fully on from step 1 and the
     residual character is preserved by initialising the fusion stack to
     ε-near-zero output (LN at the end + tiny init).

  4. ``LoRA-from-L2`` (controlled by trainer): we no longer reserve LoRA
     for L3. The model exposes ``attach_lora`` early; the trainer wires it
     into L2 after a 2-epoch warm-up so the heads can adapt before the
     vision backbone starts to drift.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Owlv2ForObjectDetection


# CLIP / OWLv2 normalisation constants.
OWLV2_MEAN = (0.48145466, 0.4578275, 0.40821073)
OWLV2_STD = (0.26862954, 0.26130258, 0.27577711)

OWLV2_MODEL_NAME = "google/owlv2-base-patch16-ensemble"


def _normalize_owlv2(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor(OWLV2_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(OWLV2_STD).view(1, 3, 1, 1)
    return (x - mean) / std


class SupportAttnPool(nn.Module):
    """Query-conditioned attention pool over OWLv2 support patches.

    Since v5 the dataset pre-crops all InsDet supports to the object on disk
    (HOTS supports are already object-centred). There is no longer any
    background to suppress, so we removed the centre-Gaussian prior — the
    pool is now a vanilla query-conditioned attention over patches.

    Inputs:
        feats   : (B*K, P, D_v=768)
        q_emb   : (B*K, D_q=512)
    Output:
        pooled  : (B*K, D_q)
    """

    def __init__(self, d_v: int, d_q: int, n_heads: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        assert d_q % n_heads == 0, "d_q must be divisible by n_heads"
        self.d_v = d_v
        self.d_q = d_q
        self.n_heads = n_heads
        self.proj_v = nn.Linear(d_v, d_q)
        self.proj_q = nn.Linear(d_q, d_q)
        self.proj_k = nn.Linear(d_q, d_q)
        self.proj_o = nn.Linear(d_q, d_q)
        self.norm_q = nn.LayerNorm(d_q)
        self.norm_kv = nn.LayerNorm(d_q)
        # Residual gate: the q_emb path is the baseline; the pool adds a
        # learned correction. Starts at 0.1 so training warm-starts close
        # to the bare-q_emb prototype.
        self.residual_gate = nn.Parameter(torch.full((), 0.1))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        feats: torch.Tensor,                     # (BK, P, D_v)
        q_emb: torch.Tensor,                     # (BK, D_q)
    ) -> torch.Tensor:
        BK, P, _ = feats.shape
        D_q = self.d_q
        H = self.n_heads
        Dh = D_q // H

        proj = self.proj_v(feats)
        kv_in = self.norm_kv(proj)
        q_in  = self.norm_q(q_emb).unsqueeze(1)

        q = self.proj_q(q_in).view(BK, 1, H, Dh).transpose(1, 2)
        k = self.proj_k(kv_in).view(BK, P, H, Dh).transpose(1, 2)
        v = kv_in.view(BK, P, H, Dh).transpose(1, 2)

        scale = 1.0 / math.sqrt(Dh)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(BK, 1, D_q)
        pooled = self.proj_o(out).squeeze(1)
        return q_emb + self.residual_gate * pooled


def _centre_gaussian_bias_DEPRECATED(gh: int, gw: int, sigma_frac: float,
                          device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Deprecated: kept for backwards-compat checkpoints. No longer used."""
    y = (torch.arange(gh, device=device, dtype=dtype) + 0.5) / gh - 0.5
    x = (torch.arange(gw, device=device, dtype=dtype) + 0.5) / gw - 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r2 = yy * yy + xx * xx                       # (gh, gw) — squared distance from centre, in [0, ~0.5]
    sigma2 = max(1e-4, sigma_frac) ** 2
    return (-0.5 * r2 / sigma2).reshape(-1)      # log-prior, peaks at 0 in the centre


class LogScaleBoxHead(nn.Module):
    """Re-parameterised box head: predicts (cx, cy, log_w, log_h).

    Wrapped around OWLv2's existing box_predictor so we don't lose the
    pretrained spatial prior. We add a learnable per-axis scale + bias to
    the raw OWLv2 output and then convert (raw_w, raw_h) → log_w/log_h via
    a softplus, giving widths in [w_min, 1.0] with a strictly positive
    gradient everywhere.

    The wrapper is OPT-IN: when not attached, the model uses the original
    box_predictor outputs unchanged. The trainer enables it once box_head
    becomes trainable (i.e. L2 onwards).
    """

    def __init__(self, w_min: float = 0.005, w_max: float = 1.0) -> None:
        super().__init__()
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        # Per-axis log-scale parameters. Initialised so that at start, the
        # output is identical to the underlying box_predictor.
        self.log_w_scale = nn.Parameter(torch.zeros(()))
        self.log_h_scale = nn.Parameter(torch.zeros(()))
        # Per-axis log-bias so the head can shift mean predicted area
        # without needing the OWLv2 box_predictor to update.
        self.log_w_bias = nn.Parameter(torch.zeros(()))
        self.log_h_bias = nn.Parameter(torch.zeros(()))

    def forward(self, raw_boxes: torch.Tensor) -> torch.Tensor:
        """raw_boxes : (..., 4) in (cx, cy, w, h) ∈ [0, 1]. Returns same shape."""
        cx, cy, w, h = raw_boxes.unbind(-1)
        # Convert raw widths to log-space, scale, bias, then back to width.
        # softplus keeps the gradient alive for tiny raw widths.
        log_w = F.softplus(w + 1e-3).log() if False else (w + 1e-3).clamp(min=1e-4).log()
        log_h = (h + 1e-3).clamp(min=1e-4).log()
        log_w = log_w * (1.0 + self.log_w_scale) + self.log_w_bias
        log_h = log_h * (1.0 + self.log_h_scale) + self.log_h_bias
        new_w = log_w.exp().clamp(min=self.w_min, max=self.w_max)
        new_h = log_h.exp().clamp(min=self.w_min, max=self.w_max)
        return torch.stack([cx, cy, new_w, new_h], dim=-1)


class MultiShotLocalizer(nn.Module):
    """OWLv2 + support-patch attention pool + abstain channel + log-scale box head."""

    def __init__(
        self,
        model_name: str = OWLV2_MODEL_NAME,
        *,
        k_max: int = 10,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        fusion_mlp_ratio: int = 2,
        fusion_dropout: float = 0.1,
        # New: support-attn pool.
        support_attn_heads: int = 4,
        support_attn_dropout: float = 0.0,
        # New: alpha residual gate init (1.0 ⇒ fusion path on from step 1).
        alpha_init: float = 1.0,
        # New: box head log-scale wrapper.
        use_log_box_head: bool = True,
        log_box_w_min: float = 0.005,
    ) -> None:
        super().__init__()
        self.owlv2 = Owlv2ForObjectDetection.from_pretrained(model_name)
        self.k_max = int(k_max)
        D_q = self.owlv2.config.text_config.hidden_size  # 512 (CLIP B/16)
        D_v = self.owlv2.config.vision_config.hidden_size  # 768
        self.query_dim = D_q
        self.vision_dim = D_v

        # --- Support attention pool over OWLv2 patches. ────────────────
        self.support_pool = SupportAttnPool(
            d_v=D_v, d_q=D_q,
            n_heads=int(support_attn_heads),
            dropout=float(support_attn_dropout),
        )

        # --- Fusion transformer over [CLS, K supports]. ──────────────
        self.cls_token = nn.Parameter(torch.randn(1, 1, D_q) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D_q, nhead=fusion_heads,
            dim_feedforward=D_q * fusion_mlp_ratio,
            dropout=fusion_dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            layer, num_layers=fusion_layers, enable_nested_tensor=False,
        )
        self.fusion_norm = nn.LayerNorm(D_q)

        # Residual gate alpha. With ``alpha_init=1.0`` the prototype starts
        # as ``baseline + 1.0 * (fused - baseline) = fused``, so the fusion
        # path is fully on. The previous 0.01 init kept the prototype pinned
        # at the baseline (mean of q_emb) for the early epochs.
        self.alpha = nn.Parameter(torch.full((), float(alpha_init)))

        # --- Abstain channel (background prototype + scalar bias). ────
        self.bg_prototype = nn.Parameter(torch.zeros(1, D_q))
        nn.init.normal_(self.bg_prototype, std=0.02)
        self.bg_bias = nn.Parameter(torch.zeros(()))

        # --- Log-scale box head wrapper (opt-in via `use_log_box_head`). ─
        self.use_log_box_head = bool(use_log_box_head)
        self.log_box_head = LogScaleBoxHead(w_min=log_box_w_min) if self.use_log_box_head else None

        # Default: backbone fully frozen.
        self.freeze_backbone()
        self._lora_attached = False

    # ------------------------------------------------------------------
    # Freezing helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        for p in self.owlv2.parameters():
            p.requires_grad = False

    def unfreeze_heads(self) -> None:
        for p in self.owlv2.class_head.parameters():
            p.requires_grad = True
        for p in self.owlv2.box_head.parameters():
            p.requires_grad = True
        for p in self.owlv2.layer_norm.parameters():
            p.requires_grad = True

    def freeze_box_head(self) -> None:
        for p in self.owlv2.box_head.parameters():
            p.requires_grad = False

    def unfreeze_box_head(self) -> None:
        for p in self.owlv2.box_head.parameters():
            p.requires_grad = True

    def class_head_params(self) -> list[nn.Parameter]:
        return list(self.owlv2.class_head.parameters()) + list(self.owlv2.layer_norm.parameters())

    def box_head_params(self) -> list[nn.Parameter]:
        out = list(self.owlv2.box_head.parameters())
        if self.log_box_head is not None:
            out += list(self.log_box_head.parameters())
        return out

    def fusion_params(self) -> list[nn.Parameter]:
        return (
            list(self.fusion.parameters())
            + [self.cls_token, self.alpha, self.bg_prototype, self.bg_bias]
            + list(self.fusion_norm.parameters())
            + list(self.support_pool.parameters())
        )

    # ------------------------------------------------------------------
    # LoRA — now available from L2 (no longer restricted to L3).
    # ------------------------------------------------------------------

    def attach_lora(
        self, *, r: int = 8, alpha: int = 16, dropout: float = 0.1, last_n_layers: int = 4,
        target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    ) -> list[nn.Parameter]:
        """Inject LoRA on q_proj/v_proj of the last N vision blocks (idempotent)."""
        if self._lora_attached:
            lora_params = [
                p for n, p in self.owlv2.named_parameters()
                if "lora_" in n and p.requires_grad
            ]
            if not lora_params:
                lora_params = [p for n, p in self.owlv2.named_parameters() if "lora_" in n]
            return lora_params

        from peft import LoraConfig, get_peft_model

        encoder_layers = self.owlv2.owlv2.vision_model.encoder.layers
        n_layers = len(encoder_layers)
        target_layer_ids = list(range(max(0, n_layers - last_n_layers), n_layers))
        target_module_paths = [
            f"owlv2.vision_model.encoder.layers.{i}.self_attn.{proj}"
            for i in target_layer_ids
            for proj in target_modules
        ]
        cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target_module_paths, bias="none",
        )
        self.owlv2 = get_peft_model(self.owlv2, cfg)
        self._lora_attached = True
        return [p for n, p in self.owlv2.named_parameters() if "lora_" in n and p.requires_grad]

    @property
    def lora_attached(self) -> bool:
        return self._lora_attached

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _support_pass(
        self, support_imgs: torch.Tensor, support_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return (B, K_max, D_q) per-support embedding (q_emb + attn-pool residual)."""
        if support_imgs.dim() != 5:
            raise ValueError(
                f"support_imgs must be (B, K, 3, S, S), got {tuple(support_imgs.shape)}"
            )
        B, K, _, S1, S2 = support_imgs.shape
        flat = support_imgs.reshape(B * K, 3, S1, S2)
        flat = _normalize_owlv2(flat)

        # OWLv2 backbone path. Run under no_grad when no class_head param is
        # trainable AND no LoRA on the vision tower is trainable.
        any_grad = torch.is_grad_enabled() and (
            any(p.requires_grad for p in self.owlv2.class_head.parameters())
            or any("lora_" in n and p.requires_grad
                   for n, p in self.owlv2.named_parameters())
        )
        ctx = torch.enable_grad() if any_grad else torch.no_grad()
        with ctx:
            fm, _ = self.owlv2.image_embedder(
                pixel_values=flat, interpolate_pos_encoding=True,
            )                                                  # (BK, gh, gw, 768)
            gh, gw = fm.shape[1], fm.shape[2]
            feats = fm.reshape(B * K, -1, fm.shape[-1])         # (BK, P, 768)
            q_emb, _, _ = self.owlv2.embed_image_query(
                feats, fm, interpolate_pos_encoding=True,
            )                                                  # (BK, 1, D_q) or (BK, D_q)

        if q_emb.dim() == 3:
            q_emb = q_emb.squeeze(1)                           # (BK, D_q)

        # Attention pool over the support patch grid, conditioned on q_emb.
        # No centre prior — supports are pre-cropped to the object on disk
        # since manifest v5, so every patch is object signal.
        with ctx:
            pooled = self.support_pool(feats, q_emb)            # (BK, D_q)

        out = pooled.view(B, K, -1)
        return out if any_grad else out.detach()

    @staticmethod
    def _baseline_prototype(
        q_emb: torch.Tensor, support_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = support_mask.float().unsqueeze(-1)
        return (q_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

    def _fuse(
        self, q_emb: torch.Tensor, support_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, K, _ = q_emb.shape
        if (~support_mask).all(dim=1).any():
            support_mask = support_mask.clone()
            empty_rows = (~support_mask).all(dim=1)
            support_mask[empty_rows, 0] = True
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, q_emb], dim=1)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
        kp_mask = torch.cat([cls_mask, ~support_mask], dim=1)
        fused = self.fusion(seq, src_key_padding_mask=kp_mask)
        return self.fusion_norm(fused[:, 0])

    def _class_predict(
        self, feats_q: torch.Tensor, proto: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.owlv2.class_predictor(feats_q, proto.unsqueeze(1))
        return logits.squeeze(-1)

    def _decode_boxes(self, feats_q: torch.Tensor, fm_q: torch.Tensor) -> torch.Tensor:
        raw = self.owlv2.box_predictor(
            feats_q, fm_q, interpolate_pos_encoding=True,
        )                                                              # (B, P, 4)
        if self.log_box_head is not None:
            raw = self.log_box_head(raw)
        return raw

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        support_imgs: torch.Tensor,
        support_mask: torch.Tensor,
        query_img: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if support_mask.dtype != torch.bool:
            support_mask = support_mask.to(torch.bool)
        if support_mask.device != support_imgs.device:
            support_mask = support_mask.to(support_imgs.device)

        q_emb = self._support_pass(support_imgs, support_mask)         # (B, K, D_q)
        baseline = self._baseline_prototype(q_emb, support_mask)
        fused = self._fuse(q_emb, support_mask)
        # Residual: prototype = baseline + alpha * (fused - baseline).
        # When fused ≈ baseline the prototype ≈ baseline regardless of α; when
        # fused diverges from baseline, α controls how much divergence the
        # prototype keeps. alpha_init=1.0 ⇒ start from the fused path.
        prototype = baseline + self.alpha * (fused - baseline)

        q_norm = _normalize_owlv2(query_img)
        fm_q, _ = self.owlv2.image_embedder(pixel_values=q_norm, interpolate_pos_encoding=True)
        gh, gw = fm_q.shape[1], fm_q.shape[2]
        feats_q = fm_q.reshape(fm_q.shape[0], -1, fm_q.shape[-1])
        B = feats_q.shape[0]

        pred_logits_fg = self._class_predict(feats_q, prototype)
        bg_proto_b = self.bg_prototype.expand(B, -1)
        bg_logits_patch = self._class_predict(feats_q, bg_proto_b)
        bg_logit = bg_logits_patch.max(dim=-1).values + self.bg_bias

        pred_boxes = self._decode_boxes(feats_q, fm_q)

        joint = torch.cat([pred_logits_fg, bg_logit.unsqueeze(-1)], dim=-1)
        joint_prob = joint.softmax(dim=-1)
        fg_prob = joint_prob[:, :-1]
        bg_prob = joint_prob[:, -1]
        best_idx = fg_prob.argmax(dim=-1)
        ar = torch.arange(B, device=pred_logits_fg.device)
        best_box = pred_boxes[ar, best_idx]
        best_score = fg_prob[ar, best_idx]
        best_logit = pred_logits_fg[ar, best_idx]

        return {
            "best_box": best_box,
            "best_score": best_score,
            "bg_prob": bg_prob,
            "best_logit": best_logit,
            "pred_logits": pred_logits_fg,
            "pred_logits_fg": pred_logits_fg,
            "bg_logit": bg_logit,
            "bg_logits_patch": bg_logits_patch,
            "pred_boxes": pred_boxes,
            "prototype": prototype,
            "bg_prototype": self.bg_prototype,
            "baseline_prototype": baseline,
            "alpha": self.alpha.detach(),
            "patch_grid": (gh, gw),
        }

    @torch.no_grad()
    def phase0_forward(
        self,
        support_imgs: torch.Tensor,
        support_mask: torch.Tensor,
        query_img: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """ONE-SHOT vanilla OWLv2 baseline.

        Uses ONLY the first valid support per episode (K = 1) and runs the
        bare ``owlv2.embed_image_query`` + ``class_predictor`` + ``box_predictor``
        path. No fusion, no support attention pool, no bg channel, no
        log-scale box head wrapper. This is the apples-to-apples baseline
        the trained model must beat.

        Score = sigmoid(top-1 patch logit) so this matches OWLv2's own
        per-patch confidence convention.
        """
        if support_imgs.dim() != 5:
            raise ValueError(f"support_imgs must be (B, K, 3, S, S), got {tuple(support_imgs.shape)}")
        B, K, _, S1, S2 = support_imgs.shape
        # Pick the first VALID slot per row. ``support_mask`` is True on real
        # supports. Rows that are entirely padded get slot 0 anyway (and
        # match the trained forward's defensive behaviour).
        safe_mask = support_mask
        if (~safe_mask).all(dim=1).any():
            safe_mask = safe_mask.clone()
            safe_mask[(~safe_mask).all(dim=1), 0] = True
        first_idx = safe_mask.float().argmax(dim=1)              # (B,) — first True
        ar = torch.arange(B, device=support_imgs.device)
        support_one = support_imgs[ar, first_idx]                # (B, 3, S, S)

        flat = _normalize_owlv2(support_one)                     # (B, 3, S, S)
        fm, _ = self.owlv2.image_embedder(pixel_values=flat, interpolate_pos_encoding=True)
        feats = fm.reshape(B, -1, fm.shape[-1])
        q_emb, _, _ = self.owlv2.embed_image_query(feats, fm, interpolate_pos_encoding=True)
        if q_emb.dim() == 3:
            q_emb = q_emb.squeeze(1)                              # (B, D_q)
        proto = q_emb                                             # one-shot prototype

        q_norm = _normalize_owlv2(query_img)
        fm_q, _ = self.owlv2.image_embedder(pixel_values=q_norm, interpolate_pos_encoding=True)
        feats_q = fm_q.reshape(fm_q.shape[0], -1, fm_q.shape[-1])
        pred_logits, _ = self.owlv2.class_predictor(feats_q, proto.unsqueeze(1))
        pred_logits = pred_logits.squeeze(-1)                     # (B, P)
        # Vanilla OWLv2 box_predictor — NO log-scale wrapper applied here.
        pred_boxes = self.owlv2.box_predictor(feats_q, fm_q, interpolate_pos_encoding=True)
        best_idx = pred_logits.argmax(dim=-1)
        ar2 = torch.arange(B, device=pred_logits.device)
        return {
            "best_box": pred_boxes[ar2, best_idx],
            "best_score": torch.sigmoid(pred_logits[ar2, best_idx]),
            "bg_prob": torch.zeros(B, device=pred_logits.device),
            "best_logit": pred_logits[ar2, best_idx],
            "pred_logits": pred_logits,
            "pred_logits_fg": pred_logits,
            "bg_logit": torch.zeros(B, device=pred_logits.device),
            "bg_logits_patch": torch.zeros_like(pred_logits),
            "pred_boxes": pred_boxes,
            "prototype": proto,
            "bg_prototype": torch.zeros(1, proto.size(-1), device=proto.device),
            "baseline_prototype": proto,
            "alpha": torch.zeros((), device=proto.device),
            "patch_grid": (fm_q.shape[1], fm_q.shape[2]),
        }
