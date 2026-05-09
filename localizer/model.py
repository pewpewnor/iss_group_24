"""Multi-shot localizer.

Architecture:

    INPUTS
        support_imgs : (B, K_max, 3, S, S)   un-normalized RGB in [0,1].
        support_mask : (B, K_max,) bool      True where the slot is real.
        query_img    : (B, 3, S, S)          un-normalized RGB.

    PER-SUPPORT (frozen OWLv2 path):
        for each support_i:
            fm_i, _ = owlv2.image_embedder(support_i, interpolate_pos_encoding=True)
            feats_i = fm_i.reshape(B*K, P, D_v=768)
            q_emb_i, _, _ = owlv2.embed_image_query(feats_i, fm_i, ...)
                                                    → (B*K, 1, D_q=512)

    FUSION (trainable; ~2.5M params):
        Stack K per-support q_emb into (B, K, D_q).
        Prepend learnable [CLS] token → (B, K+1, D_q).
        Apply 2-layer pre-LN transformer encoder with key_padding_mask.
        prototype = layer_norm(encoded[:, 0]).

    QUERY:
        fm_q = owlv2.image_embedder(query)         (B, gh, gw, 768)
        feats_q = fm_q.reshape(B, P, 768)
        pred_logits, _ = owlv2.class_predictor(feats_q, prototype.unsqueeze(1))
        pred_boxes = owlv2.box_predictor(feats_q, fm_q, interpolate_pos_encoding=True)
        best_idx = pred_logits.argmax(dim=1).squeeze(-1)
        best_box = pred_boxes[arange, best_idx]   # (B, 4) cxcywh in [0,1]

    OUTPUTS
        best_box     : (B, 4)
        best_score   : (B,)        sigmoid of top-1 logit
        pred_logits  : (B, P)
        pred_boxes   : (B, P, 4)
        prototype    : (B, D_q=512)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Owlv2ForObjectDetection


# CLIP normalization (used by OWLv2).
OWLV2_MEAN = (0.48145466, 0.4578275, 0.40821073)
OWLV2_STD = (0.26862954, 0.26130258, 0.27577711)

OWLV2_MODEL_NAME = "google/owlv2-base-patch16-ensemble"


def _normalize_owlv2(x: torch.Tensor) -> torch.Tensor:
    """Apply CLIP/OWLv2 normalization to a tensor in [0, 1]."""
    mean = x.new_tensor(OWLV2_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(OWLV2_STD).view(1, 3, 1, 1)
    return (x - mean) / std


class MultiShotLocalizer(nn.Module):
    """OWLv2 + learnable support-fusion transformer."""

    def __init__(
        self,
        model_name: str = OWLV2_MODEL_NAME,
        *,
        k_max: int = 10,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        fusion_mlp_ratio: int = 2,
        fusion_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.owlv2 = Owlv2ForObjectDetection.from_pretrained(model_name)
        self.k_max = int(k_max)
        D_q = self.owlv2.config.text_config.hidden_size  # 512
        self.query_dim = D_q

        # Learnable [CLS] token + tiny init.
        self.cls_token = nn.Parameter(torch.randn(1, 1, D_q) * 0.02)

        # Pre-LN transformer encoder.
        layer = nn.TransformerEncoderLayer(
            d_model=D_q, nhead=fusion_heads,
            dim_feedforward=D_q * fusion_mlp_ratio,
            dropout=fusion_dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=fusion_layers)
        self.fusion_norm = nn.LayerNorm(D_q)

        # Residual identity path. The prototype is computed as
        #   prototype = mean(per_support_q_emb) + alpha * fusion_correction
        # with `alpha` initialised at a small positive value so the model
        # starts NEAR zero-shot quality (~99% baseline + 1% fusion) and
        # the fusion still receives gradient from the first batch. With
        # alpha exactly 0, the gradient w.r.t. fusion params is also
        # exactly 0 (chain rule through the multiplication), creating a
        # "alpha-first warmup" that delays fusion learning by O(epochs).
        self.alpha = nn.Parameter(torch.full((), 0.01))

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
        """Stage L2: unfreeze class_head + box_head + layer_norm."""
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
        return list(self.owlv2.box_head.parameters())

    def fusion_params(self) -> list[nn.Parameter]:
        return (
            list(self.fusion.parameters())
            + [self.cls_token, self.alpha]
            + list(self.fusion_norm.parameters())
        )

    # ------------------------------------------------------------------
    # LoRA (Stage L3)
    # ------------------------------------------------------------------

    def attach_lora(
        self, *, r: int = 8, alpha: int = 16, dropout: float = 0.1, last_n_layers: int = 4,
        target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    ) -> list[nn.Parameter]:
        """Inject LoRA on q_proj/v_proj of the last N vision blocks."""
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
        lora_params = [p for n, p in self.owlv2.named_parameters() if "lora_" in n and p.requires_grad]
        return lora_params

    @property
    def lora_attached(self) -> bool:
        return self._lora_attached

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _embed_supports(
        self, support_imgs: torch.Tensor, support_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-support OWLv2 image-guided query embedding.

        Returns (B, K_max, D_q). Padded slots get whatever q_emb falls out
        of feeding zeros through OWLv2 — those slots are masked out in
        fusion attention so the value does not matter.

        Gradient policy:
          - If the *outer* call site disabled grad (eval / no_grad), we run
            under no_grad regardless of param trainability.
          - Else, if any class_head param is trainable (Stages L2 / L3),
            we keep the support-pass graph so those heads receive gradient
            from the support side too.
          - Else (Stage L1, eval mode), we run under no_grad to save memory.
        """
        if support_imgs.dim() != 5:
            raise ValueError(
                f"support_imgs must be (B, K, 3, S, S), got {tuple(support_imgs.shape)}"
            )
        B, K, _, S1, S2 = support_imgs.shape
        flat = support_imgs.reshape(B * K, 3, S1, S2)
        flat = _normalize_owlv2(flat)

        any_grad = (
            torch.is_grad_enabled()
            and any(p.requires_grad for p in self.owlv2.class_head.parameters())
        )
        ctx = torch.enable_grad() if any_grad else torch.no_grad()
        with ctx:
            fm, _ = self.owlv2.image_embedder(
                pixel_values=flat, interpolate_pos_encoding=True,
            )
            feats = fm.reshape(B * K, -1, fm.shape[-1])
            q_emb, _, _ = self.owlv2.embed_image_query(
                feats, fm, interpolate_pos_encoding=True,
            )
        if q_emb.dim() == 3:
            q_emb = q_emb.squeeze(1)
        q_emb = q_emb.view(B, K, -1)
        return q_emb if any_grad else q_emb.detach()

    @staticmethod
    def _baseline_prototype(
        q_emb: torch.Tensor, support_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean of per-support OWLv2 image-guided embeddings over valid slots.

        This is exactly what ``phase0_forward`` uses, and is the zero-shot
        quality baseline. The full prototype is built as
        ``baseline + alpha * fusion_correction``.
        """
        mask_f = support_mask.float().unsqueeze(-1)             # (B, K, 1)
        return (q_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

    def _fuse(
        self, q_emb: torch.Tensor, support_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse K per-support embeddings into a CORRECTION via [CLS] attention.

        Returns (B, D_q) — added to the baseline mean prototype with weight
        alpha (init 0). At alpha=0 the prototype equals the baseline, so L1
        starts at zero-shot quality.
        """
        B, K, _ = q_emb.shape
        if support_mask.shape != (B, K):
            raise ValueError(
                f"support_mask shape {tuple(support_mask.shape)} does not match "
                f"q_emb (B={B}, K={K})"
            )
        # Defensive guard against an entirely-masked row (no real supports);
        # MHA would produce NaN. We fall back to attending to CLS only by
        # forcing the first slot to be valid in such rows.
        if (~support_mask).all(dim=1).any():
            support_mask = support_mask.clone()
            empty_rows = (~support_mask).all(dim=1)
            support_mask[empty_rows, 0] = True
        cls = self.cls_token.expand(B, -1, -1)                  # (B, 1, D_q)
        seq = torch.cat([cls, q_emb], dim=1)                    # (B, 1+K, D_q)
        # key_padding_mask: True ⇒ ignored. CLS slot (col 0) always attended.
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
        kp_mask = torch.cat([cls_mask, ~support_mask], dim=1)   # (B, 1+K)
        fused = self.fusion(seq, src_key_padding_mask=kp_mask)
        return self.fusion_norm(fused[:, 0])                    # (B, D_q)

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

        # 1) Per-support OWLv2 query-embedding.
        q_emb = self._embed_supports(support_imgs, support_mask)         # (B, K, D_q)

        # 2) Fusion → prototype = baseline + alpha * fusion_correction.
        baseline = self._baseline_prototype(q_emb, support_mask)         # (B, D_q)
        correction = self._fuse(q_emb, support_mask)                     # (B, D_q)
        prototype = baseline + self.alpha * correction                    # (B, D_q)

        # 3) Query path.
        q_norm = _normalize_owlv2(query_img)
        fm_q, _ = self.owlv2.image_embedder(pixel_values=q_norm, interpolate_pos_encoding=True)
        gh, gw = fm_q.shape[1], fm_q.shape[2]
        feats_q = fm_q.reshape(fm_q.shape[0], -1, fm_q.shape[-1])
        pred_logits, _ = self.owlv2.class_predictor(feats_q, prototype.unsqueeze(1))
        pred_logits = pred_logits.squeeze(-1)                            # (B, P)
        pred_boxes = self.owlv2.box_predictor(
            feats_q, fm_q, interpolate_pos_encoding=True,
        )                                                                # (B, P, 4)

        best_idx = pred_logits.argmax(dim=-1)
        ar = torch.arange(pred_logits.size(0), device=pred_logits.device)
        best_box = pred_boxes[ar, best_idx]
        best_score = torch.sigmoid(pred_logits[ar, best_idx])

        return {
            "best_box": best_box,
            "best_score": best_score,
            "best_logit": pred_logits[ar, best_idx],
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "prototype": prototype,
            "baseline_prototype": baseline,
            # Detached scalar — the trainable alpha is the parameter on the
            # model itself; this is just a metric snapshot.
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
        """Zero-shot OWLv2 baseline: mean per-support query embedding then decode top-1.

        No fusion transformer — used only by Phase 0 cells.
        """
        if support_imgs.dim() != 5:
            raise ValueError(f"support_imgs must be (B, K, 3, S, S), got {tuple(support_imgs.shape)}")
        B, K, _, S1, S2 = support_imgs.shape
        flat = _normalize_owlv2(support_imgs.reshape(B * K, 3, S1, S2))
        fm, _ = self.owlv2.image_embedder(pixel_values=flat, interpolate_pos_encoding=True)
        feats = fm.reshape(B * K, -1, fm.shape[-1])
        q_emb, _, _ = self.owlv2.embed_image_query(feats, fm, interpolate_pos_encoding=True)
        if q_emb.dim() == 3:
            q_emb = q_emb.squeeze(1)
        q_emb = q_emb.view(B, K, -1)
        # Guard against fully-padded rows (no valid supports).
        safe_mask = support_mask
        if (~safe_mask).all(dim=1).any():
            safe_mask = safe_mask.clone()
            safe_mask[(~safe_mask).all(dim=1), 0] = True
        # Mean-pool over valid supports.
        mask_f = safe_mask.float().unsqueeze(-1)
        proto = (q_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        # Query path.
        q_norm = _normalize_owlv2(query_img)
        fm_q, _ = self.owlv2.image_embedder(pixel_values=q_norm, interpolate_pos_encoding=True)
        feats_q = fm_q.reshape(fm_q.shape[0], -1, fm_q.shape[-1])
        pred_logits, _ = self.owlv2.class_predictor(feats_q, proto.unsqueeze(1))
        pred_logits = pred_logits.squeeze(-1)
        pred_boxes = self.owlv2.box_predictor(feats_q, fm_q, interpolate_pos_encoding=True)
        best_idx = pred_logits.argmax(dim=-1)
        ar = torch.arange(B, device=pred_logits.device)
        return {
            "best_box": pred_boxes[ar, best_idx],
            "best_score": torch.sigmoid(pred_logits[ar, best_idx]),
            "best_logit": pred_logits[ar, best_idx],
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "prototype": proto,
        }
