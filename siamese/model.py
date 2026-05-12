"""Multi-shot existence siamese.

Frozen DINOv2-small backbone + trainable cross-attention pooling head.

INPUTS
    support_imgs : (B, K_max, 3, S, S)  un-normalized RGB in [0,1].
    support_mask : (B, K_max,) bool
    query_img    : (B, 3, S, S)         un-normalized RGB.

DINOv2 path (frozen unless LoRA in S2):
    sup_out = dinov2(supports.flatten)      → (B*K, 1+P, D=384)
    q_out   = dinov2(query)                 → (B, 1+P, D)
    sup_cls = sup_out[:, 0].view(B, K, D)
    sup_pat = sup_out[:, 1:].view(B, K, P, D)
    q_cls   = q_out[:, 0]                   (B, D)
    q_pat   = q_out[:, 1:]                  (B, P, D)

Cross-attention pool (trainable):
    flat_sup = sup_pat.reshape(B, K*P, D)
    flat_mask: (B, K*P) bool — True for padded slots (key_padding_mask).
    attended = MultiHeadAttention(query=q_pat, key=flat_sup, value=flat_sup,
                                  key_padding_mask=flat_mask)
    pooled   = attended.mean(dim=1)         (B, D)

Scalar features (computed on cosine sims between q_pat and flat_sup):
    s1 max patch sim
    s2 top-5 mean patch sim
    s3 top-5 std patch sim
    s4 mean patch sim
    s5 cosine(mean(sup_cls valid), q_cls)
    s6 entropy of softmax over flattened patch sims

Head MLP:
    feat = concat([pooled, q_cls, sup_cls_mean, scalars])  (B, 3D + 6)
    LayerNorm → Linear(.,256) → GELU → Dropout
              → Linear(256,64) → GELU → Dropout
              → Linear(64, 1) → Sigmoid

OUTPUTS
    existence_prob  : (B,)
    existence_logit : (B,)
    pooled          : (B, D)              for variance/decorrelation regularizers
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# DINOv2 normalization (ImageNet stats).
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)

DINOV2_MODEL_NAME = "facebook/dinov2-small"


def _normalize_dinov2(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor(DINOV2_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(DINOV2_STD).view(1, 3, 1, 1)
    return (x - mean) / std


N_SCALAR = 6


class MultiShotSiamese(nn.Module):
    def __init__(
        self,
        model_name: str = DINOV2_MODEL_NAME,
        *,
        k_max: int = 10,
        cross_attn_heads: int = 6,
        cross_attn_dropout: float = 0.1,
        head_hidden_1: int = 256,
        head_hidden_2: int = 64,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dinov2 = AutoModel.from_pretrained(model_name)
        D = self.dinov2.config.hidden_size  # 384
        self.embed_dim = D
        self.k_max = int(k_max)

        self.cross_attn = nn.MultiheadAttention(
            D, num_heads=cross_attn_heads,
            dropout=cross_attn_dropout, batch_first=True,
        )
        self.cross_norm_q = nn.LayerNorm(D)
        self.cross_norm_kv = nn.LayerNorm(D)

        in_dim = 3 * D + N_SCALAR
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, head_hidden_1), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(head_hidden_1, head_hidden_2), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(head_hidden_2, 1),
        )
        self.freeze_backbone()
        self._lora_attached = False

    # ------------------------------------------------------------------
    # Freezing helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        for p in self.dinov2.parameters():
            p.requires_grad = False

    def head_params(self) -> list[nn.Parameter]:
        return (
            list(self.cross_attn.parameters())
            + list(self.cross_norm_q.parameters())
            + list(self.cross_norm_kv.parameters())
            + list(self.head.parameters())
        )

    # ------------------------------------------------------------------
    # LoRA (Stage S2)
    # ------------------------------------------------------------------

    def attach_lora(
        self, *, r: int = 8, alpha: int = 16, dropout: float = 0.1, last_n_layers: int = 4,
        target_modules: tuple[str, ...] = ("query", "value"),
    ) -> list[nn.Parameter]:
        """Inject LoRA on query/value of the last N DINOv2 encoder layers.

        Idempotent: if LoRA is already attached, returns the existing LoRA
        parameter list without re-wrapping. Without this guard, on resume
        the model gets wrapped by PEFT twice (once in ``_build_model``,
        again here in the optimizer factory), which stacks adapters and
        emits PEFT's "second time" / "Already found a peft_config attribute"
        warnings.
        """
        if self._lora_attached:
            lora_params = [
                p for n, p in self.dinov2.named_parameters()
                if "lora_" in n and p.requires_grad
            ]
            if not lora_params:
                lora_params = [
                    p for n, p in self.dinov2.named_parameters() if "lora_" in n
                ]
            return lora_params

        from peft import LoraConfig, get_peft_model

        layers = self.dinov2.encoder.layer
        n_layers = len(layers)
        target_layer_ids = list(range(max(0, n_layers - last_n_layers), n_layers))
        target_module_paths = [
            f"encoder.layer.{i}.attention.attention.{proj}"
            for i in target_layer_ids
            for proj in target_modules
        ]
        cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target_module_paths, bias="none",
        )
        self.dinov2 = get_peft_model(self.dinov2, cfg)
        self._lora_attached = True
        lora_params = [p for n, p in self.dinov2.named_parameters() if "lora_" in n and p.requires_grad]
        return lora_params

    @property
    def lora_attached(self) -> bool:
        return self._lora_attached

    # ------------------------------------------------------------------
    # Scalar features
    # ------------------------------------------------------------------

    def _compute_scalars(
        self,
        flat_sup: torch.Tensor,    # (B, K*P, D)
        flat_mask: torch.Tensor,   # (B, K*P) True for PADDED slots
        q_pat: torch.Tensor,       # (B, P, D)
        sup_cls_mean: torch.Tensor,  # (B, D)
        q_cls: torch.Tensor,       # (B, D)
    ) -> torch.Tensor:
        """Return (B, N_SCALAR) feature tensor."""
        B, KP, D = flat_sup.shape
        P = q_pat.shape[1]
        # Cosine sims between every q_pat token and every flat_sup token.
        q_n = F.normalize(q_pat, dim=-1)                     # (B, P, D)
        s_n = F.normalize(flat_sup, dim=-1)                  # (B, K*P, D)
        sims = torch.einsum("bpd,bkd->bpk", q_n, s_n)        # (B, P, K*P)
        # Mask invalid support tokens.
        mask = flat_mask.unsqueeze(1).expand(-1, P, -1)      # (B, P, K*P)
        sims = sims.masked_fill(mask, -1e4)
        # Per-(B, P) max over support tokens.
        per_query_max, _ = sims.max(dim=-1)                  # (B, P)
        per_query_top5_vals, _ = sims.topk(min(5, KP), dim=-1)  # (B, P, k)
        # Now reduce over query patches.
        s1 = per_query_max.max(dim=-1).values                # (B,)
        s2 = per_query_top5_vals.mean(dim=(-1, -2))          # (B,)
        s3 = per_query_top5_vals.std(dim=(-1, -2), unbiased=False)
        s4 = per_query_max.mean(dim=-1)                      # (B,)
        s5 = F.cosine_similarity(sup_cls_mean, q_cls, dim=-1)  # (B,)
        # Entropy over flattened sims (B, P*K*P) — softmax with temperature.
        flat_sims = per_query_max                            # (B, P)
        soft = F.softmax(flat_sims, dim=-1)
        s6 = -(soft * (soft + 1e-12).log()).sum(dim=-1)
        return torch.stack([s1, s2, s3, s4, s5, s6], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"image batch must be (B, 3, S, S), got {tuple(x.shape)}")
        x = _normalize_dinov2(x)
        any_grad = (
            torch.is_grad_enabled()
            and any(p.requires_grad for p in self.dinov2.parameters())
        )
        ctx = torch.enable_grad() if any_grad else torch.no_grad()
        with ctx:
            out = self.dinov2(
                pixel_values=x, interpolate_pos_encoding=True,
            ).last_hidden_state
        return out if any_grad else out.detach()

    def forward(
        self,
        support_imgs: torch.Tensor,
        support_mask: torch.Tensor,
        query_img: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if support_imgs.dim() != 5:
            raise ValueError(f"support_imgs must be (B, K, 3, S, S), got {tuple(support_imgs.shape)}")
        if support_mask.dtype != torch.bool:
            support_mask = support_mask.to(torch.bool)
        if support_mask.device != support_imgs.device:
            support_mask = support_mask.to(support_imgs.device)
        B, K_max, _, S1, S2 = support_imgs.shape
        D = self.embed_dim

        sup_flat = support_imgs.reshape(B * K_max, 3, S1, S2)
        sup_h = self._encode(sup_flat)                       # (B*K, 1+P, D)
        q_h = self._encode(query_img)                        # (B, 1+P, D)

        sup_cls_all = sup_h[:, 0].view(B, K_max, D)          # (B, K, D)
        sup_pat = sup_h[:, 1:].view(B, K_max, -1, D)          # (B, K, P, D)
        P = sup_pat.shape[2]
        q_cls = q_h[:, 0]                                    # (B, D)
        q_pat = q_h[:, 1:]                                   # (B, P, D)

        # Defensive: avoid all-padded rows (would yield NaN in MHA softmax
        # and in division by zero of mask sum). Force the first support slot
        # to be valid for any row that has none.
        safe_mask = support_mask
        if (~safe_mask).all(dim=1).any():
            safe_mask = safe_mask.clone()
            safe_mask[(~safe_mask).all(dim=1), 0] = True

        # Mean of valid support CLS tokens.
        mask_f = safe_mask.float().unsqueeze(-1)             # (B, K, 1)
        sup_cls_mean = (sup_cls_all * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        # Cross-attention.
        flat_sup = sup_pat.reshape(B, K_max * P, D)
        # key_padding_mask: True ⇒ ignored.
        per_kp_mask = (~safe_mask).unsqueeze(-1).expand(-1, -1, P)  # (B, K, P)
        kp_mask = per_kp_mask.reshape(B, K_max * P)
        # Pre-LN.
        q_in = self.cross_norm_q(q_pat)
        kv_in = self.cross_norm_kv(flat_sup)
        attended, _ = self.cross_attn(q_in, kv_in, kv_in, key_padding_mask=kp_mask)
        pooled = attended.mean(dim=1)                        # (B, D)

        scalars = self._compute_scalars(
            flat_sup, kp_mask, q_pat, sup_cls_mean, q_cls,
        )                                                    # (B, N_SCALAR)

        feat = torch.cat([pooled, q_cls, sup_cls_mean, scalars], dim=-1)
        logit = self.head(feat).squeeze(-1)
        prob = torch.sigmoid(logit)

        return {
            "existence_prob": prob,
            "existence_logit": logit,
            "pooled": pooled,
            "q_cls": q_cls,
            "sup_cls_mean": sup_cls_mean,
        }

    @torch.no_grad()
    def phase0_forward(
        self,
        support_imgs: torch.Tensor,
        support_mask: torch.Tensor,
        query_img: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Zero-shot baseline: cosine(mean(sup_cls), q_cls), squashed to [0, 1]."""
        if support_imgs.dim() != 5:
            raise ValueError(f"support_imgs must be (B, K, 3, S, S), got {tuple(support_imgs.shape)}")
        B, K_max, _, S1, S2 = support_imgs.shape
        sup_flat = support_imgs.reshape(B * K_max, 3, S1, S2)
        sup_h = self._encode(sup_flat)
        q_h = self._encode(query_img)
        D = self.embed_dim
        sup_cls = sup_h[:, 0].view(B, K_max, D)
        q_cls = q_h[:, 0]
        # Guard against rows with no valid supports (would divide by zero).
        safe_mask = support_mask
        if (~safe_mask).all(dim=1).any():
            safe_mask = safe_mask.clone()
            safe_mask[(~safe_mask).all(dim=1), 0] = True
        mask_f = safe_mask.float().unsqueeze(-1)
        sup_cls_mean = (sup_cls * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        cos = F.cosine_similarity(sup_cls_mean, q_cls, dim=-1)  # (B,) in [-1, 1]
        prob = ((cos + 1.0) / 2.0).clamp(min=1e-6, max=1.0 - 1e-6)
        return {
            "existence_prob": prob,
            "existence_logit": torch.logit(prob),
            "pooled": q_cls,
            "q_cls": q_cls,
            "sup_cls_mean": sup_cls_mean,
        }
