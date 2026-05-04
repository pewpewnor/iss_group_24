"""Few-shot localization model.

Backbone: MobileNetV3-Large (pretrained ImageNet).

Pipeline (per episode, batch dim B):
  support_imgs   (B, K, 3, 224, 224) -> backbone P5 (7x7x160)
                                      -> ROI-align crop -> Projection (160->128->64)
                                      -> PrototypeAggregator (attention over K shots)
                                      -> proto (B, 64)

  query_img      (B, 3, 224, 224)    -> backbone P3 (28x28x40), P4 (14x14x112), P5 (7x7x160)
                                      -> FPN(P4, P5) -> 14x14x64
                                      -> + 0.5 * AvgPool(p3_lat(P3))  [stride-8 residual]
                                      -> q_feat (B, 64, 14, 14)

  proto --gate--> channel attention on q_feat -> modulated (B, 64, 14, 14)
  modulated + dot-product corr -> DetectionHead:
      ChannelAttention -> 3x DWSep -> reg (B,4,14,14) + conf (B,1,14,14)
  modulated.mean(2,3) + proto -> PresenceHead -> presence_logit (B,)
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.ops import FeaturePyramidNetwork, roi_align

IMG_SIZE = 224
GRID = 14
STRIDE = IMG_SIZE // GRID  # 16
PROTO_DIM = 64
FPN_DIM = 64
PROTO_MID = 128  # projection bottleneck hidden dim

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
    """MobileNetV3-Large exposing P3 (28x28x40), P4 (14x14x112), P5 (7x7x160).
    forward_p5_only is a fast path for the support branch (needs P5 only)."""

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
# Support branch: projection + proto aggregation
# ---------------------------------------------------------------------------


class SaliencyAttention(nn.Module):
    """Learned spatial attention pool over the support feature map.

    Replaces ROI Align — predicts a softmax-normalised (H, W) attention map
    via a small conv head, then returns the attention-weighted GAP. No bbox
    is needed at inference; the model learns where the foreground is.

    During training an auxiliary KL loss can supervise this attention to align
    with the bbox we have from cleaner.py — speeds up convergence without
    making bboxes a hard requirement.
    """

    def __init__(self, in_c: int = P5_CHANNELS, hidden: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, _, h, w = feat.shape
        logits = self.conv(feat).view(n, -1)             # (N, H*W)
        attn = torch.softmax(logits, dim=1).view(n, 1, h, w)
        pooled = (feat * attn).sum(dim=(2, 3))           # (N, C)
        return pooled, attn


class Projection(nn.Module):
    """Two-stage MLP: in_c -> mid_c -> out_c, with GELU and LayerNorm.

    Operates on a pooled (N, C) vector — pooling is now done by
    SaliencyAttention upstream, replacing the previous Conv1x1+GAP.
    """

    def __init__(
        self,
        in_c: int = P5_CHANNELS,
        mid_c: int = PROTO_MID,
        out_c: int = PROTO_DIM,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_c, mid_c, bias=False)
        self.norm1 = nn.LayerNorm(mid_c)
        self.fc2 = nn.Linear(mid_c, out_c, bias=False)
        self.norm2 = nn.LayerNorm(out_c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.fc1(x)))
        return self.norm2(self.act(self.fc2(x)))


class PrototypeAggregator(nn.Module):
    """Attention-weighted mean over K support descriptors.

    Learns which support images are more informative (sharp, unoccluded) and
    down-weights noisy/blurry ones. The attention score is a scalar per image.
    """

    def __init__(self, dim: int = PROTO_DIM) -> None:
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, support_vecs: torch.Tensor) -> torch.Tensor:
        # support_vecs: (B, K, dim)
        weights = torch.softmax(self.attn(support_vecs), dim=1)  # (B, K, 1)
        return (weights * support_vecs).sum(dim=1)               # (B, dim)


# ---------------------------------------------------------------------------
# Query branch
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


class ChannelAttention(nn.Module):
    """CBAM-style channel attention: GAP -> squeeze-excite -> per-channel scale.

    Allows the detection head to learn to up-weight the correlation channel
    when the proto is informative and suppress it when noisy.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = x.mean(dim=(2, 3))                          # (B, C)
        scale = self.fc(gap).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * scale


# ---------------------------------------------------------------------------
# Detection head + presence head
# ---------------------------------------------------------------------------


class DetectionHead(nn.Module):
    """ChannelAttention -> 3x DWSep -> reg/conf 1x1 convs."""

    def __init__(self) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(FPN_DIM + 1)  # 65 channels in
        self.block1 = _DWSep(FPN_DIM + 1, FPN_DIM)
        self.block2 = _DWSep(FPN_DIM, FPN_DIM)
        self.block3 = _DWSep(FPN_DIM, FPN_DIM)
        self.reg = nn.Conv2d(FPN_DIM, 4, 1)
        self.conf = nn.Conv2d(FPN_DIM, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.block3(self.block2(self.block1(self.channel_attn(x))))
        return self.reg(feat), self.conf(feat)


class PresenceHead(nn.Module):
    """Global binary classifier: proto + GAP(modulated_feat) -> presence logit.

    Decouples "is the object present?" from "where is it?" by operating on
    global (spatially pooled) features rather than the spatial conf map.
    """

    def __init__(self, proto_dim: int = PROTO_DIM, fpn_dim: int = FPN_DIM) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(proto_dim + fpn_dim, fpn_dim),
            nn.GELU(),
            nn.Linear(fpn_dim, 1),
        )

    def forward(self, proto: torch.Tensor, fpn_gap: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([proto, fpn_gap], dim=1)).squeeze(1)  # (B,)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class FewShotLocalizer(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = MobileNetBackbone(pretrained=pretrained)
        self.saliency = SaliencyAttention()
        self.projection = Projection()
        self.proto_agg = PrototypeAggregator()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[P4_CHANNELS, P5_CHANNELS],
            out_channels=FPN_DIM,
        )
        # Stride-8 (28x28) lateral projection added as residual to P4 output
        self.p3_lat = nn.Conv2d(P3_CHANNELS, FPN_DIM, kernel_size=1, bias=False)
        # Learned gate: proto -> per-channel attention weights over query features
        self.gate = nn.Linear(PROTO_DIM, FPN_DIM)
        self.det_head = DetectionHead()
        self.presence_head = PresenceHead()

    # ---- support branch ----------------------------------------------------

    def encode_support(
        self,
        support_imgs: torch.Tensor,
        support_bboxes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """support_imgs: (B, K, 3, 224, 224); support_bboxes: optional (B, K, 4).

        Returns:
            prototype:    (B, PROTO_DIM) from saliency-pooled features
            attn:         (B, K, 7, 7) saliency attention map
            teacher_desc: (B, K, PROTO_DIM) ROI-aligned descriptor — used as a
                          training-only feature-distillation teacher. None when
                          no bbox is provided (inference path).
        """
        b, k = support_imgs.shape[:2]
        flat = support_imgs.reshape(b * k, 3, IMG_SIZE, IMG_SIZE)
        feat = self.backbone.forward_p5_only(flat)             # (B*K, 160, 7, 7)

        # Saliency path — the inference path. Always runs.
        pooled, attn = self.saliency(feat)                     # (B*K, 160), (B*K, 1, 7, 7)
        descs = self.projection(pooled).view(b, k, -1)         # (B, K, 64)
        proto = self.proto_agg(descs)                          # (B, 64)

        # ROI Align teacher path — only when bbox is available (training).
        teacher_desc: torch.Tensor | None = None
        if support_bboxes is not None:
            boxes = support_bboxes.reshape(b * k, 4).to(feat.dtype)
            batch_idx = torch.arange(b * k, device=feat.device, dtype=feat.dtype).unsqueeze(1)
            rois = torch.cat([batch_idx, boxes], dim=1)
            cropped = roi_align(  # pyright: ignore[reportCallIssue]
                feat, rois, output_size=(7, 7), spatial_scale=1.0 / P5_STRIDE, aligned=True
            )
            roi_pooled = F.adaptive_avg_pool2d(cropped, 1).flatten(1)  # (B*K, 160)
            teacher_desc = self.projection(roi_pooled).view(b, k, -1)  # (B, K, 64)

        return proto, attn.view(b, k, attn.shape[-2], attn.shape[-1]), teacher_desc

    @torch.no_grad()
    def compute_prototype(self, support_imgs: torch.Tensor) -> torch.Tensor:
        """Inference helper: encode 5 supports into a single prototype.

        Call this once when the user provides their support set; cache the
        result and pass it to ``detect()`` per query frame to skip the
        support-branch cost on every frame.
        """
        proto, _, _ = self.encode_support(support_imgs)
        return proto

    # ---- query branch ------------------------------------------------------

    def encode_query(self, query_img: torch.Tensor) -> torch.Tensor:
        p3, p4, p5 = self.backbone(query_img)
        fpn_out = self.fpn(OrderedDict([("p4", p4), ("p5", p5)]))
        feat = fpn_out["p4"]                                   # (B, 64, 14, 14)
        # Residual contribution from stride-8 P3 features for small objects
        p3_down = F.avg_pool2d(self.p3_lat(p3), kernel_size=2)  # 28x28 -> 14x14
        return feat + 0.5 * p3_down

    # ---- shared head -------------------------------------------------------

    def _head(
        self, prototype: torch.Tensor, q_feat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        b = q_feat.shape[0]
        proto = prototype if prototype.shape[0] == b else prototype.expand(b, -1)

        # Prototype-guided channel attention
        gate = torch.sigmoid(self.gate(proto))                # (B, 64)
        modulated = q_feat * gate.view(b, -1, 1, 1)           # (B, 64, 14, 14)

        # Dot-product spatial similarity map
        corr = (q_feat * proto.view(b, -1, 1, 1)).sum(dim=1, keepdim=True)

        reg, conf = self.det_head(torch.cat([modulated, corr], dim=1))

        fpn_gap = modulated.mean(dim=(2, 3))                  # (B, 64)
        presence_logit = self.presence_head(proto, fpn_gap)   # (B,)

        return {
            "reg": reg,
            "conf": conf,
            "presence_logit": presence_logit,
            "prototype": proto,
            "corr": corr,
        }

    # ---- inference path: precomputed prototype ----------------------------

    def detect(
        self, prototype: torch.Tensor, query_img: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Real-time inference: prototype precomputed, only query branch runs.

        Use ``compute_prototype()`` once per session, then call this for
        every camera frame.
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
        if prototype is None:
            proto, support_attn, teacher_desc = self.encode_support(
                support_imgs, support_bboxes
            )
        else:
            proto, support_attn, teacher_desc = prototype, None, None
        out = self._head(proto, self.encode_query(query_img))
        out["support_attn"] = support_attn        # type: ignore[assignment]
        out["teacher_desc"] = teacher_desc        # type: ignore[assignment]
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
