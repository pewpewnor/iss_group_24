"""Few-shot localization model.

Backbone: MobileNetV3-Large. The original spec is written against MobileNetV2;
V3-Large changes the channel widths and tap indices, but the freeze policy
(everything before the first stride-16 block) and the rest of the pipeline
are identical.

Pipeline (per episode, batch dim B):
  support_imgs   (B, 5, 3, 224, 224)  --backbone-> P5: (B*5, 160, 7, 7)
                                      --roi_align-> (B*5, 160, 7, 7) cropped
                                      --proj 1x1 + GAP-> (B*5, 64)
                                      --mean over 5-> prototype (B, 64)
  query_img      (B, 3, 224, 224)     --backbone-> P4 (B,112,14,14), P5 (B,160,7,7)
                                      --FPN-> fpn_out (B, 64, 14, 14)
  prototype + fpn_out --xcorr--> corr_map (B, 1, 14, 14)
  concat(corr_map, fpn_out) -> det head -> (reg: B,4,14,14) + (conf: B,1,14,14)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.ops.roi_align import roi_align

IMG_SIZE = 224
GRID = 14
STRIDE = IMG_SIZE // GRID  # = 16
P5_STRIDE = 32             # features[15] downsamples by 32
PROTO_DIM = 64
FPN_DIM = 64

P4_CHANNELS = 112
P5_CHANNELS = 160


class MobileNetBackbone(nn.Module):
    """MobileNetV3-Large backbone exposing P4 (features[12], 14x14x112) and
    P5 (features[15], 7x7x160). features[16] (the 1x1 expansion to 960) is
    skipped to keep the lateral conv small."""

    P4_IDX = 12
    P5_IDX = 15

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = mobilenet_v3_large(weights=weights)
        self.features = m.features

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p4: torch.Tensor | None = None
        p5: torch.Tensor | None = None
        for i, block in enumerate(self.features):
            x = block(x)
            if i == self.P4_IDX:
                p4 = x
            elif i == self.P5_IDX:
                p5 = x
                break
        assert p4 is not None and p5 is not None
        return p4, p5

    def forward_p5_only(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.features):
            x = block(x)
            if i == self.P5_IDX:
                return x
        raise RuntimeError("unreachable")

    def freeze_lower(self, freeze_idx_exclusive: int = 7) -> None:
        for i, block in enumerate(self.features):
            # features beyond P5_IDX are never run in forward; keep them frozen.
            req = freeze_idx_exclusive <= i <= self.P5_IDX
            for p in block.parameters():
                p.requires_grad = req

    def unfreeze_all(self) -> None:
        for i, block in enumerate(self.features):
            req = i <= self.P5_IDX
            for p in block.parameters():
                p.requires_grad = req

    def freeze_all(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False


class Projection(nn.Module):
    """1x1 conv (P5_C -> 64) + GAP. Produces a 64-d descriptor per support crop."""

    def __init__(self, in_c: int = P5_CHANNELS, out_c: int = PROTO_DIM) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.flatten(1)


class FPN(nn.Module):
    """Top-down fusion of P4 and P5 into a 14x14x64 map."""

    def __init__(self) -> None:
        super().__init__()
        self.lat_p4 = nn.Conv2d(P4_CHANNELS, FPN_DIM, kernel_size=1)
        self.lat_p5 = nn.Conv2d(P5_CHANNELS, FPN_DIM, kernel_size=1)
        self.out = nn.Conv2d(FPN_DIM, FPN_DIM, kernel_size=3, padding=1)

    def forward(self, p4: torch.Tensor, p5: torch.Tensor) -> torch.Tensor:
        lat5 = self.lat_p5(p5)
        lat4 = self.lat_p4(p4)
        up5 = F.interpolate(lat5, scale_factor=2, mode="nearest")
        return self.out(lat4 + up5)


class _DWSep(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class DetectionHead(nn.Module):
    """Three DW-separable blocks + parallel reg/conf 1x1 convs."""

    def __init__(self) -> None:
        super().__init__()
        self.block1 = _DWSep(FPN_DIM + 1, FPN_DIM)
        self.block2 = _DWSep(FPN_DIM, FPN_DIM)
        self.block3 = _DWSep(FPN_DIM, FPN_DIM)
        self.reg = nn.Conv2d(FPN_DIM, 4, 1)
        self.conf = nn.Conv2d(FPN_DIM, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.block3(self.block2(self.block1(x)))
        return self.reg(x), self.conf(x)


def cross_correlate(prototype: torch.Tensor, query_feat: torch.Tensor) -> torch.Tensor:
    """Inner product of prototype with each spatial location. Output: (B, 1, H, W).

    The spec frames this as a grouped 1x1 conv; the equivalent dot-product over
    the channel dim is faster, numerically identical, and avoids reshape gymnastics.
    """
    b, c, h, w = query_feat.shape
    assert prototype.shape == (b, c), (prototype.shape, query_feat.shape)
    return (query_feat * prototype.view(b, c, 1, 1)).sum(dim=1, keepdim=True)


class FewShotLocalizer(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = MobileNetBackbone(pretrained=pretrained)
        self.projection = Projection()
        self.fpn = FPN()
        self.det_head = DetectionHead()

    # ---- support branch ----------------------------------------------------

    def encode_support(
        self, support_imgs: torch.Tensor, support_bboxes: torch.Tensor
    ) -> torch.Tensor:
        """support_imgs: (B, K, 3, 224, 224); support_bboxes: (B, K, 4) in 224 coords.

        Returns prototype (B, PROTO_DIM)."""
        b, k = support_imgs.shape[:2]
        flat = support_imgs.reshape(b * k, 3, IMG_SIZE, IMG_SIZE)
        feat = self.backbone.forward_p5_only(flat)  # (B*K, 160, 7, 7)
        boxes = support_bboxes.reshape(b * k, 4).to(feat.dtype)
        batch_idx = torch.arange(b * k, device=feat.device, dtype=feat.dtype).unsqueeze(1)
        rois = torch.cat([batch_idx, boxes], dim=1)
        cropped = roi_align(  # pyright: ignore[reportCallIssue]
            feat, rois, output_size=(7, 7), spatial_scale=1.0 / P5_STRIDE, aligned=True
        )
        descs = self.projection(cropped).view(b, k, -1)
        return descs.mean(dim=1)

    # ---- query branch ------------------------------------------------------

    def encode_query(self, query_img: torch.Tensor) -> torch.Tensor:
        p4, p5 = self.backbone(query_img)
        return self.fpn(p4, p5)  # (B, 64, 14, 14)

    # ---- full forward ------------------------------------------------------

    def forward(
        self,
        support_imgs: torch.Tensor,
        support_bboxes: torch.Tensor,
        query_img: torch.Tensor,
        prototype: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if prototype is None:
            prototype = self.encode_support(support_imgs, support_bboxes)
        q_feat = self.encode_query(query_img)
        corr = cross_correlate(prototype, q_feat)
        head_in = torch.cat([corr, q_feat], dim=1)
        reg, conf = self.det_head(head_in)
        return {"reg": reg, "conf": conf, "prototype": prototype, "corr": corr}


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def decode(
    reg: torch.Tensor, conf: torch.Tensor, stride: int = STRIDE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take per-cell predictions, return (bbox xyxy in 224 coords (B,4), score (B,))."""
    b, _, h, w = reg.shape
    score = torch.sigmoid(conf)  # (B, 1, H, W)
    flat_score = score.view(b, -1)
    best = flat_score.argmax(dim=1)  # (B,)
    cy = (best // w).long()
    cx = (best % w).long()
    idx = torch.arange(b, device=reg.device)

    dx = reg[idx, 0, cy, cx]
    dy = reg[idx, 1, cy, cx]
    dw = reg[idx, 2, cy, cx]
    dh = reg[idx, 3, cy, cx]

    cx_abs = (cx.float() + dx) * stride
    cy_abs = (cy.float() + dy) * stride
    w_abs = torch.exp(dw) * stride
    h_abs = torch.exp(dh) * stride

    bbox = torch.stack(
        [cx_abs - w_abs / 2, cy_abs - h_abs / 2, cx_abs + w_abs / 2, cy_abs + h_abs / 2], dim=1
    )
    return bbox, flat_score.max(dim=1).values
