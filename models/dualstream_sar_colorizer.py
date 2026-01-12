"""
Dual-Stream Transformer SAR Colorization
MODEL SKELETON ONLY (NO TRAINING)

Contract-compliant with MODEL_CONTRACT.md v1.1

What this file guarantees:
- SAR-only inference
- Dual-stream architecture (Structure + Prior)
- Cross-attention fusion
- RGB + Confidence outputs in [0,1]
- Shape + range assertions
- 1-batch forward-pass verification
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Literal, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# Determinism (verification only)
# ======================================================
def set_deterministic(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================
# Config dataclasses
# ======================================================
@dataclass(frozen=True)
class IOConfig:
    H: int
    W: int
    sar_channels: int
    rgb_channels: int


@dataclass(frozen=True)
class ViTConfig:
    patch_size: int
    embed_dim: int
    num_heads: int
    mlp_ratio: float
    dropout: float


@dataclass(frozen=True)
class StreamAConfig:
    depth: int
    in_chans: int


@dataclass(frozen=True)
class PriorOffsetConfig:
    enabled: bool
    features: List[str]
    hidden_dim: int
    num_layers: int
    out_scale: float


@dataclass(frozen=True)
class StreamBPriorConfig:
    num_prior_tokens: int
    embed_dim: int
    offset: PriorOffsetConfig


@dataclass(frozen=True)
class FusionConfig:
    num_layers: int
    num_heads: int
    dropout: float


@dataclass(frozen=True)
class DecoderConfig:
    depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    up_stages: List[Dict[str, int]]


@dataclass(frozen=True)
class VerificationConfig:
    assert_shapes: bool
    assert_ranges: bool


@dataclass(frozen=True)
class ModelConfig:
    io: IOConfig
    vit: ViTConfig
    stream_a: StreamAConfig
    stream_b: StreamBPriorConfig
    fusion: FusionConfig
    decoder: DecoderConfig
    verification: VerificationConfig


# ======================================================
# Core building blocks
# ======================================================
class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)            # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)
        return x                   # (B, N, D)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x


# ======================================================
# Stream A — SAR Encoder
# ======================================================
class ViTEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ps = cfg.vit.patch_size
        H, W = cfg.io.H, cfg.io.W
        assert H % ps == 0 and W % ps == 0

        self.patch = PatchEmbed(cfg.stream_a.in_chans, cfg.vit.embed_dim, ps)
        self.num_tokens = (H // ps) * (W // ps)

        self.pos = nn.Parameter(torch.zeros(1, self.num_tokens, cfg.vit.embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                cfg.vit.embed_dim,
                cfg.vit.num_heads,
                cfg.vit.mlp_ratio,
                cfg.vit.dropout
            )
            for _ in range(cfg.stream_a.depth)
        ])
        self.norm = nn.LayerNorm(cfg.vit.embed_dim)

    def forward(self, x):
        x = self.patch(x) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ======================================================
# Stream B — Hybrid Prior Token Bank
# ======================================================
def compute_sar_stats(sar):
    flat = sar.flatten(1)
    return torch.stack([
        flat.mean(1),
        flat.std(1, unbiased=False),
        torch.quantile(flat, 0.1, 1),
        torch.quantile(flat, 0.5, 1),
        torch.quantile(flat, 0.9, 1),
    ], dim=1)


class PriorGenerator(nn.Module):
    def __init__(self, in_dim, hidden, layers, NP, D, scale):
        super().__init__()
        net = []
        d = in_dim
        for _ in range(layers - 1):
            net += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        net += [nn.Linear(d, NP * D)]
        self.net = nn.Sequential(*net)
        self.NP, self.D, self.scale = NP, D, scale

    def forward(self, stats):
        B = stats.size(0)
        out = self.net(stats).view(B, self.NP, self.D)
        return self.scale * out


class HybridPriorBank(nn.Module):
    def __init__(self, cfg: StreamBPriorConfig):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(1, cfg.num_prior_tokens, cfg.embed_dim))
        nn.init.trunc_normal_(self.base, std=0.02)

        off = cfg.offset
        self.gen = PriorGenerator(
            in_dim=len(off.features),
            hidden=off.hidden_dim,
            layers=off.num_layers,
            NP=cfg.num_prior_tokens,
            D=cfg.embed_dim,
            scale=off.out_scale
        )

    def forward(self, sar):
        B = sar.size(0)
        base = self.base.expand(B, -1, -1)
        stats = compute_sar_stats(sar)
        return base + self.gen(stats)


# ======================================================
# Fusion — Cross Attention
# ======================================================
class CrossAttention(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, q, kv):
        x = q + self.attn(self.ln_q(q), self.ln_kv(kv), self.ln_kv(kv))[0]
        x = x + self.mlp(self.ln2(x))
        return x


# ======================================================
# Decoder
# ======================================================
class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.vit.embed_dim
        self.blocks = nn.ModuleList([
            TransformerBlock(D, cfg.decoder.num_heads, cfg.decoder.mlp_ratio, cfg.decoder.dropout)
            for _ in range(cfg.decoder.depth)
        ])
        self.norm = nn.LayerNorm(D)

        self.ps = cfg.vit.patch_size
        self.H, self.W = cfg.io.H, cfg.io.W

        layers = []
        in_ch = D
        for st in cfg.decoder.up_stages:
            layers += [
                nn.Upsample(scale_factor=st["scale"], mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, st["out_channels"], 3, padding=1),
                nn.GELU(),
            ]
            in_ch = st["out_channels"]
        self.up = nn.Sequential(*layers)

        self.rgb = nn.Sequential(nn.Conv2d(in_ch, 3, 1), nn.Sigmoid())
        self.conf = nn.Sequential(nn.Conv2d(in_ch, 1, 1), nn.Sigmoid())

    def forward(self, z):
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)

        B, N, D = z.shape
        h = self.H // self.ps
        z = z.transpose(1, 2).view(B, D, h, h)
        z = self.up(z)
        if z.shape[-2:] != (self.H, self.W):
            z = F.interpolate(z, (self.H, self.W))
        return {"rgb": self.rgb(z), "conf": self.conf(z)}


# ======================================================
# Full Model
# ======================================================
class DualStreamSARColorizer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = ViTEncoder(cfg)
        self.prior = HybridPriorBank(cfg.stream_b)
        self.fusion = nn.ModuleList([
            CrossAttention(
                cfg.vit.embed_dim,
                cfg.fusion.num_heads,
                cfg.vit.mlp_ratio,
                cfg.fusion.dropout
            )
            for _ in range(cfg.fusion.num_layers)
        ])
        self.dec = Decoder(cfg)

    def forward(self, sar, *, mode: Literal["train", "infer"]):
        if self.cfg.verification.assert_shapes:
            assert sar.shape[1:] == (1, self.cfg.io.H, self.cfg.io.W)

        za = self.enc(sar)
        zb = self.prior(sar)

        for blk in self.fusion:
            za = blk(za, zb)

        out = self.dec(za)

        if self.cfg.verification.assert_ranges:
            assert out["rgb"].min() >= 0 and out["rgb"].max() <= 1
            assert out["conf"].min() >= 0 and out["conf"].max() <= 1

        return out


# ======================================================
# 1-batch forward verification
# ======================================================
if __name__ == "__main__":
    set_deterministic()

    cfg = ModelConfig(
        io=IOConfig(256, 256, 1, 3),
        vit=ViTConfig(16, 512, 8, 4.0, 0.1),
        stream_a=StreamAConfig(12, 1),
        stream_b=StreamBPriorConfig(
            256, 512,
            PriorOffsetConfig(True, ["mean", "std", "p10", "p50", "p90"], 256, 2, 0.1)
        ),
        fusion=FusionConfig(2, 8, 0.1),
        decoder=DecoderConfig(
            6, 8, 4.0, 0.1,
            [{"scale": 2, "out_channels": 256},
             {"scale": 2, "out_channels": 128},
             {"scale": 2, "out_channels": 64},
             {"scale": 2, "out_channels": 32}]
        ),
        verification=VerificationConfig(True, True),
    )

    model = DualStreamSARColorizer(cfg).eval()
    sar = torch.rand(2, 1, 256, 256)
    out = model(sar, mode="infer")

    print("RGB:", out["rgb"].shape)
    print("CONF:", out["conf"].shape)
    print("✅ Forward pass OK")
