from __future__ import annotations

import os
import sys
from typing import Any, Dict

import torch
import yaml

# Ensure repo root is on PYTHONPATH so "models.*" imports work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.dualstream_sar_colorizer import (  # noqa: E402
    DualStreamSARColorizer,
    ModelConfig,
    IOConfig,
    ViTConfig,
    StreamAConfig,
    StreamBPriorConfig,
    PriorOffsetConfig,
    FusionConfig,
    DecoderConfig,
    VerificationConfig,
    set_deterministic,
)


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cfg_from_yaml_dict(d: Dict[str, Any]) -> ModelConfig:
    # --- IO ---
    H = int(d["io"]["image_size"]["H"])
    W = int(d["io"]["image_size"]["W"])
    io = IOConfig(
        H=H,
        W=W,
        sar_channels=int(d["io"]["sar_channels"]),
        rgb_channels=int(d["io"]["rgb_channels"]),
    )

    # --- ViT ---
    vit = ViTConfig(
        patch_size=int(d["vit"]["patch_size"]),
        embed_dim=int(d["vit"]["embed_dim"]),
        num_heads=int(d["vit"]["num_heads"]),
        mlp_ratio=float(d["vit"]["mlp_ratio"]),
        dropout=float(d["vit"]["dropout"]),
    )

    # --- Stream A ---
    stream_a = StreamAConfig(
        depth=int(d["stream_a"]["depth"]),
        in_chans=int(d["stream_a"]["in_chans"]),
    )

    # --- Stream B prior ---
    off = d["stream_b_prior"]["conditional_offset"]
    offset = PriorOffsetConfig(
        enabled=bool(off["enabled"]),
        features=list(off["features"]),
        hidden_dim=int(off["hidden_dim"]),
        num_layers=int(off["num_layers"]),
        out_scale=float(off["out_scale"]),
    )

    stream_b = StreamBPriorConfig(
        num_prior_tokens=int(d["stream_b_prior"]["num_prior_tokens"]),
        embed_dim=int(d["stream_b_prior"]["embed_dim"]),
        offset=offset,
    )

    # --- Fusion ---
    fusion = FusionConfig(
        num_layers=int(d["fusion"]["num_layers"]),
        num_heads=int(d["fusion"]["num_heads"]),
        dropout=float(d["fusion"]["dropout"]),
    )

    # --- Decoder ---
    up_stages = list(d["decoder"]["upsample_head"]["stages"])
    decoder = DecoderConfig(
        depth=int(d["decoder"]["depth"]),
        num_heads=int(d["decoder"]["num_heads"]),
        mlp_ratio=float(d["decoder"]["mlp_ratio"]),
        dropout=float(d["decoder"]["dropout"]),
        up_stages=up_stages,
    )

    # --- Verification ---
    verification = VerificationConfig(
        assert_shapes=bool(d["verification"]["assert_shapes"]),
        assert_ranges=bool(d["verification"]["assert_ranges"]),
    )

    # --- Safety checks tied to contract ---
    ps = vit.patch_size
    if H % ps != 0 or W % ps != 0:
        raise ValueError(f"Image size {(H,W)} must be divisible by patch_size={ps}")
    num_tokens = (H // ps) * (W // ps)
    if int(d["stream_b_prior"]["num_prior_tokens"]) != num_tokens:
        raise ValueError(
            f"num_prior_tokens must match patch tokens. "
            f"Expected {num_tokens}, got {d['stream_b_prior']['num_prior_tokens']}"
        )

    return ModelConfig(
        io=io,
        vit=vit,
        stream_a=stream_a,
        stream_b=stream_b,
        fusion=fusion,
        decoder=decoder,
        verification=verification,
    )


def assert_in_01(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} has NaN/Inf")
    mn = x.min().item()
    mx = x.max().item()
    if mn < -1e-6 or mx > 1.0 + 1e-6:
        raise ValueError(f"{name} out of [0,1]: min={mn:.6f}, max={mx:.6f}")


def main() -> None:
    set_deterministic(1234)

    cfg_path = os.path.join(REPO_ROOT, "configs", "model_v1_1.yaml")
    d = load_yaml(cfg_path)
    cfg = cfg_from_yaml_dict(d)

    model = DualStreamSARColorizer(cfg).eval()

    B = 2
    sar = torch.rand(B, 1, cfg.io.H, cfg.io.W, dtype=torch.float32)

    with torch.no_grad():
        out = model(sar, mode="infer")

    rgb = out["rgb"]
    conf = out["conf"]

    print("rgb:", rgb.shape, rgb.dtype, (rgb.min().item(), rgb.max().item()))
    print("conf:", conf.shape, conf.dtype, (conf.min().item(), conf.max().item()))

    # Contract assertions
    assert rgb.shape == (B, 3, cfg.io.H, cfg.io.W)
    assert conf.shape == (B, 1, cfg.io.H, cfg.io.W)
    assert_in_01(rgb, "rgb")
    assert_in_01(conf, "conf")

    print("✅ YAML-driven 1-batch forward verification PASSED.")


if __name__ == "__main__":
    main()
