import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

sys.path.insert(0, os.path.abspath("."))

from models import mavss_sr_arch


torch.backends.cudnn.benchmark = True


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def measure_runtime_memory(model, x, warmup=10, runs=50):
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(runs):
            _ = model(x)
        end.record()

        torch.cuda.synchronize()

    avg_time_ms = start.elapsed_time(end) / runs
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return avg_time_ms, peak_mem_mb


class WindowAttention2D(nn.Module):
    """
    A 2D window-based self-attention module used to replace VSSM
    in the high-frequency branch while keeping the remaining
    MAVSS-SR architecture unchanged.

    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(self, dim, window_size=16, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.attn = hat_arch.WindowAttention(
            dim=dim,
            window_size=hat_arch.to_2tuple(window_size),
            num_heads=num_heads,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0
        )

        relative_position_index = self.calculate_rpi()
        self.register_buffer("relative_position_index", relative_position_index)

    def calculate_rpi(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1

        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def forward(self, x):
        b, c, h, w = x.shape
        ws = self.window_size

        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        _, _, hp, wp = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()
        x_windows = hat_arch.window_partition(x, ws)
        x_windows = x_windows.view(-1, ws * ws, c)

        attn_windows = self.attn(
            x_windows,
            rpi=self.relative_position_index,
            mask=None
        )

        attn_windows = attn_windows.view(-1, ws, ws, c)
        x = hat_arch.window_reverse(attn_windows, ws, hp, wp)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x[:, :, :h, :w]


def build_mavss_model():
    model = mavss_sr_arch.MAVSSSR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=256,
        num_heads=[8, 8, 8, 8, 8, 8],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv"
    )
    return model


def evaluate_model(name, model, x):
    model = model.cuda()
    model.eval()

    with torch.inference_mode():
        y = model(x)

    params_m = count_params(model)

    try:
        flops, _ = profile(model, inputs=(x,), verbose=False)
        flops_g = flops / 1e9
    except Exception as e:
        flops_g = None
        print(f"{name} FLOPs calculation failed:", repr(e))

    runtime_ms, memory_mb = measure_runtime_memory(model, x, warmup=10, runs=50)

    print(f"========== {name} ==========")
    print(f"Device:      {torch.cuda.get_device_name(0)}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape:{tuple(y.shape)}")
    print("--------------------------------------------")
    print(f"Params:      {params_m:.3f} M")
    if flops_g is not None:
        print(f"FLOPs:       {flops_g:.3f} G")
    else:
        print("FLOPs:       N/A")
    print(f"Runtime:     {runtime_ms:.3f} ms")
    print(f"Peak Memory: {memory_mb:.2f} MB")
    print("============================================")

    del model
    torch.cuda.empty_cache()


def main():
    device = "cuda"
    x = torch.randn(1, 3, 64, 64).to(device)

    # 1. Original MAVSS-SR with VSSM
    original_vssm = hat_arch.VSSM
    model_vssm = build_mavss_model()
    evaluate_model("MAVSS-SR w/ VSSM", model_vssm, x)

    # 2. Attention-based variant:
    # Replace only the VSSM module in the high-frequency branch
    # with window-based self-attention.
    hat_arch.VSSM = WindowAttention2D
    model_attn = build_mavss_model()
    evaluate_model("MAVSS-SR w/ Window Attention", model_attn, x)

    # Restore original VSSM definition
    hat_arch.VSSM = original_vssm


if __name__ == "__main__":
    main()
