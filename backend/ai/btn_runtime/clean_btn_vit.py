"""
Self-contained Phase 12 CleanBTNVit architecture for VibeShift production inference.

Implements BatchTransformerNetwork + CleanBTNVit (strictly batch-independent output_VIT
pathway at 112x112) without hardcoded training-weight disk paths, because all 790
parameters (MobileFaceNet landmark branch, IR-50 backbone, multi-scale cross-attention,
and 2-layer ViT classification head) are stored inside clean_btn_vit_phase12.pt.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from backend.ai.btn_runtime.BatchTransformer import BatchTransformer
from backend.ai.btn_runtime.ir50 import Backbone
from backend.ai.btn_runtime.mobilefacenet import MobileFaceNet
from backend.ai.btn_runtime.vit_model import DropPath, PatchEmbed, VisionTransformer


def window_partition(x: torch.Tensor, window_size: int, h_w: int, w_w: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, h_w, window_size, w_w, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


class window(nn.Module):
    def __init__(self, window_size: int, dim: int):
        super().__init__()
        self.window_size = window_size
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = self.norm(x)
        shortcut = x
        h_w = H // self.window_size
        w_w = W // self.window_size
        x_windows = window_partition(x, self.window_size, h_w, w_w)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        return x_windows, shortcut


class WindowAttentionGlobal(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        qkv_bias: bool = True,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        window_size_tuple = (window_size, window_size)
        self.window_size = window_size_tuple
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size_tuple[0] - 1) * (2 * window_size_tuple[1] - 1),
                num_heads,
            )
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, q_global: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        B = q_global.shape[0]
        head_dim = C // self.num_heads
        B_dim = B_ // B
        kv = self.qkv(x).reshape(B_, N, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q_global = q_global.repeat(1, B_dim, 1, 1, 1)
        q = q_global.reshape(B_, self.num_heads, N, head_dim)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _to_channel_last(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1)


def _to_channel_first(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2)


def _to_query(x: torch.Tensor, N: int, num_heads: int, dim_head: int) -> torch.Tensor:
    B = x.shape[0]
    return x.reshape(B, 1, N, num_heads, dim_head).permute(0, 1, 3, 2, 4)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_reverse(
    windows: torch.Tensor, window_size: int, H: int, W: int, h_w: int, w_w: int
) -> torch.Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, h_w, w_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class feedforward(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale=None,
    ):
        super().__init__()
        if layer_scale is not None and isinstance(layer_scale, (int, float)):
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
        self.window_size = window_size
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, attn_windows: torch.Tensor, shortcut: torch.Tensor) -> torch.Tensor:
        B, H, W, C = shortcut.shape
        h_w = H // self.window_size
        w_w = W // self.window_size
        x = window_reverse(attn_windows, self.window_size, H, W, h_w, w_w)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm(x)))
        return x


class BatchTransformerNetwork(nn.Module):
    """
    Production-clean BatchTransformerNetwork matching Phase 12 checkpoint keys exactly,
    without loading separate MobileFaceNet or IR-50 training weights from disk.
    """

    def __init__(
        self,
        img_size: int = 224,
        num_classes: int = 7,
        window_size=(28, 14, 7),
        num_heads=(2, 4, 8),
        dims=(64, 128, 256),
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_heads = list(num_heads)
        self.dim_head = [dim // nh for nh, dim in zip(num_heads, dims)]
        self.num_classes = num_classes
        self.window_size = list(window_size)
        self.N = [win * win for win in window_size]

        self.face_landback = MobileFaceNet([112, 112], 136)
        for param in self.face_landback.parameters():
            param.requires_grad = False

        self.VIT = VisionTransformer(depth=2, embed_dim=embed_dim)
        self.ir_back = Backbone(50, 0.0, "ir")

        self.attn1 = WindowAttentionGlobal(dim=dims[0], num_heads=num_heads[0], window_size=window_size[0])
        self.attn2 = WindowAttentionGlobal(dim=dims[1], num_heads=num_heads[1], window_size=window_size[1])
        self.attn3 = WindowAttentionGlobal(dim=dims[2], num_heads=num_heads[2], window_size=window_size[2])
        self.window1 = window(window_size=window_size[0], dim=dims[0])
        self.window2 = window(window_size=window_size[1], dim=dims[1])
        self.window3 = window(window_size=window_size[2], dim=dims[2])
        self.conv1 = nn.Conv2d(in_channels=dims[0], out_channels=dims[0], kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=dims[1], out_channels=dims[1], kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=dims[2], out_channels=dims[2], kernel_size=3, stride=2, padding=1)

        dpr = [x.item() for x in torch.linspace(0, 0.5, 5)]
        self.ffn1 = feedforward(dim=dims[0], window_size=window_size[0], layer_scale=1e-5, drop_path=dpr[0])
        self.ffn2 = feedforward(dim=dims[1], window_size=window_size[1], layer_scale=1e-5, drop_path=dpr[1])
        self.ffn3 = feedforward(dim=dims[2], window_size=window_size[2], layer_scale=1e-5, drop_path=dpr[2])

        self.conv4 = nn.Conv2d(in_channels=dims[0], out_channels=dims[1], kernel_size=3, stride=2, padding=1)
        self.window4 = window(window_size=window_size[1], dim=dims[1])
        self.attn4 = WindowAttentionGlobal(dim=dims[1], num_heads=num_heads[1], window_size=window_size[1])
        self.ffn4 = feedforward(dim=dims[1], window_size=window_size[1], layer_scale=1e-5, drop_path=dpr[2])

        self.conv5 = nn.Conv2d(in_channels=dims[1], out_channels=dims[2], kernel_size=3, stride=2, padding=1)
        self.window5 = window(window_size=window_size[2], dim=dims[2])
        self.attn5 = WindowAttentionGlobal(dim=dims[2], num_heads=num_heads[2], window_size=window_size[2])
        self.ffn5 = feedforward(dim=dims[2], window_size=window_size[2], layer_scale=1e-5, drop_path=dpr[2])

        self.conv7 = nn.Conv2d(in_channels=dims[2], out_channels=dims[2], kernel_size=7, stride=7, padding=0)
        self.conv8 = nn.Conv2d(in_channels=dims[2], out_channels=7, kernel_size=1, stride=1, padding=0)
        self.LL7 = nn.Linear(dims[2], num_classes)
        self.last_face_conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        self.embed_q = nn.Sequential(
            nn.Conv2d(dims[0], 768, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1),
        )
        self.embed_k = nn.Sequential(
            nn.Conv2d(dims[1], 768, kernel_size=3, stride=2, padding=1),
        )
        self.embed_v = PatchEmbed(img_size=14, patch_size=14, in_c=256, embed_dim=768)
        self.attention = BatchTransformer(7)


class CleanBTNVit(nn.Module):
    """
    Wraps BatchTransformerNetwork and executes ONLY the strictly batch-independent
    output_VIT pathway at 112x112 resolution (identical to Phase 12 evaluation).
    """

    def __init__(self, btn_model: BatchTransformerNetwork | None = None, head_drop: float = 0.0):
        super().__init__()
        self.btn = btn_model if btn_model is not None else BatchTransformerNetwork()
        self.head_drop = nn.Dropout(p=head_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.btn
        x112 = (
            F.interpolate(x, size=112, mode="bilinear", align_corners=False)
            if x.shape[-1] != 112 or x.shape[-2] != 112
            else x
        )
        with torch.no_grad():
            x_face1, x_face2, x_face3 = m.face_landback(x112)
        x_face3 = m.last_face_conv(x_face3)
        x_face1 = _to_channel_last(x_face1)
        x_face2 = _to_channel_last(x_face2)
        x_face3 = _to_channel_last(x_face3)

        q1 = _to_query(x_face1, m.N[0], m.num_heads[0], m.dim_head[0])
        q2 = _to_query(x_face2, m.N[1], m.num_heads[1], m.dim_head[1])
        q3 = _to_query(x_face3, m.N[2], m.num_heads[2], m.dim_head[2])

        x_ir1, x_ir2, x_ir3 = m.ir_back(x112)
        F1, F2, F3 = m.conv1(x_ir1), m.conv2(x_ir2), m.conv3(x_ir3)
        x_w1, sc1 = m.window1(F1)
        x_w2, sc2 = m.window2(F2)
        x_w3, sc3 = m.window3(F3)

        S1, S2, S3 = m.attn1(x_w1, q1), m.attn2(x_w2, q2), m.attn3(x_w3, q3)
        S1, S2, S3 = m.ffn1(S1, sc1), m.ffn2(S2, sc2), m.ffn3(S3, sc3)
        S1, S2, S3 = _to_channel_first(S1), _to_channel_first(S2), _to_channel_first(S3)

        S1 = m.embed_q(S1).flatten(2).transpose(1, 2)
        S2 = m.embed_k(S2).flatten(2).transpose(1, 2)
        S3 = m.embed_v(S3)
        S = torch.cat([S1, S2, S3], dim=1)

        feat = m.VIT.forward_features(S)
        feat = m.VIT.se_block(feat)
        feat = self.head_drop(feat)
        return m.VIT.head(feat)
