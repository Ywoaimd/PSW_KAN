import torch
import torch.nn as nn
import torch.nn.functional as F


class AdpWaveletBlock(nn.Module):
    """轻量自适应小波样块（1D，时序专用）
    输入/输出: (B*N, T, d_model)
    结构: depthwise conv 抽取低/高频 + 1x1 混合 + 通道门控 + LayerNorm
    """

    def __init__(self, d_model: int, reduction: int = 4, kernel_sizes=(5, 3), dropout: float = 0.0):
        super().__init__()
        k_low, k_high = kernel_sizes
        pad_low = k_low // 2
        pad_high = k_high // 2

        # depthwise: 保持每通道独立的小波近似滤波
        self.low = nn.Conv1d(d_model, d_model, kernel_size=k_low, padding=pad_low, groups=d_model, bias=False)
        self.high = nn.Conv1d(d_model, d_model, kernel_size=k_high, padding=pad_high, groups=d_model, bias=False)

        # 1x1 混合将 [low, high] 融合回 d_model
        self.mix = nn.Conv1d(2 * d_model, d_model, kernel_size=1, bias=True)

        # 通道自适应门控（Squeeze-Excite）
        hidden = max(1, d_model // reduction)
        self.se_fc1 = nn.Linear(d_model, hidden)
        self.se_fc2 = nn.Linear(hidden, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BN, T, C)
        x_in = x
        x = x.transpose(1, 2)  # (BN, C, T)

        low = self.low(x)
        # 高频分量对输入做轻滤波，避免与低频完全重叠
        high = self.high(x - low)

        feat = torch.cat([low, high], dim=1)  # (BN, 2C, T)
        feat = self.mix(feat)  # (BN, C, T)

        # 通道门控（全局平均池化 over T）
        gap = feat.mean(dim=2)  # (BN, C)
        gate = torch.sigmoid(self.se_fc2(F.gelu(self.se_fc1(gap))))  # (BN, C)
        feat = feat * gate.unsqueeze(-1)

        feat = feat.transpose(1, 2)  # (BN, T, C)
        feat = self.dropout(feat)
        out = self.norm(x_in + feat)
        return out


