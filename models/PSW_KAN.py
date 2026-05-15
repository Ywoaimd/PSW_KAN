import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.ChebyKANLayer import ChebyKANLinear
from modules.PatchLite import PatchEmbedding, AttentionLayer, Encoder, EncoderLayer
from modules.AdpWavelet import AdpWaveletBlock
# Local SE with a minimum bottleneck width for small-channel datasets.
class SEAttentionMin(nn.Module):
    def __init__(self, channel: int, ratio: int = 4, dropout_rate: float = 0.0, min_hidden: int = 4):
        super().__init__()
        hidden = max(channel // ratio, min_hidden)
        hidden = min(hidden, channel)
        layers = [
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
        ]
        if dropout_rate and dropout_rate > 0.0:
            layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Linear(hidden, channel, bias=False))
        self.fc = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N)
        b, c, _ = x.size()
        pooled = F.adaptive_avg_pool1d(x, 1).view(b, c)
        w = self.sigmoid(self.fc(pooled)).view(b, c, 1)
        return x * w

# PatchLite：内置 PatchTST 复制版，独立命名以区分外部实现


class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features, order):
        super().__init__()
        self.fc1 = ChebyKANLinear(
            in_features,
            out_features,
            order
        )

    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, -1).contiguous()
        return x


class FrequencyDecomp(nn.Module):

    def __init__(self, configs):
        super(FrequencyDecomp, self).__init__()
        self.configs = configs

    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high_res = self.frequency_interpolation(
                out_low.transpose(1, 2),
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers - i)),
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers - i - 1))
            ).transpose(1, 2)
            out_high_left = out_high - out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]
            out_level_list.append(out_high_left)
        out_level_list.reverse()
        return out_level_list

    def frequency_interpolation(self, x, seq_len, target_len):
        len_ratio = seq_len / target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros([x_fft.size(0), x_fft.size(1), target_len // 2 + 1], dtype=x_fft.dtype).to(x_fft.device)
        out_fft[:, :, :seq_len // 2 + 1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out


class FrequencyMixing(nn.Module):

    def __init__(self, configs):
        super(FrequencyMixing, self).__init__()
        self.configs = configs

        self.front_block = M_KAN(
            configs.d_model,
            self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers)),
            order=configs.begin_order
        )

        self.front_blocks = torch.nn.ModuleList([
            M_KAN(
                configs.d_model,
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers - i - 1)),
                order=i + configs.begin_order + 1
            )
            for i in range(configs.down_sampling_layers)
        ])

    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_low = self.front_block(out_low)
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high = self.front_blocks[i](out_high)
            out_high_res = self.frequency_interpolation(
                out_low.transpose(1, 2),
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers - i)),
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers - i - 1))
            ).transpose(1, 2)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]
            out_level_list.append(out_low)
        out_level_list.reverse()
        return out_level_list

    def frequency_interpolation(self, x, seq_len, target_len):
        len_ratio = seq_len / target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros([x_fft.size(0), x_fft.size(1), target_len // 2 + 1], dtype=x_fft.dtype).to(x_fft.device)
        out_fft[:, :, :seq_len // 2 + 1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out


class M_KAN(nn.Module):
    def __init__(self, d_model, seq_len, order):
        super().__init__()
        self.channel_mixer = nn.Sequential(
            ChebyKANLayer(d_model, d_model, order)
        )
        self.conv = BasicConv(d_model, d_model, kernel_size=3, degree=order, groups=d_model)

    def forward(self, x):
        x1 = self.channel_mixer(x)
        x2 = self.conv(x)
        out = x1 + x2
        return out


class BasicConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, degree, stride=1, padding=0, dilation=1, groups=1, act=False, bn=False, bias=False, dropout=0.):
        super(BasicConv, self).__init__()
        self.out_channels = c_out
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class PatchLiteBlock(nn.Module):
    """基于内部 PatchLite 的轻量分支：patch-embedding + 1层 encoder + 扁平预测头，输出 (B,pred_len,N)。"""
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.patch_len = int(getattr(configs, 'patch_len', 32))
        self.stride = int(getattr(configs, 'patch_stride', 12))
        padding = self.stride
        patch_dropout = float(getattr(configs, 'pyra_dropout', configs.dropout))

        self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, padding, patch_dropout)

        n_heads = int(getattr(configs, 'pyra_n_head', getattr(configs, 'n_heads', 4)))
        attn = AttentionLayer(
            __import__('modules.PatchLite.SelfAttention_Family', fromlist=['FullAttention']).FullAttention(
                False, getattr(configs, 'factor', 5), attention_dropout=patch_dropout, output_attention=False
            ),
            d_model=configs.d_model,
            n_heads=n_heads
        )
        enc_layer = EncoderLayer(attn, configs.d_model, getattr(configs, 'd_ff', configs.d_model * 2), dropout=patch_dropout, activation=configs.activation)
        self.encoder = Encoder([enc_layer], norm_layer=nn.LayerNorm(configs.d_model))

        self.head_nf = configs.d_model * int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.head_nf, configs.pred_len),
            nn.Dropout(getattr(configs, 'pyra_dropout', configs.dropout))
        )

    def forward(self, x_enc):
        # x_enc: (B,T,N)
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)
        enc_out, _ = self.encoder(enc_out)

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out).permute(0, 2, 1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
        return dec_out


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.res_blocks = nn.ModuleList([FrequencyDecomp(configs)
                                         for _ in range(configs.e_layers)])
        self.add_blocks = nn.ModuleList([FrequencyMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature

        self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.layer = configs.e_layers
        self.normalize_layers = torch.nn.ModuleList([
            Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
            for i in range(configs.down_sampling_layers + 1)
        ])
        self.projection_layer = nn.Linear(
            configs.d_model, 1, bias=True)
        self.predict_layer = nn.Linear(
            configs.seq_len,
            configs.pred_len,
        )

        # Decide whether to enable SE from dataset channel count (enc_in),
        # with use_se override. Avoid instantiation when disabled to keep RNG order.
        se_threshold = int(getattr(configs, 'se_param_threshold', 1))
        self.se_min_hidden = int(getattr(configs, 'se_min_hidden', 12))
        param_count = int(getattr(configs, 'enc_in', 0))
        default_se = param_count >= se_threshold
        cfg_flag = getattr(configs, 'use_se', None)
        self.se_enabled = (bool(cfg_flag) if cfg_flag is not None else bool(default_se))
        self.se_alpha_min = float(getattr(configs, 'se_alpha_min', 0.0))
        self.se_alpha_max = float(getattr(configs, 'se_alpha_max', 1.0))
        self.se_alpha_c0 = int(getattr(configs, 'se_alpha_c0', 8))
        self.se_alpha_c1 = int(getattr(configs, 'se_alpha_c1', 128))
        self.se_alpha_mode = str(getattr(configs, 'se_alpha_mode', 'power')).lower()
        self.se_alpha_pow = float(getattr(configs, 'se_alpha_pow', 2.0))
        if self.se_alpha_c1 <= self.se_alpha_c0:
            self.se_alpha_c1 = self.se_alpha_c0 + 1
        cfg_alpha = getattr(configs, 'se_alpha', None)
        if cfg_alpha is None:
            if param_count <= self.se_alpha_c0:
                se_alpha = self.se_alpha_min
            elif param_count >= self.se_alpha_c1:
                se_alpha = self.se_alpha_max
            else:
                if self.se_alpha_mode == 'log':
                    c = math.log1p(param_count)
                    c0 = math.log1p(self.se_alpha_c0)
                    c1 = math.log1p(self.se_alpha_c1)
                    t = (c - c0) / float(c1 - c0)
                else:
                    t = (param_count - self.se_alpha_c0) / float(self.se_alpha_c1 - self.se_alpha_c0)
                    if self.se_alpha_mode == 'power':
                        t = t ** self.se_alpha_pow
                    elif self.se_alpha_mode == 'sqrt':
                        t = math.sqrt(t)
                se_alpha = self.se_alpha_min + t * (self.se_alpha_max - self.se_alpha_min)
            self.se_alpha = float(max(self.se_alpha_min, min(self.se_alpha_max, se_alpha)))
        else:
            self.se_alpha = float(cfg_alpha)
        if self.se_alpha <= 0.0:
            self.se_enabled = False

        # SE：仅在启用时实例化，禁用时不创建模块以保持与无 SE 版本随机初始化一致
        if self.se_enabled:
            self.se_attention_layers = nn.ModuleList([
                SEAttentionMin(
                    channel=configs.d_model,
                    ratio=4,
                    dropout_rate=0.1,
                    min_hidden=self.se_min_hidden,
                )
                for _ in range(configs.e_layers)
            ])
            self.se_after_prediction = SEAttentionMin(
                channel=configs.d_model,
                ratio=4,
                dropout_rate=0.1,
                min_hidden=self.se_min_hidden,
            )
        else:
            self.se_attention_layers = None
            self.se_after_prediction = None

        # 顶层 AdpWavelet 微残差（当 pred_len != 720 时启用；禁用时不实例化以避免影响 RNG 序列）
        self.adpwave_enabled = (int(self.configs.pred_len) != 720)
        if self.adpwave_enabled:
            self.adpwave = AdpWaveletBlock(d_model=configs.d_model, reduction=4, kernel_sizes=(5, 3), dropout=0.0)
            self.adpwave_tau = float(getattr(configs, 'adpwave_tau', 0.15))
        else:
            self.adpwave = None
            self.adpwave_tau = 0.0

        # PatchLite 并联分支
        self.patch_alpha = float(getattr(configs, 'patch_alpha', 0.6))
        self.patch_block = PatchLiteBlock(configs)

    def forecast(self, x_enc):
        # 保留原始输入以供 PatchLite 分支
        x_enc_orig = x_enc

        # 多尺度处理
        x_enc_levels = self.__multi_level_process_inputs(x_enc)
        x_list = []
        for i, x in zip(range(len(x_enc_levels)), x_enc_levels):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        enc_out_list = []
        for i, x in zip(range(len(x_list)), x_list):
            enc_out = self.enc_embedding(x, None)
            enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.res_blocks[i](enc_out_list)
            enc_out_list = self.add_blocks[i](enc_out_list)

            # 每层后应用 SE（仅启用时）
            if self.se_enabled:
                for j in range(len(enc_out_list)):
                    BN, T_, d_model = enc_out_list[j].shape
                    N = self.configs.enc_in
                    B = BN // N
                    if BN % N != 0:
                        raise ValueError(f"Tensor size mismatch: BN ({BN}) is not divisible by N ({N})")

                    enc_out_reshaped = enc_out_list[j].reshape(B, N, T_, d_model)
                    enc_out_reshaped = enc_out_reshaped.permute(0, 2, 1, 3)
                    enc_out_for_attention = enc_out_reshaped.reshape(B * T_, N, d_model)
                    enc_out_for_attention = enc_out_for_attention.transpose(1, 2)

                    enc_out_attended = self.se_attention_layers[i](enc_out_for_attention)
                    if self.se_alpha != 1.0:
                        enc_out_attended = enc_out_for_attention + self.se_alpha * (enc_out_attended - enc_out_for_attention)

                    enc_out_attended = enc_out_attended.transpose(1, 2)
                    enc_out_attended = enc_out_attended.reshape(B, T_, N, d_model)
                    enc_out_attended = enc_out_attended.permute(0, 2, 1, 3).reshape(B * N, T_, d_model)

                    enc_out_list[j] = enc_out_attended

        # 顶层 AdpWavelet 微残差（在层内 SE 之后，预测之前；仅启用时应用）
        if self.adpwave_enabled:
            x_top = enc_out_list[0]
            x_top_enh = self.adpwave(x_top)
            enc_out_list[0] = (1.0 - self.adpwave_tau) * x_top + self.adpwave_tau * x_top_enh

        # 预测层（可选预测后 SE）
        dec_out = enc_out_list[0]
        dec_out = self.predict_layer(dec_out.permute(0, 2, 1)).permute(0, 2, 1)

        BN, T_, d_model = dec_out.shape
        N = self.configs.enc_in
        B = BN // N
        if BN % N != 0:
            raise ValueError(f"Tensor size mismatch: BN ({BN}) is not divisible by N ({N})")

        if self.se_enabled:
            dec_out_reshaped = dec_out.reshape(B, N, T_, d_model)
            dec_out_reshaped = dec_out_reshaped.permute(0, 2, 1, 3)
            dec_out_for_attention = dec_out_reshaped.reshape(B * T_, N, d_model)
            dec_out_for_attention = dec_out_for_attention.transpose(1, 2)

            dec_out_attended = self.se_after_prediction(dec_out_for_attention)
            if self.se_alpha != 1.0:
                dec_out_attended = dec_out_for_attention + self.se_alpha * (dec_out_attended - dec_out_for_attention)

            dec_out_attended = dec_out_attended.transpose(1, 2)
            dec_out_attended = dec_out_attended.reshape(B, T_, N, d_model)
            dec_out_attended = dec_out_attended.permute(0, 2, 1, 3).reshape(B * N, T_, d_model)

            dec_out = dec_out_attended

        # 投影并反标准化
        dec_out = self.projection_layer(dec_out).reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
        y_timekan = self.normalize_layers[0](dec_out, 'denorm')

        # PatchLite 辅助分支
        y_patch = self.patch_block(x_enc_orig)

        # 融合
        y = self.patch_alpha * y_timekan + (1.0 - self.patch_alpha) * y_patch
        return y

    def __multi_level_process_inputs(self, x_enc):
        down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
        x_enc = x_enc_sampling_list
        return x_enc

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out
        else:
            raise ValueError('Other tasks implemented yet')
