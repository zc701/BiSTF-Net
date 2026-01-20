import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import mne
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
from einops import rearrange


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 mlp_dwconv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = DWConv(hidden_features) if mlp_dwconv else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        B, N, C = x.shape
        try:
            H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.dwconv(x)
            x = x.flatten(2).transpose(1, 2)
        except:
            pass
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BiFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_dwconv=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       mlp_dwconv=mlp_dwconv)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SCAF_Module(nn.Module):
    def __init__(self, input_dim, d_model, heads, drop):
        super().__init__()
        self.eeg_proj = nn.Linear(input_dim, d_model)
        self.fnirs_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, d_model))
        self.biformer_block = BiFormerBlock(dim=d_model, num_heads=heads, mlp_ratio=2.0, qkv_bias=True, drop=drop,
                                            attn_drop=drop)
        self.norm_eeg = nn.LayerNorm(d_model)
        self.norm_fnirs = nn.LayerNorm(d_model)
        self.out_proj_eeg = nn.Linear(d_model, d_model)
        self.out_proj_fnirs = nn.Linear(d_model, d_model)

    def forward(self, eeg, fnirs):
        eeg_p = self.norm_eeg(self.eeg_proj(eeg))
        fnirs_p = self.norm_fnirs(self.fnirs_proj(fnirs))
        q_eeg = eeg_p.unsqueeze(1)
        k_fnirs = fnirs_p.unsqueeze(1)
        combined_features = torch.cat([q_eeg, k_fnirs], dim=1)
        combined_features = combined_features + self.pos_embedding
        output = self.biformer_block(combined_features)
        eeg_output_fused = self.out_proj_eeg(output[:, 0, :])
        fnirs_output_fused = self.out_proj_fnirs(output[:, 1, :])
        ef_loss = F.mse_loss(eeg_output_fused, fnirs_output_fused)
        fusion_feature = eeg_output_fused + fnirs_output_fused
        return fusion_feature, None, None, ef_loss


class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        mean_x = torch.mean(x, dim=1, keepdim=True)
        mean_y = torch.mean(y, dim=1, keepdim=True)
        xm = x - mean_x
        ym = y - mean_y
        r_num = torch.sum(xm * ym, dim=1)
        r_den = torch.sqrt(torch.sum(xm ** 2, dim=1) * torch.sum(ym ** 2, dim=1))
        r = r_num / (r_den + 1e-8)
        return -torch.mean(r)


def calculate_padding(kernel_size, stride, dilation=1):
    return ((stride - 1) * dilation - stride + kernel_size) // 2


class BiCMG_Module(nn.Module):
    def __init__(self, eeg_dim, fnirs_dim, n_heads=4, head_dim=16, dropout=0.1, downsample_factor=4):
        super().__init__()
        inner_dim = n_heads * head_dim
        self.n_heads = n_heads
        self.scale = head_dim ** -0.5
        self.downsample_factor = downsample_factor
        self.norm_eeg = nn.LayerNorm(eeg_dim)
        self.norm_fnirs_for_eeg = nn.LayerNorm(fnirs_dim)
        self.to_q_eeg = nn.Linear(eeg_dim, inner_dim, bias=False)
        self.to_kv_fnirs = nn.Linear(fnirs_dim, inner_dim * 2, bias=False)
        self.to_out_eeg = nn.Sequential(nn.Linear(inner_dim, eeg_dim), nn.Dropout(dropout))
        self.norm_fnirs = nn.LayerNorm(fnirs_dim)
        self.norm_eeg_for_fnirs = nn.LayerNorm(eeg_dim)
        self.to_q_fnirs = nn.Linear(fnirs_dim, inner_dim, bias=False)
        self.to_kv_eeg = nn.Linear(eeg_dim, inner_dim * 2, bias=False)
        self.to_out_fnirs = nn.Sequential(nn.Linear(inner_dim, fnirs_dim), nn.Dropout(dropout))

    def forward(self, eeg_feat, fnirs_feat):
        B, C_e, D, H, W = eeg_feat.shape
        if eeg_feat.shape[2:] != fnirs_feat.shape[2:]:
            fnirs_feat = F.interpolate(fnirs_feat, size=(D, H, W), mode='trilinear', align_corners=False)
        pooled_d, pooled_h, pooled_w = max(D // self.downsample_factor, 1), max(H // self.downsample_factor, 1), max(
            W // self.downsample_factor, 1)
        eeg_proxy = F.adaptive_avg_pool3d(eeg_feat, (pooled_d, pooled_h, pooled_w))
        fnirs_proxy = F.adaptive_avg_pool3d(fnirs_feat, (pooled_d, pooled_h, pooled_w))
        eeg_seq_proxy = rearrange(eeg_proxy, 'b c d h w -> b (d h w) c')
        fnirs_seq_proxy = rearrange(fnirs_proxy, 'b c d h w -> b (d h w) c')
        eeg_norm = self.norm_eeg(eeg_seq_proxy)
        fnirs_norm_eeg = self.norm_fnirs_for_eeg(fnirs_seq_proxy)
        q_eeg = self.to_q_eeg(eeg_norm)
        k_fnirs, v_fnirs = self.to_kv_fnirs(fnirs_norm_eeg).chunk(2, dim=-1)
        q = rearrange(q_eeg, 'b n (h d) -> b h n d', h=self.n_heads)
        k = rearrange(k_fnirs, 'b n (h d) -> b h n d', h=self.n_heads)
        v = rearrange(v_fnirs, 'b n (h d) -> b h n d', h=self.n_heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        eeg_update_low_res = self.to_out_eeg(out)
        eeg_update_low_res_3d = rearrange(eeg_update_low_res, 'b (d h w) c -> b c d h w', d=pooled_d, h=pooled_h,
                                          w=pooled_w)
        eeg_update = F.interpolate(eeg_update_low_res_3d, size=(D, H, W), mode='trilinear', align_corners=False)
        eeg_enhanced = eeg_feat + eeg_update
        fnirs_norm = self.norm_fnirs(fnirs_seq_proxy)
        eeg_norm_fnirs = self.norm_eeg_for_fnirs(eeg_seq_proxy)
        q_fnirs = self.to_q_fnirs(fnirs_norm)
        k_eeg, v_eeg = self.to_kv_eeg(eeg_norm_fnirs).chunk(2, dim=-1)
        q = rearrange(q_fnirs, 'b n (h d) -> b h n d', h=self.n_heads)
        k = rearrange(k_eeg, 'b n (h d) -> b h n d', h=self.n_heads)
        v = rearrange(v_eeg, 'b n (h d) -> b h n d', h=self.n_heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        fnirs_update_low_res = self.to_out_fnirs(out)
        fnirs_update_low_res_3d = rearrange(fnirs_update_low_res, 'b (d h w) c -> b c d h w', d=pooled_d, h=pooled_h,
                                            w=pooled_w)
        fnirs_update = F.interpolate(fnirs_update_low_res_3d, size=(D, H, W), mode='trilinear', align_corners=False)
        fnirs_enhanced = fnirs_feat + fnirs_update
        bicmg_loss = torch.tensor(0.0, device=eeg_feat.device)
        return eeg_enhanced, fnirs_enhanced, bicmg_loss


class ConvBlock(nn.Module):
    def __init__(self, fnirs_filter, eegfusion_filter, fnirs_size, eegfusion_size, fnirs_stride, eegfusion_stride,
                 tem_kernel_size, in_channels=None):
        super().__init__()
        fnirs_padding = (
            calculate_padding(fnirs_size[0], fnirs_stride[0]), calculate_padding(fnirs_size[1], fnirs_stride[1]),
            calculate_padding(fnirs_size[2], fnirs_stride[2]))
        eegfusion_padding = (calculate_padding(eegfusion_size[0], eegfusion_stride[0]),
                             calculate_padding(eegfusion_size[1], eegfusion_stride[1]),
                             calculate_padding(eegfusion_size[2], eegfusion_stride[2]))
        self.fnirs_conv = nn.Sequential(
            nn.Conv3d(in_channels['fnirs'], fnirs_filter, kernel_size=fnirs_size, stride=fnirs_stride,
                      padding=fnirs_padding), nn.BatchNorm3d(fnirs_filter), nn.ELU())
        self.eegfusion_conv = nn.Sequential(
            nn.Conv3d(in_channels['eegfusion'], eegfusion_filter, kernel_size=eegfusion_size, stride=eegfusion_stride,
                      padding=eegfusion_padding), nn.BatchNorm3d(eegfusion_filter), nn.ELU())
        downsample = 4 if fnirs_filter == 8 else 2
        self.bicmg = BiCMG_Module(eeg_dim=eegfusion_filter, fnirs_dim=fnirs_filter, n_heads=4, head_dim=16, dropout=0.1,
                                  downsample_factor=downsample)
        self.dropout = nn.Dropout(0.5)

    def forward(self, eegfusion_input, fnirs_input):
        fnirs_feature = self.fnirs_conv(fnirs_input)
        eegfusion_feature_conv = self.eegfusion_conv(eegfusion_input)
        eegfusion_guided, fnirs_guided, bicmg_loss = self.bicmg(eegfusion_feature_conv, fnirs_feature)
        eegfusion_guided = self.dropout(eegfusion_guided)
        fnirs_guided = self.dropout(fnirs_guided)
        return eegfusion_guided, fnirs_guided, bicmg_loss


class ATA_Module(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, n_heads=8, num_gru_layers=2, dropout=0.1):
        super().__init__()
        self.in_channels, self.hidden_dim, self.n_heads = in_channels, hidden_dim, n_heads
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_dim, num_layers=num_gru_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if num_gru_layers > 1 else 0)
        self.gru_output_dim = hidden_dim * 2
        self.norm1, self.dropout1 = nn.LayerNorm(self.gru_output_dim), nn.Dropout(dropout)
        self.q_proj = nn.Linear(in_channels, self.gru_output_dim)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=self.gru_output_dim, num_heads=n_heads,
                                                          dropout=dropout, batch_first=True)
        self.norm2, self.dropout2 = nn.LayerNorm(self.gru_output_dim), nn.Dropout(dropout)
        self.pcc_loss = PearsonCorrelationLoss()

    def forward(self, eeg_feat, fnirs_feat_sequence):
        B, C, D_fnirs, H, W = fnirs_feat_sequence.shape
        eeg_pooled = F.adaptive_avg_pool3d(eeg_feat, 1).view(B, C)
        fnirs_pooled_sequence = F.adaptive_avg_pool2d(
            fnirs_feat_sequence.permute(0, 2, 1, 3, 4).reshape(B * D_fnirs, C, H, W), 1).view(B, D_fnirs, C)
        gru_outputs, _ = self.gru(fnirs_pooled_sequence)
        x = self.dropout1(self.norm1(gru_outputs))
        query = self.q_proj(eeg_pooled).unsqueeze(1)
        attn_output, attn_weights = self.multi_head_attention(query, x, x)
        aligned_fnirs = self.norm2(query + self.dropout2(attn_output)).squeeze(1)
        ata_loss = self.pcc_loss(query.squeeze(1), aligned_fnirs)
        return aligned_fnirs, ata_loss


class BiSTFNet(nn.Module):
    def __init__(self, verbose=False):
        super().__init__()
        self.conv1 = ConvBlock(
            fnirs_filter=8, eegfusion_filter=8,
            fnirs_size=(2, 2, 5), eegfusion_size=(2, 2, 13),
            fnirs_stride=(2, 2, 2), eegfusion_stride=(2, 2, 6),
            tem_kernel_size=5,
            in_channels={'fnirs': 22, 'eegfusion': 1}
        )
        self.conv2 = ConvBlock(
            fnirs_filter=16, eegfusion_filter=16,
            fnirs_size=(2, 2, 3), eegfusion_size=(2, 2, 5),
            fnirs_stride=(2, 2, 2), eegfusion_stride=(2, 2, 2),
            tem_kernel_size=3,
            in_channels={'fnirs': 8, 'eegfusion': 8}
        )
        self.ata = ATA_Module(in_channels=16, hidden_dim=128, n_heads=8, num_gru_layers=2, dropout=0.1)
        ata_output_dim = 128 * 2
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.eeg_proj_for_scaf = nn.Linear(16, ata_output_dim)
        self.scaf = SCAF_Module(input_dim=ata_output_dim, d_model=64, heads=2, drop=0.3)
        self.classifier_fc = nn.Linear(64, 2)
        self.eeg_classifier_fc = nn.Linear(16, 2)
        self.verbose = verbose

    def forward(self, eeg, fnirs):
        eeg = eeg.permute(0, 4, 3, 1, 2)
        B_fnirs, T_seq, H, W, T_point, C_fnirs = fnirs.shape
        fnirs_permuted = fnirs.permute(0, 1, 5, 4, 2, 3)
        fnirs_reshaped = fnirs_permuted.reshape(B_fnirs, T_seq * C_fnirs, T_point, H, W)

        eegfusion1, fnirs1, bicmg1_loss = self.conv1(eeg, fnirs_reshaped)
        eegfusion2, fnirs2, bicmg2_loss = self.conv2(eegfusion1, fnirs1)

        aligned_fnirs_feat, ata_loss = self.ata(eegfusion2, fnirs2)

        if self.verbose:
            self.verbose = False

        B, C, D, H, W = eegfusion2.shape
        eeg_pooled = self.pool(eegfusion2.view(B, C, -1)).squeeze(-1)
        eeg_for_scaf = self.eeg_proj_for_scaf(eeg_pooled)
        main_feat, _, _, ef_loss = self.scaf(eeg_for_scaf, aligned_fnirs_feat)
        class_logits = self.classifier_fc(main_feat)
        eeg_logits = self.eeg_classifier_fc(eeg_pooled)

        return {
            'class_output': class_logits,
            'eeg_output': eeg_logits,
            'losses': {
                'bicmg1_loss': bicmg1_loss,
                'bicmg2_loss': bicmg2_loss,
                'ef_loss': ef_loss,
                'ata_loss': ata_loss
            }
        }
