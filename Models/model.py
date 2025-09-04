import copy
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Models.Attention import *
<<<<<<< HEAD
from Models.DeformableAttention import DeformableEncoder, AdaptiveFeatureAggregation
=======
>>>>>>> 1df5b5cb40be153a6eae12a1aeda023519836a03


def Encoder_factory(config):
    model = EEG2Rep(config, num_classes=config['num_labels'])
    return model


class EEG2Rep(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        """
         channel_size: number of EEG channels
         seq_len: number of timepoints in a window
        """
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']  # d_x
        # Embedding Layer -----------------------------------------------------------
        config['pooling_size'] = 2  # Max pooling size in input embedding
        seq_len = int(seq_len / config['pooling_size'])  # Number of patches (l)
        self.InputEmbedding = InputEmbedding(config)  # input (Batch,Channel, length) -> output (Batch, l, d_x)
        self.PositionalEncoding = PositionalEmbedding(seq_len, emb_size)
        # -------------------------------------------------------------------------
        self.momentum = config['momentum']
        self.device = config['device']
        self.mask_ratio = config['mask_ratio']
        self.mask_len = int(config['mask_ratio'] * seq_len)
        self.mask_token = nn.Parameter(torch.randn(emb_size, ))
<<<<<<< HEAD
        # 使用可变形空间注意力编码器替代标准编码器
        self.use_deformable = config.get('use_deformable', False)  # 默认不使用可变形注意力
        if self.use_deformable:
            self.contex_encoder = DeformableEncoder(config)
        else:
            self.contex_encoder = Encoder(config)
=======
        self.contex_encoder = Encoder(config)
>>>>>>> 1df5b5cb40be153a6eae12a1aeda023519836a03
        self.target_encoder = copy.deepcopy(self.contex_encoder)
        self.Predictor = Predictor(emb_size, config['num_heads'], config['dim_ff'], 1, config['pre_layers'])
        self.predict_head = nn.Linear(emb_size, config['num_labels'])
        self.Norm = nn.LayerNorm(emb_size)
        self.Norm2 = nn.LayerNorm(emb_size)
        self.gap = nn.AdaptiveAvgPool1d(1)
<<<<<<< HEAD
        # 自适应特征聚合，替代简单的平均池化
        self.use_adaptive_aggregation = config.get('use_adaptive_aggregation', False)  # 默认不使用
        if self.use_adaptive_aggregation:
            self.adaptive_aggregation = AdaptiveFeatureAggregation(emb_size)
=======
>>>>>>> 1df5b5cb40be153a6eae12a1aeda023519836a03

    def copy_weight(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.contex_encoder.parameters(), self.target_encoder.parameters()):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.contex_encoder.parameters(), self.target_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data

    def linear_prob(self, x):
        with (torch.no_grad()):
            patches = self.InputEmbedding(x)
            patches = self.Norm(patches)
            patches = patches + self.PositionalEncoding(patches)
            patches = self.Norm2(patches)
            out = self.contex_encoder(patches)
            out = out.transpose(2, 1)
            out = self.gap(out)
            return out.squeeze()

    def pretrain_forward(self, x):
        patches = self.InputEmbedding(x)  # (Batch, l, d_x)
        patches = self.Norm(patches)
        patches = patches + self.PositionalEncoding(patches)
        patches = self.Norm2(patches)

        # MaskMix: 生成两组不同掩码
        rep_mask_token1 = self.mask_token.repeat(patches.shape[0], patches.shape[1], 1)
        rep_mask_token1 = rep_mask_token1 + self.PositionalEncoding(rep_mask_token1)
        index = np.arange(patches.shape[1])
        index_chunk1 = Semantic_Subsequence_Preserving(index, 2, self.mask_ratio)
        v_index1 = np.ravel(index_chunk1)
        m_index1 = np.setdiff1d(index, v_index1)
        visible1 = patches[:, v_index1, :]
        rep_mask_token1 = rep_mask_token1[:, m_index1, :]
        rep_contex1 = self.contex_encoder(visible1)
        with torch.no_grad():
            rep_target1 = self.target_encoder(patches)
            rep_mask1 = rep_target1[:, m_index1, :]
        rep_mask_prediction1 = self.Predictor(rep_contex1, rep_mask_token1)

        # 第二组掩码
        rep_mask_token2 = self.mask_token.repeat(patches.shape[0], patches.shape[1], 1)
        rep_mask_token2 = rep_mask_token2 + self.PositionalEncoding(rep_mask_token2)
        index_chunk2 = Semantic_Subsequence_Preserving(index, 2, self.mask_ratio)
        v_index2 = np.ravel(index_chunk2)
        m_index2 = np.setdiff1d(index, v_index2)
        visible2 = patches[:, v_index2, :]
        rep_mask_token2 = rep_mask_token2[:, m_index2, :]
        rep_contex2 = self.contex_encoder(visible2)
        with torch.no_grad():
            rep_target2 = self.target_encoder(patches)
            rep_mask2 = rep_target2[:, m_index2, :]
        rep_mask_prediction2 = self.Predictor(rep_contex2, rep_mask_token2)

        # 池化后拼接，保证拼接shape一致
        rep_mask1_pooled = rep_mask1.mean(dim=1)
        rep_mask2_pooled = rep_mask2.mean(dim=1)
        rep_mask = torch.cat([rep_mask1_pooled, rep_mask2_pooled], dim=-1)

        rep_mask_prediction1_pooled = rep_mask_prediction1.mean(dim=1)
        rep_mask_prediction2_pooled = rep_mask_prediction2.mean(dim=1)
        rep_mask_prediction = torch.cat([rep_mask_prediction1_pooled, rep_mask_prediction2_pooled], dim=-1)

        rep_contex1_pooled = rep_contex1.mean(dim=1)
        rep_contex2_pooled = rep_contex2.mean(dim=1)
        rep_contex = torch.cat([rep_contex1_pooled, rep_contex2_pooled], dim=-1)

        rep_target1_pooled = rep_target1.mean(dim=1)
        rep_target2_pooled = rep_target2.mean(dim=1)
        rep_target = torch.cat([rep_target1_pooled, rep_target2_pooled], dim=-1)

        return [rep_mask, rep_mask_prediction, rep_contex, rep_target]

    def forward(self, x):
        patches = self.InputEmbedding(x)
        patches = self.Norm(patches)
        patches = patches + self.PositionalEncoding(patches)
        patches = self.Norm2(patches)
        out = self.contex_encoder(patches)
<<<<<<< HEAD
        
        # 使用自适应特征聚合或平均池化
        if hasattr(self, 'use_adaptive_aggregation') and self.use_adaptive_aggregation:
            global_features = self.adaptive_aggregation(out)
        else:
            global_features = torch.mean(out, dim=1)
            
        return self.predict_head(global_features)
=======
        return self.predict_head(torch.mean(out, dim=1))
>>>>>>> 1df5b5cb40be153a6eae12a1aeda023519836a03


class InputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        channel_size = config['Data_shape'][1]  # EEG通道数
        seq_len = config['Data_shape'][2]       # 时间点数
        emb_size = config['emb_size']
        # PatchEmbedding结构：
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (channel_size, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # 池化获得patch
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
        )
        # 动态权重分支：门控网络，输出与主卷积通道数一致的权重
        self.dynamic_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(40, 40),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T) -> (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)
        feat = self.shallownet(x)  # (B, 40, 1, patch数)
        # 动态门控：生成权重并加权主特征
        gate = self.dynamic_gate(feat)  # (B, 40)
        gate = gate.unsqueeze(-1).unsqueeze(-1)  # (B, 40, 1, 1)
        feat = feat * gate  # (B, 40, 1, patch数)
        x = self.projection(feat)  # (B, emb_size, 1, patch数)
        x = x.squeeze(2)        # (B, emb_size, patch数)
        x = x.transpose(1, 2)   # (B, patch数, emb_size)
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        d_model = config['emb_size']
        attn_heads = config['num_heads']
        # d_ffn = 4 * d_model
        d_ffn = config['dim_ff']
        layers = config['layers']
        dropout = config['dropout']
        enable_res_parameter = True
        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x


def Semantic_Subsequence_Preserving(time_step_indices, chunk_count, target_percentage):
    # Get the total number of time steps
    total_time_steps = len(time_step_indices)
    # Calculate the desired total time steps for the selected chunks
    target_total_time_steps = int(total_time_steps * target_percentage)

    # Calculate the size of each chunk
    chunk_size = target_total_time_steps // chunk_count

    # Randomly select starting points for each chunk with minimum distance
    start_points = [random.randint(0, total_time_steps - chunk_size)]
    # Randomly select starting points for each subsequent chunk with minimum distance
    for _ in range(chunk_count - 1):
        next_start_point = random.randint(0, total_time_steps - chunk_size)
        start_points.append(next_start_point)

    # Select non-overlapping chunks using indices
    selected_chunks_indices = [time_step_indices[start:start + chunk_size] for start in start_points]

    return selected_chunks_indices


class Predictor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Predictor, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter) for i in range(layers)])

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)
        return rep_mask_token


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


