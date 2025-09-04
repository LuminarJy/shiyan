import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableSpatialAttention(nn.Module):
    """
    可变形空间注意力机制
    结合了可变形卷积的思想和空间注意力，使模型能够自适应地关注EEG信号中的重要区域
    """
    def __init__(self, emb_size, num_heads, seq_len, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_dim = emb_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 标准注意力组件
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)
        
        # 可变形偏移量预测器
        self.offset_predictor = nn.Sequential(
            nn.Linear(emb_size, emb_size // 2),
            nn.GELU(),
            nn.Linear(emb_size // 2, 2 * num_heads)  # 每个头有2D偏移量(x,y)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(emb_size, emb_size)
        self.norm = nn.LayerNorm(emb_size)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 预测偏移量
        offsets = self.offset_predictor(x)  # [B, seq_len, 2*num_heads]
        offsets = offsets.view(batch_size, seq_len, self.num_heads, 2)  # [B, seq_len, num_heads, 2]
        
        # 标准注意力计算
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 重塑为多头格式
        q = q.permute(0, 2, 1, 3)  # [B, num_heads, seq_len, head_dim]
        k = k.permute(0, 2, 1, 3)  # [B, num_heads, seq_len, head_dim]
        v = v.permute(0, 2, 1, 3)  # [B, num_heads, seq_len, head_dim]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, seq_len, seq_len]
        
        # 应用可变形偏移
        # 为每个查询位置创建参考点网格
        device = x.device
        y_grid, x_grid = torch.meshgrid(
            torch.arange(seq_len, device=device),
            torch.arange(seq_len, device=device)
        )
        grid = torch.stack([x_grid, y_grid], dim=-1).float()  # [seq_len, seq_len, 2]
        
        # 对每个头和每个批次应用偏移
        deformed_attn_scores = []
        
        for h in range(self.num_heads):
            head_offsets = offsets[:, :, h, :]  # [B, seq_len, 2]
            
            # 扩展偏移以匹配网格形状
            head_offsets = head_offsets.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [B, seq_len, seq_len, 2]
            
            # 应用偏移到网格
            deformed_grid = grid.unsqueeze(0) + head_offsets  # [B, seq_len, seq_len, 2]
            
            # 归一化到[-1, 1]范围用于grid_sample
            norm_grid = 2.0 * deformed_grid / (seq_len - 1) - 1.0
            
            # 使用grid_sample应用可变形采样
            # 重塑注意力分数以适应grid_sample
            head_scores = attn_scores[:, h].unsqueeze(1)  # [B, 1, seq_len, seq_len]
            
            # 应用grid_sample
            deformed_scores = F.grid_sample(
                head_scores, 
                norm_grid, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=True
            )  # [B, 1, seq_len, seq_len]
            
            deformed_attn_scores.append(deformed_scores.squeeze(1))
        
        # 合并所有头的结果
        deformed_attn_scores = torch.stack(deformed_attn_scores, dim=1)  # [B, num_heads, seq_len, seq_len]
        
        # 应用softmax
        attn_weights = F.softmax(deformed_attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        out = torch.matmul(attn_weights, v)  # [B, num_heads, seq_len, head_dim]
        
        # 重塑回原始形状
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.emb_size)
        
        # 输出投影
        out = self.output_projection(out)
        
        return out


class DeformableTransformerBlock(nn.Module):
    """
    使用可变形空间注意力的Transformer块
    """
    def __init__(self, d_model, attn_heads, seq_len, d_ffn, enable_res_parameter, dropout=0.1):
        super(DeformableTransformerBlock, self).__init__()
        self.attn = DeformableSpatialAttention(d_model, attn_heads, seq_len, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.enable_res_parameter = enable_res_parameter
        if enable_res_parameter:
            self.res_param1 = nn.Parameter(torch.tensor(1e-8))
            self.res_param2 = nn.Parameter(torch.tensor(1e-8))
        
    def forward(self, x, mask=None):
        # 第一个子层: 可变形空间注意力
        if self.enable_res_parameter:
            x = self.norm1(x + self.res_param1 * self.attn(x))
            x = self.norm2(x + self.res_param2 * self.ffn(x))
        else:
            x = self.norm1(x + self.attn(x))
            x = self.norm2(x + self.ffn(x))
        return x


class DeformableEncoder(nn.Module):
    """
    使用可变形空间注意力的编码器
    """
    def __init__(self, config):
        super(DeformableEncoder, self).__init__()
        d_model = config['emb_size']
        attn_heads = config['num_heads']
        seq_len = int(config['Data_shape'][2] / config['pooling_size'])  # 与原始模型保持一致
        d_ffn = config['dim_ff']
        layers = config['layers']
        dropout = config['dropout']
        enable_res_parameter = True
        
        # 可变形Transformer块
        self.TRMs = nn.ModuleList(
            [DeformableTransformerBlock(d_model, attn_heads, seq_len, d_ffn, enable_res_parameter, dropout) for i in range(layers)]
        )
        
    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x


class AdaptiveFeatureAggregation(nn.Module):
    """
    自适应特征聚合，替代简单的平均池化
    """
    def __init__(self, emb_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(emb_size, emb_size // 4),
            nn.LayerNorm(emb_size // 4),
            nn.GELU(),
            nn.Linear(emb_size // 4, 1)
        )
        
    def forward(self, x):
        # x: [B, seq_len, emb_size]
        
        # 计算注意力权重
        attn_weights = self.attention(x)  # [B, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 加权聚合
        weighted_x = x * attn_weights
        aggregated = weighted_x.sum(dim=1)  # [B, emb_size]
        
        return aggregated
