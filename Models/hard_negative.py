import torch
import torch.nn.functional as F

def get_hard_negatives(features, labels):
    """
    features: (batch_size, feature_dim)
    labels: (batch_size,)
    返回每个anchor的hard negative索引
    """
    features = F.normalize(features, dim=1)
    sim_matrix = torch.matmul(features, features.t())  # (B, B)
    mask = torch.eye(features.size(0), device=features.device).bool()
    sim_matrix.masked_fill_(mask, -1e9)
    label_matrix = labels.unsqueeze(1) != labels.unsqueeze(0)
    sim_matrix[~label_matrix] = -1e9
    hard_neg_indices = sim_matrix.argmax(dim=1)
    return hard_neg_indices

def triplet_loss(features, labels, hard_neg_indices, margin=0.2):
    """
    features: (batch_size, feature_dim)
    labels: (batch_size,)
    hard_neg_indices: (batch_size,)
    """
    anchor = features
    positive = features  # 自监督场景下正样本为自身
    negative = features[hard_neg_indices]
    loss = F.triplet_margin_loss(anchor, positive, negative, margin=margin, reduction='mean')
    return loss 