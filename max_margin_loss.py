import torch

def max_margin_loss(inp, margin):
    pos_cos = inp[:,0]
    neg_cos, _ = torch.max(inp[:,1:], dim=1)
    return torch.mean(neg_cos - pos_cos + margin)
