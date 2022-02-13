import torch
import torch.nn as nn

class SOSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, anchors, positives, negatives):
        # anchors, positives and negatives are N x D (N: number of images, D: dimensionality)
        dist_an = torch.sum(torch.pow(anchors - negatives, 2), dim=1)
        dist_pn = torch.sum(torch.pow(positives - negatives, 2), dim=1)
        nq = anchors.size(dim=0)
        return torch.sum(torch.pow(dist_an - dist_pn, 2)) ** 0.5 / nq