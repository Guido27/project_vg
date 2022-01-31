import torch
import torch.nn as nn
import torch.nn.functional as F


def loss(anchors, positives, negatives):
    dist_pos = ((anchors - positives) ** 2).sum(1)
    dist_neg = ((anchors - negatives) ** 2).sum(1)

    dist = -torch.cat((dist_pos, dist_neg), 1)
    dist = F.log_softmax(dist, 1)
    loss = (-dist[:, 0]).mean()

    return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, anchors, positives, negatives):
        # anchors, positives and negatives are N x D (N: number of images, D: dimensionality)
        dist_pos = torch.sum(torch.pow(anchors - positives, 2), dim=1)
        dist_neg = torch.sum(torch.pow(anchors - negatives, 2), dim=1)
        nq = anchors.size(dim=0)
        return torch.sum(torch.clamp(dist_pos - dist_neg + self.margin, min=0)) / nq


class SOSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, anchors, positives, negatives):
        # anchors, positives and negatives are N x D (N: number of images, D: dimensionality)
        dist_an = torch.sum(torch.pow(anchors - negatives, 2), dim=1)
        dist_pn = torch.sum(torch.pow(positives - negatives, 2), dim=1)
        nq = anchors.size(dim=0)
        return torch.sum(torch.pow(dist_an - dist_pn, 2)) ** 0.5 / nq


class SAREJointLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, anchors, positives, negatives):
        dist_pos = torch.sum(torch.pow(anchors - positives, 2), dim=1)
        dist_neg = torch.sum(torch.pow(anchors - negatives, 2), dim=1)
        c_pq = torch.exp(-dist_pos) / (torch.exp(-dist_pos) + torch.exp(-dist_neg))
        batch_loss = -torch.log(c_pq)
        nq = anchors.size(dim=0)
        return torch.sum(batch_loss) / nq
