import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, anchors, positives, negatives):
        # anchors, positives and negatives are N x D (N: number of images, D: dimensionality)
        dist_pos = torch.sum(torch.pow(anchors - positives, 2), dim=1)
        dist_neg = torch.sum(torch.pow(anchors - negatives, 2), dim=1)
        return torch.mean(torch.clamp(dist_pos - dist_neg + self.margin, min=0))


class SOSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, anchors, positives, negatives):
        # anchors, positives and negatives are N x D (N: number of images, D: dimensionality)
        dist_an = torch.sum(torch.pow(anchors - negatives, 2), dim=1)
        dist_pn = torch.sum(torch.pow(positives - negatives, 2), dim=1)
        nq = anchors.size(dim=0)
        return torch.sum(torch.pow(dist_an - dist_pn, 2)) ** 0.5 / nq


class SAREIndLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, anchors, positives, negatives):
        dist_pos = torch.sum(torch.pow(anchors - positives, 2), dim=1)
        dist_neg = torch.sum(torch.pow(anchors - negatives, 2), dim=1)

        # c_pq = torch.exp(-dist_pos) / (torch.exp(-dist_pos) + torch.exp(-dist_neg))
        # batch_loss = -torch.log(c_pq)
        # nq = anchors.size(dim=0)
        # return torch.sum(batch_loss) / nq

        # OR
        return torch.mean(dist_pos + torch.logaddexp(-dist_pos, -dist_neg))
