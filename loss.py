import torch
import torch.nn.functional as F

def loss(anchors, positives, negatives):
    dist_pos = ((anchors - positives) ** 2).sum(1)
    dist_neg = ((anchors - negatives) ** 2).sum(1)

    dist = - torch.cat((dist_pos, dist_neg), 1)
    dist = F.log_softmax(dist, 1)
    loss = (- dist[:, 0]).mean()

    return loss


def triplet_loss(anchors, positives, negatives, margin=0.1):
    # x is D x N
    # dim = x.size(0) # D
    # nq = torch.sum(label.data==-1).item() # number of tuples
    # S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    # xa = x[:, label.data==-1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
    # xp = x[:, label.data==1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
    # xn = x[:, label.data==0]

    dist_pos = torch.sum(torch.pow(anchors - positives, 2), dim=0)
    dist_neg = torch.sum(torch.pow(anchors - negatives, 2), dim=0)

    # nq = 
    print(anchors.size())

    return torch.sum(torch.clamp(dist_pos - dist_neg + margin, min=0)) / nq