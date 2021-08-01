import torch
import torch.nn as nn
import torch.nn.functional as F


def check_type_forward(in_types):
    assert len(in_types) == 3

    x0_type, x1_type, y_type = in_types
    assert x0_type.size() == x1_type.shape
    assert x1_type.size()[0] == y_type.shape[0]
    assert x1_type.size()[0] > 0
    assert x0_type.dim() == 2
    assert x1_type.dim() == 2
    assert y_type.dim() == 1


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        check_type_forward((x0, x1, y))

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

# class ContrastiveLoss(nn.Module):
#
#     def __init__(self, margin):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#         self.eps = 1e-9
#
#     def forward(self, output1, output2, target, size_average=True):
#         distances = (output2 - output1).pow(2).sum(1)
#         losses = 0.5 * (target.float() * distances +
#                         (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
#         return losses.mean() if size_average else losses.sum()
