import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetLoss(nn.Module):

    def __init__(self, reg_weight):
        super(PointNetLoss, self).__init__()
        self.reg_weight = reg_weight

    def forward(self, *input):
        pred, target, feat_trans = input
        nll_loss = F.nll_loss(pred, target)
        iden = torch.eye(feat_trans.shape[-1]).unsqueeze(0).cuda()
        orth_loss = (iden - torch.bmm(feat_trans, feat_trans.transpose(2, 1))).norm(p=2)
        loss = nll_loss + self.reg_weight * orth_loss

        return loss
