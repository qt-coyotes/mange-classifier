import torch
from torch import nn


class MacroSoftFBetaLoss(nn.Module):
    def __init__(self, beta=1, eps=torch.finfo(torch.float32).eps):
        super().__init__()
        self.beta = beta
        self.beta2 = beta ** 2
        self.eps = eps

    def forward(self, yhat, y):
        tp = (yhat * y).sum(axis=0)
        precision = tp / (yhat.sum(axis=0) + self.eps)
        recall = tp / (y.sum(axis=0) + self.eps)
        fbeta = (
            (1 + self.beta2) * precision * recall
            / (self.beta2 * precision + recall + self.eps)
        )
        return 1 - fbeta.mean()
