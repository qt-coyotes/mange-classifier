import torch
from torch import Tensor, nn


class BinaryMacroSoftFBetaLoss(nn.Module):
    def __init__(self, beta=1, eps=torch.finfo(torch.float32).eps):
        super().__init__()
        self.beta = beta
        self.beta2 = beta ** 2
        self.eps = eps

    def forward(self, yhat: Tensor, y: Tensor):
        tp = (yhat * y).sum(axis=0)
        precision = tp / (yhat.sum(axis=0) + self.eps)
        recall = tp / (y.sum(axis=0) + self.eps)
        fbeta = (
            (1 + self.beta2) * precision * recall
            / (self.beta2 * precision + recall + self.eps)
        )
        return 1 - fbeta.mean()


class BinaryExpectedCostLoss(nn.Module):
    def __init__(
            self,
            ctp: float = 0.,
            cfp: float = 1.,
            cfn: float = 50.,
            ctn: float = 0.
    ):
        """
        Args:
            ctp: Cost of true positive
            cfp: Cost of false positive
            cfn: Cost of false negative
            ctn: Cost of true negative
        """
        super().__init__()
        self.ctp = ctp
        self.cfp = cfp
        self.cfn = cfn
        self.ctn = ctn

    def forward(self, yhat: Tensor, y: Tensor):
        tp = (yhat * y).sum(axis=0)
        fp = (yhat * (1 - y)).sum(axis=0)
        fn = ((1 - yhat) * y).sum(axis=0)
        tn = ((1 - yhat) * (1 - y)).sum(axis=0)
        n = tp + tn + fp + fn
        cost = (self.ctp * tp + self.cfp * fp + self.cfn * fn + self.ctn * tn) / n
        return cost.mean()
