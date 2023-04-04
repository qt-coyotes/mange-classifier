import torch
from torch import Tensor, nn


class BinaryMacroSoftFBetaLoss(nn.Module):
    def __init__(self, beta=1, eps=torch.finfo(torch.float32).eps):
        super().__init__()
        self.beta = beta
        self.beta2 = beta**2
        self.eps = eps

    def forward(self, yhat: Tensor, y: Tensor):
        tp = (yhat * y).sum(axis=0)
        precision = tp / (yhat.sum(axis=0) + self.eps)
        recall = tp / (y.sum(axis=0) + self.eps)
        fbeta = (
            (1 + self.beta2)
            * precision
            * recall
            / (self.beta2 * precision + recall + self.eps)
        )
        return 1 - fbeta.mean()


class BinarySurrogateFBetaLoss(nn.Module):
    def __init__(self, beta=1, eps=torch.finfo(torch.float32).eps):
        super().__init__()
        self.beta = beta
        self.beta2 = beta**2
        self.clip_log_x = torch.exp(torch.tensor(-100.0))
        self.eps = eps

    def forward(self, yhat: Tensor, y: Tensor):
        p = y.mean(axis=0)
        return (
            -y * self.log(yhat)
            + (1 - y)
            * self.log(self.beta2 * p / (1 - p + self.eps) + yhat)
        ).mean()

    def log(self, x: Tensor):
        return torch.log(torch.max(x, self.clip_log_x))


class BinaryExpectedCostLoss(nn.Module):
    def __init__(
        self,
        ctp: float = 0.0,
        cfp: float = 1.0,
        cfn: float = 50.0,
        ctn: float = 0.0,
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
        cost = (
            self.ctp * tp + self.cfp * fp + self.cfn * fn + self.ctn * tn
        ) / n
        return cost.mean()


class HybridLoss(nn.Module):
    def __init__(
        self,
        loss_a: nn.Module,
        loss_b: nn.Module
    ):
        """
        Args:
            loss_a: Loss function to use for the first epoch
            loss_b: Loss function to use for the remaining epochs
        """
        super().__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.max_batch_idx = -1

    def forward(self, yhat: Tensor, y: Tensor, batch_idx: int):
        if batch_idx > self.max_batch_idx:
            self.max_batch_idx = batch_idx
            return self.loss_a(yhat, y)
        else:
            return self.loss_b(yhat, y)
