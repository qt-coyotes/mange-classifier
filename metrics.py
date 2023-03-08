from torch import Tensor
from torchmetrics.classification.stat_scores import BinaryStatScores
from torchmetrics.utilities.compute import _safe_divide


class BinaryExpectedCost(BinaryStatScores):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
            self,
            ctp: float = 0.,
            cfp: float = 1.,
            cfn: float = 5.,
            ctn: float = 0.
    ):
        """
        Args:
            costs: A 1D tensor of the costs of: TN, FP, FN, TP.
        """
        super().__init__()
        self.ctp = ctp
        self.cfp = cfp
        self.cfn = cfn
        self.ctn = ctn

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        return _safe_divide(
            self.ctp * tp + self.cfp * fp + self.cfn * fn + self.ctn * tn,
            tp + tn + fp + fn
        )
