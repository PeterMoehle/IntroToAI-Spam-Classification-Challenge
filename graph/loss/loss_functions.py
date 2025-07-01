from ..abc_loss_graph import LossGraph
from ..abc_node import Node
from ..nodes.function_node import FunctionNode
from ..nodes.value_node import ValueNode
from ..functions.arithmetic import Addition, ElementwiseMultiplication, Max, Sum, Subtraction
from ..functions.math_functions import Log, Exp


class HingeLoss(LossGraph):
    """
    Hinge Loss implementation for SVM-style classification.

    Computes max(0, 1 - y * prediction) where y is in {-1, 1}.
    Converts binary labels (0,1) to (-1,1) internally.
    """

    def _build_loss(self) -> Node:
        # Convert binary labels (0,1) to (-1,1) for hinge loss: y_hinge = 2*y - 1
        two = ValueNode(self.pred_idx + 1, 2.0, name_prefix="two", is_learnable=False)
        one = ValueNode(self.pred_idx + 2, 1.0, name_prefix="one", is_learnable=False)
        neg_one = ValueNode(self.pred_idx + 3, -1.0, name_prefix="neg", is_learnable=False)
        zero = ValueNode(self.pred_idx + 4, 0.0, name_prefix="zero", is_learnable=False)

        y_scaled = FunctionNode(self.pred_idx + 5, ElementwiseMultiplication(), [two, self.target])
        neg_one_mult_one = FunctionNode(self.pred_idx + 6, ElementwiseMultiplication(), [neg_one, one])
        y_hinge = FunctionNode(self.pred_idx + 7, Addition(), [y_scaled, neg_one_mult_one])

        # Compute margin: y * prediction
        margin = FunctionNode(self.pred_idx + 8, ElementwiseMultiplication(), [y_hinge, self.prediction])

        # Compute 1 - margin
        one_minus_margin = FunctionNode(self.pred_idx + 9, Subtraction(), [one, margin])

        # Max(0, 1 - y*prediction)
        hinge_per_sample = FunctionNode(self.pred_idx + 10, Max(), [zero, one_minus_margin])

        # Sum over all samples
        total_loss = FunctionNode(self.pred_idx + 11, Sum(), [hinge_per_sample])

        return total_loss


class LogisticLoss1(LossGraph):
    """
    Logistic Loss for binary classification using log(1 + exp(-y * pred)) form.

    Assumes y ∈ {-1, +1} and prediction is a raw score (logit).
    """

    def _build_loss(self) -> Node:
        # Constants
        neg_one = ValueNode(self.pred_idx + 1, -1.0, name_prefix="neg", is_learnable=False)
        one = ValueNode(self.pred_idx + 2, 1.0, name_prefix="one", is_learnable=False)

        # Compute -y * pred
        neg_y = FunctionNode(self.pred_idx + 3, ElementwiseMultiplication(), [neg_one, self.target])
        neg_y_pred = FunctionNode(self.pred_idx + 4, ElementwiseMultiplication(), [neg_y, self.prediction])

        # Compute exp(-y * pred)
        exp_node = FunctionNode(self.pred_idx + 5, Exp(), [neg_y_pred])

        # Compute 1 + exp(-y * pred)
        sum_node = FunctionNode(self.pred_idx + 6, Addition(), [one, exp_node])

        # Compute log(1 + exp(-y * pred))
        log_node = FunctionNode(self.pred_idx + 7, Log(), [sum_node])

        # Sum over all samples
        total_loss = FunctionNode(self.pred_idx + 8, Sum(), [log_node])

        return total_loss


class LogisticLoss2(LossGraph):
    """
    Logistic Loss (Cross-Entropy) for binary classification.

    Computes -y*log(pred) - (1-y)*log(1-pred) where y is in {0, 1}.
    """

    def _build_loss(self) -> Node:
        # First term: -y * log(pred)
        neg_one = ValueNode(self.pred_idx + 1, -1.0, name_prefix="neg", is_learnable=False)
        one = ValueNode(self.pred_idx + 2, 1.0, name_prefix="one", is_learnable=False)

        log_pred = FunctionNode(self.pred_idx + 3, Log(), [self.prediction])
        neg_target = FunctionNode(self.pred_idx + 4, ElementwiseMultiplication(), [neg_one, self.target])
        neg_target_log_pred = FunctionNode(self.pred_idx + 5, ElementwiseMultiplication(), [neg_target, log_pred])

        # Second term: -(1-y) * log(1-pred)
        one_minus_target = FunctionNode(self.pred_idx + 6, Subtraction(), [one, self.target])
        one_minus_pred = FunctionNode(self.pred_idx + 7, Subtraction(), [one, self.prediction])

        log_one_minus_pred = FunctionNode(self.pred_idx + 8, Log(), [one_minus_pred])
        neg_one_mult_one_minus_target = FunctionNode(self.pred_idx + 9, ElementwiseMultiplication(), [neg_one, one_minus_target])
        second_term = FunctionNode(self.pred_idx + 10, ElementwiseMultiplication(), [neg_one_mult_one_minus_target, log_one_minus_pred])

        # Combine terms and sum over all samples
        combined = FunctionNode(self.pred_idx + 11, Addition(), [neg_target_log_pred, second_term])
        total_loss = FunctionNode(self.pred_idx + 12, Sum(), [combined])

        return total_loss


class MSELoss(LossGraph):
    """
    Mean Squared Error Loss for regression-style training.

    Computes sum((prediction - target)²) over all samples.
    """

    def _build_loss(self) -> Node:
        # Compute prediction - target
        neg_one = ValueNode(self.pred_idx + 1, -1.0, name_prefix="neg", is_learnable=False)

        neg_one_mult_target = FunctionNode(self.pred_idx + 2, ElementwiseMultiplication(), [neg_one, self.target])
        diff = FunctionNode(self.pred_idx + 3, Addition(), [self.prediction, neg_one_mult_target])

        # Square the difference
        squared_diff = FunctionNode(self.pred_idx + 4, ElementwiseMultiplication(), [diff, diff])

        # Sum over all samples
        total_loss = FunctionNode(self.pred_idx + 5, Sum(), [squared_diff])

        return total_loss
