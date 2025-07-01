from ..abc_node import Node
from ..nodes.function_node import FunctionNode
from ..nodes.value_node import ValueNode
from .loss_functions import HingeLoss, LogisticLoss1, LogisticLoss2, MSELoss


class LossGraphFactory:
    """
    Factory class for creating loss function computation graphs.

    Provides a unified interface for creating different types of loss functions
    based on string identifiers.
    """

    @staticmethod
    def get_loss(output_node: FunctionNode, target_node: ValueNode, loss_type: str = "mse") -> Node:
        """Create a loss computation graph based on the specified type."""
        if loss_type == "hinge":
            loss = HingeLoss(prediction=output_node, target=target_node).get_loss_node()
        elif loss_type == "logistic1":
            loss = LogisticLoss1(prediction=output_node, target=target_node).get_loss_node()
        elif loss_type == "logistic2":
            loss = LogisticLoss2(prediction=output_node, target=target_node).get_loss_node()
        elif loss_type == "mse":
            loss = MSELoss(prediction=output_node, target=target_node).get_loss_node()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. "
                             f"Supported types: 'hinge', 'logistic', 'mse'")

        return loss