from abc import ABC, abstractmethod

from .abc_node import Node


class LossGraph(ABC):
    """
    Abstract base class for loss function computation graphs.

    Loss graphs encapsulate the computation of loss functions and their
    gradients within the computational graph framework.
    """

    def __init__(self, prediction: Node, target: Node):
        """Initialize the loss graph."""
        self.prediction = prediction
        self.pred_idx = prediction.id
        self.target = target
        self.loss_node = self._build_loss()

    @abstractmethod
    def _build_loss(self) -> Node:
        """Build the loss computation graph."""
        raise NotImplementedError("Subclasses must implement _build_loss method")

    def get_loss_node(self) -> Node:
        """Get the loss computation node."""
        return self.loss_node