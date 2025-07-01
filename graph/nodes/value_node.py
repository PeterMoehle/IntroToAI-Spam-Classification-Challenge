from typing import Any
import numpy as np

from ..abc_node import Node


class ValueNode(Node):
    """
    Node that holds a constant value or learnable parameter.

    ValueNodes store either constant values (like input data) or learnable
    parameters (like weights and biases) that can be updated during training.
    """

    def __init__(self, idx: int, value: Any, name_prefix: str = "const", is_learnable: bool = True):
        """Initialize a value node."""
        super().__init__(idx)
        self.value = np.array(value, dtype=float)
        self.name = f"{name_prefix}{self.id}"
        self.is_learnable = is_learnable
        self.gradient = np.zeros_like(self.value, dtype=float)
        self.node_type = "ValueNode"
        self.is_parameter = name_prefix in ["W", "b", "w"]

    def forward(self) -> np.ndarray:
        """Return the stored value."""
        return self.value