import numpy as np

from abc import ABC, abstractmethod
from typing import Optional


class Node(ABC):
    """
    Abstract base class for all computational graph nodes.

    Nodes represent operations or values in the computational graph and
    support forward computation and backward gradient propagation.
    """

    def __init__(self, idx: int):
        """Initialize a node with a unique identifier."""
        self.id = idx
        self.value: Optional[np.ndarray] = None
        self.gradient: Optional[np.ndarray] = None
        self.name: str = f"α{self.id}"
        self.node_type: str = "Node"

    @abstractmethod
    def forward(self) -> np.ndarray:
        """Compute the forward pass for this node."""
        raise NotImplementedError("Subclasses must implement forward method")

    def backward(self, upstream_gradient: np.ndarray) -> None:
        """Accumulate gradients with proper shape handling."""
        if self.gradient is None and self.value is not None:
            self.gradient = np.zeros_like(self.value, dtype=float)

        if self.gradient is not None:
            # Handle shape broadcasting for gradient accumulation
            if upstream_gradient.shape != self.gradient.shape:
                if upstream_gradient.size == 1:
                    # Scalar upstream gradient
                    self.gradient += upstream_gradient.item()
                elif self.gradient.size == 1:
                    # Scalar gradient, sum upstream gradient
                    self.gradient += np.sum(upstream_gradient)
                else:
                    # Try to broadcast or sum appropriately
                    try:
                        self.gradient += upstream_gradient
                    except ValueError:
                        # Sum over batch dimension if needed
                        if len(upstream_gradient.shape) > len(self.gradient.shape):
                            axis_to_sum = tuple(range(len(upstream_gradient.shape) - len(self.gradient.shape)))
                            self.gradient += np.sum(upstream_gradient, axis=axis_to_sum)
                        else:
                            self.gradient += np.sum(upstream_gradient)
            else:
                self.gradient += upstream_gradient

    def reset_gradient(self) -> None:
        """Reset the gradient to zero."""
        if self.value is not None:
            self.gradient = np.zeros_like(self.value, dtype=float)
        else:
            self.gradient = None

    def __repr__(self) -> str:
        return f"Node(id={self.id}, name={self.name})"