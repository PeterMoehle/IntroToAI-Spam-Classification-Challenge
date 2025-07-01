import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, Any


class Model(ABC):
    """Abstract base class for all trainable models in the evaluation system."""

    def __init__(self, name: str, epochs: int = 100, lr: float = 0.01):
        """Initialize the model with basic parameters."""
        self.name = name
        self.epochs = epochs
        self.lr = lr

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True) -> Dict[str, Any]:
        """Train the model on the given data."""
        raise NotImplementedError("Subclasses must implement train method")

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for the given input features."""
        raise NotImplementedError("Subclasses must implement predict method")

    @abstractmethod
    def get_binary_predictions(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Get binary class predictions from the model output."""
        raise NotImplementedError("Subclasses must implement get_binary_predictions method")