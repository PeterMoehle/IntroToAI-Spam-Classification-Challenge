import numpy as np

from typing import Dict, Any, Optional

from .abc_model import Model
from graph.abc_graph_model import GraphModel
from graph.nodes.function_node import FunctionNode
from graph.nodes.value_node import ValueNode
from graph.functions.arithmetic import DotProduct, Addition
from graph.functions.activation import ReLU, Sigmoid, Tanh
from graph.loss.loss_factory import LossGraphFactory


class NNClassifierGraph(GraphModel):
    """
    Neural Network classifier using computational graphs.

    Implements a two-layer neural network with customizable loss functions
    and activation functions for spam classification.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, hidden_dim: int = 32, loss_type: str = "logistic"):
        """Initialize the neural network classifier."""
        input_dim = X.shape[1]
        self.loss_type = loss_type

        # Create input and target nodes
        self.x = ValueNode(1, X, name_prefix="x", is_learnable=False)
        self.y = ValueNode(2, y, name_prefix="y", is_learnable=False)

        # Initialize network parameters
        # Using HE-Initialization. Provides much better results than other variants.
        self.W1 = ValueNode(3, np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim),
                            name_prefix="W1", is_learnable=True)
        self.b1 = ValueNode(4, np.zeros((1, hidden_dim)), name_prefix="b1", is_learnable=True)
        self.W2 = ValueNode(5, np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim),
                            name_prefix="W2", is_learnable=True)
        self.b2 = ValueNode(6, np.zeros((1, 1)), name_prefix="b2", is_learnable=True)

        # Build network architecture
        z1 = FunctionNode(8, DotProduct(), [self.x, self.W1])
        z1_b = FunctionNode(7, Addition(), [z1, self.b1])
        a1 = FunctionNode(9, ReLU(), [z1_b])
        z2 = FunctionNode(11, DotProduct(), [a1, self.W2])
        z2_b = FunctionNode(10, Addition(), [z2, self.b2])

        # Choose output activation based on loss type
        if loss_type == "hinge":
            output = FunctionNode(12, Tanh(), [z2_b])
        else:
            output = FunctionNode(12, Sigmoid(), [z2_b])

        # Create loss node
        loss_node = LossGraphFactory.get_loss(output_node=output, target_node=self.y, loss_type=loss_type)

        super().__init__(input_nodes=[self.x], target_node=self.y, output_node=output, loss_node=loss_node)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Make predictions with appropriate output transformation."""
        self.input_nodes[0].value = input_data
        for node in self.predict_pass_order:
            node.forward()

        predictions = self.output_node.value

        # Transform predictions based on loss type
        if self.loss_type == "hinge":
            # Convert tanh output [-1,1] to probabilities [0,1]
            predictions = (predictions + 1) / 2

        return predictions

    def get_binary_predictions(self, input_data: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Get binary class predictions."""
        probabilities = self.predict(input_data)
        return (probabilities > threshold).astype(int)


class NeuralNetwork(Model):
    def __init__(self, name: str, epochs: int = 20, lr: float = 0.01, hidden_dim: int = 32, loss_type: str = "logistic"):
        super().__init__(name, epochs, lr)
        self.hidden_dim = hidden_dim
        self.loss_type = loss_type
        self.model: Optional[NNClassifierGraph] = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True) -> Dict[str, Any]:
        self.model = NNClassifierGraph(X_train, y_train, hidden_dim=self.hidden_dim, loss_type=self.loss_type)
        return self.model.train(X_train, y_train, epochs=self.epochs, lr=self.lr, verbose=verbose)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be trained before predicting.")
        return self.model.predict(X)

    def get_binary_predictions(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be trained before predicting.")
        return self.model.get_binary_predictions(X, threshold=threshold)