import numpy as np

from typing import Optional, Dict, Any

from models import Model
from graph.abc_graph_model import GraphModel
from graph.nodes.function_node import FunctionNode
from graph.nodes.value_node import ValueNode
from graph.functions.arithmetic import DotProduct, Addition
from graph.functions.activation import ReLU, Sigmoid, Tanh
from graph.loss.loss_factory import LossGraphFactory


class LinearClassifierGraph(GraphModel):
    """
    Linear binary classifier using the computational graph framework.
    Model: y_hat = sigmoid(xW + b)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        input_dim = X.shape[1]

        # Input and target
        self.x = ValueNode(1, X, name_prefix="x", is_learnable=False)
        self.y = ValueNode(2, y, name_prefix="y", is_learnable=False)

        # Parameters
        self.W = ValueNode(3, np.random.randn(input_dim, 1) * np.sqrt(2.0 / input_dim), name_prefix="W", is_learnable=True)
        self.b = ValueNode(4, np.zeros((1, 1)), name_prefix="b", is_learnable=True)

        # Graph: xW + b → sigmoid → loss
        z = FunctionNode(5, DotProduct(), [self.x, self.W])
        z_b = FunctionNode(6, Addition(), [z, self.b])
        output = FunctionNode(7, Sigmoid(), [z_b])

        # Loss
        loss_node = LossGraphFactory.get_loss(output_node=output, target_node=self.y, loss_type="hinge")

        super().__init__(input_nodes=[self.x], target_node=self.y, output_node=output, loss_node=loss_node)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        self.input_nodes[0].value = input_data
        for node in self.predict_pass_order:
            node.forward()
        return self.output_node.value

    def get_binary_predictions(self, input_data: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict(input_data) > threshold).astype(int)



class LinearClassifierGD(Model):
    """
    Linear classifier that uses the Graph-based gradient descent model.
    """

    def __init__(self, name: str, epochs: int = 100, lr: float = 0.01):
        super().__init__(name, epochs, lr)
        self.model: Optional[LinearClassifierGraph] = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True) -> Dict[str, Any]:
        self.model = LinearClassifierGraph(X_train, y_train)
        return self.model.train(X_train, y_train, epochs=self.epochs, lr=self.lr, verbose=verbose)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained.")
        return self.model.predict(X)

    def get_binary_predictions(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained.")
        return self.model.get_binary_predictions(X, threshold)



class LinearClassifierClosedForm(Model):
    """
    Linear classifier using the closed-form least squares solution.
    """

    def __init__(self, name: str):
        super().__init__(name, epochs=1, lr=0)  # Not used
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True) -> Dict[str, Any]:
        # Add bias term to X
        X_augmented = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

        # Solve using normal equation: w = (X^T X)^-1 X^T y
        pseudo_inverse = np.linalg.pinv(X_augmented)
        w = np.dot(pseudo_inverse, y_train)

        self.weights = w[:-1]
        self.bias = w[-1]

        # Compute training loss (MSE)
        predictions = np.dot(X_train, self.weights) + self.bias
        loss = np.mean((predictions - y_train) ** 2)

        if verbose:
            print(f"Closed-form solution found. Final MSE loss: {loss:.6f}")

        return {"loss": [loss], "epoch": [0]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_output = np.dot(X, self.weights) + self.bias
        return 1 / (1 + np.exp(-linear_output))

    def get_binary_predictions(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict(X)
        return (probabilities > threshold).astype(int)

