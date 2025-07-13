import numpy as np

from typing import Dict, Any

from .abc_model import Model


class DistanceCalculator:
    @staticmethod
    def calculate_distance(distance: str, x1: np.ndarray, x2: np.ndarray) -> np.float64:
        if distance == 'euclidean':
            return np.linalg.norm(x1 - x2)
        elif distance == 'manhattan':
            return np.sum(np.abs(x1 - x2)).astype(float)
        else:
            raise ValueError(f"Unknown distance type: '{distance}'")

class KNN(Model):
    def __init__(self, name: str, k: int, distance: str = 'euclidean'):
        super().__init__(name)
        self.k = k
        self.distance = distance
        self.X_train = None
        self.y_train = None

    def _get_closest(self, distances: np.ndarray) -> np.ndarray:
        """Returns the indices of the k smallest distances."""
        return np.argsort(distances)[:self.k]

    def train(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True) -> Dict[str, Any]:
        self.X_train = X_train
        self.y_train = y_train
        if verbose:
            print(f"Training completed with {X_train.shape[0]} samples.")
        return {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for x in X:
            distances = np.empty(self.X_train.shape[0])
            for i in range(self.X_train.shape[0]):
                distances[i] = np.array(DistanceCalculator.calculate_distance(self.distance, self.X_train[i], x))
            closest_indices = self._get_closest(distances)
            closest_labels = self.y_train[closest_indices]
            
            # Voting (for classification): majority label
            values, counts = np.unique(closest_labels, return_counts=True)
            majority_label = values[np.argmax(counts)]
            predictions.append(majority_label)
        return np.array(predictions)


    def get_binary_predictions(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        preds = self.predict(X)
        return (preds >= threshold).astype(int)