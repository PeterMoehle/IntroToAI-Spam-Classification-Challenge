import numpy as np

from typing import List, Dict, Any, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

from models.abc_model import Model


class Evaluator:
    """
    K-Fold cross-validation evaluator for machine learning models.

    Provides comprehensive evaluation of models using k-fold cross-validation
    and collects performance statistics including accuracy and standard deviation.
    """

    def __init__(self, models: List[Model], n_splits: int = 5, random_state: int = 42):
        """Initialize the evaluator with models and evaluation parameters."""
        self.models = models
        self.n_splits = n_splits
        self.random_state = random_state
        np.random.seed(random_state)
        self.results: Dict[str, Dict[str, Any]] = {}

    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """Evaluate all models using k-fold cross-validation."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for model in self.models:
            print(f"\n=== K-Fold Evaluation for Model: {model.name} ===")
            fold_accuracies = []
            fold_f1_scores = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Train model (only show verbose output for first fold)
                model.train(X_train, y_train, verbose=(verbose and fold == 0))

                # Evaluate model
                y_pred = model.get_binary_predictions(X_test).flatten()
                y_true = y_test.astype(int).flatten()
                accuracy = accuracy_score(y_true, y_pred)
                fold_accuracies.append(accuracy)
                f1_score_value = f1_score(y_true, y_pred)
                fold_f1_scores.append(f1_score_value)

                print(f"Fold {fold + 1}/{self.n_splits}: Accuracy = {accuracy:.4f}; F1 Score = {f1_score_value:.4f}")

            # Compute statistics
            mean_acc = np.mean(fold_accuracies)
            std_acc = np.std(fold_accuracies)

            mean_f1 = np.mean(fold_f1_scores)
            std_f1 = np.std(fold_f1_scores)

            self.results[model.name] = {
                "fold_accuracies": fold_accuracies,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "mean_f1": mean_f1,
                "std_f1": std_f1
            }

            print(f"\nModel: {model.name}")
            print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

        return self.results

    def print_summary(self):
        """Print a summary of all evaluation results."""
        print("\n" + "=" * 60)
        print("FINAL K-FOLD COMPARISON RESULTS")
        print("=" * 60)
        for model_name, result in self.results.items():
            mean_acc = result['mean_accuracy']
            std_acc = result['std_accuracy']
            mean_f1 = result['mean_f1']
            std_f1 = result['std_f1']
            print(f"{model_name:<50}: ACC({mean_acc:.4f} ± {std_acc:.4f}); F1({mean_f1:.4f} {std_f1:.4f})")

    def best_model(self, metric: str) -> Tuple[str, Model]:
        """Identify the best performing model based on mean metric."""
        if not self.results:
            return ""
        if metric not in ["mean_accuracy", "std_accuracy", "mean_f1", "std_f1"]:
            raise ValueError(f"Invalid metric: {metric}")

        # Find model name with the best value for the given metric
        best_model_name = max(self.results, key=lambda name: self.results[name][metric])
        # Find the corresponding model object
        best_model_obj = next(model for model in self.models if model.name == best_model_name)
        return best_model_name, best_model_obj

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Get the complete evaluation results."""
        return self.results

    def test_model(self, X_new: np.ndarray, y_new: np.ndarray, model: Model = None, metric: str = "mean_accuracy") -> Tuple[float, float]:
        """Test a specific model (or the best model) on new, unseen data."""
        if model is None:
            model_name, model = self.best_model(metric)
            print(f"Using best model based on {metric}: {model_name}")

        # Make predictions
        y_pred = model.get_binary_predictions(X_new).flatten()
        y_true = y_new.astype(int).flatten()

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"\n=== Evaluation on New Data: {model.name} ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return accuracy, f1
