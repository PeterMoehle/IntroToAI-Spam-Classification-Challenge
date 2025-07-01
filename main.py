from models import NaiveBayes, NeuralNetwork
from utils import DataLoader, Evaluator


def main():
    print("=" * 60)
    print("SPAM CLASSIFICATION CHALLENGE")
    print("=" * 60)
    print("Loading and preprocessing data...")

    # Load spam classification data
    X, y = DataLoader.load_spam_data('./data_train')

    # Define models to evaluate
    models = [
        NaiveBayes(name="Naive Bayes"),
        #NeuralNetwork(name="NN (Hinge Loss, Hidden=32, Epochs=15, LR=0.005)",
        #    hidden_dim=32, epochs=15, lr=0.005, loss_type="hinge"
        #),
        # NeuralNetwork(name="NN (Logistic Loss2, Hidden=16, Epochs=20, LR=0.01)",
        #    hidden_dim=16, epochs=20, lr=0.01, loss_type="logistic2"
        # ),
        #NeuralNetwork(name="NN (MSE Loss, Hidden=64, Epochs=25, LR=0.001)",
        #    hidden_dim=64, epochs=25, lr=0.001, loss_type="mse"
        #),
    ]

    print(f"\nEvaluating {len(models)} models...")

    # Create evaluator and run k-fold cross-validation
    evaluator = Evaluator(models=models, n_splits=3, random_state=42)

    results = evaluator.evaluate(X, y, verbose=True)
    evaluator.print_summary()

    best_model = evaluator.best_model()
    if best_model:
        print(f"\nBest performing model: {best_model}")
        best_acc = results[best_model]['mean_accuracy']
        best_std = results[best_model]['std_accuracy']
        print(f"Best accuracy: {best_acc:.4f} ± {best_std:.4f}")

    print("\nEvaluation completed successfully!")


if __name__ == '__main__':
    main()