from models import LinearClassifierGD
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
        LinearClassifierGD(name="LinearClassifier Gradient Descent", epochs=10, lr=0.1),
    ]

    print(f"\nEvaluating {len(models)} models...")

    # Create evaluator and run k-fold cross-validation
    evaluator = Evaluator(models=models, n_splits=3, random_state=42)

    results = evaluator.evaluate(X, y, verbose=True)

    evaluator.print_summary()

    best_acc_name, best_model_acc = evaluator.best_model("mean_accuracy")
    if best_model_acc:
        print(f"\nBest performing model (ACC): {best_acc_name}")
        best_acc = results[best_acc_name]['mean_accuracy']
        best_std = results[best_acc_name]['std_accuracy']
        print(f"Best accuracy: {best_acc:.4f} ± {best_std:.4f}")

    best_f1_name, best_model_f1 = evaluator.best_model("mean_f1")
    if best_model_f1:
        print(f"\nBest performing model (F1): {best_f1_name}")
        best_f1 = results[best_f1_name]['mean_f1']
        best_std = results[best_f1_name]['std_f1']
        print(f"Best f1-score: {best_f1:.4f} ± {best_std:.4f}")

    X_test, y_test = DataLoader.load_spam_data('./data_test')

    evaluator.test_model(X_test, y_test, best_model_f1, metric="mean_f1")

if __name__ == '__main__':
    main()