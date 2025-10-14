import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from hw1_part2_preprocessing import UnifiedDataPreprocessor

warnings.filterwarnings("ignore")

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
INCOME_MAPPING = {"<=50K": 0, ">50K": 1}


def train_and_evaluate_cart(x_train, y_train, x_test, y_test):
    """Trains and evaluates a CART model using scikit-learn's DecisionTreeClassifier.

    Uses Gini impurity criterion with default parameters and no pruning, which may lead to overfitting.
    Computes training/test accuracy, overfitting gap, classification report, and confusion matrix.

    Args:
        x_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target labels.
        x_test (pd.DataFrame): Test feature data.
        y_test (pd.Series): Test target labels.

    Returns:
        tuple: (model, test_pred, metrics)
            - model (DecisionTreeClassifier): Trained CART model.
            - test_pred (np.ndarray): Predictions on test set.
            - metrics (dict): Dictionary containing evaluation metrics (train_accuracy, test_accuracy,
              overfitting_gap, tree_nodes, tree_depth, train_samples, test_samples, features).

    Raises:
        ValueError: If input data is invalid or incompatible.
    """
    print("\nTraining CART model (Gini impurity, no pruning)...")

    model = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        ccp_alpha=0.0,
    )

    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    overfitting_gap = train_accuracy - test_accuracy

    metrics = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "overfitting_gap": overfitting_gap,
        "tree_nodes": model.tree_.node_count,
        "tree_depth": model.tree_.max_depth,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "features": x_train.shape[1],
    }

    print("\n=== CART Model Results ===")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Overfitting Gap: {overfitting_gap:.4f}")
    print(f"Tree Nodes: {model.tree_.node_count:,}")
    print(f"Tree Max Depth: {model.tree_.max_depth}")

    print("\n=== Test Set Classification Report ===")
    print(
        classification_report(
            y_test,
            test_pred,
            target_names=["<=50K", ">50K"],
            digits=4,
            zero_division=0,
        )
    )

    print("\n=== Test Set Confusion Matrix ===")
    cm = confusion_matrix(y_test, test_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["True <=50K", "True >50K"],
        columns=["Pred <=50K", "Pred >50K"],
    )
    print(cm_df)

    total_test = len(y_test)
    print(f"\n=== Error Analysis ===")
    print(f"Total Test Samples: {total_test:,}")
    print(f"Correct Predictions: {int(test_accuracy * total_test):,} ({test_accuracy:.1%})")
    print(
        f"Incorrect Predictions: {int((1 - test_accuracy) * total_test):,} ({(1 - test_accuracy):.1%})"
    )

    true_pos = cm[1, 1]
    false_neg = cm[1, 0]
    recall_high_income = (
        true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    )
    print(f">50K Class Recall: {recall_high_income:.4f} (Key Business Metric)")

    print("\n" + "=" * 60)
    return model, test_pred, metrics


def main():
    """Executes the complete CART model training and evaluation pipeline.

    Loads and preprocesses data, trains the CART model, and outputs evaluation metrics.

    Returns:
        tuple: (model, metrics, test_pred, y_test)
            - model (DecisionTreeClassifier): Trained CART model.
            - metrics (dict): Evaluation metrics.
            - test_pred (np.ndarray): Predictions on test set.
            - y_test (pd.Series): True test labels.

    Raises:
        FileNotFoundError: If required data files (adult.data, adult.test) are missing.
        Exception: For other unexpected errors during execution.
    """
    try:
        print("1. Loading and preprocessing data...")
        preprocessor = UnifiedDataPreprocessor()
        x_train, x_test, y_train, y_test = preprocessor.get_processed_data(
            discretize=False, validation_split=0.0
        )
        feature_names = preprocessor.get_feature_names()
        print(f"Number of Features: {len(feature_names)}")
        print(f"Feature Names: {feature_names}")
        print(f"Training Set Shape: {x_train.shape}, Test Set Shape: {x_test.shape}")

        print("\n2. Training CART model...")
        model, test_pred, metrics = train_and_evaluate_cart(x_train, y_train, x_test, y_test)

        print("\n=== Execution Summary ===")
        print("✅ CART baseline completed")
        print(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"   Overfitting Gap: {metrics['overfitting_gap']:.4f}")
        print(f"   Tree Complexity: {metrics['tree_nodes']:,} nodes")

        return model, metrics, test_pred, y_test

    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        print("Please ensure data/adult.data and data/adult.test exist")
        raise
    except Exception as e:
        print(f"❌ Execution Error: {e}")
        raise


if __name__ == "__main__":
    model, metrics, test_pred, y_test = main()