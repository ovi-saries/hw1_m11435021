"""
Cost Complexity Pruning Analysis for CART Decision Tree.

This module performs cost complexity pruning analysis on a CART decision tree using the adult dataset.
It generates plots for pruning metrics and compares classification accuracy for three ccp_alpha settings.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree

from hw1_part2_preprocessing import UnifiedDataPreprocessor

warnings.filterwarnings("ignore")

# Constants
RESULTS_DIR = "results/part3"
PLOT_DPI = 300
FIGURE_SIZE = (20, 10)
FONT_SIZE = 10
MAX_TREE_DEPTH = 3
RANDOM_STATE = 42


def train_and_evaluate_cart_pruned(
    x_train, y_train, x_test, y_test, ccp_alpha, feature_names, model_name
):
    """Train and evaluate a CART model with specified ccp_alpha.

    Args:
        x_train (array-like): Training feature data.
        y_train (array-like): Training target data.
        x_test (array-like): Testing feature data.
        y_test (array-like): Testing target data.
        ccp_alpha (float): Cost complexity pruning parameter.
        feature_names (list): Names of features.
        model_name (str): Name of the model for display purposes.

    Returns:
        tuple: Trained model, test predictions, and evaluation metrics.
    """
    print(f"\nTraining CART model (ccp_alpha={ccp_alpha})...")

    model = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        ccp_alpha=ccp_alpha,
    )

    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    overfitting_gap = train_accuracy - test_accuracy

    metrics = {
        "ccp_alpha": ccp_alpha,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "overfitting_gap": overfitting_gap,
        "tree_nodes": model.tree_.node_count,
        "tree_depth": model.tree_.max_depth,
    }

    print(f"\n=== {model_name} (ccp_alpha={ccp_alpha}) ===")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print(f"Overfitting Gap: {overfitting_gap:.4f}")
    print(f"Number of Nodes: {model.tree_.node_count:,}")
    print(f"Tree Depth: {model.tree_.max_depth}")

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
        cm, index=["True <=50K", "True >50K"], columns=["Pred <=50K", "Pred >50K"]
    )
    print(cm_df)

    plt.figure(figsize=FIGURE_SIZE)
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["<=50K", ">50K"],
        filled=True,
        max_depth=MAX_TREE_DEPTH,
        fontsize=FONT_SIZE,
    )
    plt.title(f"CART Decision Tree (ccp_alpha={ccp_alpha:.6f}, {model_name})")
    output_dir = Path(__file__).parent.parent.parent / "hw1_m11435021" / RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / f"cart_tree_ccp_alpha_{ccp_alpha:.6f}.png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )
    plt.close()

    return model, test_pred, metrics


def analyze_pruning_path(x_train, y_train, x_test, y_test):
    """Analyze the pruning path to identify candidate ccp_alpha values.

    Args:
        x_train (array-like): Training feature data.
        y_train (array-like): Training target data.
        x_test (array-like): Testing feature data.
        y_test (array-like): Testing target data.

    Returns:
        tuple: Three selected ccp_alpha values and pruning metrics dictionary.
    """
    print("\nAnalyzing pruning path...")
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    path = model.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    train_accuracies = []
    test_accuracies = []
    node_counts = []
    depths = []

    for ccp_alpha in ccp_alphas:
        model = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=ccp_alpha)
        model.fit(x_train, y_train)
        train_accuracies.append(accuracy_score(y_train, model.predict(x_train)))
        test_accuracies.append(accuracy_score(y_test, model.predict(x_test)))
        node_counts.append(model.tree_.node_count)
        depths.append(model.tree_.max_depth)

    idx_optimal = np.argmax(test_accuracies)
    ccp_alpha_light = 0.0
    ccp_alpha_optimal = ccp_alphas[idx_optimal]
    ccp_alpha_heavy = ccp_alphas[-2] if len(ccp_alphas) > 2 else ccp_alphas[-1]

    return (
        ccp_alpha_light,
        ccp_alpha_optimal,
        ccp_alpha_heavy,
        {
            "ccp_alphas": ccp_alphas[:-1],
            "impurities": impurities[:-1],
            "train_accuracies": train_accuracies[:-1],
            "test_accuracies": test_accuracies[:-1],
            "node_counts": node_counts[:-1],
            "depths": depths[:-1],
        },
    )


def plot_pruning_analysis(pruning_metrics, selected_alphas):
    """Generate plots for pruning analysis metrics.

    Args:
        pruning_metrics (dict): Dictionary containing pruning metrics.
        selected_alphas (list): List of three selected ccp_alpha values.
    """
    ccp_alphas = pruning_metrics["ccp_alphas"]
    impurities = pruning_metrics["impurities"]
    node_counts = pruning_metrics["node_counts"]
    depths = pruning_metrics["depths"]
    train_accuracies = pruning_metrics["train_accuracies"]
    test_accuracies = pruning_metrics["test_accuracies"]

    output_dir = Path(__file__).parent.parent.parent / "hw1_m11435021" / RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(
        ccp_alphas,
        impurities,
        marker="o",
        drawstyle="steps-post",
        linewidth=2,
        markersize=4,
    )
    plt.xlabel("Effective Alpha", fontsize=12)
    plt.ylabel("Total Impurity of Leaves", fontsize=12)
    plt.title("Total Impurity vs Effective Alpha for Training Set", fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "total_impurity_vs_alpha.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_dir / 'total_impurity_vs_alpha.png'}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(
        ccp_alphas,
        node_counts,
        marker="o",
        drawstyle="steps-post",
        linewidth=2,
        markersize=4,
    )
    ax1.set_xlabel("Alpha", fontsize=12)
    ax1.set_ylabel("Number of Nodes", fontsize=12)
    ax1.set_title("Number of Nodes vs Alpha", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        ccp_alphas,
        depths,
        marker="o",
        drawstyle="steps-post",
        linewidth=2,
        markersize=4,
    )
    ax2.set_xlabel("Alpha", fontsize=12)
    ax2.set_ylabel("Depth of Tree", fontsize=12)
    ax2.set_title("Depth vs Alpha", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "nodes_and_depth_vs_alpha.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_dir / 'nodes_and_depth_vs_alpha.png'}")

    plt.figure(figsize=(10, 6))
    line_train, = plt.plot(
        ccp_alphas,
        train_accuracies,
        marker="o",
        label="Train Accuracy",
        drawstyle="steps-post",
        linewidth=2,
        markersize=4,
    )
    line_test, = plt.plot(
        ccp_alphas,
        test_accuracies,
        marker="o",
        label="Test Accuracy",
        drawstyle="steps-post",
        linewidth=2,
        markersize=4,
    )

    colors = ["red", "green", "blue"]
    labels = ["Light Pruning (Overfit)", "Optimal Pruning (Balanced)", "Heavy Pruning (Underfit)"]

    for alpha, label, color in zip(selected_alphas, labels, colors):
        idx = np.argmin(np.abs(np.array(ccp_alphas) - alpha))
        plt.scatter(
            ccp_alphas[idx],
            train_accuracies[idx],
            color=color,
            s=150,
            marker="s",
            alpha=0.8,
            edgecolors="black",
            linewidth=1.5,
        )
        plt.scatter(
            ccp_alphas[idx],
            test_accuracies[idx],
            color=color,
            s=150,
            marker="D",
            alpha=0.8,
            edgecolors="black",
            linewidth=1.5,
        )
        plt.annotate(
            f"{label}\nα={ccp_alphas[idx]:.4f}",
            xy=(ccp_alphas[idx], test_accuracies[idx]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    plt.xlabel("Alpha", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Accuracy vs Alpha for Training and Testing Sets (CART Pruning Analysis)", fontsize=14, pad=20)
    plt.legend(handles=[line_train, line_test], fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_alpha_cart.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_dir / 'accuracy_vs_alpha_cart.png'}")

    print("\n=== Comparison of Three ccp_alpha Settings ===")
    print(
        f"Light Pruning (α={selected_alphas[0]:.6f}): Train Accuracy={train_accuracies[0]:.4f}, Test Accuracy={test_accuracies[0]:.4f}"
    )
    print(
        f"Optimal Pruning (α={selected_alphas[1]:.6f}): Train Accuracy={train_accuracies[np.argmax(test_accuracies)]:.4f}, Test Accuracy={max(test_accuracies):.4f}"
    )
    print(
        f"Heavy Pruning (α={selected_alphas[2]:.6f}): Train Accuracy={train_accuracies[-2]:.4f}, Test Accuracy={test_accuracies[-2]:.4f}"
    )


def main():
    """Perform pruning analysis and compare three ccp_alpha settings.

    Returns:
        list: List of metrics dictionaries for each ccp_alpha setting.

    Raises:
        Exception: If an error occurs during execution.
    """
    try:
        print("1. Loading and preprocessing data...")
        preprocessor = UnifiedDataPreprocessor()
        x_train, x_test, y_train, y_test = preprocessor.get_processed_data(
            discretize=False, validation_split=0.0
        )
        feature_names = preprocessor.get_feature_names()
        print(f"Number of Features: {len(feature_names)}")
        print(f"Training Set Shape: {x_train.shape}, Testing Set Shape: {x_test.shape}")

        print("\n2. Analyzing pruning path...")
        ccp_alpha_light, ccp_alpha_optimal, ccp_alpha_heavy, pruning_metrics = (
            analyze_pruning_path(x_train, y_train, x_test, y_test)
        )
        print(f"Selected ccp_alpha values:")
        print(f"  Light Pruning: {ccp_alpha_light}")
        print(f"  Optimal Pruning: {ccp_alpha_optimal}")
        print(f"  Heavy Pruning: {ccp_alpha_heavy}")

        settings = [
            (ccp_alpha_light, "Light Pruning (Overfit)"),
            (ccp_alpha_optimal, "Optimal Pruning (Balanced)"),
            (ccp_alpha_heavy, "Heavy Pruning (Underfit)"),
        ]
        all_metrics = []

        for ccp_alpha, name in settings:
            model, test_pred, metrics = train_and_evaluate_cart_pruned(
                x_train, y_train, x_test, y_test, ccp_alpha, feature_names, name
            )
            all_metrics.append(metrics)

        print("\n3. Plotting pruning analysis...")
        plot_pruning_analysis(
            pruning_metrics, [ccp_alpha_light, ccp_alpha_optimal, ccp_alpha_heavy]
        )

        print("\n=== Summary Comparison Table ===")
        summary_df = pd.DataFrame(all_metrics)
        summary_df_formatted = summary_df[
            [
                "ccp_alpha",
                "train_accuracy",
                "test_accuracy",
                "overfitting_gap",
                "tree_nodes",
                "tree_depth",
            ]
        ].copy()
        summary_df_formatted["ccp_alpha"] = summary_df_formatted["ccp_alpha"].round(6)
        summary_df_formatted["train_accuracy"] = summary_df_formatted["train_accuracy"].round(4)
        summary_df_formatted["test_accuracy"] = summary_df_formatted["test_accuracy"].round(4)
        summary_df_formatted["overfitting_gap"] = summary_df_formatted["overfitting_gap"].round(4)
        print(summary_df_formatted.to_string(index=False))

        print("\n=== Bias-Variance Tradeoff Analysis ===")
        print(
            "Light Pruning: High variance (overfitting), high training accuracy but lower test accuracy"
        )
        print("Optimal Pruning: Balanced bias and variance, highest test accuracy")
        print("Heavy Pruning: High bias (underfitting), low training and test accuracy")

        print("\n=== Execution Completed ===")
        output_dir = Path(__file__).parent.parent.parent / "hw1_m11435021" / RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Cost complexity pruning analysis completed")
        print("Generated plots saved as:")
        print(
            f"- {output_dir / 'total_impurity_vs_alpha.png'} (Total Impurity vs Effective Alpha)"
        )
        print(
            f"- {output_dir / 'nodes_and_depth_vs_alpha.png'} (Number of Nodes and Depth vs Alpha)"
        )
        print(f"- {output_dir / 'accuracy_vs_alpha_cart.png'} (Accuracy vs Alpha)")
        print(
            f"Decision tree plots saved as {output_dir / 'cart_tree_ccp_alpha_*.png'}"
        )
        print("All plots use default matplotlib English labels")

        return all_metrics

    except Exception as e:
        print(f"❌ Execution Error: {e}")
        raise


if __name__ == "__main__":
    all_metrics = main()