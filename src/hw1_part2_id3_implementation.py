import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from hw1_part2_preprocessing import UnifiedDataPreprocessor

warnings.filterwarnings("ignore")


class ID3DecisionTree:
    """Decision tree classifier using the ID3 algorithm with optional pruning.

    Attributes:
        max_depth (int, optional): Maximum depth of the tree. Defaults to None.
        min_samples_split (int): Minimum number of samples required to split. Defaults to 2.
        tree (dict): The constructed decision tree.
        feature_names (list): Names of features used in the tree.
        majority_class (int): Majority class for handling unknown values.
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        """Initialize the ID3 decision tree.

        Args:
            max_depth (int, optional): Maximum depth of the tree. Defaults to None.
            min_samples_split (int): Minimum number of samples required to split. Defaults to 2.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_names = None
        self.majority_class = None

    def compute_entropy(self, labels):
        """Calculate the entropy of a label set.

        Args:
            labels (np.ndarray): Array of class labels.

        Returns:
            float: Entropy value of the label set.
        """
        if len(labels) == 0:
            return 0
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def compute_information_gain(self, features, labels, feature_idx):
        """Calculate the information gain for a specific feature.

        Args:
            features (np.ndarray): Feature matrix.
            labels (np.ndarray): Array of class labels.
            feature_idx (int): Index of the feature to evaluate.

        Returns:
            float: Information gain for the specified feature.
        """
        parent_entropy = self.compute_entropy(labels)
        values, counts = np.unique(features[:, feature_idx], return_counts=True)
        weighted_entropy = 0

        total_samples = len(labels)
        for value, count in zip(values, counts):
            mask = features[:, feature_idx] == value
            child_samples = len(labels[mask])
            if child_samples > 0:
                child_entropy = self.compute_entropy(labels[mask])
                weighted_entropy += (child_samples / total_samples) * child_entropy

        return parent_entropy - weighted_entropy

    def build_tree(self, features, labels, depth=0, features_to_consider=None):
        """Recursively build the decision tree.

        Args:
            features (np.ndarray): Feature matrix.
            labels (np.ndarray): Array of class labels.
            depth (int): Current depth of the tree. Defaults to 0.
            features_to_consider (list, optional): List of feature indices to consider. Defaults to None.

        Returns:
            dict: Node of the decision tree.
        """
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            return {"class": int(unique_labels[0]), "samples": len(labels)}

        if self.max_depth is not None and depth >= self.max_depth:
            majority_class = int(np.argmax(np.bincount(labels)))
            return {"class": majority_class, "samples": len(labels)}

        if len(labels) < self.min_samples_split:
            majority_class = int(np.argmax(np.bincount(labels)))
            return {"class": majority_class, "samples": len(labels)}

        if features_to_consider is None:
            features_to_consider = list(range(features.shape[1]))

        if len(features_to_consider) == 0:
            majority_class = int(np.argmax(np.bincount(labels)))
            return {"class": majority_class, "samples": len(labels)}

        best_gain = -1
        best_feature = None
        for feature in features_to_consider:
            gain = self.compute_information_gain(features, labels, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        if best_feature is None or best_gain < 1e-5:
            majority_class = int(np.argmax(np.bincount(labels)))
            return {"class": majority_class, "samples": len(labels)}

        children = {}
        feature_values = np.unique(features[:, best_feature])
        for value in feature_values:
            mask = features[:, best_feature] == value
            child_features, child_labels = features[mask], labels[mask]
            if len(child_labels) == 0:
                majority_class = int(np.argmax(np.bincount(labels)))
                children[value] = {"class": majority_class, "samples": 0}
            else:
                new_features = [f for f in features_to_consider if f != best_feature]
                children[value] = self.build_tree(child_features, child_labels, depth + 1, new_features)

        return {
            "feature": best_feature,
            "children": children,
            "samples": len(labels),
            "class": int(np.argmax(np.bincount(labels))),
        }

    def fit(self, features, labels):
        """Train the decision tree model.

        Args:
            features (np.ndarray): Training feature matrix.
            labels (np.ndarray): Training class labels.

        Returns:
            ID3DecisionTree: Trained model instance.
        """
        self.majority_class = int(np.argmax(np.bincount(labels)))
        self.tree = self.build_tree(features, labels)
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        return self

    def predict_one(self, sample, node):
        """Predict the class for a single sample.

        Args:
            sample (np.ndarray): Single feature vector.
            node (dict): Current node in the decision tree.

        Returns:
            int: Predicted class label.
        """
        if "class" in node and "children" not in node:
            return node["class"]

        if "children" in node:
            feature_idx = node["feature"]
            feature_value = sample[feature_idx]
            if feature_value in node["children"]:
                return self.predict_one(sample, node["children"][feature_value])
            return node.get("class", self.majority_class)

        return node.get("class", self.majority_class)

    def predict(self, features):
        """Predict classes for a dataset.

        Args:
            features (np.ndarray): Feature matrix to predict.

        Returns:
            np.ndarray: Predicted class labels.

        Raises:
            ValueError: If the model is not fitted.
        """
        if self.tree is None:
            raise ValueError("Model must be fitted before prediction")

        predictions = []
        for i in range(features.shape[0]):
            try:
                pred = self.predict_one(features[i], self.tree)
                predictions.append(pred)
            except Exception as e:
                print(f"Prediction error for sample {i}: {e}")
                predictions.append(self.majority_class)

        return np.array(predictions)

    def prune(self, validation_features, validation_labels):
        """Apply post-pruning to the decision tree using validation data.

        Args:
            validation_features (np.ndarray): Validation feature matrix.
            validation_labels (np.ndarray): Validation class labels.

        Returns:
            ID3DecisionTree: Pruned model instance.
        """
        print("\nStarting post-pruning...")
        initial_acc = accuracy_score(validation_labels, self.predict(validation_features))
        print(f"Validation accuracy before pruning: {initial_acc:.4f}")

        self.tree = self._prune_node(self.tree, validation_features, validation_labels)

        final_acc = accuracy_score(validation_labels, self.predict(validation_features))
        print(f"Validation accuracy after pruning: {final_acc:.4f}")
        print(f"Accuracy change: {final_acc - initial_acc:+.4f}")

        return self

    def _prune_node(self, node, validation_features, validation_labels):
        """Recursively prune the decision tree.

        Args:
            node (dict): Current node in the decision tree.
            validation_features (np.ndarray): Validation feature matrix.
            validation_labels (np.ndarray): Validation class labels.

        Returns:
            dict: Pruned node.
        """
        if "children" not in node:
            return node

        for value in list(node["children"].keys()):
            node["children"][value] = self._prune_node(
                node["children"][value], validation_features, validation_labels
            )

        pred_before = self.predict(validation_features)
        error_before = np.mean(pred_before != validation_labels)

        original_children = node["children"]
        node["children"] = {}
        pred_after = self.predict(validation_features)
        error_after = np.mean(pred_after != validation_labels)

        if error_after <= error_before:
            del node["children"]
            del node["feature"]
            return node

        node["children"] = original_children
        return node

    def count_nodes(self):
        """Count the total number of nodes in the tree.

        Returns:
            int: Number of nodes in the tree.
        """
        def count_recursive(node):
            if "children" not in node:
                return 1
            return 1 + sum(count_recursive(child) for child in node["children"].values())

        return count_recursive(self.tree) if self.tree else 0

    def get_depth(self):
        """Calculate the depth of the tree.

        Returns:
            int: Depth of the decision tree.
        """
        def depth_recursive(node):
            if "children" not in node:
                return 1
            if not node["children"]:
                return 1
            return 1 + max(depth_recursive(child) for child in node["children"].values())

        return depth_recursive(self.tree) if self.tree else 0

    def print_tree(self, max_depth=None):
        """Print the structure of the decision tree.

        Args:
            max_depth (int, optional): Maximum depth to print. Defaults to None.
        """
        def print_recursive(node, depth=0, prefix=""):
            indent = "  " * depth
            if "class" in node and "children" not in node:
                print(f"{indent}{prefix}Class: {node['class']} (samples: {node['samples']})")
                return

            feature_name = (
                self.feature_names[node["feature"]]
                if self.feature_names
                else node["feature"]
            )
            print(f"{indent}{prefix}Feature {feature_name} (samples: {node['samples']})")

            if max_depth is not None and depth >= max_depth:
                print(f"{indent}  ... (depth limit reached)")
                return

            for value, child in sorted(node["children"].items(), key=lambda x: str(x[0])):
                print_recursive(child, depth + 1, f"{value} => ")

        print_recursive(self.tree)

    def save_model(self, path):
        """Save the model to a file.

        Args:
            path (str or Path): Path to save the model.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path):
        """Load a model from a file.

        Args:
            path (str or Path): Path to the model file.

        Returns:
            ID3DecisionTree: Loaded model instance.
        """
        with open(path, "rb") as f:
            return pickle.load(f)


def train_and_evaluate_id3(use_pruning=True):
    """Train and evaluate the ID3 decision tree model.

    Args:
        use_pruning (bool): Whether to apply post-pruning. Defaults to False.

    Returns:
        dict: Dictionary containing model, accuracies, and tree statistics.

    Raises:
        FileNotFoundError: If required data files are missing.
    """
    print("\n" + "=" * 60)
    print("Starting ID3 Decision Tree Training")
    print("=" * 60)

    try:
        preprocessor = UnifiedDataPreprocessor()
        train_features, val_features, test_features, train_labels, val_labels, test_labels = (
            preprocessor.get_processed_data(
                discretize=True, n_bins=10, validation_split=0.2, random_state=42
            )
        )
        feature_names = preprocessor.get_feature_names()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure the 'data' directory exists with adult.data and adult.test files")
        return None

    print(f"\nTraining ID3 model (max_depth=10, min_samples_split=20)...")
    id3_model = ID3DecisionTree(max_depth=10, min_samples_split=20)
    id3_model.feature_names = [f"{i}: {name}" for i, name in enumerate(feature_names)]
    id3_model.fit(train_features, train_labels)

    num_nodes = id3_model.count_nodes()
    tree_depth = id3_model.get_depth()
    print(f"\nTree Statistics:")
    print(f"  Number of nodes: {num_nodes}")
    print(f"  Tree depth: {tree_depth}")

    train_pred = id3_model.predict(train_features)
    val_pred = id3_model.predict(val_features)
    train_acc_before = accuracy_score(train_labels, train_pred)
    val_acc_before = accuracy_score(val_labels, val_pred)

    print(f"\nBefore Pruning:")
    print(f"  Training accuracy: {train_acc_before:.4f}")
    print(f"  Validation accuracy: {val_acc_before:.4f}")

    if use_pruning:
        id3_model.prune(val_features, val_labels)
        num_nodes_after = id3_model.count_nodes()
        tree_depth_after = id3_model.get_depth()
        print(f"\nTree Statistics After Pruning:")
        print(f"  Number of nodes: {num_nodes_after} (reduced by {num_nodes - num_nodes_after})")
        print(f"  Tree depth: {tree_depth_after}")

        train_pred = id3_model.predict(train_features)
        val_pred = id3_model.predict(val_features)
        train_acc_after = accuracy_score(train_labels, train_pred)
        val_acc_after = accuracy_score(val_labels, val_pred)

        print(f"\nAfter Pruning:")
        print(f"  Training accuracy: {train_acc_after:.4f} ({train_acc_after - train_acc_before:+.4f})")
        print(f"  Validation accuracy: {val_acc_after:.4f} ({val_acc_after - val_acc_before:+.4f})")
    else:
        train_acc_after, val_acc_after = train_acc_before, val_acc_before
        num_nodes_after, tree_depth_after = num_nodes, tree_depth

    if test_features is not None and test_labels is not None:
        test_pred = id3_model.predict(test_features)
        test_acc = accuracy_score(test_labels, test_pred)

        print(f"\n" + "=" * 60)
        print("Test Set Results:")
        print("=" * 60)
        print(f"Test accuracy: {test_acc:.4f}")

        print("\nClassification Report (Test Set):")
        print(classification_report(test_labels, test_pred, target_names=["<=50K", ">50K"]))

        print("\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(test_labels, test_pred)
        print(cm)
        print(f"  True Negatives:  {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives:  {cm[1,1]}")

    print("\n" + "=" * 60)
    print("Decision Tree Structure Preview (Top 3 Levels):")
    print("=" * 60)
    id3_model.print_tree(max_depth=3)

    model_path = Path.cwd() / "id3_model.pkl"
    id3_model.save_model(model_path)

    return {
        "model": id3_model,
        "train_acc": train_acc_after,
        "val_acc": val_acc_after,
        "test_acc": test_acc if test_features is not None else None,
        "feature_names": feature_names,
        "num_nodes": num_nodes_after,
        "tree_depth": tree_depth_after,
    }


def save_test_predictions(actual_labels, predicted_labels, algorithm_name):
    """Save test set predictions to a CSV file.

    Args:
        actual_labels (np.ndarray): Actual class labels.
        predicted_labels (np.ndarray): Predicted class labels.
        algorithm_name (str): Name of the algorithm used.

    Returns:
        None
    """
    Path("../results").mkdir(exist_ok=True)
    pd.DataFrame({"actual": actual_labels, "predicted": predicted_labels}).to_csv(
        f"../results/{algorithm_name.lower()}_predictions.csv", index=False
    )
    print(f"✓ {algorithm_name} predictions saved")


def main():
    """Main function to execute the ID3 decision tree training and evaluation.

    Returns:
        None
    """
    print("\n" + "=" * 60)
    print("ID3 Decision Tree Implementation")
    print("Includes: Discretization, Post-Pruning, Test Set Evaluation")
    print("=" * 60)

    results = train_and_evaluate_id3(use_pruning=True)

    if results is not None:
        print("\n" + "=" * 60)
        print("Training Complete! Final Results Summary")
        print("=" * 60)
        print(f"Training accuracy: {results['train_acc']:.4f}")
        print(f"Validation accuracy: {results['val_acc']:.4f}")
        if results["test_acc"] is not None:
            print(f"Test accuracy: {results['test_acc']:.4f}")
        print(f"Decision tree nodes: {results['num_nodes']}")
        print(f"Decision tree depth: {results['tree_depth']}")

        print("\n" + "=" * 60)
        print("To compare with no pruning, run:")
        print("train_and_evaluate_id3(use_pruning=False)")
        print("=" * 60)
    else:
        print("\nTraining failed. Please check data files and paths")


if __name__ == "__main__":
    main()