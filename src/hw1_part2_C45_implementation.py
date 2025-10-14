#!/usr/bin/env python3
"""
C4.5 Decision Tree Implementation - Refactored Version
Handles node counting using @property for real-time computation.
"""

import math
import os
import time
import warnings
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

from hw1_part2_preprocessing import UnifiedDataPreprocessor


class Node:
    """Represents a node in the decision tree."""

    def __init__(self, is_leaf=False, label=None, threshold=None, attribute=None, attribute_idx=None):
        """
        Initialize a decision tree node.

        Args:
            is_leaf (bool): Whether the node is a leaf.
            label (int): Class label for leaf nodes or attribute name for internal nodes.
            threshold (float): Splitting threshold for continuous attributes.
            attribute (str): Name of the splitting attribute.
            attribute_idx (int): Index of the splitting attribute.
        """
        self.is_leaf = is_leaf
        self.label = label
        self.attribute = attribute
        self.attribute_idx = attribute_idx
        self.threshold = threshold
        self.children = {}
        self.samples = 0
        self.class_distribution = {}

    def __repr__(self):
        """
        Return a string representation of the node.

        Returns:
            str: String describing the node.
        """
        if self.is_leaf:
            return f"Leaf({self.label}, samples={self.samples})"
        return f"Node({self.attribute}, threshold={self.threshold}, samples={self.samples})"


class C45DecisionTree:
    """C4.5 Decision Tree implementation with post-pruning and real-time node counting."""

    DEFAULT_CLASSES = [0, 1]  # 0: <=50K, 1: >50K

    def __init__(
        self,
        max_depth=None,
        min_samples_split=20,
        min_samples_leaf=10,
        min_gain_ratio=0.01,
        pruning=True,
        validation_split=0.1,
        data_dir="../data",
    ):
        """
        Initialize the C4.5 Decision Tree.

        Args:
            max_depth (int): Maximum depth of the tree.
            min_samples_split (int): Minimum number of samples required to split an internal node.
            min_samples_leaf (int): Minimum number of samples required at a leaf node.
            min_gain_ratio (float): Minimum gain ratio threshold for splitting.
            pruning (bool): Whether to enable post-pruning.
            validation_split (float): Proportion of data to use for validation (for pruning).
            data_dir (str): Directory path for data files.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_ratio = min_gain_ratio
        self.pruning = pruning
        self.validation_split = validation_split
        self.data_dir = data_dir

        self.tree = None
        self.is_fitted = False
        self.classes = self.DEFAULT_CLASSES
        self.attributes = []
        self.attribute_types = {}
        self.attribute_indices = {}
        self.continuous_thresholds = {}
        self.preprocessor = None
        self.train_time = 0
        self.predict_time = 0
        self.x_test = None
        self.y_test = None

    @property
    def n_nodes(self):
        """
        Compute the total number of nodes in the tree.

        Returns:
            int: Total number of nodes.
        """
        if self.tree is None:
            return 0
        return self._count_all_nodes(self.tree)

    @property
    def n_leaves(self):
        """
        Compute the number of leaf nodes in the tree.

        Returns:
            int: Number of leaf nodes.
        """
        if self.tree is None:
            return 0
        return self._count_leaves(self.tree)

    def _count_all_nodes(self, node):
        """
        Recursively count all nodes in the tree.

        Args:
            node (Node): Current node to count.

        Returns:
            int: Number of nodes in the subtree.
        """
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        count = 1
        for child in node.children.values():
            count += self._count_all_nodes(child)
        return count

    def _count_leaves(self, node):
        """
        Recursively count leaf nodes in the tree.

        Args:
            node (Node): Current node to count.

        Returns:
            int: Number of leaf nodes in the subtree.
        """
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        count = 0
        for child in node.children.values():
            count += self._count_leaves(child)
        return count

    def _setup_attributes(self):
        """
        Set up attribute types and indices based on the preprocessor.
        """
        self.attributes = self.preprocessor.get_feature_names()
        for attr in self.attributes:
            self.attribute_types[attr] = (
                "continuous" if attr in self.preprocessor.CONTINUOUS_FEATURES else "discrete"
            )
        self.attribute_indices = {attr: idx for idx, attr in enumerate(self.attributes)}

    def fit(self):
        """
        Train the C4.5 Decision Tree.

        Returns:
            C45DecisionTree: Self reference.
        """
        print("=" * 60)
        print("C4.5 Decision Tree Training Started")
        print("=" * 60)
        start_time = time.time()

        self.preprocessor = UnifiedDataPreprocessor(data_dir=self.data_dir)
        if self.pruning and self.validation_split > 0:
            x_train, x_val, x_test, y_train, y_val, y_test = self.preprocessor.get_processed_data(
                discretize=False,
                validation_split=self.validation_split,
                random_state=42,
                verbose=True,
            )
            self.x_test = x_test
            self.y_test = y_test
        else:
            x_train, x_test, y_train, y_test = self.preprocessor.get_processed_data(
                discretize=False,
                validation_split=0.0,
                verbose=True,
            )
            x_val, y_val = None, None
            self.x_test = x_test
            self.y_test = y_test

        self._setup_attributes()
        print(f"✓ Attributes set up: {len(self.attributes)} features")

        print("⏳ Preprocessing continuous attributes...")
        self._preprocess_continuous_attributes(x_train)

        print("⏳ Building decision tree structure...")
        self.tree = self._build_tree(x_train, y_train, list(range(len(self.attributes))), depth=0)

        nodes_before_prune = self.n_nodes
        if self.pruning and x_val is not None:
            print("⏳ Performing post-pruning...")
            self._post_prune(self.tree, x_val, y_val)
            print(f"✓ Pruning completed: {nodes_before_prune} → {self.n_nodes} nodes")

        self.is_fitted = True
        self.train_time = time.time() - start_time
        self._print_tree_statistics()
        print(f"\n✓ Training completed, total time: {self.train_time:.2f} seconds")
        print("=" * 60)

        return self

    def _preprocess_continuous_attributes(self, x):
        """
        Preprocess continuous attributes by sorting and caching candidate thresholds.

        Args:
            x (np.ndarray): Feature matrix.
        """
        for attr in self.attributes:
            if self.attribute_types[attr] == "continuous":
                attr_idx = self.attribute_indices[attr]
                values = np.sort(np.unique(x[:, attr_idx]))
                if len(values) > 1:
                    thresholds = [(values[i] + values[i + 1]) / 2 for i in range(min(len(values) - 1, 100))]
                    self.continuous_thresholds[attr] = thresholds
                else:
                    self.continuous_thresholds[attr] = []

    def _build_tree(self, x, y, available_attrs, depth):
        """
        Recursively build the decision tree using C4.5 algorithm.

        Args:
            x (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
            available_attrs (list): Indices of available attributes for splitting.
            depth (int): Current depth of the tree.

        Returns:
            Node: Root node of the subtree.
        """
        n_samples = len(x)
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)

        if n_samples < self.min_samples_split or len(class_counts) == 1:
            return self._create_leaf(majority_class, class_counts, n_samples)
        if not available_attrs or (self.max_depth is not None and depth >= self.max_depth):
            return self._create_leaf(majority_class, class_counts, n_samples)

        best_attr_idx, best_threshold, best_subsets = self._find_best_split(x, y, available_attrs)
        if best_attr_idx is None:
            return self._create_leaf(majority_class, class_counts, n_samples)

        best_attr = self.attributes[best_attr_idx]
        node = Node(
            is_leaf=False,
            attribute=best_attr,
            attribute_idx=best_attr_idx,
            threshold=best_threshold,
        )
        node.samples = n_samples
        node.class_distribution = class_counts

        remaining_attrs = [a for a in available_attrs if a != best_attr_idx]
        for branch_value, (x_subset, y_subset) in best_subsets.items():
            if len(x_subset) >= self.min_samples_leaf:
                child = self._build_tree(x_subset, y_subset, remaining_attrs, depth + 1)
                node.children[branch_value] = child
            else:
                subset_counts = Counter(y_subset)
                subset_majority = max(subset_counts, key=subset_counts.get)
                node.children[branch_value] = self._create_leaf(subset_majority, subset_counts, len(x_subset))

        if not node.children:
            return self._create_leaf(majority_class, class_counts, n_samples)

        return node

    def _create_leaf(self, label, class_counts, n_samples):
        """
        Create a leaf node.

        Args:
            label (int): Class label for the leaf.
            class_counts (Counter): Distribution of classes in the node.
            n_samples (int): Number of samples in the node.

        Returns:
            Node: Leaf node.
        """
        leaf = Node(is_leaf=True, label=label)
        leaf.samples = n_samples
        leaf.class_distribution = class_counts
        return leaf

    def _find_best_split(self, x, y, available_attrs):
        """
        Find the best attribute and threshold for splitting.

        Args:
            x (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
            available_attrs (list): Indices of available attributes.

        Returns:
            tuple: (best_attr_idx, best_threshold, best_subsets)
        """
        best_gain_ratio = self.min_gain_ratio
        best_attr_idx = None
        best_threshold = None
        best_subsets = None

        for attr_idx in available_attrs:
            attr = self.attributes[attr_idx]
            if self.attribute_types[attr] == "discrete":
                subsets = self._split_discrete(x, y, attr_idx)
                if len(subsets) > 1:
                    gain_ratio = self._calculate_gain_ratio(y, subsets)
                    if gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_attr_idx = attr_idx
                        best_threshold = None
                        best_subsets = subsets
            else:
                if attr in self.continuous_thresholds:
                    for threshold in self.continuous_thresholds[attr]:
                        subsets = self._split_continuous(x, y, attr_idx, threshold)
                        if len(subsets) == 2:
                            gain_ratio = self._calculate_gain_ratio(y, subsets)
                            if gain_ratio > best_gain_ratio:
                                best_gain_ratio = gain_ratio
                                best_attr_idx = attr_idx
                                best_threshold = threshold
                                best_subsets = subsets

        return best_attr_idx, best_threshold, best_subsets

    def _split_discrete(self, x, y, attr_idx):
        """
        Split data based on a discrete attribute.

        Args:
            x (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
            attr_idx (int): Index of the attribute to split on.

        Returns:
            dict: Subsets of data for each attribute value.
        """
        subsets = {}
        for i in range(len(x)):
            value = str(x[i, attr_idx])
            if value not in subsets:
                subsets[value] = ([], [])
            subsets[value][0].append(x[i])
            subsets[value][1].append(y[i])
        return {k: (np.array(v[0]), np.array(v[1])) for k, v in subsets.items() if len(v[0]) > 0}

    def _split_continuous(self, x, y, attr_idx, threshold):
        """
        Split data based on a continuous attribute and threshold.

        Args:
            x (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
            attr_idx (int): Index of the attribute to split on.
            threshold (float): Threshold value for splitting.

        Returns:
            dict: Subsets of data for left and right branches.
        """
        mask = x[:, attr_idx] <= threshold
        return {"left": (x[mask], y[mask]), "right": (x[~mask], y[~mask])}

    def _calculate_gain_ratio(self, y, subsets):
        """
        Compute the gain ratio for a split.

        Args:
            y (np.ndarray): Target labels.
            subsets (dict): Subsets of data after splitting.

        Returns:
            float: Gain ratio of the split.
        """
        gain = self._information_gain(y, subsets)
        split_info = self._split_information(y, subsets)
        return 0 if split_info == 0 else gain / split_info

    def _information_gain(self, y, subsets):
        """
        Compute the information gain for a split.

        Args:
            y (np.ndarray): Target labels.
            subsets (dict): Subsets of data after splitting.

        Returns:
            float: Information gain of the split.
        """
        total_entropy = self._entropy(y)
        n_total = len(y)
        weighted_entropy = 0
        for x_subset, y_subset in subsets.values():
            if len(y_subset) > 0:
                weight = len(y_subset) / n_total
                weighted_entropy += weight * self._entropy(y_subset)
        return total_entropy - weighted_entropy

    def _split_information(self, y, subsets):
        """
        Compute the split information for a split.

        Args:
            y (np.ndarray): Target labels.
            subsets (dict): Subsets of data after splitting.

        Returns:
            float: Split information.
        """
        n_total = len(y)
        split_info = 0
        for x_subset, y_subset in subsets.values():
            if len(y_subset) > 0:
                proportion = len(y_subset) / n_total
                split_info -= proportion * math.log2(proportion)
        return split_info

    def _entropy(self, y):
        """
        Compute the entropy of a set of labels.

        Args:
            y (np.ndarray): Target labels.

        Returns:
            float: Entropy value.
        """
        if len(y) == 0:
            return 0
        class_counts = Counter(y)
        entropy = 0
        n_samples = len(y)
        for count in class_counts.values():
            if count > 0:
                p = count / n_samples
                entropy -= p * math.log2(p)
        return entropy

    def _post_prune(self, node, x_val, y_val):
        """
        Perform post-pruning on the decision tree using pessimistic pruning.

        Args:
            node (Node): Current node to prune.
            x_val (np.ndarray): Validation feature matrix.
            y_val (np.ndarray): Validation target labels.
        """
        if node is None or node.is_leaf:
            return
        for child in list(node.children.values()):
            self._post_prune(child, x_val, y_val)
        error_before = self._evaluate_node(node, x_val, y_val)
        original_is_leaf = node.is_leaf
        original_label = node.label
        original_children = node.children.copy()
        majority_class = max(node.class_distribution, key=node.class_distribution.get)
        node.is_leaf = True
        node.label = majority_class
        node.children = {}
        error_after = self._evaluate_node(node, x_val, y_val)
        if error_after > error_before:
            node.is_leaf = original_is_leaf
            node.label = original_label
            node.children = original_children

    def _evaluate_node(self, node, x, y):
        """
        Evaluate the error rate of a node on validation data.

        Args:
            node (Node): Node to evaluate.
            x (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.

        Returns:
            float: Error rate.
        """
        if len(x) == 0:
            return 0
        predictions = np.array([self._predict_one(node, x_i) for x_i in x])
        errors = np.sum(predictions != y)
        return errors / len(y)

    def predict(self):
        """
        Predict using the trained decision tree.

        Returns:
            tuple: Predicted labels and true labels.

        Raises:
            ValueError: If the model has not been trained.
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() first.")
        print("\n" + "=" * 60)
        print("C4.5 Prediction Started")
        print("=" * 60)
        start_time = time.time()
        x_test = self.x_test
        y_test = self.y_test
        print(f"✓ Using loaded test data: {len(x_test)} samples")
        predictions = np.array([self._predict_one(self.tree, x) for x in x_test])
        self.predict_time = time.time() - start_time
        print(f"✓ Prediction completed, time: {self.predict_time:.2f} seconds")
        print(f"  Average prediction speed: {len(x_test)/self.predict_time:.0f} samples/second")
        print("=" * 60)
        return predictions, y_test

    def _predict_one(self, node, x):
        """
        Predict the class for a single sample.

        Args:
            node (Node): Current node in the tree.
            x (np.ndarray): Feature vector.

        Returns:
            int: Predicted class label.
        """
        if node.is_leaf:
            return node.label
        attr_idx = node.attribute_idx
        attr_value = x[attr_idx]
        if node.threshold is None:
            branch_key = str(attr_value)
            if branch_key in node.children:
                return self._predict_one(node.children[branch_key], x)
            return max(node.class_distribution, key=node.class_distribution.get)
        branch_key = "left" if attr_value <= node.threshold else "right"
        if branch_key in node.children:
            return self._predict_one(node.children[branch_key], x)
        return max(node.class_distribution, key=node.class_distribution.get)

    def evaluate(self, y_true, y_pred):
        """
        Evaluate prediction results.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            dict: Evaluation metrics including accuracy, confusion matrix, predictions, and true labels.
        """
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred, labels=self.classes)
        print(f"                Predicted")
        print(f"              <=50K  >50K")
        print(f"Actual <=50K   {cm[0,0]:6d} {cm[0,1]:5d}")
        print(f"       >50K    {cm[1,0]:6d} {cm[1,1]:5d}")
        print("\nClassification Report:")
        class_names = ["<=50K", ">50K"]
        print(classification_report(y_true, y_pred, target_names=class_names))
        print("Prediction Distribution:")
        pred_counts = Counter(y_pred)
        for i, cls_name in enumerate(class_names):
            count = pred_counts.get(i, 0)
            percentage = count / len(y_pred) * 100
            print(f"  {cls_name}: {count:5d} ({percentage:5.2f}%)")
        print("=" * 60)
        return {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "predictions": y_pred,
            "true_labels": y_true,
        }

    def _print_tree_statistics(self):
        """
        Display decision tree statistics.
        """
        print("\n" + "-" * 60)
        print("Decision Tree Statistics:")
        print(f"  Total nodes: {self.n_nodes}")
        print(f"  Leaf nodes: {self.n_leaves}")
        print(f"  Internal nodes: {self.n_nodes - self.n_leaves}")
        print(f"  Tree depth: {self._calculate_depth(self.tree)}")
        print(f"  Average leaf samples: {self._calculate_avg_leaf_samples(self.tree):.1f}")
        print("-" * 60)

    def _calculate_depth(self, node):
        """
        Calculate the depth of the tree.

        Args:
            node (Node): Current node.

        Returns:
            int: Depth of the subtree.
        """
        if node is None or node.is_leaf:
            return 1
        if not node.children:
            return 1
        return 1 + max(self._calculate_depth(child) for child in node.children.values())

    def _calculate_avg_leaf_samples(self, node):
        """
        Calculate the average number of samples in leaf nodes.

        Args:
            node (Node): Current node.

        Returns:
            float: Average number of samples in leaf nodes.
        """
        if node is None:
            return 0
        if node.is_leaf:
            return node.samples
        if not node.children:
            return node.samples
        leaf_samples = []
        self._collect_leaf_samples(node, leaf_samples)
        return sum(leaf_samples) / len(leaf_samples) if leaf_samples else 0

    def _collect_leaf_samples(self, node, leaf_samples):
        """
        Collect the number of samples in all leaf nodes.

        Args:
            node (Node): Current node.
            leaf_samples (list): List to store leaf node sample counts.
        """
        if node is None:
            return
        if node.is_leaf:
            leaf_samples.append(node.samples)
        else:
            for child in node.children.values():
                self._collect_leaf_samples(child, leaf_samples)


def main():
    """
    Execute the complete C4.5 Decision Tree workflow with unified data preprocessing.
    """
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + " " * 8 + "C4.5 Decision Tree - Unified Preprocessing" + " " * 14 + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60 + "\n")

    data_dir = "../data"
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory {data_dir} not found")
        return

    model = C45DecisionTree(
        max_depth=15,
        min_samples_split=50,
        min_samples_leaf=20,
        min_gain_ratio=0.01,
        pruning=True,
        validation_split=0.1,
        data_dir=data_dir,
    )
    model.fit()
    predictions, true_labels = model.predict()
    results = model.evaluate(true_labels, predictions)
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + " " * 20 + "Execution Completed" + " " * 26 + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    print(f"\n🎯 Final accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"⏱️  Total execution time: {model.train_time + model.predict_time:.2f} seconds")
    print(f"   - Training time: {model.train_time:.2f} seconds")
    print(f"   - Prediction time: {model.predict_time:.2f} seconds")
    print("\n✅ C4.5 Decision Tree executed successfully!\n")


if __name__ == "__main__":
    main()