#!/usr/bin/env python3

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

warnings.filterwarnings("ignore")

# Constants
DATA_DIR = "data"
RESULTS_DIR = "results"
OUTPUT_FILENAME = "decision_trees_comparison.xlsx"


def setup_paths() -> Dict[str, Path]:
    """Set up project directory paths.

    Creates necessary directories if they do not exist.

    Returns:
        Dict[str, Path]: Dictionary containing paths for source, data, and results directories.
    """
    current_dir = Path(__file__).parent
    project_root = current_dir.parent

    paths = {
        "src": current_dir,
        "data": project_root / DATA_DIR,
        "results": project_root / RESULTS_DIR,
    }

    paths["results"].mkdir(exist_ok=True, parents=True)

    return paths


class AlgorithmRunner:
    """Interface for running decision tree algorithms."""

    def __init__(self, paths: Dict[str, Path]):
        """Initialize AlgorithmRunner with project paths.

        Args:
            paths (Dict[str, Path]): Dictionary of project directory paths.
        """
        self.paths = paths
        self.results = {}

    def run_id3(self) -> Dict[str, Any]:
        """Execute ID3 decision tree algorithm.

        Returns:
            Dict[str, Any]: Dictionary containing algorithm name, true labels, predictions,
                test accuracy, and model.

        Raises:
            Exception: If ID3 execution fails.
        """
        print("\n" + "=" * 70)
        print("Running ID3 Decision Tree")
        print("=" * 70)

        try:
            from sklearn.metrics import accuracy_score
            from hw1_part2_id3_implementation import ID3DecisionTree
            from hw1_part2_preprocessing import UnifiedDataPreprocessor

            preprocessor = UnifiedDataPreprocessor(data_dir=str(self.paths["data"]))
            x_train, x_val, x_test, y_train, y_val, y_test = (
                preprocessor.get_processed_data(
                    discretize=True,
                    n_bins=10,
                    validation_split=0.2,
                    random_state=42,
                    verbose=False,
                )
            )

            print("Training ID3 model...")
            model = ID3DecisionTree(max_depth=10, min_samples_split=20)
            feature_names = preprocessor.get_feature_names()
            model.feature_names = [f"{i}: {name}" for i, name in enumerate(feature_names)]
            model.fit(x_train, y_train)

            print("Performing ID3 post-pruning...")
            model.prune(x_val, y_val)

            y_pred = model.predict(x_test)
            test_acc = accuracy_score(y_test, y_pred)

            print(f"✓ ID3 completed - Test accuracy: {test_acc:.4f}")

            return {
                "name": "ID3",
                "y_true": y_test,
                "y_pred": y_pred,
                "test_acc": test_acc,
                "model": model,
            }

        except Exception as e:
            print(f"✗ ID3 execution error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_cart(self) -> Dict[str, Any]:
        """Execute CART decision tree algorithm.

        Returns:
            Dict[str, Any]: Dictionary containing algorithm name, true labels, predictions,
                test accuracy, model, and metrics.

        Raises:
            Exception: If CART execution fails.
        """
        print("\n" + "=" * 70)
        print("Running CART Decision Tree")
        print("=" * 70)

        try:
            from hw1_part2_cart_implementation import train_and_evaluate_cart
            from hw1_part2_preprocessing import UnifiedDataPreprocessor

            preprocessor = UnifiedDataPreprocessor(data_dir=str(self.paths["data"]))
            x_train, x_test, y_train, y_test = preprocessor.get_processed_data(
                discretize=False, validation_split=0.0, verbose=False
            )

            model, test_pred, metrics = train_and_evaluate_cart(
                x_train, y_train, x_test, y_test
            )

            print(f"✓ CART completed - Test accuracy: {metrics['test_accuracy']:.4f}")

            return {
                "name": "CART",
                "y_true": y_test,
                "y_pred": test_pred,
                "test_acc": metrics["test_accuracy"],
                "model": model,
                "metrics": metrics,
            }

        except Exception as e:
            print(f"✗ CART execution error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_c45(self) -> Dict[str, Any]:
        """Execute C4.5 decision tree algorithm.

        Returns:
            Dict[str, Any]: Dictionary containing algorithm name, true labels, predictions,
                test accuracy, and model.

        Raises:
            Exception: If C4.5 execution fails.
        """
        print("\n" + "=" * 70)
        print("Running C4.5 Decision Tree")
        print("=" * 70)

        try:
            from hw1_part2_C45_implementation import C45DecisionTree

            model = C45DecisionTree(
                max_depth=15,
                min_samples_split=50,
                min_samples_leaf=20,
                min_gain_ratio=0.01,
                pruning=True,
                validation_split=0.1,
                data_dir=str(self.paths["data"]),
            )

            model.fit()

            y_pred, y_true = model.predict()

            test_acc = accuracy_score(y_true, y_pred)

            print(f"✓ C4.5 completed - Test accuracy: {test_acc:.4f}")

            return {
                "name": "C4.5",
                "y_true": y_true,
                "y_pred": y_pred,
                "test_acc": test_acc,
                "model": model,
            }

        except Exception as e:
            print(f"✗ C4.5 execution error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_c50(self) -> Dict[str, Any]:
        """Execute C5.0 decision tree algorithm.

        Returns:
            Dict[str, Any]: Dictionary containing algorithm name, true labels, predictions,
                test accuracy, and results.

        Raises:
            Exception: If C5.0 execution fails.
        """
        print("\n" + "=" * 70)
        print("Running C5.0 Decision Tree")
        print("=" * 70)

        try:
            from hw1_part2_C50_implementation import C50Config, C50Experiment

            config = C50Config(
                data_dir=self.paths["data"],
                results_dir=self.paths["results"] / "c50_temp",
                use_validation=True,
                validation_size=0.2,
                n_bins=10,
                trials=10,
                use_cost_matrix=False,
                random_state=42,
                verbose=False,
            )

            experiment = C50Experiment(config)
            results = experiment.run()

            test_results = results["test_results"]
            y_true = test_results["y_true"]
            y_pred = test_results["y_pred"]
            test_acc = test_results["accuracy"]

            print(f"✓ C5.0 completed - Test accuracy: {test_acc:.4f}")

            return {
                "name": "C5.0",
                "y_true": y_true,
                "y_pred": y_pred,
                "test_acc": test_acc,
                "results": results,
            }

        except Exception as e:
            print(f"✗ C5.0 execution error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_all(self) -> Dict[str, Any]:
        """Execute all decision tree algorithms.

        Returns:
            Dict[str, Any]: Dictionary containing results of all algorithms.
        """
        print("\n" + "█" * 70)
        print("█" + " " * 68 + "█")
        print("█" + " " * 15 + "Decision Tree Comparison System" + " " * 28 + "█")
        print("█" + " " * 68 + "█")
        print("█" * 70)

        start_time = time.time()

        algorithms = {
            "ID3": self.run_id3,
            "CART": self.run_cart,
            "C4.5": self.run_c45,
            "C5.0": self.run_c50,
        }

        for algo_name, runner in algorithms.items():
            result = runner()
            if result is not None:
                self.results[algo_name] = result
            else:
                print(f"⚠️ {algo_name} skipped (execution failed)")

        total_time = time.time() - start_time

        print("\n" + "=" * 70)
        print(f"All algorithms completed, total time: {total_time:.2f} seconds")
        print(f"Successfully executed: {len(self.results)}/4 algorithms")
        print("=" * 70)

        return self.results


class MetricsCalculator:
    """Calculator for evaluation metrics."""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            Dict[str, float]: Dictionary containing accuracy, precision, recall, F1-score,
                specificity, NPV, and confusion matrix components.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "npv": npv,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "confusion_matrix": cm,
        }

    @staticmethod
    def calculate_all_metrics(results: Dict[str, Any]) -> pd.DataFrame:
        """Calculate metrics for all algorithms.

        Args:
            results (Dict[str, Any]): Dictionary of algorithm results.

        Returns:
            pd.DataFrame: DataFrame containing metrics for each algorithm.
        """
        metrics_list = []

        for algo_name, result in results.items():
            metrics = MetricsCalculator.calculate_metrics(
                result["y_true"], result["y_pred"]
            )

            metrics_list.append(
                {
                    "algorithm": algo_name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "specificity": metrics["specificity"],
                    "npv": metrics["npv"],
                    "true_negatives": metrics["true_negatives"],
                    "false_positives": metrics["false_positives"],
                    "false_negatives": metrics["false_negatives"],
                    "true_positives": metrics["true_positives"],
                }
            )

        return pd.DataFrame(metrics_list)


class ExcelExporter:
    """Manager for exporting results to Excel."""

    def __init__(self, output_path: Path):
        """Initialize ExcelExporter with output file path.

        Args:
            output_path (Path): Path to the output Excel file.
        """
        self.output_path = output_path

    def export_comparison(self, results: Dict[str, Any], metrics_df: pd.DataFrame):
        """Export comprehensive comparison results to Excel.

        Args:
            results (Dict[str, Any]): Dictionary of algorithm results.
            metrics_df (pd.DataFrame): DataFrame containing evaluation metrics.

        Raises:
            Exception: If Excel export fails.
        """
        print("\n" + "=" * 70)
        print("Generating Excel comparison report")
        print("=" * 70)

        try:
            with pd.ExcelWriter(self.output_path, engine="openpyxl") as writer:
                self._export_predictions(results, writer)
                self._export_metrics(metrics_df, writer)
                self._export_confusion_matrices(results, writer)
                self._export_detailed_stats(results, metrics_df, writer)

            print(f"✓ Excel file saved: {self.output_path}")
            print(f"  File size: {self.output_path.stat().st_size / 1024:.2f} KB")

        except Exception as e:
            print(f"✗ Excel export error: {e}")
            raise

    def _export_predictions(self, results: Dict[str, Any], writer):
        """Export prediction comparison to Excel sheet.

        Args:
            results (Dict[str, Any]): Dictionary of algorithm results.
            writer: Excel writer object.
        """
        print("  Generating Sheet 1: Prediction Comparison...")

        first_algo = list(results.keys())[0]
        y_true = results[first_algo]["y_true"]
        n_samples = len(y_true)

        pred_df = pd.DataFrame(
            {
                "sample_id": range(1, n_samples + 1),
                "true_label": ["<=50K" if y == 0 else ">50K" for y in y_true],
                "true_label_code": y_true,
            }
        )

        for algo_name, result in results.items():
            y_pred = result["y_pred"]
            pred_df[f"{algo_name}_predicted"] = [
                "<=50K" if y == 0 else ">50K" for y in y_pred
            ]
            pred_df[f"{algo_name}_correct"] = (y_true == y_pred).astype(int)

        pred_cols = [f"{algo}_predicted" for algo in results.keys()]
        pred_df["all_agree"] = pred_df[pred_cols].nunique(axis=1) == 1

        pred_df.to_excel(writer, sheet_name="prediction_comparison", index=False)

        print(f"    ✓ {n_samples} test samples processed")

    def _export_metrics(self, metrics_df: pd.DataFrame, writer):
        """Export evaluation metrics to Excel sheet.

        Args:
            metrics_df (pd.DataFrame): DataFrame containing evaluation metrics.
            writer: Excel writer object.
        """
        print("  Generating Sheet 2: Metrics Comparison...")

        metrics_formatted = metrics_df.copy()
        for col in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "specificity",
            "npv",
        ]:
            if col in metrics_formatted.columns:
                metrics_formatted[col] = metrics_formatted[col].round(4)

        metrics_formatted.to_excel(writer, sheet_name="metrics_comparison", index=False)

        print(f"    ✓ Metrics for {len(metrics_df)} algorithms included")

    def _export_confusion_matrices(self, results: Dict[str, Any], writer):
        """Export confusion matrices to Excel sheet.

        Args:
            results (Dict[str, Any]): Dictionary of algorithm results.
            writer: Excel writer object.
        """
        print("  Generating Sheet 3: Confusion Matrix Comparison...")

        cm_data = []

        for algo_name, result in results.items():
            metrics = MetricsCalculator.calculate_metrics(
                result["y_true"], result["y_pred"]
            )
            cm = metrics["confusion_matrix"]

            cm_data.append(
                {
                    "algorithm": algo_name,
                    "tn_<=50k_predicted_as_<=50k": int(cm[0, 0]),
                    "fp_<=50k_predicted_as_>50k": int(cm[0, 1]),
                    "fn_>50k_predicted_as_<=50k": int(cm[1, 0]),
                    "tp_>50k_predicted_as_>50k": int(cm[1, 1]),
                    "total_samples": int(cm.sum()),
                    "correct_predictions": int(cm[0, 0] + cm[1, 1]),
                    "wrong_predictions": int(cm[0, 1] + cm[1, 0]),
                }
            )

        cm_df = pd.DataFrame(cm_data)
        cm_df.to_excel(writer, sheet_name="confusion_matrix_comparison", index=False)

        print(f"    ✓ {len(results)} confusion matrices included")

    def _export_detailed_stats(self, results: Dict[str, Any], metrics_df: pd.DataFrame, writer):
        """Export detailed statistics to Excel sheet.

        Args:
            results (Dict[str, Any]): Dictionary of algorithm results.
            metrics_df (pd.DataFrame): DataFrame containing evaluation metrics.
            writer: Excel writer object.
        """
        print("  Generating Sheet 4: Detailed Statistics...")

        stats_data = []

        for algo_name, result in results.items():
            y_true = result["y_true"]
            y_pred = result["y_pred"]

            true_neg_count = np.sum(y_true == 0)
            true_pos_count = np.sum(y_true == 1)
            pred_neg_count = np.sum(y_pred == 0)
            pred_pos_count = np.sum(y_pred == 1)

            stats_data.append(
                {
                    "algorithm": algo_name,
                    "total_samples": len(y_true),
                    "true_negative_count": int(true_neg_count),
                    "true_positive_count": int(true_pos_count),
                    "predicted_negative_count": int(pred_neg_count),
                    "predicted_positive_count": int(pred_pos_count),
                    "class_balance": f"{true_pos_count}/{true_neg_count}",
                    "accuracy": result["test_acc"],
                }
            )

        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name="detailed_statistics", index=False)

        print("    ✓ Statistics completed")


def main():
    """Execute the main program workflow."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 10 + "Decision Tree Comparison System - Full Version" + " " * 25 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    start_time = time.time()

    try:
        print("\nStep 1/5: Setting up project paths")
        paths = setup_paths()
        print(f"  ✓ Data directory: {paths['data']}")
        print(f"  ✓ Results directory: {paths['results']}")

        print("\nStep 2/5: Running decision tree algorithms")
        runner = AlgorithmRunner(paths)
        results = runner.run_all()

        if len(results) == 0:
            print("✗ No algorithms executed successfully, program terminated")
            return

        print("\nStep 3/5: Calculating evaluation metrics")
        metrics_df = MetricsCalculator.calculate_all_metrics(results)
        print("  ✓ Metrics calculation completed")
        print("\n" + "=" * 70)
        print("Metrics Summary:")
        print("=" * 70)
        print(
            metrics_df[
                ["algorithm", "accuracy", "precision", "recall", "f1_score"]
            ].to_string(index=False)
        )
        print("=" * 70)

        print("\nStep 4/5: Generating Excel comparison report")
        output_path = paths["results"] / OUTPUT_FILENAME
        exporter = ExcelExporter(output_path)
        exporter.export_comparison(results, metrics_df)

        total_time = time.time() - start_time

        print("\n" + "█" * 70)
        print("█" + " " * 68 + "█")
        print("█" + " " * 25 + "Execution Completed" + " " * 35 + "█")
        print("█" + " " * 68 + "█")
        print("█" * 70)

        print(f"\n✅ Comparison system executed successfully!")
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print(f"📊 Successfully executed: {len(results)}/4 algorithms")
        print(f"📁 Output file: {output_path}")
        print(
            f"\nBest algorithm: {metrics_df.loc[metrics_df['accuracy'].idxmax(), 'algorithm']} "
            f"(Accuracy: {metrics_df['accuracy'].max():.4f})"
        )
        print("\n" + "=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\n✗ Program execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()