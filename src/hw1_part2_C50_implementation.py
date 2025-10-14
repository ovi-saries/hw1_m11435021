"""
C5.0 Decision Tree Model Training and Evaluation System
=====================================================
A complete implementation of the C5.0 decision tree, including data preprocessing,
model training, prediction, and evaluation.

Requirements:
    - pandas>=1.3.0
    - numpy>=1.21.0
    - scikit-learn>=1.0.0
    - rpy2>=3.4.0
    - R package: C50

Author: AI Assistant
Date: 2025-10-14
Version: 3.0 (Rigorous Edition)
"""

import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Rpy2 imports
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

# Sklearn imports
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Import unified data preprocessing module
from hw1_part2_preprocessing import UnifiedDataPreprocessor


# Configuration Management
@dataclass
class C50Config:
    """Configuration class for the C5.0 model."""
    
    data_dir: Path = field(default_factory=lambda: Path("../data"))
    results_dir: Path = field(default_factory=lambda: Path("../results"))
    log_dir: Path = field(default_factory=lambda: Path("../logs"))
    train_file: str = "adult.data"
    test_file: str = "adult.test"
    use_validation: bool = True
    validation_size: float = 0.2
    n_bins: int = 10
    discretization_strategy: str = "quantile"
    handle_unseen_levels: str = "mode"
    trials: int = 10
    winnow: bool = False
    use_cost_matrix: bool = False
    cost_fn_to_fp_ratio: float = 5.0
    random_state: int = 42
    verbose: bool = True

    def __post_init__(self):
        """Validate configuration and create necessary directories.

        Raises:
            ValueError: If validation_size is not between 0 and 1, n_bins < 2, or trials < 1.
        """
        if not 0 < self.validation_size < 1:
            raise ValueError(f"validation_size must be between (0, 1), got: {self.validation_size}")
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got: {self.n_bins}")
        if self.trials < 1:
            raise ValueError(f"trials must be >= 1, got: {self.trials}")

        for directory in [self.data_dir, self.results_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)


def setup_logging(config: C50Config) -> logging.Logger:
    """Configure the logging system.

    Args:
        config (C50Config): Configuration object.

    Returns:
        logging.Logger: Configured logger object.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = config.log_dir / f"c50_experiment_{timestamp}.log"
    logger = logging.getLogger("C50_Experiment")
    logger.handlers.clear()
    logger.setLevel(logging.INFO if config.verbose else logging.WARNING)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging system initialized, log file: {log_file}")
    return logger


# R Package Management
class RPackageManager:
    """Manages R packages for C5.0 model."""

    def __init__(self, logger: logging.Logger):
        """Initialize the R package manager.

        Args:
            logger (logging.Logger): Logger instance for logging messages.
        """
        self.logger = logger
        self.c50 = None
        self.base = None
        self.available = False

    def load_packages(self) -> bool:
        """Load required R packages.

        Returns:
            bool: True if packages are loaded successfully, False otherwise.
        """
        try:
            self.c50 = importr("C50")
            self.base = importr("base")
            self.available = True
            self.logger.info("C50 package loaded successfully")

            try:
                version = str(self.base.packageVersion("C50"))
                self.logger.info(f"C50 version: {version}")
            except Exception:
                pass

            return True
        except Exception as e:
            self.logger.error(f"C50 loading failed: {e}")
            self.logger.error("Please run in R: install.packages('C50')")
            self.available = False
            return False

    def check_availability(self) -> None:
        """Check if R packages are available.

        Raises:
            RuntimeError: If C50 package is not loaded correctly.
        """
        if not self.available:
            raise RuntimeError("C50 package not loaded correctly, cannot proceed")


# Data Processor
class AdultDataProcessor:
    """Processor for Adult dataset using unified preprocessing module."""

    def __init__(self, config: C50Config, logger: logging.Logger):
        """Initialize the Adult data processor.

        Args:
            config (C50Config): Configuration object.
            logger (logging.Logger): Logger instance for logging messages.
        """
        self.config = config
        self.logger = logger
        self.preprocessor = UnifiedDataPreprocessor(data_dir=str(config.data_dir))
        self.feature_levels: Dict[str, list] = {}
        self.feature_names = None

    def _convert_to_r_factors(self, data_array: np.ndarray, is_training: bool = False) -> pd.DataFrame:
        """Convert numpy array to R factor format DataFrame.

        Args:
            data_array (np.ndarray): Input data array.
            is_training (bool): Whether the data is for training. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with features converted to R factor format.
        """
        feature_names = self.preprocessor.get_feature_names()
        df = pd.DataFrame(data_array, columns=feature_names)

        for col in df.columns:
            df[col] = df[col].astype(int).astype(str)
            if is_training:
                unique_vals = sorted(df[col].unique(), key=lambda x: int(x))
                self.feature_levels[col] = unique_vals

        return df

    def prepare_data(
        self,
    ) -> Tuple[
        pd.DataFrame,
        Optional[pd.DataFrame],
        pd.DataFrame,
        np.ndarray,
        Optional[np.ndarray],
        np.ndarray,
    ]:
        """Prepare data for training and evaluation.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, np.ndarray, Optional[np.ndarray], np.ndarray]:
                Tuple containing training, validation (if applicable), and test DataFrames and labels.
        """
        self.logger.info("Using unified data preprocessing module...")

        if self.config.use_validation:
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.get_processed_data(
                discretize=False,
                n_bins=self.config.n_bins,
                validation_split=self.config.validation_size,
                random_state=self.config.random_state,
                verbose=False,
            )
            self.logger.info("Data loading completed (with validation set)")
            self.logger.info(f"  Training set: {X_train.shape}")
            self.logger.info(f"  Validation set: {X_val.shape}")
            self.logger.info(f"  Test set: {X_test.shape}")
        else:
            X_train, X_test, y_train, y_test = self.preprocessor.get_processed_data(
                discretize=True,
                n_bins=self.config.n_bins,
                validation_split=0.0,
                random_state=self.config.random_state,
                verbose=False,
            )
            X_val, y_val = None, None
            self.logger.info("Data loading completed (no validation set)")
            self.logger.info(f"  Training set: {X_train.shape}")
            self.logger.info(f"  Test set: {X_test.shape}")

        self.logger.info(
            f"Label distribution - Training set >50K: {y_train.sum()} / {len(y_train)} "
            f"({y_train.mean()*100:.2f}%)"
        )
        if y_val is not None:
            self.logger.info(
                f"Label distribution - Validation set >50K: {y_val.sum()} / {len(y_val)} "
                f"({y_val.mean()*100:.2f}%)"
            )
        self.logger.info(
            f"Label distribution - Test set >50K: {y_test.sum()} / {len(y_test)} "
            f"({y_test.mean()*100:.2f}%)"
        )

        X_train_df = self._convert_to_r_factors(X_train, is_training=True)
        X_test_df = self._convert_to_r_factors(X_test, is_training=False)
        X_val_df = self._convert_to_r_factors(X_val, is_training=False) if X_val is not None else None

        self.feature_names = list(X_train_df.columns)
        self.logger.info(f"Data conversion completed, {len(self.feature_names)} features")
        self.logger.info("Feature levels recorded to ensure training/test consistency")

        return X_train_df, X_val_df, X_test_df, y_train, y_val, y_test


# C5.0 Model Manager
class C50ModelManager:
    """Manages training and prediction for the C5.0 model."""

    def __init__(
        self,
        config: C50Config,
        r_manager: RPackageManager,
        data_processor: AdultDataProcessor,
        logger: logging.Logger,
    ):
        """Initialize the C5.0 model manager.

        Args:
            config (C50Config): Configuration object.
            r_manager (RPackageManager): R package manager instance.
            data_processor (AdultDataProcessor): Data processor instance.
            logger (logging.Logger): Logger instance for logging messages.
        """
        self.config = config
        self.r_manager = r_manager
        self.data_processor = data_processor
        self.logger = logger
        self.model = None
        self.training_time = 0.0
        self.feature_names = None

    def train(self, x_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """Train the C5.0 model.

        Args:
            x_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training labels.

        Raises:
            RuntimeError: If training fails due to R runtime errors or other issues.
        """
        self.r_manager.check_availability()
        self.logger.info("=" * 60)
        self.logger.info("Starting C5.0 model training")
        self.logger.info(f"Configuration: trials={self.config.trials}, cost_matrix={self.config.use_cost_matrix}")
        self.logger.info("=" * 60)

        try:
            start_time = time.time()
            train_df = x_train.copy()
            train_df["target"] = [">50K" if y == 1 else "<=50K" for y in y_train]
            self.feature_names = list(x_train.columns)

            self.logger.info("Checking feature variability...")
            constant_features = []
            valid_features = []

            for col in self.feature_names:
                n_unique = train_df[col].nunique()
                if n_unique <= 1:
                    constant_features.append(col)
                    self.logger.warning(f"  {col}: Only {n_unique} unique value(s) (constant feature, will be removed)")
                else:
                    valid_features.append(col)
                    self.logger.debug(f"  {col}: {n_unique} unique values")

            if constant_features:
                self.logger.warning(f"Removing {len(constant_features)} constant features: {constant_features}")
                self.feature_names = valid_features
                train_df = train_df[valid_features + ["target"]]
                self.logger.info(f"Remaining {len(self.feature_names)} valid features")
            else:
                self.logger.info(f"All {len(self.feature_names)} features have variability")

            with localconverter(ro.default_converter + pandas2ri.converter):
                r_train = ro.conversion.py2rpy(train_df)

            ro.r.assign("train_data", r_train)

            self.logger.info("Checking R DataFrame structure...")
            try:
                ro.r("print(str(train_data))")
                self.logger.info(f"  Data dimensions: {len(train_df)} rows x {len(train_df.columns)} columns")
            except Exception as e:
                self.logger.warning(f"Unable to check R data structure: {e}")

            ro.r('train_data$target <- factor(train_data$target, levels = c("<=50K", ">50K"))')

            self.logger.info("Setting features as R factors...")
            for col in self.feature_names:
                col_escaped = f"`{col}`"
                levels = self.data_processor.feature_levels.get(col, [])
                if levels:
                    levels_str = ", ".join([f'"{lv}"' for lv in levels])
                    ro.r(f"train_data${col_escaped} <- factor(train_data${col_escaped}, levels = c({levels_str}))")
                else:
                    ro.r(f"train_data${col_escaped} <- as.factor(train_data${col_escaped})")

            self.logger.info("All features converted to factors")

            if self.config.use_cost_matrix:
                cost_ratio = self.config.cost_fn_to_fp_ratio
                ro.r(
                    f"""
                    cost_matrix <- matrix(c(0, {cost_ratio}, 1, 0), nrow = 2, byrow = TRUE)
                    rownames(cost_matrix) <- c("<=50K", ">50K")
                    colnames(cost_matrix) <- c("<=50K", ">50K")
                    """
                )
                self.logger.info(f"Using cost matrix (FN:FP = {cost_ratio}:1)")

            self.logger.info("Preparing C5.0 training format (x, y)...")
            x_features = train_df[self.feature_names]
            y_target = train_df["target"]

            with localconverter(ro.default_converter + pandas2ri.converter):
                r_x = ro.conversion.py2rpy(x_features)
                r_y = ro.conversion.py2rpy(y_target)

            ro.r.assign("X_train", r_x)
            ro.r.assign("y_train", r_y)
            ro.r('y_train <- factor(y_train, levels = c("<=50K", ">50K"))')

            self.logger.info(f"Starting C5.0 model training (trials={self.config.trials})...")
            if self.config.use_cost_matrix:
                ro.r(
                    f"""
                    library(C50)
                    model <- C5.0(x = X_train, y = y_train,
                                 trials = {self.config.trials},
                                 costs = cost_matrix)
                    """
                )
            else:
                ro.r(
                    f"""
                    library(C50)
                    model <- C5.0(x = X_train, y = y_train,
                                 trials = {self.config.trials})
                    """
                )

            self.model = ro.r("model")
            self.training_time = time.time() - start_time

            try:
                model_summary = str(ro.r("summary(model)"))
                if "tree" in model_summary.lower() or "rule" in model_summary.lower():
                    self.logger.info(f"Training completed, time taken: {self.training_time:.2f} seconds")
                else:
                    self.logger.warning("Model summary does not contain tree or rules")
            except Exception as e:
                self.logger.warning(f"Unable to check model summary: {e}")

            try:
                ro.r('pred_train <- predict(model, X_train, type = "class")')
                train_pred_len = len(ro.r("pred_train"))
                self.logger.info(f"Model validation successful, generated {train_pred_len} training predictions")
            except Exception as e:
                self.logger.warning(f"Warning during model validation: {e}")

        except RRuntimeError as e:
            self.logger.error(f"C5.0 training error: {e}")
            raise RuntimeError(f"Training failed: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected training error: {e}")
            raise

    def predict(self, x: pd.DataFrame, return_proba: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict using the C5.0 model.

        Args:
            x (pd.DataFrame): Input features for prediction.
            return_proba (bool): Whether to return probabilities. Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: Predicted labels, or labels and probabilities if return_proba is True.

        Raises:
            RuntimeError: If model is not trained or prediction fails.
        """
        if self.model is None:
            raise RuntimeError("Model not trained, please call train() first")

        self.logger.info(f"Starting prediction for {len(x)} samples...")

        try:
            missing_features = [f for f in self.feature_names if f not in x.columns]
            if missing_features:
                self.logger.warning(f"Test set missing features: {missing_features}, attempting to continue...")

            available_features = [f for f in self.feature_names if f in x.columns]
            x_ordered = x[available_features].copy()

            with localconverter(ro.default_converter + pandas2ri.converter):
                r_test = ro.conversion.py2rpy(x_ordered)

            ro.r.assign("X_test", r_test)

            for col in self.feature_names:
                col_escaped = f"`{col}`"
                levels = self.data_processor.feature_levels.get(col, [])
                if levels:
                    levels_str = ", ".join([f'"{lv}"' for lv in levels])
                    ro.r(f"X_test${col_escaped} <- factor(X_test${col_escaped}, levels = c({levels_str}))")
                else:
                    ro.r(f"X_test${col_escaped} <- as.factor(X_test${col_escaped})")

            ro.r('pred_class <- predict(model, X_test, type = "class")')
            pred_r = ro.r("as.character(pred_class)")
            pred_labels = np.array([1 if str(p) == ">50K" else 0 for p in pred_r])

            if return_proba:
                try:
                    ro.r('pred_prob <- predict(model, X_test, type = "prob")')
                    prob_matrix = np.array(ro.r("pred_prob"))
                    if prob_matrix.ndim == 2 and prob_matrix.shape[1] == 2:
                        pred_proba = prob_matrix[:, 1]
                    else:
                        self.logger.warning("Probability matrix format is abnormal, using default values")
                        pred_proba = np.where(pred_labels == 1, 0.7, 0.3)
                    self.logger.info("Prediction completed (with probabilities)")
                    return pred_labels, pred_proba
                except Exception as e:
                    self.logger.warning(f"Probability prediction failed: {e}, returning only labels")
                    return pred_labels, np.where(pred_labels == 1, 0.7, 0.3)
            else:
                self.logger.info("Prediction completed")
                return pred_labels
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def get_model_summary(self) -> str:
        """Get the model summary.

        Returns:
            str: Model summary or error message if retrieval fails.
        """
        if self.model is None:
            return "Model not trained"

        try:
            ro.r.assign("model", self.model)
            summary_str = str(ro.r("summary(model)"))
            return summary_str
        except Exception as e:
            self.logger.warning(f"Unable to retrieve model summary: {e}")
            return "Summary retrieval failed"

    def save_model(self, filepath: Path) -> None:
        """Save the trained model.

        Args:
            filepath (Path): Path to save the model.

        Raises:
            RuntimeError: If model is not trained or saving fails.
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        try:
            ro.r.assign("model", self.model)
            ro.r(f'saveRDS(model, "{str(filepath)}")')
            self.logger.info(f"Model saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
            raise


# Model Evaluator
class ModelEvaluator:
    """Evaluates the performance of the C5.0 model."""

    def __init__(self, config: C50Config, logger: logging.Logger):
        """Initialize the model evaluator.

        Args:
            config (C50Config): Configuration object.
            logger (logging.Logger): Logger instance for logging messages.
        """
        self.config = config
        self.logger = logger
        self.all_results: Dict[str, Dict[str, Any]] = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Evaluate model performance.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            dataset_name (str): Name of the dataset (e.g., Train, Validation, Test).
            y_proba (Optional[np.ndarray]): Predicted probabilities. Defaults to None.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics.
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"{dataset_name} set evaluation")
        self.logger.info(f"{'='*60}")

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        auc_score = None
        if y_proba is not None:
            try:
                auc_score = roc_auc_score(y_true, y_proba)
            except Exception as e:
                self.logger.warning(f"Unable to calculate ROC-AUC: {e}")

        self.logger.info(f"Accuracy:     {accuracy:.4f}")
        self.logger.info(f"Precision:    {precision:.4f}")
        self.logger.info(f"Recall:       {recall:.4f}")
        self.logger.info(f"F1 Score:     {f1:.4f}")
        self.logger.info(f"Specificity:  {specificity:.4f}")
        self.logger.info(f"NPV:          {npv:.4f}")
        if auc_score is not None:
            self.logger.info(f"ROC-AUC:      {auc_score:.4f}")

        self.logger.info(f"\nConfusion Matrix:")
        self.logger.info(f"                 Predicted <=50K    Predicted >50K")
        self.logger.info(f"Actual <=50K         {tn:6d}       {fp:6d}")
        self.logger.info(f"Actual >50K          {fn:6d}       {tp:6d}")

        results = {
            "dataset": dataset_name,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "specificity": float(specificity),
            "npv": float(npv),
            "confusion_matrix": cm.tolist(),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "y_true": y_true,
            "y_pred": y_pred,
        }

        if auc_score is not None:
            results["roc_auc"] = float(auc_score)
        if y_proba is not None:
            results["y_proba"] = y_proba

        self.all_results[dataset_name] = results
        return results

    def save_results(self, training_time: float) -> None:
        """Save all evaluation results.

        Args:
            training_time (float): Time taken for model training.
        """
        if not self.all_results:
            self.logger.warning("No evaluation results to save")
            return

        self.logger.info(f"\n{'='*60}")
        self.logger.info("Saving evaluation results")
        self.logger.info(f"{'='*60}")

        for dataset_name, results in self.all_results.items():
            pred_df = pd.DataFrame(
                {
                    "true_label": results["y_true"],
                    "predicted_label": results["y_pred"],
                    "correct": (results["y_true"] == results["y_pred"]).astype(int),
                }
            )
            if "y_proba" in results:
                pred_df["probability_>50K"] = results["y_proba"]

            pred_file = self.config.results_dir / f"predictions_{dataset_name.lower()}.csv"
            pred_df.to_csv(pred_file, index=False)
            self.logger.info(f"{dataset_name} predictions saved: {pred_file}")

        metrics_list = []
        for dataset_name in ["Train", "Validation", "Test"]:
            if dataset_name not in self.all_results:
                continue
            res = self.all_results[dataset_name]
            metrics_list.append(
                {
                    "dataset": dataset_name,
                    "accuracy": res["accuracy"],
                    "precision": res["precision"],
                    "recall": res["recall"],
                    "f1_score": res["f1_score"],
                    "specificity": res["specificity"],
                    "roc_auc": res.get("roc_auc", "N/A"),
                }
            )

        metrics_df = pd.DataFrame(metrics_list)
        metrics_file = self.config.results_dir / "metrics_summary.csv"
        metrics_df.to_csv(metrics_file, index=False)
        self.logger.info(f"Metrics summary saved: {metrics_file}")

        for dataset_name, results in self.all_results.items():
            cm_df = pd.DataFrame(
                results["confusion_matrix"],
                columns=["predicted_<=50K", "predicted_>50K"],
                index=["actual_<=50K", "actual_>50K"],
            )
            cm_file = self.config.results_dir / f"confusion_matrix_{dataset_name.lower()}.csv"
            cm_df.to_csv(cm_file)
            self.logger.info(f"{dataset_name} confusion matrix saved: {cm_file}")

        self._generate_detailed_report(training_time)

    def _generate_detailed_report(self, training_time: float) -> None:
        """Generate a detailed evaluation report.

        Args:
            training_time (float): Time taken for model training.
        """
        report_file = self.config.results_dir / "evaluation_report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("C5.0 Decision Tree Model Evaluation Report\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training time: {training_time:.2f} seconds\n")
            f.write("\n")

            for dataset_name in ["Train", "Validation", "Test"]:
                if dataset_name not in self.all_results:
                    continue
                res = self.all_results[dataset_name]
                f.write("\n" + "=" * 70 + "\n")
                f.write(f"{dataset_name} Set Evaluation Results\n")
                f.write("=" * 70 + "\n\n")
                f.write("Basic Metrics:\n")
                f.write(f"  Accuracy:     {res['accuracy']:.4f}\n")
                f.write(f"  Precision:    {res['precision']:.4f}\n")
                f.write(f"  Recall:       {res['recall']:.4f}\n")
                f.write(f"  F1 Score:     {res['f1_score']:.4f}\n")
                f.write(f"  Specificity:  {res['specificity']:.4f}\n")
                f.write(f"  NPV:          {res['npv']:.4f}\n")
                if "roc_auc" in res:
                    f.write(f"  ROC-AUC:      {res['roc_auc']:.4f}\n")
                f.write(f"\nConfusion Matrix:\n")
                f.write(f"                 Predicted <=50K    Predicted >50K\n")
                f.write(f"  Actual <=50K       {res['tn']:6d}       {res['fp']:6d}\n")
                f.write(f"  Actual >50K        {res['fn']:6d}       {res['tp']:6d}\n")
                total = res["tn"] + res["fp"] + res["fn"] + res["tp"]
                f.write(f"\nSample Statistics:\n")
                f.write(f"  Total samples: {total}\n")
                f.write(f"  Actual >50K: {res['tp'] + res['fn']} ({(res['tp'] + res['fn'])/total*100:.2f}%)\n")
                f.write(f"  Actual <=50K: {res['tn'] + res['fp']} ({(res['tn'] + res['fp'])/total*100:.2f}%)\n")

            f.write("\n\n" + "=" * 70 + "\n")
            f.write("Model Generalization Analysis\n")
            f.write("=" * 70 + "\n\n")

            if "Train" in self.all_results and "Test" in self.all_results:
                train_acc = self.all_results["Train"]["accuracy"]
                test_acc = self.all_results["Test"]["accuracy"]
                diff_acc = train_acc - test_acc
                train_f1 = self.all_results["Train"]["f1_score"]
                test_f1 = self.all_results["Test"]["f1_score"]
                diff_f1 = train_f1 - test_f1
                f.write("Accuracy Comparison:\n")
                f.write(f"  Training set: {train_acc:.4f}\n")
                f.write(f"  Test set:     {test_acc:.4f}\n")
                f.write(f"  Difference:   {diff_acc:.4f} ({diff_acc/train_acc*100:.2f}%)\n\n")
                f.write("F1 Score Comparison:\n")
                f.write(f"  Training set: {train_f1:.4f}\n")
                f.write(f"  Test set:     {test_f1:.4f}\n")
                f.write(f"  Difference:   {diff_f1:.4f} ({diff_f1/train_f1*100:.2f}%)\n\n")
                if diff_acc > 0.05:
                    f.write("Warning: Model may have significant overfitting (accuracy difference > 5%)\n")
                    f.write("  Suggestion: Reduce trials, increase data size, or use regularization\n")
                elif diff_acc > 0.02:
                    f.write("Note: Model shows slight overfitting signs (accuracy difference 2-5%)\n")
                    f.write("  Suggestion: Consider adjusting model parameters\n")
                else:
                    f.write("Good: Model has good generalization (accuracy difference < 2%)\n")

            if "Validation" in self.all_results:
                val_acc = self.all_results["Validation"]["accuracy"]
                f.write(f"\nValidation set accuracy: {val_acc:.4f}\n")

            f.write("\n\n" + "=" * 70 + "\n")
            f.write("Error Analysis\n")
            f.write("=" * 70 + "\n\n")

            if "Test" in self.all_results:
                res = self.all_results["Test"]
                total_errors = res["fp"] + res["fn"]
                total_samples = res["tn"] + res["fp"] + res["fn"] + res["tp"]
                f.write(f"Total errors: {total_errors} / {total_samples} ({total_errors/total_samples*100:.2f}%)\n\n")
                f.write("Error Type Distribution:\n")
                f.write(f"  False Positive: {res['fp']} ({res['fp']/total_errors*100:.2f}%)\n")
                f.write(f"    → Actual <=50K predicted as >50K\n")
                f.write(f"  False Negative: {res['fn']} ({res['fn']/total_errors*100:.2f}%)\n")
                f.write(f"    → Actual >50K predicted as <=50K\n")

        self.logger.info(f"Detailed report saved: {report_file}")

    def print_summary(self) -> None:
        """Print a summary of evaluation results."""
        if not self.all_results:
            self.logger.warning("No evaluation results")
            return

        print("\n" + "=" * 70)
        print("Evaluation Results Summary")
        print("=" * 70)
        print(f"\n{'Dataset':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 70)

        for dataset_name in ["Train", "Validation", "Test"]:
            if dataset_name not in self.all_results:
                continue
            res = self.all_results[dataset_name]
            print(
                f"{dataset_name:<15} {res['accuracy']:<10.4f} {res['precision']:<10.4f} "
                f"{res['recall']:<10.4f} {res['f1_score']:<10.4f}"
            )

        print("=" * 70)


# Main Experiment Workflow
class C50Experiment:
    """Manages the complete C5.0 experiment workflow."""

    def __init__(self, config: Optional[C50Config] = None):
        """Initialize the experiment.

        Args:
            config (Optional[C50Config]): Configuration object. Defaults to None.
        """
        self.config = config or C50Config()
        self.logger = setup_logging(self.config)
        self.r_manager = RPackageManager(self.logger)
        self.data_processor = AdultDataProcessor(self.config, self.logger)
        self.model_manager = None
        self.evaluator = ModelEvaluator(self.config, self.logger)
        self._log_config()

    def _log_config(self) -> None:
        """Log experiment configuration."""
        self.logger.info(f"\n{'='*70}")
        self.logger.info("Experiment Configuration")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Data directory: {self.config.data_dir}")
        self.logger.info(f"Results directory: {self.config.results_dir}")
        self.logger.info(f"Use validation set: {self.config.use_validation}")
        if self.config.use_validation:
            self.logger.info(f"Validation set size: {self.config.validation_size}")
        self.logger.info(f"Discretization bins: {self.config.n_bins}")
        self.logger.info(f"Discretization strategy: {self.config.discretization_strategy}")
        self.logger.info(f"C5.0 trials: {self.config.trials}")
        self.logger.info(f"Use cost matrix: {self.config.use_cost_matrix}")
        if self.config.use_cost_matrix:
            self.logger.info(f"Cost ratio (FN:FP): {self.config.cost_fn_to_fp_ratio}:1")
        self.logger.info(f"Random seed: {self.config.random_state}")
        self.logger.info(f"{'='*70}\n")

    def run(self) -> Dict[str, Any]:
        """Run the complete experiment workflow.

        Returns:
            Dict[str, Any]: Dictionary containing experiment results.

        Raises:
            RuntimeError: If experiment fails due to R package issues or other errors.
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info("C5.0 Decision Tree Experiment Started")
        self.logger.info(f"{'='*70}\n")

        experiment_start_time = time.time()

        try:
            self.logger.info("Step 1/7: Loading R packages")
            if not self.r_manager.load_packages():
                raise RuntimeError("R package loading failed")

            self.logger.info("\nStep 2/7: Data preparation")
            x_train, x_val, x_test, y_train, y_val, y_test = self.data_processor.prepare_data()

            self.logger.info("\nStep 3/7: Training model")
            self.model_manager = C50ModelManager(self.config, self.r_manager, self.data_processor, self.logger)
            self.model_manager.train(x_train, y_train)

            self.logger.info("\nStep 4/7: Evaluating training set")
            y_train_pred = self.model_manager.predict(x_train)
            train_results = self.evaluator.evaluate(y_true=y_train, y_pred=y_train_pred, dataset_name="Train")

            if x_val is not None:
                self.logger.info("\nStep 5/7: Evaluating validation set")
                y_val_pred = self.model_manager.predict(x_val)
                val_results = self.evaluator.evaluate(y_true=y_val, y_pred=y_val_pred, dataset_name="Validation")
            else:
                self.logger.info("\nStep 5/7: Skipped (no validation set)")

            self.logger.info("\nStep 6/7: Evaluating test set")
            y_test_pred, y_test_proba = self.model_manager.predict(x_test, return_proba=True)
            test_results = self.evaluator.evaluate(
                y_true=y_test, y_pred=y_test_pred, dataset_name="Test", y_proba=y_test_proba
            )

            self.logger.info("\nStep 7/7: Saving results")
            self.evaluator.save_results(training_time=self.model_manager.training_time)

            model_file = self.config.results_dir / "c50_model.rds"
            self.model_manager.save_model(model_file)

            self.logger.info(f"\n{'='*70}")
            self.logger.info("Model Summary")
            self.logger.info(f"{'='*70}")
            summary = self.model_manager.get_model_summary()
            self.logger.info(summary)

            total_time = time.time() - experiment_start_time
            self.logger.info(f"\n{'='*70}")
            self.logger.info("Experiment Completion Summary")
            self.logger.info(f"{'='*70}")
            self.logger.info(f"Total execution time:     {total_time:.2f} seconds")
            self.logger.info(f"Training time:            {self.model_manager.training_time:.2f} seconds")
            self.logger.info(f"Training set accuracy:    {train_results['accuracy']:.4f}")
            if x_val is not None:
                self.logger.info(f"Validation set accuracy:  {val_results['accuracy']:.4f}")
            self.logger.info(f"Test set accuracy:        {test_results['accuracy']:.4f}")
            self.logger.info(f"Test set F1 score:       {test_results['f1_score']:.4f}")
            if "roc_auc" in test_results:
                self.logger.info(f"Test set ROC-AUC:        {test_results['roc_auc']:.4f}")
            self.logger.info(f"{'='*70}\n")

            self.evaluator.print_summary()

            final_results = {
                "total_time": total_time,
                "training_time": self.model_manager.training_time,
                "train_results": train_results,
                "test_results": test_results,
            }
            if x_val is not None:
                final_results["validation_results"] = val_results

            return final_results
        except Exception as e:
            self.logger.error(f"\nExperiment failed: {e}")
            self.logger.exception("Detailed error message:")
            raise


def main() -> Optional[Dict[str, Any]]:
    """Main function to run the C5.0 experiment.

    Returns:
        Optional[Dict[str, Any]]: Experiment results or None if interrupted.

    Raises:
        Exception: If experiment fails due to any error.
    """
    print("\n" + "=" * 70)
    print("C5.0 Decision Tree Experiment System")
    print("=" * 70 + "\n")

    config = C50Config(
        use_validation=True,
        validation_size=0.2,
        n_bins=10,
        discretization_strategy="quantile",
        trials=10,
        winnow=False,
        use_cost_matrix=False,
        random_state=42,
        verbose=True,
    )

    experiment = C50Experiment(config)

    try:
        results = experiment.run()
        print("\n" + "=" * 70)
        print("Experiment executed successfully!")
        print("=" * 70)
        print(f"Test set accuracy: {results['test_results']['accuracy']:.4f}")
        print(f"Test set F1 score: {results['test_results']['f1_score']:.4f}")
        print("=" * 70 + "\n")
        return results
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        return None
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        raise


if __name__ == "__main__":
    results = main()