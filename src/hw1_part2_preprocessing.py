import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

warnings.filterwarnings("ignore")


class UnifiedDataPreprocessor:
    """Handles unified data preprocessing for decision tree models."""

    COLUMNS = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]

    CONTINUOUS_FEATURES = [
        "age",
        "fnlwgt",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    ]

    CATEGORICAL_FEATURES = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    def __init__(self, data_dir="../data"):
        """Initializes the preprocessor with the data directory.

        Args:
            data_dir (str): Path to the directory containing data files.

        Raises:
            FileNotFoundError: If training or test data files are not found.
        """
        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / "adult.data"
        self.test_path = self.data_dir / "adult.test"
        self.label_encoders = {}
        self.discretizer = None

        if not self.train_path.exists():
            raise FileNotFoundError(f"Training data file not found: {self.train_path}")
        if not self.test_path.exists():
            raise FileNotFoundError(f"Test data file not found: {self.test_path}")


    def load_raw_data(self, verbose=True):
        """Loads raw training and test data.

        Args:
            verbose (bool): If True, prints loading information.

        Returns:
            tuple: (train_df, test_df) containing raw dataframes.
        """
        if verbose:
            print("=" * 60)
            print("Loading raw data")
            print("=" * 60)

        train_df = pd.read_csv(
            self.train_path, names=self.COLUMNS, na_values=" ?", skipinitialspace=True
        )

        test_df = pd.read_csv(
            self.test_path,
            names=self.COLUMNS,
            na_values=" ?",
            skiprows=1,
            skipinitialspace=True,
        )

        if verbose:
            print(f"Training data shape: {train_df.shape}")
            print(f"Test data shape: {test_df.shape}")

        return train_df, test_df


    def clean_data(self, train_df, test_df, verbose=True):
        """Cleans training and test data by handling missing values and standardizing labels.

        Args:
            train_df (pd.DataFrame): Training dataframe.
            test_df (pd.DataFrame): Test dataframe.
            verbose (bool): If True, prints cleaning information.

        Returns:
            tuple: (train_df, test_df) containing cleaned dataframes.
        """
        if verbose:
            print("\nCleaning data...")

        train_before = len(train_df)
        test_before = len(test_df)

        train_df = train_df.replace([" ?", " ? ", "?"], np.nan)
        test_df = test_df.replace([" ?", " ? ", "?"], np.nan)

        test_df["income"] = test_df["income"].str.replace(".", "", regex=False)

        for df in [train_df, test_df]:
            df["income"] = df["income"].str.strip()

        train_df = train_df.dropna().reset_index(drop=True)
        test_df = test_df.dropna().reset_index(drop=True)

        valid_labels = ["<=50K", ">50K"]
        train_df = train_df[train_df["income"].isin(valid_labels)].reset_index(
            drop=True
        )
        test_df = test_df[test_df["income"].isin(valid_labels)].reset_index(drop=True)

        if verbose:
            print(
                f"Training data: {train_before} → {len(train_df)} "
                f"(removed {train_before - len(train_df)} rows)"
            )
            print(
                f"Test data: {test_before} → {len(test_df)} "
                f"(removed {test_before - len(test_df)} rows)"
            )

        return train_df, test_df


    def encode_categorical(self, train_df, test_df, verbose=True):
        """Encodes categorical features using LabelEncoder.

        Args:
            train_df (pd.DataFrame): Training dataframe.
            test_df (pd.DataFrame): Test dataframe.
            verbose (bool): If True, prints encoding information.

        Returns:
            tuple: (train_df, test_df) containing encoded dataframes.
        """
        if verbose:
            print("\nEncoding categorical features...")

        train_df = train_df.copy()
        test_df = test_df.copy()

        for feature in self.CATEGORICAL_FEATURES:
            le = LabelEncoder()
            le.fit(train_df[feature].astype(str))
            train_df[feature] = le.transform(train_df[feature].astype(str))

            test_values = test_df[feature].astype(str)
            unseen_mask = ~test_values.isin(le.classes_)

            if unseen_mask.any():
                mode_class = train_df[feature].mode()[0]
                if verbose:
                    print(
                        f"  {feature}: Found {unseen_mask.sum()} unseen categories, "
                        f"replaced with mode {mode_class}"
                    )
                test_values[unseen_mask] = le.classes_[mode_class]

            test_df[feature] = le.transform(test_values)
            self.label_encoders[feature] = le

        if verbose:
            print(f"✓ Encoding complete for {len(self.CATEGORICAL_FEATURES)} categorical features")

        return train_df, test_df


    def get_processed_data(
        self,
        discretize=False,
        n_bins=10,
        validation_split=0.0,
        random_state=42,
        verbose=True,
    ):
        """Processes data and prepares it for model training.

        Args:
            discretize (bool): If True, discretizes continuous features.
            n_bins (int): Number of bins for discretization.
            validation_split (float): Proportion of training data to use for validation.
            random_state (int): Random seed for reproducibility.
            verbose (bool): If True, prints processing information.

        Returns:
            tuple: If validation_split > 0, returns (X_train, X_val, X_test, y_train, y_val, y_test).
                   Otherwise, returns (X_train, X_test, y_train, y_test).
        """
        train_df, test_df = self.load_raw_data(verbose)
        train_df, test_df = self.clean_data(train_df, test_df, verbose)
        train_df, test_df = self.encode_categorical(train_df, test_df, verbose)

        if discretize:
            train_df, test_df = self._discretize_continuous(train_df, test_df, n_bins, verbose)

        feature_cols = self.CONTINUOUS_FEATURES + self.CATEGORICAL_FEATURES
        X_train_full = train_df[feature_cols].values
        y_train_full = (train_df["income"] == ">50K").astype(int).values
        X_test = test_df[feature_cols].values
        y_test = (test_df["income"] == ">50K").astype(int).values

        if verbose:
            print(f"\n{'='*60}")
            print("Data preparation complete")
            print(f"{'='*60}")
            print(f"Number of features: {len(feature_cols)}")
            print(f"  - Continuous features: {len(self.CONTINUOUS_FEATURES)}")
            print(f"  - Categorical features: {len(self.CATEGORICAL_FEATURES)}")
            print(f"Discretized: {discretize}")

        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=validation_split,
                random_state=random_state,
                stratify=y_train_full,
            )

            if verbose:
                print(f"\nData split:")
                print(f"  Training set: {X_train.shape}")
                print(f"  Validation set: {X_val.shape}")
                print(f"  Test set: {X_test.shape}")
                print(f"\nLabel distribution (>50K proportion):")
                print(f"  Training set: {y_train.mean():.2%}")
                print(f"  Validation set: {y_val.mean():.2%}")
                print(f"  Test set: {y_test.mean():.2%}")
                print(f"{'='*60}\n")

            return X_train, X_val, X_test, y_train, y_val, y_test

        else:
            if verbose:
                print(f"\nData shapes:")
                print(f"  Training set: {X_train_full.shape}")
                print(f"  Test set: {X_test.shape}")
                print(f"\nLabel distribution (>50K proportion):")
                print(f"  Training set: {y_train_full.mean():.2%}")
                print(f"  Test set: {y_test.mean():.2%}")
                print(f"{'='*60}\n")

            return X_train_full, X_test, y_train_full, y_test


    def _discretize_continuous(self, train_df, test_df, n_bins, verbose):
        """Discretizes continuous features.

        Args:
            train_df (pd.DataFrame): Training dataframe.
            test_df (pd.DataFrame): Test dataframe.
            n_bins (int): Number of bins for discretization.
            verbose (bool): If True, prints discretization information.

        Returns:
            tuple: (train_df, test_df) containing discretized dataframes.
        """
        if verbose:
            print(f"\nDiscretizing continuous features (n_bins={n_bins})...")

        train_df = train_df.copy()
        test_df = test_df.copy()

        self.discretizer = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
        )

        train_continuous = train_df[self.CONTINUOUS_FEATURES].values
        test_continuous = test_df[self.CONTINUOUS_FEATURES].values

        train_discretized = self.discretizer.fit_transform(train_continuous)
        test_discretized = self.discretizer.transform(test_continuous)

        for i, col in enumerate(self.CONTINUOUS_FEATURES):
            train_df[col] = train_discretized[:, i].astype(int)
            test_df[col] = test_discretized[:, i].astype(int)

        if verbose:
            print(f"✓ Discretization complete, each feature divided into {n_bins} bins")

        return train_df, test_df


    def get_feature_names(self):
        """Returns the list of feature names.

        Returns:
            list: List of feature names.
        """
        return self.CONTINUOUS_FEATURES + self.CATEGORICAL_FEATURES


    def get_feature_info(self):
        """Returns detailed information about features.

        Returns:
            dict: Dictionary containing feature information.
        """
        return {
            "all_features": self.get_feature_names(),
            "continuous": self.CONTINUOUS_FEATURES,
            "categorical": self.CATEGORICAL_FEATURES,
            "n_features": len(self.CONTINUOUS_FEATURES) + len(self.CATEGORICAL_FEATURES),
        }


def example_usage():
    """Demonstrates usage of the UnifiedDataPreprocessor."""
    print("\n" + "=" * 60)
    print("Unified Data Preprocessor Usage Example")
    print("=" * 60 + "\n")

    preprocessor = UnifiedDataPreprocessor(data_dir="../data")

    print("\n[Example 1] CART model usage (no discretization, no validation set)")
    print("-" * 60)
    X_train, X_test, y_train, y_test = preprocessor.get_processed_data(
        discretize=False, validation_split=0.0, verbose=True
    )
    print(f"CART can use directly:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

    print("\n[Example 2] C4.5 model usage (no discretization, with validation set)")
    print("-" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.get_processed_data(
        discretize=False, validation_split=0.1, random_state=42, verbose=True
    )
    print(f"C4.5 can use directly:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}")

    print("\n[Example 3] ID3 model usage (with discretization, with validation set)")
    print("-" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.get_processed_data(
        discretize=True, n_bins=10, validation_split=0.2, random_state=42, verbose=True
    )
    print(f"ID3 can use directly:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}")

    print("\n[Example 4] C5.0 model usage (with discretization, with validation set)")
    print("-" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.get_processed_data(
        discretize=True, n_bins=10, validation_split=0.2, random_state=42, verbose=True
    )
    print(f"C5.0 can use directly:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}")

    print("\n[Feature Information]")
    print("-" * 60)
    feature_info = preprocessor.get_feature_info()
    print(f"Total features: {feature_info['n_features']}")
    print(f"Continuous features: {feature_info['continuous']}")
    print(f"Categorical features: {feature_info['categorical']}")

    print("\n" + "=" * 60)
    print("✓ All examples completed")
    print("=" * 60 + "\n")


def get_data_for_cart():
    """Fetches data for CART model.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) for CART model.
    """
    preprocessor = UnifiedDataPreprocessor()
    return preprocessor.get_processed_data(discretize=False, validation_split=0.0)


def get_data_for_c45():
    """Fetches data for C4.5 model.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test) for C4.5 model.
    """
    preprocessor = UnifiedDataPreprocessor()
    return preprocessor.get_processed_data(
        discretize=False, validation_split=0.1, random_state=42
    )


def get_data_for_id3():
    """Fetches data for ID3 model.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test) for ID3 model.
    """
    preprocessor = UnifiedDataPreprocessor()
    return preprocessor.get_processed_data(
        discretize=True, n_bins=10, validation_split=0.2, random_state=42
    )


def get_data_for_c50():
    """Fetches data for C5.0 model.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test) for C5.0 model.
    """
    preprocessor = UnifiedDataPreprocessor()
    return preprocessor.get_processed_data(
        discretize=True, n_bins=10, validation_split=0.2, random_state=42
    )


if __name__ == "__main__":
    example_usage()