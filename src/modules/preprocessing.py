import os
from typing import Dict, List, Tuple, Union

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

def remove_duplicates_and_columns(
    dataset: pd.DataFrame,
    columns_to_drop: List[str] = [],
    missingness_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Remove duplicate rows and drop columns with high missingness.

    Args:
        dataset (pd.DataFrame):
            Input dataset containing features and target.
        columns_to_drop (List[str], optional):
            List of specific columns to drop regardless of missingness. Default is [].
        missingness_threshold (float, optional):
            Proportion of missing values above which a column is dropped. Default is 0.5.

    Returns:
        pd.DataFrame:
            Cleaned dataset with duplicates removed and selected columns dropped.
    """
    
    # Drop duplicate rows
    dataset = dataset.drop_duplicates()

    # Calculate missing fraction per column
    missing_frac = dataset.isnull().mean()

    # Identify columns with missing fraction above threshold
    highly_missing = missing_frac[missing_frac > missingness_threshold].index.tolist()

    # Combine user-specified columns with highly missing columns
    to_drop = columns_to_drop + highly_missing

    # Drop selected columns
    dataset = dataset.drop(to_drop, axis = 1)

    return dataset

def label_classes(
    dataset: pd.DataFrame,
    y: str,
    label_map: Dict[Union[str, int], Union[str, int]]
) -> pd.DataFrame:
    """
    Relabel the target column using a mapping dictionary.

    Args:
        dataset (pd.DataFrame):
            Input dataset containing features and target.
        y (str):
            Name of the target column to relabel.
        label_map (Dict[Union[str, int], Union[str, int]]):
            Dictionary mapping original class values to new labels.

    Returns:
        pd.DataFrame:
            Dataset with the target column relabeled.
    """
    
    # Apply mapping function to target column
    dataset[y] = dataset[y].map(lambda x: label_map[x])

    return dataset

def preprocess_dataset(
    dataset: pd.DataFrame,
    y: str, categorical_columns: List[str],
    numeric_columns: List[str],
    ordinal_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess dataset by splitting into train/test sets and applying 
    appropriate encoding/imputation/scaling pipelines.

    Args:
        dataset (pd.DataFrame):
            Input dataset containing features and target.
        y (str):
            Name of the target column.
        categorical_columns (List[str]):
            List of categorical feature column names.
        numeric_columns (List[str]):
            List of numeric feature column names.
        ordinal_columns (List[str]):
            List of ordinal categorical column names.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - train: Preprocessed training dataset including transformed features and target.
            - test: Preprocessed test dataset including transformed features and target.
    """
    
    # Split features and target
    X = dataset.drop(y, axis = 1)
    y_series = dataset[y]

    # Train/test split (stratified by target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_series, test_size = 0.2, stratify = y_series, random_state = 42
    )

    # Determine encoding strategy for categorical columns
    categorical_nunique = X[categorical_columns].nunique()

    # Binary categorical columns are treated as ordinal
    binary_columns = categorical_nunique[categorical_nunique == 2].index.tolist()
    ordinal_columns = list(set(ordinal_columns + binary_columns))

    # One-hot encode columns with <= 10 categories (excluding binary)
    ohe_columns = categorical_nunique[
        (categorical_nunique <= 10) & (categorical_nunique != 2)
    ].index.tolist()

    # Target encode columns with > 10 categories
    target_encoding_columns = categorical_nunique[
        categorical_nunique > 10
    ].index.tolist()

    # Define preprocessing steps
    # Numeric: median imputation + standard scaling
    numeric_preprocessor = Pipeline(
        steps = [
            ("imputation_median", SimpleImputer(strategy = "median")),
            ("standard_scaler", StandardScaler()),
        ]
    )

    # One-hot encoding: mode imputation + one-hot encoder
    ohe_preprocessor = Pipeline(
        steps = [
            ("imputation_mode", SimpleImputer(strategy = "most_frequent")),
            ("ohe_encoder", OneHotEncoder(drop = "if_binary")),
        ]
    )

    # Target encoding: mode imputation + target encoder
    te_preprocessor = Pipeline(
        steps = [
            ("imputation_mode", SimpleImputer(strategy = "most_frequent")),
            ("target_encoder", TargetEncoder(target_type = "binary", random_state = 42)),
        ]
    )

    # Ordinal encoding: mode imputation + ordinal encoder
    ordinal_preprocessor = Pipeline(
        steps = [
            ("imputation_mode", SimpleImputer(strategy = "most_frequent")),
            ("ordinal_encoder", OrdinalEncoder()),
        ]
    )

    # Combine into column transformer
    preprocessor = ColumnTransformer(
        transformers = [
            ("numerical", numeric_preprocessor, numeric_columns),
            ("ohe", ohe_preprocessor, ohe_columns),
            ("te", te_preprocessor, target_encoding_columns),
            ("ordinal", ordinal_preprocessor, ordinal_columns),
        ],
        n_jobs = -1,
        verbose_feature_names_out = False,
        sparse_threshold = 0,
        remainder = "passthrough",
    )

    # Fit preprocessor (y_train needed for target encoding)
    preprocessor.fit(X_train, y_train)

    # Transform train/test sets
    X_train = pd.DataFrame(
        preprocessor.transform(X_train),
        index = X_train.index,
        columns = preprocessor.get_feature_names_out(),
    )
    X_test = pd.DataFrame(
        preprocessor.transform(X_test),
        index = X_test.index,
        columns = preprocessor.get_feature_names_out(),
    )

    # Recombine features and target
    train = pd.concat([X_train, y_train], axis = 1)
    test = pd.concat([X_test, y_test], axis = 1)

    return train, test

def save_processed_dataset(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_path: str,
    test_path: str,
    compression: str = "gzip"
) -> None:
    """
    Save processed train and test datasets to disk as compressed CSV files.

    Args:
        train (pd.DataFrame):
            Processed training dataset including features and target.
        test (pd.DataFrame):
            Processed test dataset including features and target.
        train_path (str):
            Path to save the training dataset file.
        test_path (str):
            Path to save the test dataset file.
        compression (str, optional):
            Compression format to use when saving files. Default is 'gzip'.
    """
    
    # Group datasets and their respective paths
    datasets = [train, test]
    paths = [train_path, test_path]

    # Save each dataset to disk
    for dataset, path in zip(datasets, paths):
        # Create parent directory if it does not exist
        directory = os.path.join(*path.split(os.sep)[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save dataset as CSV with compression
        dataset.to_csv(path, index = False, compression = compression)

    return