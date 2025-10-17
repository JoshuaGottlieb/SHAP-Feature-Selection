import os
import numpy as np
import pandas as pd
from typing import Union, List, Optional, Any

def feature_select(shap_values: np.ndarray, kind: Union[str, List[str]] = "sum",
                   sum_threshold: float = 1.0, min_strength: float = 0.0,
                   max_threshold: float = 0.0, decay_threshold: float = 0.0) -> np.ndarray:
    """
    Selects the most relevant feature indices based on SHAP importance values using
    one or more configurable selection strategies.

    This function ranks features by their SHAP importance values (in descending order)
    and determines how many of the top features to retain according to one or more
    threshold-based strategies. The output is a NumPy array of selected column indices.

    Args:
        shap_values (np.ndarray):
            A 1D array of SHAP importance values corresponding to features.
            Each value represents the overall contribution strength of one feature.
        kind (Union[str, List[str]], optional):
            The selection strategy or list of strategies to apply. Multiple strategies
            can be combined, in which case the most restrictive (smallest) feature
            subset is chosen. Supported options include:
              - `'sum'`: retain features cumulatively contributing up to
                `sum_threshold` proportion of total SHAP importance.
              - `'max'`: retain features with SHAP values above
                `max_threshold` proportion of the maximum SHAP value.
              - `'decay'`: retain features until the relative decay between
                consecutive SHAP values falls below `decay_threshold`.
            Defaults to `'sum'`.
        sum_threshold (float, optional):
            The cumulative proportion (0–1) of total SHAP importance to retain when using
            `'sum'` mode. For example, `0.8` retains the features that together account
            for 80% of total importance. Defaults to `1.0` (retain all features).
        min_strength (float, optional):
            Minimum proportion of total SHAP importance required for a feature
            to be considered strong in `'sum'` mode. Defaults to `0.0` (no cutoff).
        max_threshold (float, optional):
            Minimum proportion (0–1) of the maximum SHAP value required for a feature
            to be retained in `'max'` mode. Defaults to `0.0` (retain all features).
        decay_threshold (float, optional):
            Minimum ratio of consecutive SHAP values required to continue retaining
            features in `'decay'` mode. When the ratio between a feature’s SHAP value
            and the previous feature’s SHAP value falls below this threshold,
            selection stops. Defaults to `0.0` (no restriction).

    Returns:
        np.ndarray:
            An array of selected feature indices, sorted by decreasing SHAP importance.
            The final index (equal to the number of features) is appended to represent
            the target column position, assuming it appears last in the dataset.

    Notes:
        - If multiple strategies are specified, the smallest subset of selected
          features across all applied criteria is returned.
        - SHAP values should be non-negative or absolute magnitudes of contributions.
        - The `'decay'` strategy may behave conservatively when there are early
          large drops in SHAP values (see TODO note in the code).
    """

    # Ensure `kind` is always a list
    if isinstance(kind, str):
        kind = [kind]

    # Sort features by SHAP importance (descending)
    sorted_shap_values = np.sort(shap_values)[::-1]
    sort_order = np.argsort(shap_values)[::-1]
    n_features = len(sorted_shap_values)
    filter_idx = n_features - 1  # Default: keep all

    # Strategy 1: Retain features contributing up to a proportion of total SHAP sum
    if "sum" in kind:
        total_shap = shap_values.sum()
        cumsum = sorted_shap_values.cumsum()

        min_strength_mask = sorted_shap_values >= min_strength * total_shap

        filter_idx = min(
            filter_idx,
            np.searchsorted(cumsum, sum_threshold * total_shap, side = 'right'), # Index for total contribution
            n_features if np.all(min_strength_mask) or np.all(~min_strength_mask) else np.argmin(min_strength_mask)
        )

    # Strategy 2: Retain features above a fraction of the max SHAP value
    if "max" in kind:
        max_value = sorted_shap_values[0]
        max_strength_mask = sorted_shap_values >= max_threshold * max_value
        
        filter_idx = min(
            filter_idx,
            n_features if np.all(max_strength_mask) or np.all(~max_strength_mask) else np.argmin(max_strength_mask)
        )

    # Strategy 3: Retain features until SHAP values decay below a threshold
    # TODO: Fix this to somehow ignore early drop offs (e.g.: 2nd SHAP is < 60% of the max SHAP but 3rd SHAP is 80% of 2nd)
    if "decay" in kind:
        percent_previous = sorted_shap_values[1:] / sorted_shap_values[:-1]
        decay_mask = percent_previous >= decay_threshold
        filter_idx = min(
            filter_idx,
            n_features if np.all(decay_mask) or np.all(~decay_mask) else np.argmin(decay_mask) + 1
        )

    # Select the top-ranked features based on computed threshold
    selection = sort_order[:filter_idx] if filter_idx < n_features - 1 else sort_order

    # Concatenate the target column index (last column)
    selection = np.concatenate([selection, [n_features]])
    
    return selection

def create_feature_selected_dataset(idx: np.ndarray, train: pd.DataFrame, test: pd.DataFrame,
                                    root_dir: str, dataset_name: str, model_type: str,
                                    selection_strategy: str, selection_threshold: float,
                                    verbose: bool = True) -> None:
    """
    Creates and saves feature-selected versions of training and testing datasets
    based on the provided feature indices.

    The function subsets both the training and testing dataframes using the
    feature indices (`idx`), then saves the reduced datasets as compressed CSV
    files (`.csv.gz`) in organized subdirectories under `root_dir`.

    Args:
        idx (np.ndarray):
            Array of column indices to retain, typically produced by a feature
            selection process such as SHAP-based ranking.
        train (pd.DataFrame):
            Full training dataset from which features will be selected.
        test (pd.DataFrame):
            Full testing dataset from which features will be selected.
        root_dir (str):
            Base directory where the reduced datasets will be saved.
            Two subdirectories are expected: `train/` and `test/`.
        dataset_name (str):
            Name of the dataset, used for naming output files.
        model_type (str):
            Model identifier (e.g., `"rf"`, `"xgb"`, `"nn"`) included in filenames
            for traceability.
        selection_strategy (str):
            Name of the feature selection strategy applied (e.g., `"sum"`, `"max"`,
            `"decay"`, or combined forms like `"sum+decay"`).
        selection_threshold (float):
            Threshold parameter value used for selection (e.g., 0.8 for 80% cumulative
            SHAP contribution). Included in filenames for reproducibility.
        verbose (bool, optional):
            If True (default), prints informative messages about the process, including
            selected feature count, file paths, and dataset shapes. Set to False to
            suppress output.
    """

    # Subset datasets based on selected feature indices
    reduced_train = train.iloc[:, idx]
    reduced_test = test.iloc[:, idx]

    if verbose:
        print(f"[INFO] Selected {len(idx) - 1} features out of {train.shape[1] - 1} total columns.")

    # Construct output file paths
    reduced_train_path = os.path.join(
        root_dir,
        "train",
        f"{dataset_name}-train-{model_type}-{selection_strategy}-{selection_threshold}.csv.gz"
    )
    reduced_test_path = os.path.join(
        root_dir,
        "test",
        f"{dataset_name}-test-{model_type}-{selection_strategy}-{selection_threshold}.csv.gz"
    )

    # Ensure target directories exist
    os.makedirs(os.path.dirname(reduced_train_path), exist_ok = True)
    os.makedirs(os.path.dirname(reduced_test_path), exist_ok = True)

    # Save datasets with gzip compression
    reduced_train.to_csv(reduced_train_path, index = False, compression = "gzip")
    reduced_test.to_csv(reduced_test_path, index = False, compression = "gzip")

    if verbose:
        print(f"[INFO] Saved reduced training dataset -> {reduced_train_path}")
        print(f"[INFO] Saved reduced testing dataset  -> {reduced_test_path}")
        print(f"[INFO] Reduced dataset shapes: train = {reduced_train.shape}, test = {reduced_test.shape}\n")

    return