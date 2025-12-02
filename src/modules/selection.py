import os
import time
from typing import List, Union

import numpy as np
import pandas as pd
from ReliefF import ReliefF
from skfda.preprocessing.dim_reduction.variable_selection import (
    MinimumRedundancyMaximumRelevance as MRMR,
)
from skfda.representation.grid import FDataGrid
from sklearn.feature_selection import mutual_info_classif

from modules.utils import load_dataset, load_object

def shap_select(
    shap_values: np.ndarray,
    kind: Union[str, List[str]] = "sum",
    sum_threshold: float = 1.0,
    min_strength: float = 0.0,
    max_threshold: float = 0.0
) -> np.ndarray:
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
    Returns:
        np.ndarray:
            An array of selected feature indices, sorted by decreasing SHAP importance.
            The final index (equal to the number of features) is appended to represent
            the target column position, assuming it appears last in the dataset.

    Notes:
        - If multiple strategies are specified, the smallest subset of selected
          features across all applied criteria is returned.
        - SHAP values should be non-negative or absolute magnitudes of contributions.
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
            max(np.searchsorted(cumsum, sum_threshold * total_shap, side = 'right'), 1), # Index for total contribution
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

    # Select the top-ranked features based on computed threshold
    selection = sort_order[:filter_idx] if filter_idx < n_features - 1 else sort_order

    # Concatenate the target column index (last column)
    selection = np.concatenate([selection, [n_features]])
    
    return selection

def create_feature_selected_dataset(
    idx: np.ndarray,
    train: pd.DataFrame,
    test: pd.DataFrame,
    root_dir: str,
    dataset_name: str,
    model_type: str,
    selection_strategy: str,
    selection_threshold: float,
    verbose: bool = True
) -> None:
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
    if model_type != "":
        model_type = f'-{model_type}'

    reduced_train_path = os.path.join(
        root_dir,
        "train",
        f"{dataset_name}-train{model_type}-{selection_strategy}-{selection_threshold}.csv.gz"
    )
    reduced_test_path = os.path.join(
        root_dir,
        "test",
        f"{dataset_name}-test{model_type}-{selection_strategy}-{selection_threshold}.csv.gz"
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

def run_shap_selection(
    dataset_names: List[str],
    model_types: List[str],
    strategies: List[str],
    sum_thresholds: List[float],
    max_thresholds: List[float],
    train_dir: str,
    test_dir: str,
    explanations_dir: str,
    save_dir: str,
    metrics_dir: str
) -> None:
    """
    Perform SHAP-based feature selection across multiple datasets and models.  
    Loads precomputed SHAP explanations for each dataset-model pair, applies specified
    feature selection strategies ('sum' or 'max'), and saves reduced datasets containing
    only the selected features. Selection statistics and timing information are recorded
    in a metrics file.
    
    Args:
        dataset_names (List[str]): Names of datasets to process.
        model_types (List[str]): Model types to include (e.g., ['rf', 'xgb', 'logreg']).
        strategies (List[str]): SHAP feature selection strategies to apply:
            - 'sum': Select features by cumulative SHAP contribution.
            - 'max': Select features exceeding a SHAP magnitude threshold.
        sum_thresholds (List[float]): Thresholds for cumulative SHAP contribution (used with
            'sum' strategy). Example: [0.7, 0.8, 0.9, 0.95].
        max_thresholds (List[float]): Thresholds for SHAP magnitude filtering (used with
            'max' strategy). Example: [0.01, 0.05, 0.1, 0.15].
        train_dir (str): Directory containing preprocessed training datasets (CSV or
            compressed CSV).
        test_dir (str): Directory containing preprocessed testing datasets (CSV or
            compressed CSV).
        explanations_dir (str): Directory containing SHAP explanation pickle files for each
            dataset-model pair.
        save_dir (str): Directory to save reduced feature-selected datasets.
        metrics_dir (str): Directory to save SHAP feature selection metrics (e.g., timing
            and feature counts).
    """

    fit_metrics = []  # Collect metrics such as number of features and SHAP fit time

    # Iterate through datasets
    for dataset in dataset_names:
        set_name_snake = dataset.replace('-', '_')  # Standardize dataset name for file paths
        print(f"\n[INFO] Processing dataset: {dataset}")

        # Load training and testing data
        train = load_dataset(os.path.join(train_dir, f"{set_name_snake}-train.csv.gz"))
        test = load_dataset(os.path.join(test_dir, f"{set_name_snake}-test.csv.gz"))
        print(f"[INFO] Loaded data — train: {train.shape}, test: {test.shape}")

        # Process each model type
        for model_type in model_types:
            print(f"[INFO] Model type: {model_type}")

            shap_path = os.path.join(
                explanations_dir,
                dataset,
                f"{set_name_snake}-{model_type}-sampling-global.pickle.xz"
            )

            # Load SHAP explanation data
            try:
                shap_values = load_object(shap_path)
                print(f"[INFO] Loaded SHAP values from: {shap_path}")
                print(f"[INFO] Number of SHAP values: {len(shap_values)}")
            except Exception as e:
                print(f"[WARNING] Skipping model '{model_type}' — failed to load SHAP values ({e})")
                continue

            # Apply each feature selection strategy
            for strategy in strategies:
                print(f"[INFO] Applying strategy: {strategy}")

                # Strategy 1: cumulative SHAP contribution ('sum')
                if strategy == 'sum':
                    for sum_threshold in sum_thresholds:
                        print(f"[INFO] Using 'sum' threshold = {sum_threshold:.2f}")

                        # Select top features based on cumulative SHAP importance
                        idx = shap_select(
                            shap_values = shap_values['mean_shap'],
                            kind = 'sum',
                            sum_threshold = sum_threshold
                        )
                        print(f"[INFO] Selected {len(idx) - 1} features")

                        # Save reduced datasets
                        create_feature_selected_dataset(
                            idx = idx,
                            train = train,
                            test = test,
                            root_dir = save_dir,
                            dataset_name = set_name_snake,
                            model_type = model_type,
                            selection_strategy = strategy,
                            selection_threshold = sum_threshold
                        )

                        # Record selection metrics
                        fit_metrics.append({
                            'dataset': dataset,
                            'model': model_type,
                            'strategy': f'{strategy}_{sum_threshold}',
                            'k': len(idx) - 1,
                            'time': shap_values.get('time', None)
                        })
                        print(f"[DONE] Saved reduced dataset for 'sum' ({sum_threshold:.2f})")

                # Strategy 2: SHAP magnitude threshold ('max')
                elif strategy == 'max':
                    for max_threshold in max_thresholds:
                        print(f"[INFO] Using 'max' threshold = {max_threshold:.2f}")

                        # Select top features based on SHAP magnitude
                        idx = shap_select(
                            shap_values = shap_values['mean_shap'],
                            kind = 'max',
                            max_threshold = max_threshold
                        )
                        print(f"[INFO] Selected {len(idx) - 1} features")

                        # Save reduced datasets
                        create_feature_selected_dataset(
                            idx = idx,
                            train = train,
                            test = test,
                            root_dir = save_dir,
                            dataset_name = set_name_snake,
                            model_type = model_type,
                            selection_strategy = strategy,
                            selection_threshold = max_threshold
                        )

                        # Record selection metrics
                        fit_metrics.append({
                            'dataset': dataset,
                            'model': model_type,
                            'strategy': f'{strategy}_{max_threshold}',
                            'k': len(idx) - 1,
                            'time': shap_values.get('time', None)
                        })
                        print(f"[DONE] Saved reduced dataset for 'max' ({max_threshold:.2f})")

        print(f"[INFO] Completed all models for dataset: {dataset}")

    # Save SHAP feature selection metrics summary
    os.makedirs(metrics_dir, exist_ok = True)
    metrics_path = os.path.join(metrics_dir, 'SHAP_fit_statistics.csv')
    fit_metrics_df = pd.DataFrame(fit_metrics)
    fit_metrics_df.to_csv(metrics_path, index = False)
    print(f"\n[INFO] SHAP feature selection metrics saved to: {metrics_path}")

    return

def run_feature_selection(
    dataset_names: List[str],
    shap_fit_metrics: pd.DataFrame,
    train_dir: str,
    test_dir: str,
    save_dir: str,
    metrics_dir: str
) -> None:
    """
    Perform multiple feature selection methods (Mutual Information, ReliefF, MRMR, FCBF)
    across datasets, generate reduced versions, and record timing metrics.

    Args:
        dataset_names (List[str]): Names of datasets to process.
        k_percentages (List[float]): List of feature retention percentages (e.g., [0.25, 0.5, 0.75]).
        train_dir (str): Directory containing training datasets.
        test_dir (str): Directory containing testing datasets.        
        save_dir (str): Directory to save reduced datasets.
        metrics_dir (str): Directory to save timing metrics summary.
    """   
    
    # Initialize list to store timing metrics for feature selection
    fit_metrics = []

    # Iterate through all datasets
    for dataset in dataset_names:
        # Convert dataset name to snake_case for consistent file naming
        set_name_snake = dataset.replace('-', '_')
        print(f"[INFO] Processing dataset: {dataset}")

        # Load preprocessed training and testing splits
        train = load_dataset(os.path.join(train_dir, f'{set_name_snake}-train.csv.gz'))
        test = load_dataset(os.path.join(test_dir, f'{set_name_snake}-test.csv.gz'))
        print(f"[INFO] Loaded training and testing data (train shape: {train.shape}, test shape: {test.shape})")

        # Separate features (X) and target (y)
        X = train.iloc[:, :-1]
        y = train.iloc[:, -1]
        p = X.shape[1]  # Number of features

        # Temporary container for per-method feature rankings and timings
        save_data = []

        # Mutual Information
        print("[INFO] Computing Mutual Information feature ranking...")
        t_0 = time.time()
        mutual_info = mutual_info_classif(X, y, random_state = 42)
        sorted_idx = np.argsort(mutual_info)[::-1]
        calc_time = time.time() - t_0
        save_data.append(('mutual_info', sorted_idx, calc_time))
        print(f"[DONE] Mutual Information computed in {calc_time:.2f} seconds.")

        # ReliefF
        print("[INFO] Computing ReliefF feature ranking...")
        t_0 = time.time()
        relief = ReliefF(
            n_neighbors = 3,          # Low neighbor count for efficiency
            n_features_to_keep = p    # Keep all for sorting later
        )
        relief.fit(X.to_numpy(), y.to_numpy())  # ReliefF requires array-like inputs
        sorted_idx = relief.top_features
        calc_time = time.time() - t_0
        save_data.append(('relieff', sorted_idx, calc_time))
        print(f"[DONE] ReliefF computed in {calc_time:.2f} seconds.")

        # MRMR
        print("[INFO] Computing MRMR feature ranking...")
        t_0 = time.time()
        mrmr = MRMR(
            n_features_to_select = p, # Keep all for later selection
            method = 'MID'            # Mutual Information Difference
        )
        mrmr.fit(FDataGrid(data_matrix = X.to_numpy()), y)
        sorted_idx = mrmr.get_support(indices = True)
        calc_time = time.time() - t_0
        save_data.append(('mrmr', sorted_idx, calc_time))
        print(f"[DONE] MRMR computed in {calc_time:.2f} seconds.")

        shap_metrics_slice = shap_fit_metrics.loc[shap_fit_metrics.dataset == dataset]

        # Generate Reduced Datasets for Mutual Info, ReliefF, MRMR
        for strategy, sorted_idx, calc_time in save_data:
            print(f"[INFO] Generating reduced datasets for {strategy}...")
            for k in sorted(shap_metrics_slice.k.unique()):
                # Select top-k features + target column
                idx = np.concatenate([sorted_idx[:k], [p]])

                # Create and save reduced train/test datasets
                create_feature_selected_dataset(
                    idx = idx,
                    train = train,
                    test = test,
                    root_dir = save_dir,
                    dataset_name = set_name_snake,
                    model_type = "",
                    selection_strategy = strategy,
                    selection_threshold = k
                )

                # Record timing info
                fit_metrics.append({
                    'dataset': dataset,
                    'strategy': strategy,
                    'k': k,
                    'time': calc_time
                })

        print(f"[DONE] Completed processing for dataset: {dataset}\n")

    # Save Timing Metrics Summary
    metrics_path = os.path.join(metrics_dir, 'non_SHAP_fit_statistics.csv')
    print(f"[INFO] Writing timing metrics to {metrics_path}")
    fit_metrics = pd.DataFrame(fit_metrics)
    fit_metrics.to_csv(metrics_path, index = False)
    print("[DONE] All feature selection results saved successfully.\n")

    return