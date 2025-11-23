import os
from typing import Dict, Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from pandas.io.formats.style import Styler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score
)

from modules.selection import shap_select

from modules.utils import load_object

# ---- Statistics Generating Functions ----

def generate_crossval_statistics(models: Dict[str, Dict[str, Dict[str, Any]]],
                                 train_sets: Dict[str, Dict[str, pd.DataFrame]],
                                 n_splits: int = 5, random_state: int = 42,
                                 metrics_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Perform stratified k-fold cross-validation for all dataset-model-selection combinations
    and compute mean performance metrics (PR-AUC, ROC-AUC, F1). Optionally saves results to CSV.

    Args:
        models (Dict[str, Dict[str, Dict[str, Any]]]): Nested dictionary of fitted models
            indexed as [dataset_name][model_type][selection_type].
        train_sets (Dict[str, Dict[str, pd.DataFrame]]): Nested dictionary of corresponding
            training datasets indexed similarly.
        n_splits (int, optional): Number of folds for StratifiedKFold. Defaults to 5.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        metrics_dir (Optional[str], optional): Directory to save results as CSV. If None, results
            are not saved. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing averaged cross-validation statistics for each model.
    """
    model_stats = []

    # Loop through datasets, model types, and selection strategies
    for dataset_name, type_dict in models.items():
        for model_type, model_dict in type_dict.items():
            for selection_type, model in model_dict.items():
                try:
                    # Determine the key used to retrieve the matching dataset
                    if 'max' in selection_type or 'sum' in selection_type:
                        key = f'{model_type}-{selection_type}'
                    else:
                        key = selection_type

                    # Retrieve the corresponding training dataset
                    train = train_sets[dataset_name][key]
                    X = train.iloc[:, :-1]
                    y = train.iloc[:, -1]

                    # Initialize stratified k-fold
                    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)

                    # Track metrics per fold
                    pr_aucs, roc_aucs, recalls, precisions, f1s = [], [], [], [], []

                    for train_index, test_index in skf.split(X, y):
                        X_test_fold = X.iloc[test_index, :]
                        y_test_fold = y.iloc[test_index]

                        # Get model predictions and probabilities
                        y_pred = model.predict(X_test_fold)
                        if model_type != 'svc':
                            y_proba = model.predict_proba(X_test_fold)[:, 1]
                        else:
                            y_proba = model.decision_function(X_test_fold)

                        # Compute metrics per fold
                        pr_aucs.append(average_precision_score(y_test_fold, y_proba))
                        roc_aucs.append(roc_auc_score(y_test_fold, y_proba))
                        recalls.append(recall_score(y_test_fold, y_pred))
                        precisions.append(precision_score(y_test_fold, y_pred))
                        f1s.append(f1_score(y_test_fold, y_pred))

                    # Store averaged results
                    model_stats.append({
                        'dataset': dataset_name,
                        'model': model_type,
                        'selection_type': selection_type,
                        'k': X.shape[1],
                        'pr_auc': np.mean(pr_aucs),
                        'roc_auc': np.mean(roc_aucs),
                        'recall': np.mean(recalls),
                        'precision': np.mean(precisions),
                        'f1': np.mean(f1s)
                    })

                except Exception as e:
                    print(f"[ERROR] Failed to evaluate model for {dataset_name}, {model_type}, {selection_type}: {e}")
                    continue

    # Convert results to DataFrame
    results_df = pd.DataFrame(model_stats)

    # Optionally save to CSV
    if metrics_dir is not None:
        # Add fit time statistics
        results_df = add_fit_time_statistics(results_df, metrics_dir)

        # Save to path
        metrics_path = os.path.join(metrics_dir, 'crossval_statistics.csv')
        results_df.to_csv(metrics_path, index = False)
        print(f"[INFO] Cross-validation statistics saved to {metrics_path}")

    return results_df

def generate_test_statistics(models: Dict[str, Dict[str, Dict[str, Any]]],
                             test_sets: Dict[str, Dict[str, pd.DataFrame]],
                             metrics_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Evaluate trained models on held-out test data and compute performance metrics.

    Parameters
    ----------
    models : dict
        Nested dict of fitted models indexed as [dataset][model][selection_type].
    test_sets : dict
        Nested dict of corresponding test sets with same structure as `models`.
    metrics_dir : str, optional
        Directory to save results and merge timing data. If None, results are returned only.

    Returns
    -------
    pd.DataFrame
        Test performance metrics (PR-AUC, ROC-AUC, Recall, Precision, F1) per model.
    """
    model_stats = []

    for dataset_name, type_dict in models.items():
        for model_type, model_dict in type_dict.items():
            for selection_type, model in model_dict.items():
                try:
                    # Select correct test set key
                    if 'max' in selection_type or 'sum' in selection_type:
                        key = f'{model_type}-{selection_type}'
                    else:
                        key = selection_type
                    
                    test = test_sets[dataset_name][key]
                    X, y = test.iloc[:, :-1], test.iloc[:, -1]

                    # Predict and compute probabilities
                    y_pred = model.predict(X)
                    y_proba = model.decision_function(X) if model_type == 'svc' else model.predict_proba(X)[:, 1]

                    # Compute metrics
                    model_stats.append({
                        'dataset': dataset_name,
                        'model': model_type,
                        'selection_type': selection_type,
                        'k': X.shape[1],
                        'pr_auc': average_precision_score(y, y_proba),
                        'roc_auc': roc_auc_score(y, y_proba),
                        'recall': recall_score(y, y_pred),
                        'precision': precision_score(y, y_pred),
                        'f1': f1_score(y, y_pred)
                    })

                except Exception as e:
                    print(f"[ERROR] {dataset_name} | {model_type} | {selection_type}: {e}")
                    continue

    results_df = pd.DataFrame(model_stats)

    # Optionally merge timing info and save
    if metrics_dir is not None:
        results_df = add_fit_time_statistics(results_df, metrics_dir)
        path = os.path.join(metrics_dir, 'test_statistics.csv')
        results_df.to_csv(path, index = False)
        print(f"[INFO] Test statistics saved to {path}")

    return results_df

def add_fit_time_statistics(results_df: pd.DataFrame, metrics_dir: str) -> pd.DataFrame:
    """
    Merge fit-time statistics (SHAP and non-SHAP) into a results statistics dataframe.

    This function reads two CSV files from the provided metrics directory:
        - SHAP_fit_statistics.csv
        - non_SHAP_fit_statistics.csv
    It then updates the 'time' column in the results dataframe based on the dataset, model, and selection_type.

    Args:
        results_df (pd.DataFrame): Dataframe containing metrics for each combination.
            Must include 'dataset', 'model', 'selection_type' columns.
        metrics_dir (str): Directory containing the metric CSV files.

    Returns:
        pd.DataFrame: Updated statistics with 'time' values filled in.
    """
    # Load metric summary files
    shap_fit_stats = pd.read_csv(os.path.join(metrics_dir, 'SHAP_fit_statistics.csv'))
    non_shap_fit_stats = pd.read_csv(os.path.join(metrics_dir, 'non_SHAP_fit_statistics.csv'))

    # Initialize time column to zero
    results_df['time'] = 0.0

    # Iterate through all combinations of dataset/model/selection_type
    for dataset in results_df['dataset'].unique():
        for model in results_df['model'].unique():
            for selection_type in results_df['selection_type'].unique():

                # Skip full (non-reduced) models
                if selection_type == 'full':
                    continue

                try:
                    # Case 1: SHAP-based selection ('max' or 'sum')
                    if 'max' in selection_type or 'sum' in selection_type:
                        match = shap_fit_stats.loc[
                            (shap_fit_stats['dataset'] == dataset.replace('_', '-')) &
                            (shap_fit_stats['model'] == model)
                        ]
                        if not match.empty:
                            time_val = match['time'].iloc[0]
                            results_df.loc[
                                (results_df['dataset'] == dataset) &
                                (results_df['model'] == model) &
                                (results_df['selection_type'] == selection_type),
                                'time'
                            ] = time_val

                    # Case 2: Non-SHAP-based selection (e.g., mutual_info, relieff, mrmr)
                    elif '_' in selection_type:
                        s_type = '_'.join(selection_type.split('_')[:-1])
                        match = non_shap_fit_stats.loc[
                            (non_shap_fit_stats['dataset'] == dataset.replace('_', '-')) &
                            (non_shap_fit_stats['strategy'] == s_type)
                        ]
                        if not match.empty:
                            time_val = match['time'].iloc[0]
                            results_df.loc[
                                (results_df['dataset'] == dataset) &
                                (results_df['model'] == model) &
                                (results_df['selection_type'] == selection_type),
                                'time'
                            ] = time_val

                except Exception as e:
                    print(f"[ERROR] Failed to match timing for {dataset}, {model}, {selection_type}: {e}")
                    continue

    return results_df

# ---- Feature and Statistic Extraction Functions ----

def dataset_model_slice(df: pd.DataFrame, dataset: str, model: str) -> pd.DataFrame:
    """
    Filter a DataFrame for a given dataset–model pair and simplify selection labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing model evaluation or fit statistics.
    dataset : str
        Target dataset name.
    model : str
        Target model type.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with simplified 'selection_type' labels.
    """
    # Create a copy of the filtered subset
    df_slice = df.loc[(df.dataset == dataset) & (df.model == model)].copy()

    # Simplify selection type names (e.g., "mutual_info_50" -> "mutual_info")
    df_slice.loc[:, "selection_type"] = df_slice["selection_type"].apply(
        lambda x: "_".join(x.split("_")[:-1]) if "_" in x else x
    )

    return df_slice

def select_lowest_k(
    dataframe: pd.DataFrame,
    groupby_col: str,
    transform: Callable,
    include_selection_type: bool = True,
    group_cols: list[str] = None,
) -> pd.DataFrame:
    """
    Select rows with the lowest 'k' value per group after applying a transformation
    to a grouping column, with flexible grouping columns.

    Args:
        dataframe (pd.DataFrame):
            Input DataFrame containing at least the columns in group_cols,
            plus [groupby_col, 'k'], and optionally 'selection_type'.
        groupby_col (str):
            Column name used for the transformation.
        transform (Callable):
            Function applied via groupby().transform(), such as min, max,
            median, or a custom function returning a scalar.
        include_selection_type (bool, optional):
            Whether to append 'selection_type' to the grouping keys
            (if the column exists). Defaults to True.
        group_cols (list[str], optional):
            List of columns to group by (default: ['dataset', 'model']).

    Returns:
        pd.DataFrame:
            Filtered DataFrame containing one row per grouping combination
            and groupby_col value with the smallest 'k' after applying
            the transformation.
    """

    df = dataframe.copy()

    # Exclude "full" selection_type when present
    if "selection_type" in df.columns:
        df = df[df["selection_type"] != "full"]

    # Default grouping columns
    if group_cols is None:
        group_cols = ["dataset", "model"]

    # Optionally append selection_type
    if include_selection_type and "selection_type" in df.columns:
        group_cols = group_cols + ["selection_type"]

    # Apply transform to the target column within groups
    groupby_mask = (
        df.groupby(group_cols)[groupby_col].transform(transform)
        == df[groupby_col]
    )

    masked_data = df.loc[groupby_mask]

    # Select lowest k inside each (group_cols + groupby_col) subgroup
    k_mask = (
        masked_data
        .groupby(group_cols + [groupby_col])["k"]
        .transform("min")
        == masked_data["k"]
    )

    return masked_data.loc[k_mask].drop_duplicates()

# def select_lowest_k(dataframe: pd.DataFrame, groupby_col: str,
#                     transform: Callable, include_selection_type: bool = True) -> pd.DataFrame:
#     """
#     Select rows with the lowest 'k' value per group after applying a transformation
#     to a grouping column, optionally including 'selection_type' in the grouping.

#     This function filters the input DataFrame to exclude entries where
#     'selection_type' == 'full' (if present), applies a specified transformation
#     (e.g., 'min', 'max', 'median') to the specified grouping column, and returns
#     only those rows that both match the transformed group value and have the
#     smallest 'k' for that group.

#     Args:
#         dataframe (pd.DataFrame): Input DataFrame containing at least the columns:
#             ['dataset', 'model', groupby_col, 'k'], and optionally 'selection_type'.
#         groupby_col (str): The column name used for computing the transformation.
#         transform (Callable): A transformation function applied via `groupby().transform()`,
#             such as `min`, `max`, `median`, or a custom callable returning a scalar.
#         include_selection_type (bool, optional): Whether to include 'selection_type' as part
#             of the grouping key. Defaults to True.

#     Returns:
#         pd.DataFrame: A filtered DataFrame containing one row per unique combination of
#         grouping keys (depending on `include_selection_type`) and 'groupby_col' with the
#         smallest 'k' value, after applying the specified transformation.
#     """
#     # Exclude rows where the selection type is 'full' (if column exists)
#     df = dataframe.copy()
#     if 'selection_type' in df.columns:
#         df = df[df.selection_type != 'full']

#     # Define grouping columns based on user preference
#     group_cols = ['dataset', 'model']
#     if include_selection_type and 'selection_type' in df.columns:
#         group_cols.append('selection_type')

#     # Identify rows where the groupby_col value equals the transformed value within each group
#     groupby_mask = (
#         df.groupby(group_cols)[groupby_col].transform(transform) == df[groupby_col]
#     )

#     # Keep only rows matching the transformed group value
#     masked_data = df.loc[groupby_mask]

#     # Within each subgroup, select rows with the minimum 'k' value
#     k_mask = (
#         masked_data.groupby(group_cols + [groupby_col])['k']
#         .transform('min') == masked_data['k']
#     )

#     # Drop duplicates to ensure one representative per group and return the filtered DataFrame
#     return masked_data.loc[k_mask].drop_duplicates()
    
def extract_best_k(cv_stats: pd.DataFrame, test_stats: pd.DataFrame,
                   metrics_dir: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """
    Extract optimal 'k' configurations from cross-validation and test statistics.

    This function performs the following steps:

    1. Merge cross-validation (CV) and test statistics on shared identifiers
       ['dataset', 'model', 'selection_type', 'k'].
    2. Clean and normalize selection type names:
        - Removes numeric suffixes (e.g., 'mutual_info_50' → 'mutual_info').
        - Collapses SHAP variants into a single category ('sum' or 'max' → 'shap').
    3. Compute overfitting metrics:
        - `pr_auc_diff` = PR-AUC_CV - PR-AUC_test (smaller indicates lower overfitting).
        - `pr_auc_diff_percent` = pr_auc_diff / PR-AUC_test.
    4. Compute feature set metrics relative to the full feature set:
        - `k_percent`: proportion of features used relative to the full set.
        - `pr_auc_test_percent`: test PR-AUC relative to full-feature test performance.
    5. Identify "best k" configurations:
        - Lowest overfitting: configurations minimizing `pr_auc_diff`.
        - Maximum test performance: configurations maximizing `pr_auc_test`.
        - Strongest local peak: local maxima in the PR-AUC_test curve, filtered
          by the 25th percentile threshold.
        - Earliest peak: among local peaks, the configuration with the lowest `k`.
    6. Optionally save the resulting DataFrames as CSV files in the specified directory.

    Args:
        cv_stats (pd.DataFrame): Cross-validation results with columns:
            ['dataset', 'model', 'selection_type', 'k', 'pr_auc'].
        test_stats (pd.DataFrame): Test set results with matching identifiers and 'pr_auc'.
        metrics_dir (str, optional): Path to save output CSVs. If None, results are returned
            but not saved.

    Returns:
        dict[str, pd.DataFrame]: Dictionary containing the following keys:
            - 'lowest_overfit': Rows with minimum PR-AUC difference.
            - 'max_test': Rows with maximum test PR-AUC.
            - 'strongest_peak': Rows representing strongest local PR-AUC peaks.
            - 'earliest_peak': Rows representing earliest local peaks.
    """

    # Merge CV and test data on dataset, model, selection_type, and k
    cv_test = cv_stats.merge(
        test_stats,
        on=['dataset', 'model', 'selection_type', 'k'],
        suffixes=('_cv', '_test'),
        how='inner'
    )[
        ['dataset', 'model', 'selection_type', 'k', 'pr_auc_cv', 'pr_auc_test']
    ]

    # Simplify selection_type names by removing trailing numeric suffixes
    cv_test["selection_type"] = cv_test["selection_type"].str.replace(
        r"_[^_]+$", "", regex=True
    )

    # Normalize SHAP variants into a single category
    cv_test["selection_type"] = cv_test["selection_type"].replace(
        {"sum": "shap", "max": "shap"}
    )

    # Compute overfitting metrics
    cv_test["pr_auc_diff"] = cv_test["pr_auc_cv"] - cv_test["pr_auc_test"]
    cv_test["pr_auc_diff_percent"] = cv_test["pr_auc_diff"] / cv_test["pr_auc_test"]

    # Vectorized computation of percentage of features kept relative to full feature set
    full_k = (
        cv_test[cv_test["selection_type"] == "full"]
        .groupby("dataset")["k"]
        .max()
        .rename("k_full")
    )
    cv_test = cv_test.merge(full_k, on="dataset", how="left")
    cv_test["k_percent"] = 100 * cv_test["k"] / cv_test["k_full"]
    cv_test.drop(columns="k_full", inplace=True)

    # Vectorized computation of PR-AUC test percentage relative to full feature set
    full_pr_auc = (
        cv_test[cv_test["selection_type"] == "full"]
        .groupby(["dataset", "model"])["pr_auc_test"]
        .max()
        .rename("full_pr_auc")
    )
    cv_test = cv_test.merge(full_pr_auc, on=["dataset", "model"], how="left")
    cv_test["pr_auc_test_percent"] = 100 * cv_test["pr_auc_test"] / cv_test["full_pr_auc"]
    cv_test.drop(columns="full_pr_auc", inplace=True)

    # Identify best-k configurations by lowest overfitting
    lowest_overfit = select_lowest_k(cv_test, groupby_col='pr_auc_diff', transform='min')

    # Identify best-k configurations by maximum test performance
    max_test = select_lowest_k(cv_test, groupby_col='pr_auc_test', transform='max')

    # Identify strongest local peaks in PR-AUC curves
    peak_slice = []
    grouped = cv_test.sort_values("k").groupby(["dataset", "selection_type", "model"])

    for (dataset, s_type, m_type), sub in grouped:
        if s_type == "full":
            continue

        # 25th percentile threshold for candidate peaks
        threshold = sub["pr_auc_test"].quantile(0.25)

        # Use shifted values to detect local maxima
        prev_vals = sub["pr_auc_test"].shift(1)
        next_vals = sub["pr_auc_test"].shift(-1)

        peak_mask = (
            (sub["pr_auc_test"] > prev_vals) &
            (sub["pr_auc_test"] > next_vals) &
            (sub["pr_auc_test"] >= threshold)
        )

        candidate_idx = sub.index[peak_mask].tolist()

        # Include smallest k if its performance is competitive
        min_k_idx = sub.nsmallest(1, "k").index[0]
        if sub.loc[min_k_idx, "pr_auc_test"] > sub.loc[candidate_idx, "pr_auc_test"].max():
            candidate_idx.append(min_k_idx)

        # Fallback: if no peaks, select global max PR-AUC
        if not candidate_idx:
            max_idx = sub["pr_auc_test"].idxmax()
            candidate_idx.append(max_idx)

        peak_slice.extend(candidate_idx)

    # Combine identified peaks across all datasets/models
    peak_slice = cv_test.loc[sorted(set(peak_slice))].copy()

    # Among peaks, select strongest (max PR-AUC) and earliest (min k)
    strongest_peak = select_lowest_k(peak_slice, groupby_col='pr_auc_test', transform='max')
    earliest_peak = select_lowest_k(peak_slice, groupby_col='k', transform='min')

    results = {
        "lowest_overfit": lowest_overfit,
        "max_test": max_test,
        "strongest_peak": strongest_peak,
        "earliest_peak": earliest_peak,
    }

    # Save results as CSV files if a metrics directory is provided
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
        for name, df in results.items():
            df.to_csv(os.path.join(metrics_dir, f"{name}_best_k.csv"), index=False)

    return results

# def calculate_performance_counts(results: Dict[str, pd.DataFrame],
#                                  metrics_dir: Optional[str] = None) -> pd.DataFrame:
#     """
#     Compute and summarize performance counts and average k percentages
#     across multiple result DataFrames, optionally saving the results to CSV.

#     This function iterates over a dictionary of result DataFrames, applying
#     custom aggregation rules depending on the dataset index:
#         - For the first dataset, selects entries with the minimum `pr_auc_diff`.
#         - For subsequent datasets, selects entries with the maximum `pr_auc_test`.

#     The function then counts how many times each feature selection method
#     appears and computes the mean `k_percent` for each method. The results
#     are concatenated into a single summary table.

#     Args:
#         results (Dict[str, pd.DataFrame]): 
#             Dictionary mapping dataset names (keys) to result DataFrames (values).
#         metrics_dir (Optional[str], optional): 
#             Directory path where the resulting summary table should be saved as
#             `performance_counts.csv`. If None, the file is not saved. Defaults to None.

#     Returns:
#         pd.DataFrame: 
#             A DataFrame summarizing counts and mean k percentages per selection type.
#     """
#     performance_counts = pd.DataFrame()

#     for i, (name, df) in enumerate(results.items()):
#         # Define selection criterion depending on dataset order
#         if i == 0:
#             groupby_col = 'pr_auc_diff'
#             transform = 'min'
#         else:
#             groupby_col = 'pr_auc_test'
#             transform = 'max'

#         # Select the best-performing subset per criterion
#         subframe = select_lowest_k(
#             df,
#             groupby_col = groupby_col,
#             transform = transform,
#             include_selection_type = False
#         ).drop_duplicates(subset = groupby_col, keep = False)

#         # Count occurrences of each selection method
#         df_counts = subframe['selection_type'].value_counts().to_frame(
#             name = f'{name.replace("_", " ").title()} Count'
#         )

#         # Compute mean k_percent per selection type
#         df_k = np.round(
#             subframe.groupby('selection_type')['k_percent'].mean().to_frame(
#                 name = f'{name.replace("_", " ").title()} Mean K%'
#             ),
#             2
#         )

#         # Merge into master performance table
#         performance_counts = pd.concat([performance_counts, df_counts, df_k], axis = 1).fillna(0)

#     # Clean and format final table
#     performance_counts.index.name = 'Selection Type'
#     performance_counts = performance_counts.reset_index().sort_values('Selection Type')
#     performance_counts['Selection Type'] = performance_counts['Selection Type'].replace({
#         'relieff': 'ReliefF',
#         'shap': 'SHAP',
#         'mutual_info': 'Mutual Information',
#         'mrmr': 'mRMR'
#     })

#     # Set index for readability
#     performance_counts = performance_counts.set_index('Selection Type')

#     # Optionally save to CSV
#     if metrics_dir is not None:
#         os.makedirs(metrics_dir, exist_ok = True)
#         save_path = os.path.join(metrics_dir, "performance_counts.csv")
#         performance_counts.to_csv(save_path, index = True)

#     return performance_counts

def calculate_performance_counts(
    results: Dict[str, pd.DataFrame],
    metrics_dir: Optional[str] = None,
    subset_cols: Union[str, list[str], None] = "selection_type",
) -> pd.DataFrame:
    """
    Compute and summarize performance counts and average k percentages
    across multiple result DataFrames, optionally saving the results to CSV.

    Args:
        results (Dict[str, pd.DataFrame]):
            Dictionary mapping dataset names to result DataFrames.
        metrics_dir (Optional[str], optional):
            If provided, saves the summary CSV in this directory.
        subset_cols (str or list[str], optional):
            Column(s) to use for value_counts and groupby operations.
            Defaults to 'selection_type'.

    Returns:
        pd.DataFrame:
            Summary table with counts and mean k_percent per selection group.
    """

    # Normalize subset_cols to list for consistency
    if subset_cols is None:
        subset_cols = "selection_type"
    # elif isinstance(subset_cols, str):
    #     subset_cols = [subset_cols]

    group_key = subset_cols  # used for groupby and counts

    performance_counts = pd.DataFrame()

    for i, (name, df) in enumerate(results.items()):
        # Select criterion depending on dataset order
        if i == 0:
            groupby_col = "pr_auc_diff"
            transform = "min"
        else:
            groupby_col = "pr_auc_test"
            transform = "max"

        # Run lowest-k selection (ignores selection_type grouping)
        subframe = select_lowest_k(
            df,
            groupby_col=groupby_col,
            transform=transform,
            include_selection_type=False,
        ).drop_duplicates(subset=groupby_col, keep=False)

        # ----- Counts -----
        df_counts = (
            subframe[group_key]
            .value_counts()
            .to_frame(name=f'{name.replace("_", " ").title()} Count')
        )

        # ----- Mean K% -----
        df_k = (
            subframe.groupby(group_key)["k_percent"]
            .mean()
            .round(2)
            .to_frame(name=f'{name.replace("_", " ").title()} Mean K%')
        )

        # Merge into master table
        performance_counts = pd.concat(
            [performance_counts, df_counts, df_k], axis=1
        ).fillna(0)

    # Clean final table
    
    
    performance_counts = performance_counts.sort_index()
    # print(performance_counts)
    performance_counts.index.name = (
        "Selection Group" if isinstance(group_key, list) else group_key.replace("_", " ").title()
    )
    
    performance_counts = performance_counts.reset_index()
    # print(performance_counts)

    # Rename common selection types (only applies when subset_cols = ['selection_type'])
    rename_map = {
        "relieff": "ReliefF",
        "shap": "SHAP",
        "mutual_info": "Mutual Information",
        "mrmr": "mRMR",
    }
    if subset_cols == "selection_type":
        performance_counts['Selection Type'] = \
            performance_counts['Selection Type'].replace(rename_map)

    # Re-index after optional name mapping
    performance_counts = performance_counts.set_index(performance_counts.columns[0])

    # Optional save
    if metrics_dir is not None:
        os.makedirs(metrics_dir, exist_ok=True)
        save_path = os.path.join(metrics_dir, "performance_counts.csv")
        performance_counts.to_csv(save_path, index=True)

    return performance_counts
    
def extract_average_time_data(time_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute average timing statistics for selection types and SHAP models.

    Args:
        time_data (pd.DataFrame): DataFrame containing columns
            ['selection_type', 'avg_time', 'model', 'model_avg_time'].

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - average_time_data: Average time per selection type.
            - average_shap_time: Average time per model for SHAP-based selection.
    """

    # Compute average time per selection type
    average_time_data = (
        time_data.groupby('selection_type')['avg_time']
        .mean()
        .reset_index()
    )

    # Normalize selection type names
    average_time_data['selection_type'] = average_time_data['selection_type'].replace({
        'mrmr': 'mRMR',
        'mutual_info': 'Mutual Information',
        'relieff': 'ReliefF',
        'shap': 'SHAP'
    })

    average_time_data.columns = ['Selection Type', 'Average Time (s)']

    # Compute average SHAP time per model
    average_shap_time = (
        time_data[time_data['selection_type'] == 'shap']
        .groupby('model')['model_avg_time']
        .mean()
        .reset_index()
    )

    # Normalize model names
    average_shap_time['model'] = average_shap_time['model'].replace({
        'dt': 'Decision Tree',
        'logreg': 'Logistic Regression',
        'rf': 'Random Forest',
        'svc': 'Support Vector Machine',
        'xgb': 'XGBoost'
    })

    average_shap_time.columns = ['Model Type', 'Average Time (s)']

    return average_time_data, average_shap_time

def calculate_shap_counts(results: Dict[str, pd.DataFrame], cv_stats: pd.DataFrame,
                          metrics_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate how many times SHAP-based feature selection appears in each of the
    result categories (e.g., lowest overfit, max test, strongest peak, earliest peak).

    Optionally saves the resulting summary table to a CSV file when a directory
    path is provided.

    Args:
        results (dict): Dictionary mapping result names to DataFrames containing the
            selected rows (e.g., {'max_test': df, 'strongest_peak': df, ...}).
        cv_stats (pd.DataFrame): Cross-validation statistics DataFrame used to count
            occurrences of each selection type.
        metrics_dir (str, optional): Directory where the output CSV file will be saved.
            If None, results are returned but not written to disk.

    Returns:
        pd.DataFrame: A summary table containing SHAP selection type counts across
            all result categories, with readable labels.
    """

    shap_counts = pd.DataFrame()

    # Collect SHAP counts for each metric category
    for metric, frame in results.items():
        idx = frame[frame['selection_type'] == 'shap'].index.tolist()

        shap_metric_counts = (
            cv_stats.loc[idx]
            .value_counts('selection_type')
            .to_frame(name = metric)
        )

        shap_counts = (
            pd.concat([shap_counts, shap_metric_counts], axis = 1)
            .fillna(0)
        )

    shap_counts = shap_counts.astype(int).sort_values(
        'earliest_peak', ascending = False
    )

    # Final display formatting
    shap_counts = shap_counts.reset_index()
    shap_counts['selection_type'] = shap_counts['selection_type'].apply(
        lambda x: ' = '.join(x.title().split('_'))
    )

    shap_counts.columns = [
        'Selection Type',
        'Lowest Overfit Count',
        'Max Test Count',
        'Strongest Peak Count',
        'Earliest Peak Count'
    ]

    # Optional saving
    if metrics_dir is not None:
        os.makedirs(metrics_dir, exist_ok = True)
        save_path = os.path.join(metrics_dir, "shap_feature_selection_counts.csv")
        shap_counts.to_csv(save_path, index = False)

    return shap_counts

# ---- Plotting and Styling Functions ----

def custom_barplot(data, x: str, y: str, hue: Optional[str] = None,
                   hue_order: Optional[List[str]] = None, xticklabels: Optional[List[str]] = None,
                   xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                   title: Optional[str] = None, rotation: int = 0,
                   legend_labels: Optional[List[str]] = None,
                   legend_kwargs: Optional[Dict[str, Any]] = None, figsize: tuple = (12, 4)) -> plt.Axes:
    """
    Create a customized bar plot using Seaborn with flexible labeling and formatting options.

    Args:
        data (pd.DataFrame): Input DataFrame containing data for plotting.
        x (str): Column name to use for the x-axis.
        y (str): Column name to use for the y-axis.
        hue (str, optional): Column name for color grouping.
        hue_order (List[str], optional): Custom ordering of hue levels passed to seaborn.
        xticklabels (List[str], optional): Custom labels for x-axis ticks.
        xlabel (str, optional): Label for the x-axis. Defaults to no label.
        ylabel (str, optional): Label for the y-axis.
        title (str, optional): Title of the plot.
        rotation (int, optional): Rotation angle for x-tick labels.
        legend_labels (List[str], optional): Custom labels for the legend.
        legend_kwargs (dict, optional): Additional kwargs passed to ax.legend().
        figsize (tuple, optional): Figure size.

    Returns:
        matplotlib.axes.Axes: The Matplotlib Axes object for further customization.
    """

    # Create figure and axis
    fig, ax = plt.subplots(figsize = figsize)

    # Bar plot
    sns.barplot(
        data = data,
        x = x,
        y = y,
        hue = hue,
        hue_order = hue_order,
        errorbar = None,
        ax = ax
    )

    # Custom xticklabels
    if xticklabels is not None:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(xticklabels)

    # Labels and title
    ax.set_xlabel(xlabel if xlabel is not None else "")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    # Rotate x-ticks
    ax.tick_params(axis = "x", rotation = rotation)

    # Legend handling
    if hue is not None:
        handles, labels = ax.get_legend_handles_labels()
        if legend_labels is not None:
            labels = legend_labels

        if legend_kwargs is None:
            legend_kwargs = {}

        ax.legend(handles = handles, labels = labels, **legend_kwargs)

    return ax


def custom_scatterplot(data, x: str, y: str, hue: Optional[str] = None,
                       hue_order: Optional[List[str]] = None, xlabel: Optional[str] = None,
                       ylabel: Optional[str] = None, title: Optional[str] = None,
                       legend_labels: Optional[List[str]] = None,
                       legend_kwargs: Optional[Dict[str, Any]] = None, rotation: int = 0,
                       figsize: tuple = (12, 4)) -> plt.Axes:
    """
    Create a customized scatter plot using Seaborn with flexible labeling and formatting options.

    Args:
        data (pd.DataFrame): Input DataFrame to plot.
        x (str): Column name for x-axis values.
        y (str): Column name for y-axis values.
        hue (str, optional): Column name for color grouping. Defaults to None.
        hue_order (List[str], optional): Custom ordering of hue levels to pass to seaborn.
        xlabel (str, optional): Label for the x-axis. If None, no label is applied.
        ylabel (str, optional): Label for the y-axis. If None, no label is applied.
        title (str, optional): Plot title. Defaults to None.
        legend_labels (List[str], optional): Custom labels for the legend. If None, seaborn defaults are used.
        legend_kwargs (dict, optional): Additional keyword arguments passed to ax.legend().
        rotation (int, optional): Rotation for x-tick labels. Defaults to 0.
        figsize (tuple, optional): Figure size. Defaults to (12, 4).

    Returns:
        matplotlib.axes.Axes: The configured Axes object.
    """

    # Create figure and axis
    fig, ax = plt.subplots(figsize = figsize)

    # Scatterplot
    sns.scatterplot(
        data = data,
        x = x,
        y = y,
        hue = hue,
        hue_order = hue_order,
        ax = ax
    )

    # Axis labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel("")

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Title
    if title is not None:
        ax.set_title(title)

    # Rotate x-tick labels if needed
    ax.tick_params(axis = "x", rotation = rotation)

    # Legend customization
    if hue is not None:
        handles, labels = ax.get_legend_handles_labels()

        # Override seaborn-generated labels if user supplies custom ones
        if legend_labels is not None:
            labels = legend_labels

        if legend_kwargs is None:
            legend_kwargs = {}

        ax.legend(handles = handles, labels = labels, **legend_kwargs)

    return ax

# def style_dataframe(df: pd.DataFrame, hide_index: bool = True,
#                     format_dict: Optional[dict] = None, cell_width: str = '90px',
#                     index_width: str = '150px') -> Styler:
#     """
#     Apply consistent styling to a DataFrame for display in Jupyter notebooks.

#     Args:
#         df (pd.DataFrame): The DataFrame to style.
#         hide_index (bool, optional): Whether to hide the index column. Defaults to True.
#         format_dict (dict, optional): Dictionary mapping column names to formatting strings
#             (e.g., {'col1': '{:.2f}'}). If None, numeric columns are automatically formatted.
#         cell_width (str, optional): Width of data cells. Defaults to '90px'.
#         index_width (str, optional): Width of index column. Defaults to '150px'.

#     Returns:
#         pd.io.formats.style.Styler: Styled DataFrame object.
#     """
#     styled = df.style.set_table_styles(
#         [
#             # Column headers
#             {'selector': 'th', 'props': [
#                 ('text-align', 'center'),
#                 ('color', 'black'),
#                 # ('border', '1px solid black'),
#                 ('border-bottom', '1px solid black'),
#                 # ('border-top', '1px solid black'),
#                 # ('border-right', '1px solid black'),
#                 ('background-color', 'white'),
#                 ('font-size', '12px'),
#                 ('font-weight', 'bold')
#             ]},
#             # Index column (row header)
#             {'selector': 'th.row_heading', 'props': [
#                 ('text-align', 'center'),
#                 ('width', index_width),
#                 # ('border', '1px solid black'),
#                 # ('border-top', '1px solid black'),
#                 ('border-right', '1px solid black'),
#                 ('background-color', 'white'),
#                 ('color', 'black')
#             ]},
#             # Data cells
#             {'selector': 'td', 'props': [
#                 ('text-align', 'center'),
#                 ('color', 'black'),
#                 # ('border', '1px solid black'),
#                 # ('border-bottom', '1px solid black'),
#                 # ('border-right', '1px solid black'),
#                 ('background-color', 'white'),
#                 ('width', cell_width),
#                 ('font-size', '12px'),
#                 ('font-weight', 'bold')
#             ]}
#         ]
#     )
#     if format_dict:
#         styled = styled.format(format_dict)

#     if hide_index:
#         styled = styled.hide(axis = 'index')

#     styled = styled.set_properties(**{'width': cell_width})

#     return styled

def style_dataframe(
    df: pd.DataFrame,
    hide_index: bool = True,
    format_dict: Optional[dict] = None,
    cell_width: str = '90px',
    index_width: str = '150px',
    alternate_rows: bool = False,
    alt_color_1: str = "#ffffff",
    alt_color_2: str = "#f5f5f5",
    header_bg: str = "#e6e6e6",
    header_font_color: str = "black",
) -> Styler:
    """
    Apply consistent styling to a DataFrame for display in Jupyter notebooks,
    with optional alternating row background colors and unified header styling
    for standard headers and MultiIndex headers.
    """

    def _alternate_row_bg_full(df_in: pd.DataFrame) -> pd.DataFrame:
        """Create DataFrame of style strings for row-wise alternating backgrounds."""
        n_rows, n_cols = df_in.shape
        bg = pd.DataFrame('', index=df_in.index, columns=df_in.columns)
        for i in range(n_rows):
            color = alt_color_1 if (i % 2) == 0 else alt_color_2
            bg.iloc[i, :] = f'background-color: {color}'
        return bg

    styled = df.style.set_table_styles(
        [
            # GENERAL TH — applies to *all* header cells including MultiIndex
            {'selector': 'th', 'props': [
                ('text-align', 'center'),
                ('color', header_font_color),
                ('background-color', header_bg),
                ('font-size', '12px'),
                ('font-weight', 'bold'),
            ]},

            # Column header cells (more specific, overrides general th)
            {'selector': 'th.col_heading', 'props': [
                ('border-bottom', '1px solid black'),
            ]},

            # Row index header cells
            {'selector': 'th.row_heading', 'props': [
                ('width', index_width),
                ('border-right', '1px solid black'),
            ]},

            # Data cells
            {'selector': 'td', 'props': [
                ('text-align', 'center'),
                ('color', 'black'),
                # ('background-color', 'white'),
                ('width', cell_width),
                ('font-size', '12px'),
                ('font-weight', 'bold'),
            ]},
        ]
    )

    # Formatting
    if format_dict:
        styled = styled.format(format_dict)

    # Hide index if requested
    if hide_index:
        styled = styled.hide(axis='index')

    # Apply alternating row backgrounds
    if alternate_rows:
        styled = styled.apply(_alternate_row_bg_full, axis=None)

    # Ensure widths apply
    styled = styled.set_properties(**{'width': cell_width})

    return styled


# def style_dataframe(
#     df: pd.DataFrame,
#     hide_index: bool = True,
#     format_dict: Optional[dict] = None,
#     cell_width: str = "90px",
#     index_width: str = "150px",
#     alternate_rows: bool = False,
#     striped_cols: Optional[list] = None,
#     striped_index_level: Optional[Union[int, str]] = None,
#     alt_color_1: str = "#ffffff",
#     alt_color_2: str = "#f5f5f5",
#     header_bg: str = "#e6e6e6",
#     header_font_color: str = "black",
# ) -> Styler:
#     """
#     Style a DataFrame with optional alternating row shading, optionally applying
#     alternating shading to a specific level of a MultiIndex.

#     Args:
#         striped_cols (list, optional):
#             Columns to apply striping to. If None, all columns are striped.
#         striped_index_level (int or str, optional):
#             MultiIndex level (by integer position or name) to use for row striping.
#             If None, index is not striped separately.
#     """

#     # Determine column mask for striping
#     if striped_cols is None:
#         col_mask = pd.Series(True, index=df.columns)
#     else:
#         col_mask = pd.Series(False, index=df.columns)
#         for col in striped_cols:
#             if col in df.columns:
#                 col_mask.loc[col] = True
#             else:
#                 raise KeyError(f"Column '{col}' not found in DataFrame.columns.")

#     def _alternate_row_styles(data: pd.DataFrame) -> pd.DataFrame:
#         """Generate CSS for alternating row colors on selected columns."""
#         styles = pd.DataFrame("", index=data.index, columns=data.columns)

#         if striped_index_level is not None and isinstance(data.index, pd.MultiIndex):
#             # Compute stripe colors based on unique values in the given level
#             if isinstance(striped_index_level, int):
#                 level_vals = data.index.get_level_values(striped_index_level)
#             else:
#                 level_vals = data.index.get_level_values(striped_index_level)

#             # Track alternating blocks per unique value
#             unique_vals = pd.Series(level_vals).factorize()[0]  # 0,1,2,...
#             for i, idx in enumerate(data.index):
#                 bg = alt_color_1 if (unique_vals[i] % 2 == 0) else alt_color_2
#                 for col in data.columns:
#                     if col_mask[col]:
#                         styles.at[idx, col] = f"background-color: {bg}"
#         else:
#             # Default: stripe by row index
#             for i, idx in enumerate(data.index):
#                 bg = alt_color_1 if (i % 2 == 0) else alt_color_2
#                 for col in data.columns:
#                     if col_mask[col]:
#                         styles.at[idx, col] = f"background-color: {bg}"

#         return styles

#     # Base styles
#     styled = df.style.set_table_styles(
#         [
#             {"selector": "th", "props": [
#                 ("background-color", header_bg),
#                 ("color", header_font_color),
#                 ("font-size", "12px"),
#                 ("font-weight", "bold"),
#                 ("text-align", "center"),
#             ]},
#             {"selector": "th.col_heading", "props": [("border-bottom", "1px solid black")]},
#             {"selector": "th.row_heading", "props": [
#                 ("width", index_width),
#                 ("border-right", "1px solid black"),
#                 ("text-align", "center"),
#             ]},
#             {"selector": "td", "props": [
#                 ("text-align", "center"),
#                 ("color", "black"),
#                 # ("background-color", "white"),
#                 ("width", cell_width),
#                 ("font-size", "12px"),
#                 ("font-weight", "bold"),
#             ]},
#         ]
#     )

#     if format_dict:
#         styled = styled.format(format_dict)

#     if alternate_rows:
#         styled = styled.apply(_alternate_row_styles, axis=None)

#     if hide_index:
#         styled = styled.hide(axis="index")

#     styled = styled.set_properties(**{"width": cell_width})

#     return styled





# def style_dataframe(
#     df: pd.DataFrame,
#     hide_index: bool = True,
#     format_dict: Optional[dict] = None,
#     cell_width: str = '90px',
#     index_width: str = '150px',
#     alternate_rows: bool = False,
#     alt_color_1: str = "#ffffff",
#     alt_color_2: str = "#f5f5f5",
#     header_bg: str = "#e6e6e6",
#     header_font_color: str = "black",
# ) -> Styler:
#     """
#     Apply consistent styling to a DataFrame for display in Jupyter notebooks,
#     with optional alternating row background colors and unified header styling
#     for both column headers and index headers.

#     Args:
#         df (pd.DataFrame): The DataFrame to style.
#         hide_index (bool, optional): Whether to hide the index column. Defaults to True.
#         format_dict (dict, optional): Formatting rules for columns.
#         cell_width (str, optional): Width of data cells.
#         index_width (str, optional): Width of index column.
#         alternate_rows (bool, optional): Enable alternating row colors.
#         alt_color_1 (str, optional): Background for even rows.
#         alt_color_2 (str, optional): Background for odd rows.
#         header_bg (str, optional): Background color for *all* header cells.
#         header_font_color (str, optional): Text color for header labels.

#     Returns:
#         Styler: Styled DataFrame.
#     """

#     def _alternate_row_colors(row):
#         """Return row-level background color instructions."""
#         color = alt_color_1 if row.name % 2 == 0 else alt_color_2
#         return [f"background-color: {color}"] * len(row)

#     styled = df.style.set_table_styles(
#         [
#             # Column headers
#             {'selector': 'th.col_heading', 'props': [
#                 ('text-align', 'center'),
#                 ('color', header_font_color),
#                 ('background-color', header_bg),
#                 ('border-bottom', '1px solid black'),
#                 ('font-size', '12px'),
#                 ('font-weight', 'bold'),
#             ]},
#             # Index header (row index label)
#             {'selector': 'th.row_heading', 'props': [
#                 ('text-align', 'center'),
#                 ('width', index_width),
#                 ('border-right', '1px solid black'),
#                 ('color', header_font_color),
#                 ('background-color', header_bg),   # <<< MATCH COLUMN HEADER STYLE
#                 ('font-size', '12px'),
#                 ('font-weight', 'bold'),
#             ]},
#             # Data cells
#             {'selector': 'td', 'props': [
#                 ('text-align', 'center'),
#                 ('color', 'black'),
#                 ('background-color', 'white'),
#                 ('width', cell_width),
#                 ('font-size', '12px'),
#                 ('font-weight', 'bold'),
#             ]},
#         ]
#     )

#     if format_dict:
#         styled = styled.format(format_dict)

#     if hide_index:
#         styled = styled.hide(axis='index')

#     # Alternating row colors
#     if alternate_rows:
#         styled = styled.apply(_alternate_row_colors, axis=1)

#     styled = styled.set_properties(**{'width': cell_width})

#     return styled
    
def style_pivot(
    dataframe: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    aggregation: Union[str, callable] = "mean",
    column_labels: Optional[list] = None,
    index_labels: Optional[list] = None,
    hide_index: bool = True,
    precision: str = "{:.4f}",
    cell_width: str = "120px",
    caption: Optional[str] = None,
) -> pd.io.formats.style.Styler:
    """
    Create a pivot table with custom labels and apply consistent styling with style_dataframe.

    Args:
        dataframe (pd.DataFrame):
            Source DataFrame.
        index (str):
            Column to use as pivot index.
        columns (str):
            Column to use for pivot columns.
        values (str):
            Column providing cell values.
        aggregation (str or callable, optional):
            Aggregation applied before pivoting. Defaults to "mean".
        column_labels (list, optional):
            Ordered list of labels for output pivot columns. If provided,
            replaces pivot.columns.
        index_labels (list, optional):
            Ordered list of labels for output pivot index.
        hide_index (bool, optional):
            Whether to hide index column when styling. Defaults to True.
        precision (str, optional):
            Formatting string for numeric columns (e.g. '{:.4f}').
        cell_width (str, optional):
            Width for data columns. Defaults to "120px".
        caption (str, optional):
            Caption string to add to the styled table.

    Returns:
        pd.io.formats.style.Styler:
            The styled DataFrame.
    """

    # ---- Build pivot table ----
    pivot = (
        dataframe.groupby([index, columns])[values]
        .agg(aggregation)
        .to_frame()
        .reset_index()
        .pivot(index = index, columns = columns, values = values)
    )

    # Replace columns if labels provided
    if column_labels is not None:
        pivot.columns = column_labels

    # Replace index if labels provided
    if index_labels is not None:
        pivot.index = index_labels

    pivot.index.name = ""

    # ---- Build formatting dict ----
    format_dict = {col: precision for col in pivot.columns}

    # ---- Style using helper ----
    styled = style_dataframe(
        df = pivot.reset_index(),
        hide_index = hide_index,
        format_dict = format_dict,
        cell_width = cell_width,
        alternate_rows = True
    )

    # Add left-column border for separating label column
    styled = styled.set_properties(
        subset = [""],
        **{"border-right": "1px solid black"},
    )

    # Add caption styling if provided
    if caption is not None:
        styled = styled.set_caption(caption).set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("color", "black"),
                        ("font-size", "16px"),
                        ("font-weight", "bold"),
                        ("background-color", "white"),
                    ],
                }
            ],
            overwrite = False,
        )

    return styled
    
def label_test_dataset(test_stats: pd.DataFrame, results: Dict[str, pd.DataFrame],
                       dataset: str, model: str, upper_offset: float = 0.05,
                       lower_offset: float = 0.20, connectionstyles: List[str] = None,
                       result_indices: List[int] = None, x_offsets: List[float] = None,
                       y_offsets: List[float] = None, legend_kwargs: Dict[str, Any] = None) -> plt.Axes:
    """
    Plot PR AUC versus the number of selected test features for a given dataset–model
    combination, and annotate specific test results using customizable offsets and
    connection styles.

    Args:
        test_stats (pd.DataFrame): DataFrame containing test performance statistics with
            columns such as ['dataset', 'model', 'selection_type', 'k', 'pr_auc'].
        results (dict): Dictionary mapping result category names (e.g., 'max_test',
            'strongest_peak', 'earliest_peak') to DataFrames containing test result slices.
        dataset (str): Name of the dataset to filter on.
        model (str): Name of the model to filter on.
        upper_offset (float, optional): Amount added above the maximum PR AUC to set the
            plot's upper y-limit. Defaults to 0.05.
        lower_offset (float, optional): Amount subtracted from the maximum PR AUC to set
            the plot's lower y-limit. Defaults to 0.20.
        connectionstyles (List[str], optional): List of connectionstyle strings for the
            annotation arrows (e.g., ['bar', 'arc3']). If None, defaults are applied.
        result_indices (List[int], optional): Row indices used to locate the annotated
            points inside each DataFrame provided in `results`. If None, defaults are
            applied.
        x_offsets (List[float], optional): Horizontal offsets applied to each annotation
            text position. If None, defaults are used.
        y_offsets (List[float], optional): Vertical offsets applied to each annotation
            text position. If None, defaults are used.
        legend_kwargs (dict, optional): Additional keyword arguments forwarded directly
            to `ax.legend()` to customize the legend appearance.

    Returns:
        matplotlib.axes.Axes: The axes object containing the styled PR AUC plot.
    """

    # Default styling lists
    if connectionstyles is None:
        connectionstyles = ['bar', 'arc3', 'arc3']
    if result_indices is None:
        result_indices = [1, 2, 2]
    if x_offsets is None:
        x_offsets = [-1.0, 1.0, 1.0]
    if y_offsets is None:
        y_offsets = [-0.05, 0.03, -0.05]
    if legend_kwargs is None:
        legend_kwargs = {}

    # Filter relevant rows
    subframe = dataset_model_slice(
        df = test_stats,
        dataset = dataset,
        model = model
    )

    subframe.selection_type = subframe.selection_type.replace(
        {'sum': 'shap', 'max': 'shap'}
    )

    # Consistent hue and label definitions
    hue_order = ['full', 'shap', 'mutual_info', 'relieff', 'mrmr']
    labels = ['Full', 'SHAP', 'Mutual Info', 'ReliefF', 'mRMR']

    # Plot lines and markers
    fig, ax = plt.subplots(figsize = (8, 6))

    sns.lineplot(subframe, x = 'k', y = 'pr_auc', hue = 'selection_type',
                 hue_order = hue_order, ax = ax)
    sns.scatterplot(subframe, x = 'k', y = 'pr_auc', hue = 'selection_type',
                    hue_order = hue_order, ax = ax)

    # Highlight full model point
    full_color = sns.color_palette('tab10')[0]
    sns.scatterplot(
        subframe[subframe.selection_type == 'full'],
        x = 'k',
        y = 'pr_auc',
        color = full_color,
        s = 50,
        ax = ax
    )

    # Mark best observed PR AUC
    max_score = subframe.pr_auc.max()
    min_k, max_k = subframe.k.min(), subframe.k.max()

    ax.axhline(y = max_score, linestyle = '--', color = 'black', label = 'Best Score')
    ax.annotate(
        f'Best PR AUC: {max_score:.4f}',
        xy = (min_k + 0.1, max_score + 0.005),
        fontsize = 10
    )

    # Legend cleanup and reordering
    handles, _ = ax.get_legend_handles_labels()
    num_lines = len(subframe.selection_type.unique())

    ax.legend(
        labels = labels + ['Best Score'],
        handles = [handles[num_lines]] + handles[1:num_lines] + [handles[-1]],
        **legend_kwargs
    )

    # Annotation labels and categories
    annotation_labels = [
        'Max Test Result',
        'Strongest Peak Result',
        'Earliest Peak Result'
    ]
    result_keys = ['max_test', 'strongest_peak', 'earliest_peak']

    # Add annotations with customizable offsets
    for label, key, idx, conn, dx, dy in zip(
        annotation_labels,
        result_keys,
        result_indices,
        connectionstyles,
        x_offsets,
        y_offsets
    ):
        frame = results[key]
        d = dataset_model_slice(
            df = frame,
            dataset = dataset,
            model = model
        ).sort_values('selection_type')
        row = d.iloc[idx]

        ax.annotate(
            label,
            xy = (row['k'], row['pr_auc_test']),
            xytext = (row['k'] + dx, row['pr_auc_test'] + dy),
            ha = 'center',
            size = 10,
            arrowprops = dict(arrowstyle = '->', connectionstyle = conn, lw = 2)
        )

    # Axes formatting
    ax.set_xlabel('K (number of features)')
    ax.set_ylabel('PR AUC')
    ax.set_title(
        f'Test Features vs Test PR AUC | {" ".join(dataset.split("_")[1:]).upper()} | {model.upper()}'
    )

    upper_pr_auc = subframe.pr_auc.max()
    ax.set_xlim([min_k - 1, max_k + 1])
    ax.set_ylim([upper_pr_auc - lower_offset, upper_pr_auc + upper_offset])

    return ax

def compare_features_performance(cv_slice: pd.DataFrame, test_slice: pd.DataFrame,
                                 dataset: str, model: str,
                                 figsize: Optional[Tuple[int, int]] = (16, 6),
                                 save_dir: Optional[str] = None) -> plt.Figure:
    """
    Compare Precision-Recall AUC (PR AUC) versus the number of selected features (K)
    for both cross-validation and test datasets.

    Parameters
    ----------
    cv_slice : pd.DataFrame
        DataFrame containing cross-validation results with the following columns:
        - 'k': int, number of selected features.
        - 'pr_auc': float, Precision-Recall AUC score.
        - 'selection_type': str, feature selection strategy identifier.
    test_slice : pd.DataFrame
        DataFrame containing test results with the same structure as `cv_slice`.
    dataset : str
        Name of the dataset, used in figure titles and output filenames.
    model : str
        Name of the model, used in figure titles and output filenames.
    figsize : Optional[Tuple[int, int]], default = (16, 6)
        Size of the matplotlib figure in inches.
    save_dir : Optional[str], default = None
        Directory path to save the generated figure as a PNG file.
        If provided, the figure is saved with 300 DPI and tight bounding box.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the comparison plots.

    Notes
    -----
    - The plot displays two subplots: one for cross-validation results
      and one for test results, sharing the same y-axis.
    - Each subplot shows PR AUC as a function of K across different
      feature selection strategies.
    - The best PR AUC score is annotated and marked with a horizontal dashed line.
    - If `save_dir` is specified, the figure is automatically saved to disk.
    """
    fig, axes = plt.subplots(1, 2, figsize = figsize, sharey = True)

    # Combine SHAP types together
    cv_df = cv_slice.copy()
    test_df = test_slice.copy()
    cv_df.selection_type = cv_df.selection_type.replace({'sum': 'shap', 'max': 'shap'})
    test_df.selection_type = test_df.selection_type.replace({'sum': 'shap', 'max': 'shap'})

    # Use high-end PR AUC range for compact plots
    upper_pr_auc = max(cv_df.pr_auc.max(), test_df.pr_auc.max())
    lower_pr_auc = min(cv_df.pr_auc.max(), test_df.pr_auc.max())
    min_k, max_k = cv_df.k.min(), cv_df.k.max()
    num_lines = len(cv_df.selection_type.unique())

    # Consistent hue and label ordering
    hue_order = ['full', 'shap', 'mutual_info', 'relieff', 'mrmr']
    labels = ['Full', 'SHAP', 'Mutual Info', 'ReliefF', 'mRMR']

    for i, (ax, slice_) in enumerate(zip(axes, [cv_df, test_df])):
        # Draw points and lines
        sns.lineplot(slice_, x = 'k', y = 'pr_auc', hue = 'selection_type',
                     hue_order = hue_order, ax = ax)
        sns.scatterplot(slice_, x = 'k', y = 'pr_auc', hue = 'selection_type',
                        hue_order = hue_order, ax = ax)

        # Highlight full-feature point
        sns.scatterplot(slice_[slice_.selection_type == 'full'], x = 'k', y = 'pr_auc',
                        color = sns.color_palette('tab10')[0], s = 100, ax = ax)

        # Annotate best score
        max_score = slice_.pr_auc.max()
        ax.axhline(y = max_score, linestyle = '--', color = 'black', label = 'Best Score')
        ax.annotate(f'Best PR AUC: {max_score:.4f}',
                   xy = (min_k + 0.1, max_score + 0.0025), fontsize = 12)

        # Legend configuration
        handles, _ = ax.get_legend_handles_labels()
        if i == 0:
            ax.get_legend().remove()
        else:
            ax.legend(labels = labels + ['Best Score'],
                     handles = [handles[num_lines]] + handles[1:num_lines] + [handles[-1]],
                     bbox_to_anchor = (1.01, 1), loc = 'upper left', fontsize = 12)

        # Axes setup
        ax.set_xlabel('K (number of features)', fontsize = 14)
        ax.set_ylabel('PR AUC', fontsize = 14)
        title = 'CROSS VAL' if i == 0 else 'TEST'
        ax.set_title(f'K v. PR AUC | {" ".join(dataset.split("_")[1:]).upper()} | {model.upper()} | {title}',
                    fontsize = 16)
        ax.tick_params(axis = 'both', labelsize = 12)
        ax.xaxis.set_major_locator(MaxNLocator(integer = True)) # Force X-ticks to be integer values
        ax.set_xlim([min_k - 1, max_k + 1])
        ax.set_ylim([lower_pr_auc - 0.1, upper_pr_auc + 0.01])

    plt.tight_layout()

    # Save figure if path provided
    if save_dir:
        os.makedirs(save_dir, exist_ok = True)
        file_path = os.path.join(save_dir, f'{dataset}_{model}_feature_performance.png')
        fig.savefig(file_path, dpi = 300, bbox_inches = 'tight')
        print(f"[INFO] Figure saved to {file_path}")

    return fig

def compare_feature_importances_for_dataset(
    cv_stats: pd.DataFrame, test_stats: pd.DataFrame,
    dataset: str, model_types: List[str],
    save_dir: Optional[str] = None
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[Any]]:
    """
    Generate and compare feature importance performance plots across multiple models
    for a given dataset using cross-validation and test statistics.

    Parameters
    ----------
    cv_stats : pd.DataFrame
        DataFrame containing cross-validation performance metrics for multiple models
        and datasets. Must include columns for dataset, model, and feature selection results.
    test_stats : pd.DataFrame
        DataFrame containing test performance metrics, structured identically to `cv_stats`.
    dataset : str
        The dataset identifier used to filter relevant rows from both statistics DataFrames.
    model_types : List[str]
        List of model names to include in the comparison. Each model will generate a
        separate PR AUC vs. feature count plot.
    save_dir : Optional[str], default = None
        Directory path to save the generated plots as PNG files.
        If not provided, figures are not saved.

    Returns
    -------
    Tuple[List[pd.DataFrame], List[pd.DataFrame], List[Any]]
        A tuple containing three lists:
        - cv_slices : List[pd.DataFrame]
            Filtered cross-validation subsets for each model.
        - test_slices : List[pd.DataFrame]
            Filtered test subsets for each model.
        - feature_performances : List[Any]
            The matplotlib Figure objects produced for each model comparison.
    """
    # Initialize storage lists for results
    cv_slices = []
    test_slices = []
    feature_performances = []

    # Iterate through each model type and generate comparison plots
    for model in model_types:
        # Extract relevant subset for the given dataset and model
        cv_slice = dataset_model_slice(df = cv_stats, dataset = dataset, model = model)
        test_slice = dataset_model_slice(df = test_stats, dataset = dataset, model = model)

        # Generate PR AUC vs feature count plot and optionally save it
        feature_performance = compare_features_performance(
            cv_slice = cv_slice,
            test_slice = test_slice,
            dataset = dataset,
            model = model,
            save_dir = save_dir
        )

        # Collect results for output
        cv_slices.append(cv_slice)
        test_slices.append(test_slice)
        feature_performances.append(feature_performance)

    return cv_slices, test_slices, feature_performances

def add_annotation_if_new(ax: Axes, text: str,
                          xy_position: Tuple[float, float],
                          annotations: List[Tuple[str, Tuple[float, float], object]]) -> bool:
    """
    Add a matplotlib annotation only if an annotation at the same coordinates
    has not already been recorded.

    Args:
        ax (matplotlib.axes.Axes): The axis on which the annotation will be drawn.
        text (str): The label text for the annotation.
        xy_position (Tuple[float, float]): The (x, y) coordinates for placing the annotation.
        annotations (List[Tuple[str, Tuple[float, float], object]]):
            A registry of previously added annotations, stored as
            (text, position, annotation_object) tuples.

    Returns:
        bool: True if a new annotation was added; False if the position already exists.
    """

    # Check whether an annotation at this exact coordinate has already been added.
    for existing_text, existing_pos, existing_obj in annotations:
        if existing_pos == xy_position:
            return False  # Annotation already recorded at this location.

    # Create a new annotation on the axis.
    ann_obj = ax.annotate(text, xy = xy_position, xytext = xy_position)

    # Store the annotation information so duplicates are prevented later.
    annotations.append((text, xy_position, ann_obj))

    return True

def annotated_max_shap(dataset: str, model_type: str, model_name: str,
                       shap_dir: str, results: Dict[str, pd.DataFrame],
                       stats_frame: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Generate a plot showing SHAP-based max-threshold feature selection behavior
    for a single dataset and model, including annotated benchmark points from
    different best-k strategies.

    This visualization helps compare:
        * Feature counts selected across SHAP max-thresholds
        * Best-k results from lowest overfitting, max test score,
          strongest peak, and earliest peak strategies

    Args:
        dataset (str): The dataset identifier (e.g., 'kaggle_credit_card_fraud').
        model_type (str): The model type key used in results and SHAP files
            (e.g., 'rf', 'logreg').
        model_name (str): Human-readable model name for plot labeling
            (e.g., 'Random Forest').
        shap_dir (str): Directory containing saved SHAP threshold results
            for the dataset and model.
        results (Dict[str, pd.DataFrame]): A dictionary of best-k result DataFrames.
        stats_frame (pd.DataFrame): Full statistics frame used to extract
            SHAP threshold performance and feature counts.
        ax (matplotlib.axes.Axes, optional): Axis to draw the plot onto.
            If None, a new figure and axis are created.

    Returns:
        matplotlib.axes.Axes: The axis containing the completed SHAP threshold plot.
    """

    # Load SHAP values for the dataset/model
    dataset_shap_path = os.path.join(shap_dir, f'{dataset.replace("_", "-")}')
    shap_file = [p for p in sorted(os.listdir(dataset_shap_path)) if model_type in p][0]
    shap_values = load_object(os.path.join(dataset_shap_path, shap_file))['mean_shap']

    # Compute SHAP-max selected feature counts for thresholds from 0.005 to 1.0
    thresholds = np.arange(0.005, 1, 0.005)
    max_values = [
        len(shap_select(shap_values = shap_values, kind = 'max', max_threshold = round(t, 3))) - 1
        for t in thresholds
    ]

    # Create axis if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize = (8, 6))

    colors = sns.color_palette('tab10')[:5]
    labels = [model_name, 'Lowest Overfitting', 'Max Test', 'Strongest Peak', 'Earliest Peak']

    # Plot SHAP-max curve
    sns.scatterplot(
        x = thresholds,
        y = max_values,
        color = colors[0],
        alpha = 0.75,
        ax = ax,
        label = labels[0]
    )

    # Annotation registry
    annotations = []

    # Loop through best-k result types
    for i, (metric, frame) in enumerate(results.items()):
        # Filter to shap-only entries
        mask = frame[
            (frame['dataset'] == dataset) &
            (frame['selection_type'] == 'shap')
        ].index.tolist()

        sub = stats_frame.loc[mask].copy()
        if sub.empty:
            continue

        # Extract SHAP threshold from selection_type naming
        sub['strategy'] = sub['selection_type'].apply(lambda x: x.split('_')[0])
        sub['threshold'] = sub['selection_type'].apply(lambda x: float(x.split('_')[1]))

        sub = sub[
            (sub['strategy'] == 'max') & 
            (sub['model'] == model_type)
        ]

        if sub.empty:
            continue

        # Plot benchmark point
        sns.scatterplot(
            data = sub,
            x = 'threshold',
            y = 'k',
            s = 100,
            color = colors[i + 1],
            label = labels[i + 1],
            alpha = 0.75,
            ax = ax 
        )

        # Create annotation label
        text = f'({sub["threshold"].max():.3f}, {sub["k"].max():.0f})'
        xy_pos = (sub["threshold"].max() + 0.01, sub["k"].max() + 1)

        add_annotation_if_new(ax, text, xy_pos, annotations)

    # Axis labels and title
    ax.set_xlabel('Max Threshold (Proportion of Strongest Feature)')
    ax.set_ylabel('K (Number of Features Kept)')
    ax.set_title(
        f'Max SHAP Strategy for {" ".join(dataset.split("_")[1:]).title()} Dataset and {model_name}'
    )

    return ax

def annotated_sum_shap(dataset: str, model_type: str, model_name: str,
                       shap_dir: str, results: Dict[str, pd.DataFrame],
                       stats_frame: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot SHAP-based feature counts derived from the sum strategy,
    along with annotated points representing selected thresholds
    found in a cross-validation results frame.

    Args:
        dataset (str): Name of the dataset to analyze.
        model_type (str): The model type (e.g., 'rf' or 'xgb').
        model_name (str): Readable model name for axis titles and labels.
        shap_dir (str): Base directory containing SHAP values per dataset.
        results (Dict[str, pd.DataFrame]): Mapping of metric name → results DataFrame.
        stats_frame (pd.DataFrame): Cross-validation statistics used for annotation.
        ax (matplotlib.axes.Axes, optional): Optional axis for plotting. If None,
            a new axis is created.

    Returns:
        matplotlib.axes.Axes: The axis containing the rendered plot.
    """

    # Resolve SHAP directory for this dataset
    dataset_shap_path = os.path.join(shap_dir, dataset.replace("_", "-"))
    shap_file = [p for p in sorted(os.listdir(dataset_shap_path)) if model_type in p][0]

    shap_values = load_object(
        os.path.join(dataset_shap_path, shap_file)
    )["mean_shap"]

    # Compute number of selected features for sum-based SHAP strategy
    thresholds = np.arange(0.005, 1, 0.005)
    sum_values = [
        len(shap_select(shap_values = shap_values, kind = 'sum', sum_threshold = round(t, 3))) - 1
        for t in thresholds
    ]

    # Create axis if needed
    if ax is None:
        fig, ax = plt.subplots(figsize = (8, 6))

    colors = sns.color_palette('tab10')[:5]
    labels = [model_name, 'Lowest Overfitting', 'Max Test', 'Strongest Peak', 'Earliest Peak']

    # Plot the SHAP-sum K curve
    sns.scatterplot(
        x = thresholds,
        y = sum_values,
        color = colors[0],
        alpha = 0.75,
        ax = ax,
        label = labels[0]
    )

    # Annotation registry
    annotations = []

    # For each metric, add scatter points and annotations
    for i, (metric, frame) in enumerate(results.items()):
        mask = frame[
            (frame["dataset"] == dataset) &
            (frame["selection_type"] == "shap")
        ].index.tolist()

        masked_stats = stats_frame.loc[mask].copy()
        masked_stats["strategy"] = masked_stats["selection_type"].apply(
            lambda x: x.split("_")[0]
        )
        masked_stats["threshold"] = masked_stats["selection_type"].apply(
            lambda x: float(x.split("_")[1])
        )

        # Restrict to sum strategy and the provided model_type
        subframe = masked_stats[
            (masked_stats["strategy"] == "sum") &
            (masked_stats["model"] == model_type)
        ]

        # Plot the annotation points for cross-validated thresholds
        sns.scatterplot(
            data = subframe,
            x = "threshold",
            y = "k",
            s = 100,
            color = colors[i + 1],
            alpha = 0.75,
            ax = ax,
            label = labels[i + 1]
        )

        # Add annotation if unique
        add_annotation_if_new(
            ax = ax,
            text = f"({subframe['threshold'].max():.3f}, {subframe['k'].max():.0f})",
            xy_position = (
                subframe["threshold"].max() - 0.15,
                subframe["k"].max() + 1
            ),
            annotations = annotations
        )

    ax.set_xlabel("Sum Threshold (Proportion of Total SHAP Importance)")
    ax.set_ylabel("K (Number of Features Kept)")
    ax.set_title(
        f"Sum SHAP Strategy for {' '.join(dataset.split('_')[1:]).title()} "
        f"Dataset and {model_name}"
    )

    return ax