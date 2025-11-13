import os
from typing import Dict, Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def select_lowest_k(dataframe: pd.DataFrame, groupby_col: str,
                    transform: Callable, include_selection_type: bool = True) -> pd.DataFrame:
    """
    Select rows with the lowest 'k' value per group after applying a transformation
    to a grouping column, optionally including 'selection_type' in the grouping.

    This function filters the input DataFrame to exclude entries where
    'selection_type' == 'full' (if present), applies a specified transformation
    (e.g., 'min', 'max', 'median') to the specified grouping column, and returns
    only those rows that both match the transformed group value and have the
    smallest 'k' for that group.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing at least the columns:
            ['dataset', 'model', groupby_col, 'k'], and optionally 'selection_type'.
        groupby_col (str): The column name used for computing the transformation.
        transform (Callable): A transformation function applied via `groupby().transform()`,
            such as `min`, `max`, `median`, or a custom callable returning a scalar.
        include_selection_type (bool, optional): Whether to include 'selection_type' as part
            of the grouping key. Defaults to True.

    Returns:
        pd.DataFrame: A filtered DataFrame containing one row per unique combination of
        grouping keys (depending on `include_selection_type`) and 'groupby_col' with the
        smallest 'k' value, after applying the specified transformation.
    """
    # Exclude rows where the selection type is 'full' (if column exists)
    df = dataframe.copy()
    if 'selection_type' in df.columns:
        df = df[df.selection_type != 'full']

    # Define grouping columns based on user preference
    group_cols = ['dataset', 'model']
    if include_selection_type and 'selection_type' in df.columns:
        group_cols.append('selection_type')

    # Identify rows where the groupby_col value equals the transformed value within each group
    groupby_mask = (
        df.groupby(group_cols)[groupby_col].transform(transform) == df[groupby_col]
    )

    # Keep only rows matching the transformed group value
    masked_data = df.loc[groupby_mask]

    # Within each subgroup, select rows with the minimum 'k' value
    k_mask = (
        masked_data.groupby(group_cols + [groupby_col])['k']
        .transform('min') == masked_data['k']
    )

    # Drop duplicates to ensure one representative per group and return the filtered DataFrame
    return masked_data.loc[k_mask].drop_duplicates()
    
def extract_best_k(cv_stats: pd.DataFrame, test_stats: pd.DataFrame,
                   metrics_dir: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """
    Extract optimal 'k' configurations from cross-validation and test statistics.

    This function merges cross-validation and test performance metrics, normalizes
    selection type names, computes PR-AUC differences (to assess overfitting), and
    identifies multiple variants of "best k" values:
      - Lowest overfitting (min PR-AUC difference)
      - Highest test performance (max PR-AUC)
      - Strongest local peak in test performance curves
      - Earliest local peak

    Optionally, results are saved as CSV files in a specified directory.

    Args:
        cv_stats (pd.DataFrame): DataFrame containing cross-validation metrics,
            with columns ['dataset', 'model', 'selection_type', 'k', 'pr_auc'].
        test_stats (pd.DataFrame): DataFrame containing test-set metrics,
            with matching identifiers and 'pr_auc' values.
        metrics_dir (str, optional): Directory path to save the output CSVs.
            If None, results are returned but not saved.

    Returns:
        dict[str, pd.DataFrame]: A dictionary with the following keys:
            - 'lowest_overfit': Rows with lowest PR-AUC difference.
            - 'max_test': Rows with highest test PR-AUC.
            - 'strongest_peak': Rows representing the strongest local test PR-AUC peaks.
            - 'earliest_peak': Rows representing the earliest local test PR-AUC peaks.
    """

    # Merge cross-validation and test results by shared identifiers
    cv_test = cv_stats.merge(
        test_stats,
        on = ['dataset', 'model', 'selection_type', 'k'],
        suffixes = ('_cv', '_test'),
        how = 'inner'
    )[
        ['dataset', 'model', 'selection_type', 'k', 'pr_auc_cv', 'pr_auc_test']
    ]

    # Simplify selection type names (e.g., "mutual_info_50" -> "mutual_info")
    cv_test.loc[:, "selection_type"] = cv_test["selection_type"].apply(
        lambda x: "_".join(x.split("_")[:-1]) if "_" in x else x
    )

    # Normalize naming for SHAP-based selections
    cv_test.selection_type = cv_test.selection_type.replace({'sum': 'shap', 'max': 'shap'})

    # Compute PR-AUC difference between CV and test (smaller = less overfitting)
    cv_test['pr_auc_diff'] = cv_test['pr_auc_cv'] - cv_test['pr_auc_test']

    # Compute percentage change of PR-AUC between CV and test
    cv_test['pr_auc_diff_percent'] = cv_test['pr_auc_diff'] / cv_test['pr_auc_test']
    
    # Compute percentage of features kept
    cv_test['k_percent'] = cv_test.apply(
        lambda row: 100 * row['k'] / cv_test[
            (cv_test['selection_type'] == 'full') &
            (cv_test['dataset'] == row['dataset'])
        ]['k'].max(),
        axis = 1
    )

    # Compute percentage of test performance compared to full feature set
    cv_test['pr_auc_test_percent'] = cv_test.apply(
        lambda row: 100 * row['pr_auc_test'] / cv_test[
            (cv_test['selection_type'] == 'full') &
            (cv_test['dataset'] == row['dataset']) &
            (cv_test['model'] == row['model'])
        ]['pr_auc_test'].max(),
        axis = 1
    )

    # Identify configurations with lowest overfitting
    lowest_overfit = select_lowest_k(cv_test, groupby_col = 'pr_auc_diff', transform = 'min')

    # Identify configurations with highest test performance
    max_test = select_lowest_k(cv_test, groupby_col = 'pr_auc_test', transform = 'max')

    # Initialize a DataFrame to collect local performance peaks
    peak_slice = pd.DataFrame()

    # Iterate across datasets, selection types, and models to find local PR-AUC peaks
    for dataset in cv_test.dataset.unique():
        idx = []

        for s_type in cv_test.selection_type.unique():
            if s_type == 'full':
                continue

            for m_type in cv_test.model.unique():
                # Subset for one dataset / model / selection_type
                subframe = cv_test[
                    (cv_test.dataset == dataset)
                    & (cv_test.selection_type == s_type)
                    & (cv_test.model == m_type)
                ].copy()

                if subframe.empty:
                    continue

                # Calculate 25th percentile threshold for PR-AUC
                pr_auc_threshold = subframe.pr_auc_test.quantile(0.25)

                # Shift test performance to identify local peaks
                subframe['prev'] = subframe.pr_auc_test.shift(1)
                subframe['next'] = subframe.pr_auc_test.shift(-1)

                # Find indices of local peaks above the 25th percentile
                candidate_idx = subframe[
                    (subframe.pr_auc_test > subframe.prev)
                    & (subframe.pr_auc_test > subframe.next)
                    & (subframe.pr_auc_test >= pr_auc_threshold)
                ].index.tolist()

                # Always include the smallest k if its performance is competitive
                if subframe[subframe.k == subframe.k.min()].pr_auc_test.iloc[0] > any(
                    subframe.loc[candidate_idx, :].pr_auc_test.values
                ):
                    candidate_idx.append(subframe[subframe.k == subframe.k.min()].index[0])

                # If no peaks found, fallback to the global max PR-AUC
                if not candidate_idx:
                    candidate_idx.extend(
                        subframe[subframe.pr_auc_test == subframe.pr_auc_test.max()].index.tolist()
                    )

                idx.append(candidate_idx)

        # Append all identified peaks for the current dataset
        if idx:
            peak_slice = pd.concat([peak_slice, cv_test.loc[sorted(sum(idx, [])), :]])

    # Among peaks, select those with the strongest test performance
    strongest_peak = select_lowest_k(peak_slice, groupby_col = 'pr_auc_test', transform = 'max')

    # Among peaks, select the earliest (lowest k)
    earliest_peak = select_lowest_k(peak_slice, groupby_col = 'k', transform = 'min')

    results = {
        'lowest_overfit': lowest_overfit,
        'max_test': max_test,
        'strongest_peak': strongest_peak,
        'earliest_peak': earliest_peak,
    }

    # Optionally save each results subset as CSV
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok = True)
        for name, df in results.items():
            save_path = os.path.join(metrics_dir, f"{name}_best_k.csv")
            df.to_csv(save_path, index = False)

    return results

def calculate_performance_counts(results: Dict[str, pd.DataFrame],
                                 metrics_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Compute and summarize performance counts and average k percentages
    across multiple result DataFrames, optionally saving the results to CSV.

    This function iterates over a dictionary of result DataFrames, applying
    custom aggregation rules depending on the dataset index:
        - For the first dataset, selects entries with the minimum `pr_auc_diff`.
        - For subsequent datasets, selects entries with the maximum `pr_auc_test`.

    The function then counts how many times each feature selection method
    appears and computes the mean `k_percent` for each method. The results
    are concatenated into a single summary table.

    Args:
        results (Dict[str, pd.DataFrame]): 
            Dictionary mapping dataset names (keys) to result DataFrames (values).
        metrics_dir (Optional[str], optional): 
            Directory path where the resulting summary table should be saved as
            `performance_counts.csv`. If None, the file is not saved. Defaults to None.

    Returns:
        pd.DataFrame: 
            A DataFrame summarizing counts and mean k percentages per selection type.
    """
    performance_counts = pd.DataFrame()

    for i, (name, df) in enumerate(results.items()):
        # Define selection criterion depending on dataset order
        if i == 0:
            groupby_col = 'pr_auc_diff'
            transform = 'min'
        else:
            groupby_col = 'pr_auc_test'
            transform = 'max'

        # Select the best-performing subset per criterion
        subframe = select_lowest_k(
            df,
            groupby_col = groupby_col,
            transform = transform,
            include_selection_type = False
        ).drop_duplicates(subset = groupby_col, keep = False)

        # Count occurrences of each selection method
        df_counts = subframe['selection_type'].value_counts().to_frame(
            name = f'{name.replace("_", " ").title()} Count'
        )

        # Compute mean k_percent per selection type
        df_k = np.round(
            subframe.groupby('selection_type')['k_percent'].mean().to_frame(
                name = f'{name.replace("_", " ").title()} Mean K %'
            ),
            2
        )

        # Merge into master performance table
        performance_counts = pd.concat([performance_counts, df_counts, df_k], axis = 1)

    # Clean and format final table
    performance_counts.index.name = 'Selection Type'
    performance_counts = performance_counts.reset_index()
    performance_counts['Selection Type'] = performance_counts['Selection Type'].replace({
        'relieff': 'ReliefF',
        'shap': 'SHAP',
        'mutual_info': 'Mutual Information',
        'mrmr': 'mRMR'
    })

    # Sort and set index for readability
    if 'Max Test Count' in performance_counts.columns:
        performance_counts = performance_counts.sort_values(['Max Test Count'], ascending = False)
    performance_counts = performance_counts.set_index('Selection Type')

    # Optionally save to CSV
    if metrics_dir is not None:
        os.makedirs(metrics_dir, exist_ok = True)
        save_path = os.path.join(metrics_dir, "performance_counts.csv")
        performance_counts.to_csv(save_path, index = True)

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

def style_dataframe(df: pd.DataFrame, hide_index: bool = True,
                    format_dict: Optional[dict] = None, cell_width: str = '90px',
                    index_width: str = '150px') -> Styler:
    """
    Apply consistent styling to a DataFrame for display in Jupyter notebooks.

    Args:
        df (pd.DataFrame): The DataFrame to style.
        hide_index (bool, optional): Whether to hide the index column. Defaults to True.
        format_dict (dict, optional): Dictionary mapping column names to formatting strings
            (e.g., {'col1': '{:.2f}'}). If None, numeric columns are automatically formatted.
        cell_width (str, optional): Width of data cells. Defaults to '90px'.
        index_width (str, optional): Width of index column. Defaults to '150px'.

    Returns:
        pd.io.formats.style.Styler: Styled DataFrame object.
    """
    styled = df.style.set_table_styles(
        [
            # Index column (row header)
            {'selector': 'th.row_heading', 'props': [
                ('text-align', 'center'),
                ('width', index_width),
                ('border-top', '1px solid black'),
                ('border-right', '1px solid black'),
                ('background-color', 'white'),
                ('color', 'black')
            ]},
            # Column headers
            {'selector': 'th', 'props': [
                ('text-align', 'center'),
                ('color', 'black'),
                ('border-bottom', '1px solid black'),
                ('border-top', '1px solid black'),
                ('border-right', '1px solid black'),
                ('background-color', 'white'),
                ('font-size', '12px'),
                ('font-weight', 'bold')
            ]},
            # Data cells
            {'selector': 'td', 'props': [
                ('text-align', 'center'),
                ('color', 'black'),
                ('border-bottom', '1px solid black'),
                ('border-right', '1px solid black'),
                ('background-color', 'white'),
                ('width', cell_width),
                ('font-size', '12px'),
                ('font-weight', 'bold')
            ]}
        ]
    )
    if format_dict:
        styled = styled.format(format_dict)

    if hide_index:
        styled = styled.hide(axis = 'index')

    styled = styled.set_properties(**{'width': cell_width})

    return styled

# ---- Other (To be organized) ----

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