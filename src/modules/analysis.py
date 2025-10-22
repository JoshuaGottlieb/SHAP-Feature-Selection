import os
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score
)

from modules.utils import load_object

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
    cv_stats: pd.DataFrame,
    test_stats: pd.DataFrame,
    dataset: str,
    model_types: List[str],
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

def extract_indices(df: pd.DataFrame, k_values: List[int]) -> List[int]:
    """
    Extract row indices from a DataFrame corresponding to specific combinations
    of feature selection methods and feature counts (K values).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the columns:
        - 'k' : int, number of selected features.
        - 'selection_type' : str, name of the feature selection method
          (e.g., 'max', 'sum', 'relieff', 'mrmr', 'mutual_info').
    k_values : List[int]
        A list of exactly five integers representing the desired 'k' values
        for each of the following selection methods, in order:
        1. 'max'
        2. 'sum'
        3. 'relieff'
        4. 'mrmr'
        5. 'mutual_info'

    Returns
    -------
    List[int]
        A list of integer indices corresponding to the rows in `df`
        that match the specified (k, selection_type) combinations.
    """
    # Select rows that match the specific (k, selection_type) pairs
    # Then extract their integer indices as a list
    idx = df.loc[
        ((df.k == k_values[0]) & (df.selection_type == 'max')) |
        ((df.k == k_values[1]) & (df.selection_type == 'sum')) |
        ((df.k == k_values[2]) & (df.selection_type == 'relieff')) |
        ((df.k == k_values[3]) & (df.selection_type == 'mrmr')) |
        ((df.k == k_values[4]) & (df.selection_type == 'mutual_info'))
    ].index.tolist()

    # Return the list of matched row indices
    return idx