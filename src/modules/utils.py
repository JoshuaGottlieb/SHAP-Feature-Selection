import bz2
import gzip
import lzma
import os
import pickle
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def save_object(obj: Any, path: str, compression: Optional[str] = None) -> None:
    """
    Serialize and save a Python object (e.g., model, transformer, dictionary) to disk,
    with optional compression.

    This function ensures that the destination directory exists before writing,
    supports multiple compression formats, and appends an appropriate file extension
    based on the compression type.

    Supported compression formats:
        - None: Uncompressed `.pickle`
        - 'gzip': GZIP-compressed `.pickle.gz`
        - 'bz2': BZ2-compressed `.pickle.bz2`
        - 'lzma': LZMA/XZ-compressed `.pickle.xz`

    Args:
        obj (Any):
            The Python object to serialize and save.
        path (str):
            Destination file path (without extension).
            Example: `"models/random_forest_model"`
        compression (str, optional):
            Compression type to use ('gzip', 'bz2', 'lzma', or None).
            Defaults to None (uncompressed).
    """
    # Ensure the output directory exists
    root = os.path.dirname(path)
    if root and not os.path.exists(root):
        os.makedirs(root)

    # Handle supported compression formats
    if compression in ["gzip", "bz2", "lzma"]:
        if compression == "gzip":
            ext = ".pickle.gz"
            with gzip.open(path + ext, "wb") as f:
                pickle.dump(obj, f)
        elif compression == "bz2":
            ext = ".pickle.bz2"
            with bz2.BZ2File(path + ext, "wb") as f:
                pickle.dump(obj, f)
        elif compression == "lzma":
            ext = ".pickle.xz"
            with lzma.open(path + ext, "wb") as f:
                pickle.dump(obj, f)

    else:
        # Save as an uncompressed pickle file
        if compression is not None:
            print("Warning: Unknown compression type. Defaulting to uncompressed pickle format.")
        ext = ".pickle"
        with open(path + ext, "wb") as f:
            pickle.dump(obj, f)

    # Print confirmation
    print(f"Successfully saved object to {path + ext}")

    return

def load_object(path: str) -> Any:
    """
    Load and deserialize a Python object (e.g., model, transformer, dictionary)
    from disk, automatically handling compressed pickle formats.

    This function supports the same compression extensions as `save_object()`:
        - `.pickle`: Uncompressed
        - `.pickle.gz`: GZIP-compressed
        - `.pickle.bz2`: BZ2-compressed
        - `.pickle.xz`: LZMA/XZ-compressed

    Args:
        path (str):
            Full path to the serialized object file, including its extension.
            Example: `"models/random_forest_model.pickle.gz"`

    Returns:
        Any:
            The deserialized Python object.
    """
    # Determine compression type based on file extension
    if path.endswith(".pickle.gz"):
        compression = "gzip"
    elif path.endswith(".pickle.bz2"):
        compression = "bz2"
    elif path.endswith(".pickle.xz"):
        compression = "lzma"
    else:
        compression = None

    # Load object using appropriate method
    if compression == "gzip":
        with gzip.open(path, "rb") as f:
            obj = pickle.load(f)
    elif compression == "bz2":
        with bz2.BZ2File(path, "rb") as f:
            obj = pickle.load(f)
    elif compression == "lzma":
        with lzma.open(path, "rb") as f:
            obj = pickle.load(f)
    else:
        with open(path, "rb") as f:
            obj = pickle.load(f)

    # Return the loaded Python object
    print(f"Successfully loaded object from {path}")
    
    return obj

def load_dataset(path: str, compression: str = 'gzip') -> pd.DataFrame:
    """
    Load a dataset from a CSV file into a pandas DataFrame.

    Args:
        path (str): The file path to the dataset.
        compression (str, optional): Compression type used on the CSV file 
            (e.g., 'gzip', 'bz2', 'zip', or None). Defaults to 'gzip'.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded dataset.
    """
    # Read CSV file into a DataFrame with optional compression
    df = pd.read_csv(path, compression = compression)

    return df

def load_all_datasets(dataset_names: List[str], model_types: List[str],
                      fs_types: List[str], full_dir: str, reduced_dir: str,
                      other_dir: str) -> List[Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Load all training and testing datasets (full, SHAP-reduced, and feature-selected)
    across multiple datasets and model configurations.

    Args:
        dataset_names (List[str]): List of dataset identifiers.
        model_types (List[str]): List of model types (e.g., ['rf', 'xgb']).
        fs_types (List[str]): List of feature selection method identifiers.
        full_dir (str): Directory containing full (unreduced) datasets.
        reduced_dir (str): Directory containing SHAP-based reduced datasets.
        other_dir (str): Directory containing other feature-selected datasets.

    Returns:
        Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
            Nested dictionary structured as:
            {
                'train': {dataset_name: {variant_name: DataFrame, ...}, ...},
                'test': {dataset_name: {variant_name: DataFrame, ...}, ...}
            }

    Notes:
        - Missing files or directories are skipped gracefully.
        - Each dataset split ('train' / 'test') can contain:
            - A 'full' dataset
            - Multiple SHAP-based reduced datasets (e.g., 'rf-sum_0.8')
            - Multiple other feature-selected datasets (e.g., 'fcbf', 'mrmr')
        - Informative log messages are printed for traceability.
    """
    # Initialize nested dictionaries to hold training and testing sets
    train_sets: Dict[str, Dict[str, pd.DataFrame]] = {}
    test_sets: Dict[str, Dict[str, pd.DataFrame]] = {}

    # Loop through each dataset in the list
    for dataset in dataset_names:
        # Normalize dataset name for consistent file matching
        set_name_snake = dataset.replace('-', '_')
        print(f"[INFO] Loading datasets for: {dataset}")

        # Prepare nested dicts for this dataset
        train_sets[set_name_snake] = {}
        test_sets[set_name_snake] = {}

        # Iterate through 'train' and 'test' splits
        for split, set_dict in list(zip(['train', 'test'], [train_sets[set_name_snake], test_sets[set_name_snake]])):
            print(f"[INFO] Loading {split} datasets...")

            # Load full (unreduced) dataset
            full_path = os.path.join(full_dir, split, f'{set_name_snake}-{split}.csv.gz')
            if os.path.exists(full_path):
                try:
                    set_dict['full'] = load_dataset(full_path)
                    print(f"[INFO] Loaded full {split} dataset from: {full_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to load full {split} dataset ({full_path}): {e}")
            else:
                print(f"[WARNING] Full {split} dataset not found at: {full_path}")

            # Load SHAP-reduced datasets
            shap_split_dir = os.path.join(reduced_dir, split)
            if not os.path.exists(shap_split_dir):
                print(f"[WARNING] Reduced directory missing for {split}: {shap_split_dir}")
                continue

            # Get SHAP-reduced datasets for current original dataset
            shap_paths = [path for path in os.listdir(shap_split_dir) if set_name_snake in path]

            for model_type in model_types:
                # Select paths for model type
                model_paths = [path for path in shap_paths if model_type in path]

                for subset in model_paths:
                    # Extract sampling type from string
                    sampling_type = subset.split('.csv.gz')[0].split(f'{model_type}-')[-1].replace('-', '_')
                    model_sample = f'{model_type}-{sampling_type}'
                    file_path = os.path.join(shap_split_dir, subset)
                    
                    # Load SHAP-reduced dataset
                    try:
                        set_dict[model_sample] = load_dataset(file_path)
                        print(f"[INFO] Loaded reduced {split} dataset ({model_sample}) from: {file_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to load reduced {split} dataset ({file_path}): {e}")

            # Load other feature-selected datasets
            other_split_dir = os.path.join(other_dir, split)
            if not os.path.exists(other_split_dir):
                print(f"[WARNING] Other directory missing for {split}: {other_split_dir}")
                continue

            # Get feature-selected datasets for current original dataset
            other_paths = [path for path in os.listdir(other_split_dir) if set_name_snake in path]

            for fs_type in fs_types:
                # Select paths for feature-selection type
                fs_paths = [path for path in other_paths if fs_type in path]

                for subset in fs_paths:
                    # Extract sampling type from string
                    sampling_type = subset.split('.csv.gz')[0].split(f'{split}-')[-1].replace('-', '_')
                    file_path = os.path.join(other_split_dir, subset)

                    # Load feature-reduced dataset
                    try:
                        set_dict[sampling_type] = load_dataset(file_path)
                        print(f"[INFO] Loaded other {split} dataset ({sampling_type}) from: {file_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to load other {split} dataset ({file_path}): {e}")

    print("[DONE] Completed loading all datasets.\n")
    
    return train_sets, test_sets

def load_all_models(dataset_names: List[str], model_types: List[str],
                    models_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Load all trained models for each dataset and model type.

    Args:
        dataset_names (List[str]): List of dataset identifiers.
        model_types (List[str]): List of model types.
        models_dir (str): Root directory containing subdirectories for each dataset.

    Returns:
        Dict[str, Dict[str, Dict[str, Any]]]: Nested dictionary of models:
            {dataset_name: {model_type: {sampling_type: model_object}}}
    """
    # Initialize dictionary to hold models
    models = {}

    # Loop through each dataset in the list
    for dataset in dataset_names:
        # Normalize dataset name for consistent file matching
        set_name_snake = dataset.replace('-', '_')
        print(f"[INFO] Loading models for dataset: {dataset}")

        # Initialize nested dict for this dataset
        models[set_name_snake] = {}
        set_dict = models[set_name_snake]

        # Initialize dicts for each model type
        for model_type in model_types:
            set_dict[model_type] = {}

        # Check for existence of model directory
        dataset_dir = os.path.join(models_dir, dataset)
        if not os.path.exists(dataset_dir):
            print(f"[WARNING] Model directory not found: {dataset_dir}")
            continue

        try:
            model_paths = os.listdir(dataset_dir)
        except Exception as e:
            print(f"[ERROR] Failed to list model directory {dataset_dir}: {e}")
            continue

        # Loop through all model files in the dataset directory
        for path in model_paths:
            if not path.endswith('.pickle.xz'):
                continue

            try:
                # Extract model type and sampling/selection variant from filename
                m_type = path.split(f'{set_name_snake}-')[-1].split('-')[0]
                sampling_type = (
                    path.split(f'{set_name_snake}-{m_type}-')[-1]
                    .split('.pickle.xz')[0]
                    .replace('-', '_')
                )

                # Load the model object
                file_path = os.path.join(models_dir, dataset, path)
                set_dict[m_type][sampling_type] = load_object(file_path)
                print(f"[INFO] Loaded model: {dataset} | {m_type} | {sampling_type}")

            except Exception as e:
                print(f"[ERROR] Failed to load model file ({path}): {e}")
                continue

    print("[DONE] Completed loading all models.\n")
    
    return models

def load_time_data(metrics_dir: str, save: bool = False) -> pd.DataFrame:
    """
    Load and compute timing statistics from 'crossval_statistics.csv' using fully vectorized operations.

    Steps performed:
        - Load and clean selection type names.
        - Collapse SHAP variants into a single category.
        - Compute aggregated max time per (dataset, model, selection_type).
        - Compute:
            • avg_time: mean time per selection_type
            • model_avg_time: mean time per model
        - Optionally save processed time statistics to CSV.

    Args:
        metrics_dir (str): Directory containing 'crossval_statistics.csv'.
        save (bool, optional): If True, saves the resulting dataframe as
            '<metrics_dir>/time_statistics.csv'. Defaults to False.

    Returns:
        pd.DataFrame: A processed dataframe containing timing statistics.
    """

    filepath = os.path.join(metrics_dir, "crossval_statistics.csv")
    cv_stats = pd.read_csv(filepath)

    # Simplify selection type: drop trailing suffix (e.g., mutual_info_50 → mutual_info)
    cv_stats["selection_type"] = cv_stats["selection_type"].str.replace(
        r"_[^_]+$", "", regex = True
    )

    # Normalize SHAP type names
    cv_stats["selection_type"] = cv_stats["selection_type"].replace(
        {"sum": "shap", "max": "shap"}
    )

    # Aggregate max time per group
    times = (
        cv_stats.groupby(["dataset", "model", "selection_type"], as_index = False)["time"]
        .max()
    )

    # Remove the 'full' category
    times = times[times["selection_type"] != "full"]

    # Compute average time per selection_type (vectorized)
    times["avg_time"] = (
        times.groupby("selection_type")["time"]
        .transform("mean")
    )

    # Compute average time per model (vectorized)
    times["model_avg_time"] = (
        times.groupby("model")["time"]
        .transform("mean")
    )

    # Optional export
    if save:
        output_path = os.path.join(metrics_dir, "time_statistics.csv")
        times.to_csv(output_path, index = False)

    return times

def load_model_hyperparameters(models_dir: str, metrics_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load trained models from a directory, extract key hyperparameters, and organize
    them into a pivoted DataFrame mapping datasets to model types and parameter values.

    Optionally saves the resulting DataFrame to `model_hyperparameters.csv` in the
    provided metrics directory.

    Args:
        models_dir (str): Path to the root directory containing model subfolders, where
            each subfolder corresponds to a dataset and contains model files.
        metrics_dir (Optional[str], default=None): Directory where the resulting
            CSV file (`model_hyperparameters.csv`) will be saved. If None, results
            are not saved to disk.

    Returns:
        pd.DataFrame: A pivot table with datasets as rows, model types as columns,
        and stringified hyperparameters as cell values.
    """
    models_info = []

    # Traverse datasets and collect model objects with 'full' in filename
    for dataset in os.listdir(models_dir):
        dataset_path = os.path.join(models_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue

        model_paths = [p for p in os.listdir(dataset_path) if 'full' in p]
        for path in model_paths:
            model_obj = load_object(os.path.join(dataset_path, path))
            models_info.append({
                'dataset': dataset,
                'model_type': path.split('-')[1],
                'model': model_obj
            })

    models_df = pd.DataFrame()

    # Extract relevant hyperparameters based on model type
    for model_info in models_info:
        dataset, model_type, model = tuple(model_info.values())
        model_params = model.get_params()
        df_dict = {'dataset': dataset, 'model_type': model_type}
        rel_params = {}

        if isinstance(model, LogisticRegression):
            param_kws = ['solver', 'penalty', 'C']
        elif isinstance(model, DecisionTreeClassifier):
            param_kws = ['max_depth', 'min_samples_split', 'min_samples_leaf']
        elif isinstance(model, RandomForestClassifier):
            param_kws = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
        elif isinstance(model, XGBClassifier):
            param_kws = ['n_estimators', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree']
        elif isinstance(model, SVC):
            param_kws = ['C', 'gamma']
            rel_params['kernel'] = 'rbf'
        elif isinstance(model, LinearSVC):
            param_kws = ['C']
            rel_params['kernel'] = 'linear'
        else:
            # Skip unsupported models
            continue

        # Extract parameters of interest
        for kw in param_kws:
            if kw in model_params:
                rel_params[kw] = model_params[kw]

        # Add to DataFrame
        df_dict['params'] = rel_params
        models_df = pd.concat(
            [models_df, pd.DataFrame.from_dict(df_dict, orient = 'index').T],
            ignore_index = True
        )

    # Map dataset folder names to human-readable labels
    dataset_map = {
        'kaggle-credit-card-fraud': 'Credit Card Fraud',
        'kaggle-patient-survival': 'Patient Survival',
        'uci-android-permissions': 'Android Permissions',
        'uci-breast-cancer': 'Breast Cancer',
        'uci-heart-disease': 'Heart Disease',
        'uci-indian-liver': 'Indian Liver',
        'uci-mushroom': 'Mushroom',
        'uci-secondary-mushroom': 'Secondary Mushroom',
        'uci-phishing-url': 'Phishing URL',
        'uci-spect-heart': 'SPECT Heart'
    }
    models_df['dataset'] = models_df['dataset'].replace(dataset_map)

    # Map model identifiers to readable names
    model_map = {
        'dt': 'Decision Tree',
        'logreg': 'Logistic Regression',
        'rf': 'Random Forest',
        'svc': 'Support Vector Machine',
        'xgb': 'XGBoost'
    }
    models_df['model_type'] = models_df['model_type'].replace(model_map)

    # Convert parameter dictionaries into readable HTML-style strings for display
    models_df['params'] = models_df['params'].apply(
        lambda x: '<br>'.join([f'{k} = {v}' for k, v in x.items()])
    )
    models_df.columns = ['Dataset', 'Model Type', 'Params']

    # Pivot table: datasets as rows, model types as columns, parameters as values
    pivot_df = models_df.pivot(
        columns = 'Model Type', index = 'Dataset', values = 'Params'
    )

    # Optionally save results to metrics directory
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok = True)
        save_path = os.path.join(metrics_dir, 'model_hyperparameters.csv')
        pivot_df.to_csv(save_path, index = True)

    return pivot_df