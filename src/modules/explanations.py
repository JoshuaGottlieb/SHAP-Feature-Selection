import os
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import shap

from modules.utils import load_object, save_object

def generate_shap_explanations(model_eval: Callable, data: pd.DataFrame,
                               random_state: Optional[int] = None,
                               batch: slice = None) -> Dict[str, Any]:
    """
    Generate SHAP explanations for a fitted model using the Permutation explainer,
    excluding the target column from the feature set.

    Parameters
    ----------
    model_eval : Any
        A fitted model evluation method.
    data : pd.DataFrame
        The dataset containing feature columns and a target variable.
    target_col : str, optional
        The name of the target column. If not provided, the function assumes
        the last column in `data` is the target and excludes it automatically.
    random_state : int, default = None
        Random seed for reproducibility.

    Returns
    -------
    dict
        A dictionary containing:
        - "name": the provided name
        - "time_to_explain": the total computation time in seconds
        - "explanations": the SHAP values as a NumPy array
        - "expected_values": the expected values from the SHAP explainer
        - "base_values": the base values from the SHAP results

    Notes
    -----
    - Uses `shap.explainers.Permutation` to estimate feature importance.
    - Automatically excludes the target column (by name or position).
    - Works for classification models supporting `predict_proba()`.
    - The SHAP values correspond to predicted class probabilities.
    """   
    
    # Use the full dataset for background
    background_data = data.iloc[:, :-1]

    # Apply batching, if applicable
    if batch is not None:
        feature_data = data.iloc[batch, :-1]
    else:
        feature_data = data.iloc[:, :-1]
    
    start_time = time.time()

    # Initialize the SHAP permutation explainer
    explainer = shap.PermutationExplainer(model_eval, background_data, random_state = random_state)

    print(f"[INFO] Running full SHAP explanations on all {len(feature_data)} samples...")
    
    # Compute SHAP values
    shap_values = explainer.shap_values(feature_data)

    # Extract mean absolute values
    mean_absolute_shap_values = np.mean(np.abs(shap_values), axis = 0)

    elapsed_time = time.time() - start_time

    return {
        "time": elapsed_time,
        "shap_values": shap_values,
        "mean_shap": mean_absolute_shap_values
    }

def aggregate_shap_batches(shap_dir: str, model_type: str) -> None:
    """
    Aggregate SHAP explanation batches for a given model type into a single global explanation.

    This function loads all SHAP batch files in a directory that match the given model type,
    sums their computation times, concatenates their SHAP value arrays, and computes the
    mean absolute SHAP values across all samples. The aggregated explanation is then saved
    to disk using LZMA compression.

    Args:
        shap_dir (str): Path to the directory containing SHAP batch files.
        model_type (str): Model identifier (e.g., 'rf', 'xgb') used to filter relevant files.
    """
    # List all files in the SHAP directory
    files = os.listdir(shap_dir)

    # Select SHAP batch files corresponding to the given model type
    batches = [exp for exp in files if 'batch' in exp and model_type in exp]

    # Initialize global explanation structure
    global_exp = {}

    # Load and aggregate each batch
    for i, batch in enumerate(batches):
        print(f'Aggregating batch {i + 1}')
        batch_exp = load_object(os.path.join(shap_dir, batch))
        if i == 0:
            global_exp['time'] = batch_exp['time']
            global_exp['shap_values'] = batch_exp['shap_values']
        else:
            global_exp['time'] += batch_exp['time']
            global_exp['shap_values'] = np.vstack((global_exp['shap_values'], batch_exp['shap_values']))

    # Compute mean absolute SHAP values across all samples
    global_exp['mean_shap'] = np.mean(np.abs(global_exp['shap_values']), axis = 0)

    # Derive output file path from the first batch filename
    global_exp_path = os.path.join(shap_dir, batches[0].split('-batch')[0])

    # Save aggregated SHAP explanation with compression
    save_object(global_exp, global_exp_path, compression = 'lzma')

    return