import os
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import shap

from modules.utils import load_object, save_object

def generate_shap_explanations(model_eval: Callable, data: pd.DataFrame,
                               random_state: Optional[int] = None,
                               batch: Optional[slice] = None) -> Dict[str, Any]:
    """
    Generate SHAP explanations for a fitted model using the Permutation explainer.

    This function computes SHAP values for a given model evaluation function
    (`model_eval`) using the `shap.PermutationExplainer`. The target column
    is assumed to be the last column in the dataset and is automatically excluded
    from the feature set.

    Parameters
    ----------
    model_eval : Callable
        A callable or model evaluation function that accepts feature data
        and returns predictions (e.g., `model.predict_proba` or `model.predict`).
    data : pd.DataFrame
        The full dataset containing both features and a target column as the last column.
    random_state : int, optional
        Random seed for reproducibility of the SHAP permutation process.
        Default is ``None``.
    batch : slice, optional
        Slice object specifying a subset of rows to explain.
        If ``None``, all rows in the dataset are used for SHAP computation.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - ``"time"`` : float  
          Total time (in seconds) taken to compute SHAP explanations.
        - ``"shap_values"`` : np.ndarray  
          Computed SHAP values for each feature.
        - ``"mean_shap"`` : np.ndarray  
          Mean absolute SHAP value per feature, representing global importance.

    Notes
    -----
    - Uses `shap.explainers.Permutation` for model-agnostic SHAP computation.
    - The target column (assumed to be last) is automatically excluded from the background
      and feature data used for explanation.
    - Supports both regression and classification models, depending on `model_eval`.
    - If ``batch`` is provided, explanations are computed only for the specified slice
      of rows, which is useful for large datasets.
    - The returned mean absolute SHAP values are often used for global feature ranking.
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