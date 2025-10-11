import shap
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, Union, Optional
from sklearn.model_selection import StratifiedKFold

def generate_shap_explanations(model: Any, data: pd.DataFrame, name: str, 
                               target_col: Optional[str] = None,
                               random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate SHAP explanations for a fitted model using the Permutation explainer,
    excluding the target column from the feature set.

    Parameters
    ----------
    model : Any
        A fitted model that implements a `predict_proba()` method.
    data : pd.DataFrame
        The dataset containing feature columns and a target variable.
    name : str
        A descriptive name for the model or experiment, used in the output.
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
    # Drop the target column if specified or assume last column
    if target_col and target_col in data.columns:
        feature_data = data.drop(columns = [target_col])
    else:
        feature_data = data.iloc[:, :-1]
    
    start_time = time.time()

    # Initialize the SHAP permutation explainer
    explainer = shap.explainers.Permutation(model.predict_proba, feature_data, random_state = random_state)

    # Compute SHAP values
    shap_values = explainer(feature_data)

    # Extract main components
    explanations = shap_values.values

    elapsed_time = time.time() - start_time

    return {
        "name": name,
        "time_to_explain": elapsed_time,
        "shap_values": explanations,
    }

def sample_shap_explanations(model: Any, data: pd.DataFrame, target_col: Optional[str] = None,
                             random_state: Optional[int] = None, strategy: Union[str, float] = "global",
                             k: int = 5, verbose: bool = False) -> Dict[str, Any]:
    """
    Generate SHAP explanations using different sampling strategies with K-Fold validation.

    Parameters
    ----------
    model : Any
        A fitted model with a `predict_proba()` method.
    data : pd.DataFrame
        The dataset to explain, containing features and optionally a target column.
    target_col : Optional[str], default = None
        Name of the target column. If None, the last column is assumed to be the target.
    random_state : int, default = None
        Random seed for reproducibility.
    strategy : Union[str, float], default = "global"
        Strategy for sampling data to compute SHAP values.
        - "global": Use the entire dataset.
        - "sqrt": Use sqrt(n * p) samples per fold, where n is the number of records and p is the number of features.
                    Balances the number of records against the number of features via the geometric mean.
        - "log": Use log2(n) * p samples per fold, where n is the number of records and p is the number of features.
                    Ensures at least log2(n) points are sampled per feature and scales slowly when n >> p.
        - float: Use that proportion (0 < value < 1) of data per fold.
    k : int, default = 5
        Number of folds for stratified K-Fold cross-validation.
        Must be greater than 1.
    verbose : bool, default = False
        If True, prints progress and diagnostic information.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'name': strategy used
        - 'folds': per-fold SHAP results
        - 'average_shap_values': mean SHAP values across folds
        - 'average_expected_value': mean expected SHAP value across folds
        - 'average_base_value': mean base SHAP value across folds
        - 'total_time': total time across folds
    """
    if k <= 1:
        raise ValueError("Parameter 'k' must be greater than 1.")

    # Determine target column
    if target_col is None:
        target_col = data.columns[-1]

    # Separate features and target
    X = data.drop(columns = [target_col])
    y = data[target_col]

    n = len(X)
    p = X.shape[1]

    # === FULL STRATEGY ===
    if strategy == "global":
        if verbose:
            print(f"[INFO] Running full SHAP explanations on all {n} samples...")
        result = generate_shap_explanations(model, data, name = "global",
                                            target_col = target_col, random_state = random_state)
        return result

    # === DETERMINE SAMPLE SIZE ===
    if isinstance(strategy, float):
        if not (0 < strategy < 1):
            raise ValueError("Float strategy must be between 0 and 1 (exclusive).")
        samples_per_fold = int(strategy * n)
    elif strategy == "sqrt":
        samples_per_fold = int(np.sqrt(n * p))
    elif strategy == "log":
        samples_per_fold = int(np.log2(n) * p)
    else:
        raise ValueError("Invalid strategy. Must be 'global', 'sqrt', 'log', or a float between 0 and 1.")

    # Oversampling check
    total_samples = samples_per_fold * k
    if total_samples >= n:
        raise ValueError(
            f"Sampling too large: strategy '{strategy}' with {k} folds "
            f"would oversample ({total_samples} >= {n}). Reduce k or sampling rate."
        )

    if verbose:
        print(f"[INFO] Strategy: {strategy}")
        print(f"[INFO] Samples per fold: {samples_per_fold}")
        print(f"[INFO] Total samples across folds: {total_samples}")
        print(f"[INFO] Using StratifiedKFold with {k} splits.")

    # === STRATIFIED KFOLD SETUP ===
    skf = StratifiedKFold(n_splits = k, shuffle = True, random_state = random_state)

    folds_results = {}
    total_time = 0.0
    shap_arrays = []

    for fold_idx, (_, test_idx) in enumerate(skf.split(X, y), start = 1):
        fold_data = data.iloc[test_idx]

        # Randomly subsample within the fold
        fold_sample = fold_data.sample(n = samples_per_fold, random_state = random_state)

        if verbose:
            print(f"[INFO] Processing fold {fold_idx} with {len(fold_sample)} samples...")

        fold_result = generate_shap_explanations(
            model, fold_sample, name = f"fold{fold_idx}",
            target_col = target_col, random_state = random_state
        )

        folds_results[f"fold{fold_idx}"] = fold_result
        shap_arrays.append(fold_result["shap_values"])
        total_time += fold_result["time_to_explain"]

    # === COMPUTE AVERAGE VALUES ===
    average_shap_values = np.mean(np.stack(shap_arrays, axis = 0), axis = 0)

    # === COMBINE RESULTS ===
    combined_result = {
        "name": strategy,
        "folds": folds_results,
        "average_shap_values": average_shap_values,
        "total_time": total_time
    }

    if verbose:
        print(f"[INFO] Completed all folds. Total explanation time: {total_time:.2f}s")

    return combined_result