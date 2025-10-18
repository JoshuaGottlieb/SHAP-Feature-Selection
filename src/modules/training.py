import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from modules.utils import load_dataset, load_object, save_object

def fit_model(X_train: pd.DataFrame, y_train: Union[pd.Series, pd.DataFrame],
              model_name: str, model: BaseEstimator, grid_search: bool = False,
              param_grid: Dict[str, Any] = {}, cv: int = 5, random_state: int = 42,
              save: bool = True, save_path: str = './models/temp_model.pickle',
              compression = None) -> Union[Pipeline, GridSearchCV]:
    """
    Fit a machine learning model (optionally with grid search) and save it.

    Args:
        X_train (pd.DataFrame): 
            Training feature set.
        y_train (Union[pd.Series, pd.DataFrame]): 
            Training labels.
        model_name (str): 
            Name used to identify the model within the pipeline.
        model (BaseEstimator): 
            Scikit-learn compatible estimator (e.g., LogisticRegression, RandomForestClassifier).
        grid_search (bool, optional): 
            Whether to perform hyperparameter tuning with GridSearchCV. Default is False.
        param_grid (Dict[str, Any], optional): 
            Dictionary of hyperparameters for grid search. Keys should match 
            pipeline step format (e.g., `{ 'model_name__param': [...] }`). Default is {}.
        cv (int, optional): 
            Number of cross-validation folds. Default is 5.
        random_state (int, optional): 
            Random seed for reproducibility. Default is 42.
        save (bool, optional): 
            Whether to save the trained model. Default is True.
        save_path (str, optional): 
            Path to save the trained model pickle file. Default is './models/temp_model.pickle'.
        compression (str, optional):
            Compression type to use ('gzip', 'bz2', 'lzma', or None).
            Defaults to None (uncompressed).

    Returns:
        Union[Pipeline, GridSearchCV]: 
            The trained model pipeline, optionally wrapped in a GridSearchCV object.
    """
        
    # GridSearchCV if requested
    if grid_search:
        # Define the pipeline with the given model
        pipeline = Pipeline(steps = [(model_name, model)])
        
        full_model = GridSearchCV(
            estimator = pipeline,
            param_grid = param_grid,
            cv = cv,
            scoring = 'average_precision',
            refit = 'average_precision',  # use AP score for final refit
            n_jobs = -1,
            verbose = 3
        )
    else:
        full_model = model
        
    # Fit the model on the training data
    full_model.fit(X_train, y_train)
    
    # Save the trained model if requested
    if save:
        save_object(full_model, save_path, compression = compression)
    
    return full_model

def retrain_on_reduced_features(dataset_names: List[str], model_types: List[str],
                                reduced_datasets: List[str], models_dir: str,
                                reduced_train_dir: str, selection_parse_mode: str,
                                compression: Optional[str] = 'lzma') -> None:
    """
    Retrain models on reduced feature subsets for each dataset and model type.

    Args:
        dataset_names (List[str]): List of dataset names to process.
        model_types (List[str]): List of model types (e.g., ['rf', 'xgb']).
        reduced_datasets (List[str]): Filenames of reduced feature datasets.
        models_dir (str): Directory where trained models are saved.
        reduced_train_dir (str): Directory containing reduced training datasets.
        selection_parse_mode (str): Mode for extracting the feature selection type from filenames.
            - "train": Extracts using `.split('train-')[-1]`
            - "model": Extracts using `.split(f'{model_type}-')[-1]`
        compression (str, optional): Compression type to use ('gzip', 'bz2', 'lzma', or None).
                                     Defaults to lzma.
    """

    # Loop through all datasets
    for dataset in dataset_names:
        # Standardize dataset name for consistent file naming
        set_name_snake = dataset.replace('-', '_')
        print(f'\n[INFO] Processing dataset: {dataset}')

        # Loop through all model types
        for model_type in model_types:
            print(f'[INFO] Loading full {model_type} model for {dataset}...')

            # Load the pre-trained full model
            model_path = os.path.join(
                models_dir, dataset, f'{set_name_snake}-{model_type}-full.pickle.xz'
            )
            model = load_object(model_path)

            # Identify reduced datasets corresponding to this dataset and model type
            if selection_parse_mode == "model":
                filtered_reduced_datasets = [
                    ds for ds in reduced_datasets if set_name_snake in ds and model_type in ds
                ]
            else:
                filtered_reduced_datasets = [
                    ds for ds in reduced_datasets if set_name_snake in ds
                ]
            print(f'[INFO] Found {len(filtered_reduced_datasets)} reduced datasets for {model_type}.')

            # Retrain the model for each reduced dataset
            for reduced_dataset in filtered_reduced_datasets:
                print(f'[INFO] Loading reduced dataset: {reduced_dataset}')

                # Load training data
                reduced_train_path = os.path.join(reduced_train_dir, reduced_dataset)
                reduced_train = load_dataset(reduced_train_path)
                X_train = reduced_train.iloc[:, :-1]
                y_train = reduced_train.iloc[:, -1]

                # Extract selection type depending on mode
                if selection_parse_mode == "train":
                    selection_type = reduced_dataset.split('.csv.gz')[0].split('train-')[-1]
                elif selection_parse_mode == "model":
                    selection_type = reduced_dataset.split('.csv.gz')[0].split(f'{model_type}-')[-1]
                else:
                    raise ValueError(
                        f"Invalid selection_parse_mode: '{selection_parse_mode}'. "
                        "Expected 'train' or 'model'."
                    )

                # Define save path for retrained model
                save_path = os.path.join(
                    models_dir, dataset, f'{set_name_snake}-{model_type}-{selection_type}'
                )

                # Retrain and save the model (no grid search)
                print(f'[INFO] Retraining {model_type} with selection method: {selection_type}')
                fit_model(
                    X_train = X_train,
                    y_train = y_train,
                    model_name = '',
                    model = clone(model),
                    grid_search = False,
                    save = True,
                    save_path = save_path,
                    compression = 'lzma'
                )

                print(f'[DONE] {model_type} ({selection_type}) retrained and saved for {dataset}.\n')

        print(f'[DONE] Completed retraining for all model types on {dataset}.\n')

    return