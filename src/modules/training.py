from typing import Dict, Any, Union
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
import pandas as pd
from modules.utils import save_object

def fit_model(X_train: pd.DataFrame, y_train: Union[pd.Series, pd.DataFrame],
              model_name: str, model: BaseEstimator, grid_search: bool = False,
              param_grid: Dict[str, Any] = {}, cv: int = 5, random_state: int = 42,
              save: bool = True, save_path: str = './models/temp_model.pickle') -> Union[Pipeline, GridSearchCV]:
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

    Returns:
        Union[Pipeline, GridSearchCV]: 
            The trained model pipeline, optionally wrapped in a GridSearchCV object.
    """

    # Define the pipeline with the given model
    pipeline = Pipeline(steps = [(model_name, model)])
        
    # Wrap pipeline in a GridSearchCV if requested
    if grid_search:
        full_model = GridSearchCV(
            estimator = pipeline,
            param_grid = param_grid,
            cv = cv,
            scoring = ['average_precision', 'roc_auc', 'f1', 'accuracy'],
            refit = 'average_precision',  # use AP score for final refit
            n_jobs = -1
        )
    else:
        full_model = pipeline
        
    # Fit the model on the training data
    full_model.fit(X_train, y_train)
    
    # Save the trained model if requested
    if save:
        save_object(full_model, save_path)
    
    return full_model
