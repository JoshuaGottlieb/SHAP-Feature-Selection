# SHAP-Feature-Selection

## Summary

Feature selection techniques in data science help reduce dimensionality of datasets, remove sparsity, and eliminate noisy patterns, thereby improving generalization of models to future data. Many feature selection techniques fail to optimize for the predictive power of the model, require iterating through the extensive feature space, or apply only to certain model types. Traditional feature selection techniques are classified into three groups: filter, wrapper, and embedded methods. Filter methods are fast but rely on statistical measures and are unable to use patterns learned by models. Wrapper methods require iteratively retraining models on different feature subsets, making them computationally expensive. Embedded methods operate during the model training process but are model-specific.

SHAP is a technique originally designed for the field of machine learning explainability to help explain black-box model predictions. SHAP values are assigned to features that are consistent and locally accurate with model predictions. The global model can be estimated using mean absolute SHAP values, which makes SHAP values a surrogate for feature importances. These SHAP values can be used as feature rankings for feature selection. Since SHAP is designed to be model-agnostic and explicitly leverages the patterns learned by the model, it can be used as a universal feature selection method which is model-agnostic, non-iterative, and model-aware.

Prior work using SHAP values for feature selection failed to address the issue of how to select feature subsets without resorting to computationally expensive methods. This repository contains work designed to address this gap by introducing two SHAP selection algorithms (SUM and MAX) which are designed to select features based on the summed feature importances or based on the maximum feature importance, respectively. This repository tests the SUM and MAX algorithms across five model types and [ten datasets](https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/tree/main/data/raw) and measures how SHAP selection compares to three classic filter methods. An accompanying [project paper](https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/blob/main/docs/CS668_Analytics_Capstone_Paper-Gottlieb_Chunsen_Yoo-Technical_Draft-2.pdf) is available in this repository, containing the full technical background, methodology, and analysis performed. A brief version of the work present in this repository is summarized in the [project poster](https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/blob/main/docs/Gottlieb_Chunsen_Yoo-Capstone_Project_Poster.pdf).

## Project Architecture

<img src=https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/blob/main/visualizations/Capstone-Architecture.png width = 100%, height = 100%>

The project architecture is displayed above and summarized below.
<ol>
  <li>Each dataset is processed for use in modeling.</li>
  <li>Five models are trained on each dataset (Logistic Regression, Decision Trees, Random Forests, XGBoost, and Support Vector Classifiers). Hyperparameter tuning is performed using 5-fold CV, with PR AUC as the optimization metric.</li>
  <li>The trained models are used with the Permutation explainer to calculate the SHAP values, which are aggregated and sorted into a global ranked feature list.</li>
  <li>Feature selection is performed using one of our two SHAP selection algorithms: SUM and MAX. The reduced feature training and testing sets are created, and the model is re-trained with the same hyperparameters on the reduced feature training set.</li>
  <li>Model performance is scored using the reduced feature test data for later evaluation.</li>
</ol>

The code for the MAX and SUM strategies is available under the `shap_select` function in [src/modules/selection.py](https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/blob/main/src/modules/selection.py). The SHAP selection techniques were compared with three filter methods: Mutual Information Gain, ReliefF, and Minimum-Redundancy-Maximum-Relevance (mRMR). Mutual Information Gain was implemented using [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif). ReliefF was implemented using [gitter-badger's Python implementation](https://github.com/gitter-badger/ReliefF/tree/master). mRMR was implemented using the [scikit-fda](https://github.com/GAA-UAM/scikit-fda) package.

## Repository Flow

All relevant models, datasets, and objects are saved within this project in the [data/](https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/tree/main/data), [docs/](https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/tree/main/docs), [metrics/](https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/tree/main/metrics), and [models/](https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/tree/main/models) directories. All work was performed using the notebooks under [src/](https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/tree/main/src). The repository flow is as follows:

Raw datasets are processed through the [Data Preprocessing](https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/blob/main/src/Data-Preprocessing.ipynb) notebook. Then, each model type is trained and hyperparameter tuned on each dataset in the [Hyperparameter Tuning](https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/blob/main/src/Hyperparameter-Tuning.ipynb) notebook. After training, GridSearch models are compressed into base estimator pickle files using the [Model Compression](https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/blob/main/src/Model-Compression.ipynb) notebook. The trained models are then used to generate SHAP explanations in the [Generate Explanations](https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/blob/main/src/Generate-Explanations.ipynb) notebook. Once the SHAP explanations are generated, SHAP feature selection is performed in the [SHAP Feature Selection](https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/blob/main/src/SHAP-Feature-Selection.ipynb) notebook, producing reduced datasets and retraining each model on the reduced datasets. After SHAP feature selection is done, the number of features selected per dataset is recorded and used to perform non-SHAP feature selection in the [Non-SHAP Feature Selection](https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/blob/main/src/Non-SHAP-Feature-Selection.ipynb) notebook. Finally, all of the data is merged and analyzed in the [Analysis](https://github.com/JoshuaGottlieb/SHAP-Feature-Selection/blob/main/src/Analysis.ipynb) notebook.

## Library Requirements

The libraries and version of Python used in this project are listed below.

```
Python 3.12.3

dataframe_image==0.2.7
feature_engine==1.9.3
matplotlib==3.10.6
numpy==1.26.4
pandas==2.2.3
playwright==1.51.0
ReliefF==0.1.2
scikit-fda==0.10.1
scikit-learn==1.6.1
seaborn==0.13.2
shap==0.47.1
xgboost==2.1.4
```

## Datasets

The ten datasets used in this repository are available at the locations below.

```
Android Permissions: https://archive.ics.uci.edu/dataset/722/naticusdroid+android+permissions+dataset
Breast Cancer: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
Credit Card Fraud: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease
Indian Liver Patient Dataset: https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset
Mushroom: https://archive.ics.uci.edu/dataset/73/mushroom
Patient Survival Prediction: https://www.kaggle.com/datasets/mitishaagarwal/patient
Phishing URL: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
Secondary Mushroom: https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset
SPECT Heart: https://archive.ics.uci.edu/dataset/95/spect+heart
```

## Repository Structure
```
├── data/                                 # Directory containing all data files
│   ├── raw/                                   # Raw datasets from UCI and Kaggle
│   ├── processed/                             # Processed full feature datasets
│   ├── other-feature-selection/               # MI, ReliefF, and mRMR feature reduced datasets
│   └── reduced/                               # SHAP feature reduced datasets
├── docs/                                 # Directory containing papers, presentations, posters and other documents
├── metrics/                              # Directory containing calculated statistics
├── models/                               # Directory containing pickled trained models for each dataset
├── shap-explanations/                    # Directory containing pickled SHAP explanations for trained models for each dataset
├── src/                                  # Directory containing project notebooks and source code
│   ├── Analysis.ipynb                         # Notebook containing analysis of project results and image generation
│   ├── Data-Preprocessing.ipynb               # Notebook containing code for preprocessing datasets
│   ├── Generate-Explanations.ipynb            # Notebook for generating SHAP explanations
│   ├── Hyperparameter-Tuning.ipynb            # Notebook for training and tuning models
│   ├── Model-Compression.ipynb                # Notebook for converting GridSearchCV models to compressed, pickled base estimators
│   ├── Non-SHAP-Feature-Selection.ipynb       # Notebook for selecting features using non-SHAP methods and retraining models
│   ├── SHAP-Feature-Selection.ipynb           # Notebook for selecting features using SHAP methods and retraining models
│   └── modules/                               # Source Python code for use in notebooks
|       ├── analysis.py                             # Functions for metric and prediction generation and for analysis
|       ├── explanations.py                         # Functions for generating SHAP explanations
|       ├── preprocessing.py                        # Functions for preprocessing datasets prior to model training
|       ├── selection.py                            # Functions for feature selection (SHAP and non-SHAP)
|       ├── training.py                             # Functions for model training and retraining
|       └── utils.py                                # Functions for saving and loading
├── visualizations/                       # Directory containing saved images for use in papers, presentations, and posters
├── README.md
└── requirements.txt
```
