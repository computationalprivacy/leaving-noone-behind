import os
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from src.classifiers import drop_zero_cols, fit_classifiers, scale_features
from src.data_prep import load_data, split_data
from src.feature_extractors import (apply_feature_extractor_to_datasets,
                                    fit_ohe, get_feature_extractors)
from src.shadow_data import generate_datasets
from src.utils import ignore_depreciation


def mia(path_to_data: str, path_to_metadata: str, path_to_data_split: str, target_records: list, generator_name: str,
        n_synth: int = None, n_datasets: int = 1000, epsilon: float = 0.0, models: list=['random_forest', 'logistic_regression'], output_path: str = './output/files/'):
    """
    Membership Inference Attack (MIA) function to evaluate data privacy risks.

    Parameters
    -----------
    path_to_data: str
        Path to the data file.
    path_to_metadata: str
        Path to the metadata file.
    path_to_data_split: str
        Path to the data split information.
    target_records: list
        List of target records for MIA.
    generator_name: str
        Name of the data generator being used.
    n_synth: int
        Number of synthetic records to use. Defaults to the size of df_target if not provided.
    n_datasets: int
        Number of datasets to generate. Defaults to 1000.
    epsilon: float
        Differential privacy parameter. Defaults to 0.0.
    output_path: str
        Path to store output files. Defaults to './output/files/'.

    Returns
    -------
    dict: A dictionary containing the MIA results for each target record.
    """

    df, categorical_cols, continuous_cols, meta_data = load_data(path_to_data, path_to_metadata)
    df_aux, df_eval, df_target = split_data(df, path_to_data_split)

    if n_synth is None:
        n_synth = len(df_target)

    mia_results = []

    for tr in tqdm(target_records):
        mia_result = train_evaluate_mia(df_aux=df_aux, df_target=df_target, meta_data=meta_data, target_record_id=tr, df_eval=df_eval,
                            generator_name=generator_name, continuous_cols=continuous_cols, categorical_cols=categorical_cols,
                             n_synth=n_synth, n_datasets=n_datasets, epsilon=epsilon,
                             models = models)
        mia_results.append(mia_result)

    os.makedirs(output_path, exist_ok=True)
    
    with open(output_path+'mia_results.pickle', 'wb') as f:
        pickle.dump(mia_results, f)
    return mia_results

def train_evaluate_mia(df_aux:pd.DataFrame, df_target: pd.DataFrame, meta_data: list, target_record_id: int, df_eval: pd.DataFrame,
                            generator_name: str, continuous_cols: list, categorical_cols: list,
                             n_synth: int = 1000, n_datasets: int = 1000, seeds_train: list = None, seeds_eval: list = None, epsilon: float = 0.0,
                             models: list = None, cv: bool = False):
    """
    Train and evaluate a membership inference attack (MIA) using shadow datasets and target record.

    Parameters
    -----------
        df_aux: pd.DataFrame
            Auxiliary dataset used for generating shadow datasets.
        df_target: pd.DataFrame
            Dataset containing the target record for MIA.
        meta_data: list
            Metadata information used for feature extraction and generating synthetic datasets.
        target_record_id: int
            The ID of the target record for MIA.
        df_eval: pd.DataFrame
            Evaluation dataset used for testing the trained models.
        generator_name: str
            Name of the data generator used for generating synthetic datasets.
        continuous_cols: list
            A list of column names representing continuous features.
        categorical_cols: list
            A list of column names representing categorical features.
        n_synth: int, optional
            Number of synthetic records to generate for each shadow dataset (default is 1000).
        n_datasets: int, optional
            Number of shadow datasets to generate (default is 1000).
        seeds_train: list, optional
            List of seeds used for training dataset generation (default is None).
        seeds_eval: list, optional
            List of seeds used for evaluation dataset generation (default is None).
        epsilon: float, optional
            Differential privacy parameter for synthetic dataset generation (default is 0.0).
        models: list, optional
            A list of model names to use for training the meta-classifier (default is ['random_forest']).
        cv: bool, optional
            Whether to use cross-validation during model training (default is False).
        output_path: str, optional
            Path to save output files (default is './output/files/').

    Returns
    -------
        tuple: A tuple containing:
            - target_record_id (int): The ID of the target record used for MIA.
            - model_metrics (dict): A dictionary containing AUC and accuracy metrics for each trained model.
    """
    
    target_record = df_target.loc[[target_record_id]]
    print('Generating shadow datasets...')
    datasets_and_labels = generate_datasets(df_aux=df_aux, df_target=df_target, meta_data=meta_data,
                                            target_record_id=target_record_id, df_eval=df_eval,
                                            generator_name=generator_name, n_synth=n_synth, n_datasets=n_datasets,
                                            seeds_train=seeds_train, seeds_eval=seeds_eval, epsilon=epsilon) 
    datasets_train = [d for d in datasets_and_labels if d[2] is True]
    datasets_eval = [d for d in datasets_and_labels if d[2] is False]
    
    # fit one-hot encoding
    ohe, ohe_column_names = fit_ohe(df_aux, categorical_cols, meta_data)

    # Compute the query-based features
    QUERY_FEATURE_EXTRACTORS = [('query', range(1, df_aux.shape[1] + 1), 1e6, {'categorical':(1,), 'continuous': (3,)})]

    feature_extractors, do_ohe = get_feature_extractors(QUERY_FEATURE_EXTRACTORS)
    
    ignore_depreciation()
    print('Extracting training features...')
    features_and_labels = apply_feature_extractor_to_datasets(datasets_train=datasets_train, datasets_eval=datasets_eval, target_record=target_record, ohe=ohe, ohe_columns=categorical_cols,
                                        ohe_column_names=ohe_column_names, continuous_cols=continuous_cols, feature_extractors=feature_extractors, do_ohe=do_ohe)
    
    X_train = pd.concat([d[0] for d in features_and_labels if d[2] is True])
    y_train = pd.Series([d[1] for d in features_and_labels if d[2] is True])

    X_eval = pd.concat([d[0] for d in features_and_labels if d[2] is False])
    y_eval = pd.Series([d[1] for d in features_and_labels if d[2] is False])
    
    X_train, X_eval = drop_zero_cols(X_train, X_eval)
    X_train, X_eval = scale_features(X_train, X_eval)

    print('training meta-classifier')
    print(models)
    
    trained_models = fit_classifiers(X_train, y_train, cv=cv, models=models)

    model_metrics = dict()

    for i,m in enumerate(trained_models):
        preds = m.predict_proba(X_eval)
        accuracy = accuracy_score(y_eval, (preds[:,1]>0.5)*1)
        auc = roc_auc_score(y_eval, preds[:,1])
        model_metrics[models[i]] = {'auc':auc, 'accuracy':accuracy}
    return target_record_id, model_metrics
    