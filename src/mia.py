import asyncio
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.classifiers import drop_zero_cols, fit_classifiers, scale_features
from src.data_prep import load_data, split_data
from src.feature_extractors import (apply_feature_extractor,
                                    apply_feature_extractor_train_eval,
                                    fit_ohe, get_feature_extractors)
from src.shadow_data import (generate_evaluation_datasets,
                             generate_shadow_datasets)
from src.utils import ignore_depreciation


def mia(path_to_data: str, path_to_metadata: str, path_to_data_split: str, target_records: list, generator_name: str,
        n_original: int = None, n_synth: int = None, n_datasets: int = 1000, epsilon: float = 0.0):
    
    model_metrics = asyncio.run(train_evaluate_mia_record_array(path_to_data=path_to_data, path_to_metadata=path_to_metadata, path_to_data_split=path_to_data_split,
                                                       target_records=target_records, generator_name=generator_name, n_original=n_original, n_synth=n_synth, 
                                                       n_datasets=n_datasets, epsilon=epsilon))
    metrics_df = pd.DataFrame({
        'record': model_metrics.keys(),
        'auc':[model_metrics[k]['random_forest']['auc'] for k in model_metrics.keys()],
        'accuracy':[model_metrics[k]['random_forest']['accuracy'] for k in model_metrics.keys()],
    })
    return metrics_df

# def mia(path_to_data: str, path_to_metadata: str, path_to_data_split: str, target_records: list, generator_name: str,
#         n_original: int = None, n_synth: int = None, n_datasets: int = 1000, epsilon: float = 0.0):
    
#     df, categorical_cols, continuous_cols, meta_data = load_data(path_to_data, path_to_metadata)
#     df_aux, df_eval, df_target = split_data(df, path_to_data_split)

#     if n_original is None:
#         n_original = len(df_target)
#     if n_synth is None:
#         n_synth = len(df_target)

#     metrics_per_record = dict()

#     for tr in target_records:
#         print(f'Running MIA for target record {tr}')
#         model_metrics = train_evaluate_mia(df_aux=df_aux, df_target=df_target, df_eval=df_eval, meta_data=meta_data, target_record_id=tr, generator_name=generator_name,
#                            continuous_cols=continuous_cols, categorical_cols=categorical_cols, n_original = n_original, n_synth= n_synth,
#                            n_datasets = n_datasets, epsilon=epsilon)
#         metrics_per_record[tr] = model_metrics
#     return metrics_per_record

async def train_evaluate_mia_record_array(path_to_data: str, path_to_metadata: str, path_to_data_split: str, target_records: list, generator_name: str,
        n_original: int = None, n_synth: int = None, n_datasets: int = 1000, epsilon: float = 0.0):
    df, categorical_cols, continuous_cols, meta_data = load_data(path_to_data, path_to_metadata)
    df_aux, df_eval, df_target = split_data(df, path_to_data_split)

    if n_original is None:
        n_original = len(df_target)
    if n_synth is None:
        n_synth = len(df_target)

    metrics_per_record = dict()
    tasks=list()

    for tr in target_records:
        tasks.append(asyncio.create_task(
            train_evaluate_mia(df_aux=df_aux, df_target=df_target, df_eval=df_eval, metrics_per_record=metrics_per_record, meta_data=meta_data,
                               target_record_id=tr, generator_name=generator_name, continuous_cols=continuous_cols, categorical_cols=categorical_cols,
                               n_original = n_original, n_synth= n_synth, n_datasets = n_datasets, epsilon=epsilon)
        ))
    for i in range(len(tasks)):
        await tasks[i]
    return metrics_per_record

async def train_evaluate_mia(df_aux:pd.DataFrame, df_target: pd.DataFrame, meta_data: list, target_record_id: int, df_eval: pd.DataFrame,
                             metrics_per_record: dict,
                             generator_name: str, continuous_cols: list, categorical_cols: list, n_original: int = 1000, n_synth: int = 1000,
                             n_datasets: int = 1000, seeds: list = None, epsilon: float = 0.0, models: list = ['random_forest'],
                             cv: bool = False):
    
    target_record = df_target.loc[[target_record_id]]
    print('Generating shadow datasets...')

    synthetic_datasets_train, y_train = generate_shadow_datasets(df_aux=df_aux, df_target=df_target, meta_data=meta_data, target_record_id=target_record_id, generator_name=generator_name,
                                n_original=n_original, n_synth=n_synth, n_datasets=n_datasets, seeds=seeds, epsilon=epsilon)

    print('Generating evaluation datasets...')
    synthetic_datasets_eval, y_eval = generate_evaluation_datasets(df_target=df_target, meta_data=meta_data, target_record_id=target_record_id,
                                                                df_eval=df_eval, generator_name=generator_name, n_synth=n_synth, n_datasets=n_datasets,
                                                                seeds=seeds, epsilon=epsilon)
    
    # fit one-hot encoding
    ohe, ohe_column_names = fit_ohe(df_aux, categorical_cols, meta_data)

    # Compute the query-based features
    QUERY_FEATURE_EXTRACTORS = [('query', range(1, df_aux.shape[1] + 1), 1e6, {'categorical':(1,), 'continuous': (3,)})]

    feature_extractors, do_ohe = get_feature_extractors(QUERY_FEATURE_EXTRACTORS)
    
    ignore_depreciation()
    print('Extracting training features...')
    X_train, X_eval = apply_feature_extractor_train_eval(datasets_train=synthetic_datasets_train, datasets_eval=synthetic_datasets_eval, target_record=target_record,
                                                         ohe=ohe, ohe_columns=categorical_cols, ohe_column_names=ohe_column_names, continuous_cols=continuous_cols,
                                                         feature_extractors=feature_extractors, do_ohe=do_ohe)
    
    X_train, X_eval = drop_zero_cols(X_train, X_eval)
    X_train, X_eval = scale_features(X_train, X_eval)

    print('training meta-classifier')
    
    trained_models = fit_classifiers(X_train, y_train, cv=cv, models=models)

    model_metrics = dict()

    for i,m in enumerate(trained_models):
        preds = m.predict_proba(X_eval)
        accuracy = accuracy_score(y_eval, (preds[:,1]>0.5)*1)
        auc = roc_auc_score(y_eval, preds[:,1])
    
        with open(f'{target_record_id}_{models[i]}_risk_score.pickle', 'wb') as f:
            pickle.dump({'auc':auc, 'accuracy':accuracy}, f)
        
        model_metrics[models[i]] = {'auc':auc, 'accuracy':accuracy}
    
    metrics_per_record[target_record_id] = model_metrics
    


    