from itertools import product
from random import sample

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from src.feature_extractors import apply_ohe


def correlation_comparison(real_data, synthetic_data):
    corr_real = np.array(real_data.corr(numeric_only = False))
    corr_synth = np.array(synthetic_data.corr(numeric_only = False))
    return np.linalg.norm(np.abs(corr_real - corr_synth), ord='fro') #Frobenius norm between the two matrices

def get_ml_target(dataset_path: str) -> str:
    # for now hard coded: for each dataset, determine a column to do (binary) prediction on
    if 'Adult' in dataset_path:
        ml_target = 'salary'
    elif 'Census' in dataset_path:
        ml_target = 'Sex'
    else:
        raise ValueError('Computing ML utility will not work as target is not specified')
    return ml_target

def machinelearning_comparison(real_data, synthetic_data, y_column, ohe, ohe_column_names,
                               cont_cols, cat_cols, ML_models):
    # first apply ohe
    real_data_ohe = apply_ohe(real_data, ohe, cat_cols, ohe_column_names, cont_cols)
    y = real_data_ohe[f"{y_column}_0"]
    X = real_data_ohe.drop([f"{y_column}_0", f"{y_column}_1"],axis=1)
    synth_data_ohe = apply_ohe(synthetic_data, ohe, cat_cols, ohe_column_names, cont_cols)
    y_synth = synth_data_ohe[f"{y_column}_0"]
    X_synth = synth_data_ohe.drop([f"{y_column}_0", f"{y_column}_1"],axis=1)

    # do train/test split
    X_train_real,X_test,y_train_real,y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
    X_train_synth,_,y_train_synth,_ = train_test_split(X_synth, y_synth, test_size=0.2, random_state = 10)

    dict_to_return = {}
    for model in ML_models:
        if model == 'MLP':
            model_real = MLPClassifier(random_state = 1)
            model_synth = MLPClassifier(random_state = 1)
        elif model == 'LogReg':
            model_real = LogisticRegression(random_state=1)
            model_synth = LogisticRegression(random_state=1)
        model_real.fit(X_train_real, y_train_real)
        model_synth.fit(X_train_synth, y_train_synth)
        acc_real = accuracy_score(y_test, model_real.predict(X_test))
        acc_synth = accuracy_score(y_test, model_synth.predict(X_test))
        dict_to_return[model] = {'Real':acc_real, 'Synth' :acc_synth}

    return dict_to_return

def query(data, k_columns, row, cont_cols, cat_cols):
    length = len(data)
    for i, k_column in enumerate(k_columns):
        if k_column in cat_cols:
            data = data[data[k_column] == row[i]]
        else:
            data = data[data[k_column] <= row[i]]
    return len(data)/length

def diff_queries(real_data, synthetic_data, k_columns, cont_cols, cat_cols, cont_sample=50):
    values = []
    somme = 0

    for column in k_columns:
        # consider all values when column is categorical
        if column in cat_cols:
            possible = set(real_data[column].to_numpy())
        # if col is continuous, take a sample of the values
        else:
            possible = set(real_data[column].sample(cont_sample).to_numpy())
        values.append(possible)

    allrows = product(*values)
    n_queries = 0
    for row in allrows:
        count_real = query(real_data, k_columns, row, cont_cols, cat_cols)
        count_synth = query(synthetic_data, k_columns, row, cont_cols, cat_cols)
        somme += abs(count_real - count_synth)
        n_queries += 1
    return somme / n_queries


def k_way_marginals_comparison(real_data, synthetic_data, k, p,
                               cont_cols, cat_cols):
    columns = list(real_data.columns)

    somme = 0
    for _ in range(p):
        indices = range(len(columns))
        rand_indices = sample(indices, k)
        rand_indices_ = sorted(rand_indices)
        k_columns = [columns[i] for i in rand_indices_]
        somme += diff_queries(real_data, synthetic_data, k_columns, cont_cols, cat_cols)

    return somme / p

def compute_utility(datasets: list, dataset_path: str, cont_cols: list, cat_cols: list,
                    ohe, ohe_column_names, n_datasets: int = 10, ML_models = ('LogReg',)):
    # format of the datasets: list of dictionaries.
    # Each dictionary has the format {'Real':{'With':df1,'Without':df2},'Synth':{'With':df1,'Without':df2}} for with or without a target
    # note that for simplicity we are using k=1, 2, 3 and p=5.

    if len(datasets) < n_datasets:
        n_datasets = len(datasets)

    ml_target = get_ml_target(dataset_path)

    utilities = {}

    for i in tqdm(range(n_datasets)):
        dataset = datasets[i]
        real_datasets = dataset['Real']
        synth_datasets = dataset['Synth']
        types = ['With', 'Without']
        utilities[i] = {}
        for type_ in types:
            all_metrics = {}

            for k in range(1, 4):
                all_metrics[f"{k}-way"] = k_way_marginals_comparison(real_datasets[type_], synth_datasets[type_], k, 5,
                                                                     cont_cols, cat_cols)

            all_metrics["correlation"] = correlation_comparison(real_datasets[type_], synth_datasets[type_])

            all_metrics[f"ML_acc_{ml_target}"] = machinelearning_comparison(real_datasets[type_], synth_datasets[type_],
                                                                    ml_target, ohe, ohe_column_names, cont_cols, cat_cols, ML_models)
            utilities[i][type_] = all_metrics

    aggregate_utility = {}
    for type_ in types:
        aggregate_utility_type = {}
        for metric in all_metrics.keys():
            if metric == f"ML_acc_{ml_target}":
                for model in ML_models:
                    for data_type in ('Real', 'Synth'):
                        aggregate_utility_type[f"ML_acc_{ml_target}_{model}_{data_type}"] = \
                            np.mean([utilities[i][type_][metric][model][data_type] for i in range(n_datasets)])
            else:
                aggregate_utility_type[metric] = np.mean([utilities[i][type_][metric] for i in range(n_datasets)])
        aggregate_utility[type_] = aggregate_utility_type

    return aggregate_utility