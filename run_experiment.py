import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from src.distance import compute_achilles
import time

from src.data_prep import load_data, split_data
from src.distance import top_n_vulnerable_records
from src.plots import plot_achilles, calculate_statistics
from src.mia import mia

import nest_asyncio
nest_asyncio.apply()

def run_exp():
    path_to_data = 'data/adult/Adult_dataset.csv'
    path_to_metadata = 'data/adult/Adult_metadata_discretized.json'

    df, categorical_cols, continuous_cols, meta_data = load_data(path_to_data, path_to_metadata)
    df_aux, df_eval, df_target = split_data(df, 'data/adult/1000_indices.pickle')

    print('Calculating Achilles scores...')
    t1 = time.time()
    all_dists = compute_achilles(df_target, categorical_cols, continuous_cols, meta_data, 5)
    t2 = time.time()

    print(f'time taken = {(t2-t1)}')

    top_n_records = top_n_vulnerable_records(all_dists, 100)

    t1 = time.time()
    

    mia(path_to_data=path_to_data, path_to_metadata=path_to_metadata, path_to_data_split='data/adult/1000_indices.pickle',
        target_records=top_n_records[0:1], generator_name='SYNTHPOP',
            n_original=1000, n_synth=1000, n_datasets=10, epsilon= 0.0, output_path = './output/files/')
    t2 = time.time()
    time_taken = t2-t1

    print(f'time taken for MIAs = {time_taken}')



if __name__ == "__main__":
    run_exp()