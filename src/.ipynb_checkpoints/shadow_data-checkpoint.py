### Add pipeline to create data for shadow modeling
import pandas as pd
import numpy as np
import pickle as pickle
from tqdm import tqdm
from random import sample, choice
from src.generators import Generator
from src.distance import compute_distances
from src.feature_extractors import fit_ohe, apply_ohe

def create_shadow_training_data_membership(df: pd.DataFrame, meta_data: list,
                                target_record: pd.DataFrame, generator: Generator,
                                n_original: int, n_synth: int, n_pos: int, seeds: list, return_indices=False, num_indices=None) -> tuple:
    datasets = []
    datasets_utility = []
    labels = []
    indices_list = []
    
    assert len(seeds) == n_pos * 2

    for i in tqdm(range(n_pos)):
        indices_sub = sample(list(df.index), n_original - 1)
        df_sub = df.loc[indices_sub]
        df_w_target = pd.concat([df_sub, target_record], axis=0)
        indices_wo_target = sample(list(df.index), n_original)
        df_wo_target = df.loc[indices_wo_target]  
        try:
            # let's create a synthetic dataset from data with the target record
            synthetic_from_target = generator.fit_generate(dataset=df_w_target, metadata=meta_data,
                                                           size=n_synth, seed = seeds[2 * i])
            datasets.append(synthetic_from_target)
            labels.append(1)
        except:
            print('error')
            print(df_w_target)

        # let's create a synthetic dataset from data without the target record
        synthetic_wo_target = generator.fit_generate(dataset=df_wo_target, metadata=meta_data,
                                                       size=n_synth, seed = seeds[2 * i + 1])

        datasets.append(synthetic_wo_target)
        labels.append(0)
        
        datasets_utility.append({'Real':{'With':df_w_target,'Without':df_wo_target}, 'Synth':{'With':synthetic_from_target,'Without':synthetic_wo_target}})
        indices_list.append(df_w_target.index)
        indices_list.append(indices_wo_target)
    if return_indices:
        return datasets, labels, datasets_utility, indices_list[0:num_indices]
    else:
        return datasets, labels, datasets_utility


def create_shadow_training_data_membership_specific_2(df_sub: pd.DataFrame, meta_data: list,
                                target_record: pd.DataFrame, generator: Generator,
                                n_original: int, n_synth: int, n_pos: int, seeds: list, reference_record: pd.DataFrame) -> tuple:
        
    #df_sub has 999 records. 
    datasets = []
    datasets_utility = []
    labels = []
    
    assert len(seeds) == n_pos * 2

    df_w_target = pd.concat([df_sub, target_record], axis=0)

    df_wo_target = pd.concat([df_sub, reference_record], axis=0)
        

    # let's create a synthetic dataset from data with the target record (only 1 model is trained)
    synthetic_from_target = generator.fit_generate(dataset=df_w_target, metadata=meta_data,
                                                       size=n_synth*n_pos, seed = seeds[0])
    for i in range(n_pos):
        datasets.append(synthetic_from_target.iloc[i*n_synth:(i+1)*n_synth])
        labels.append(1)

    # let's create a synthetic dataset from data without the target record (only 1 model is trained)
    synthetic_wo_target = generator.fit_generate(dataset=df_wo_target, metadata=meta_data,
                                                       size=n_synth*n_pos, seed = seeds[1])
    for i in range(n_pos):
        datasets.append(synthetic_wo_target.iloc[i*n_synth:(i+1)*n_synth])
        labels.append(0)
        
    datasets_utility.append({'Real':{'With':df_w_target,'Without':df_wo_target}, 'Synth':{'With':synthetic_from_target,'Without':synthetic_wo_target}})

    return datasets, labels, datasets_utility

def create_shadow_training_data_membership_specific_100(df_sub: pd.DataFrame, meta_data: list,
                                target_record: pd.DataFrame, generator: Generator,
                                n_original: int, n_synth: int, n_pos: int, seeds: list, reference_record: pd.DataFrame) -> tuple:
        
    #df_sub has 999 records. 
    datasets = []
    datasets_utility = []
    labels = []
    
    assert len(seeds) == n_pos * 2

    df_w_target = pd.concat([df_sub, target_record], axis=0)

    df_wo_target = pd.concat([df_sub, reference_record], axis=0)
        

    # let's create a synthetic dataset from data with and without the target record (200 models are trained)
    for i in range(n_pos):
        synthetic_from_target = generator.fit_generate(dataset=df_w_target, metadata=meta_data,
                                                       size=n_synth, seed = seeds[2*i])
        datasets.append(synthetic_from_target)
        labels.append(1)
        
        synthetic_wo_target = generator.fit_generate(dataset=df_wo_target, metadata=meta_data,
                                                       size=n_synth, seed = seeds[2*i+1])
        
        datasets.append(synthetic_wo_target)
        labels.append(0)
        
    datasets_utility.append({'Real':{'With':df_w_target,'Without':df_wo_target}, 'Synth':{'With':synthetic_from_target,'Without':synthetic_wo_target}})

    return datasets, labels, datasets_utility

def create_shadow_training_data_membership_specific_sep_generators(df_sub: pd.DataFrame, meta_data: list,
                                target_record: pd.DataFrame, generator: Generator,
                                n_original: int, n_synth: int,
                                n_pos: int, seeds: list, reference_record: pd.DataFrame) -> tuple:

    print('training each test generator with a different seed')
    #df_sub has 999 records. 
    datasets = []
    labels = []
    
    assert len(seeds) == n_pos * 2

    df_w_target = pd.concat([df_sub, target_record], axis=0)

    df_wo_target = pd.concat([df_sub, reference_record], axis=0)
    
    for i in tqdm(range(n_pos)):
        
        synthetic_w_target = generator.fit_generate(dataset=df_w_target, metadata=meta_data,
                                                   size=n_synth, seed=seeds[2*i])
        datasets.append(synthetic_w_target)
        labels.append(1)
        
    for i in tqdm(range(n_pos)):
        
        synthetic_wo_target = generator.fit_generate(dataset=df_wo_target, metadata=meta_data,
                                                    size=n_synth, seed=seeds[2*i+1])
        datasets.append(synthetic_wo_target)
        labels.append(0)

    return datasets, labels

def create_shadow_training_data_membership_specific_vary_xref(df_sub: pd.DataFrame, meta_data: list,
                                target_record: pd.DataFrame, generator: Generator,
                                n_original: int, n_synth: int,
                                n_pos: int, seeds: list, df_test: pd.DataFrame, shuffle: bool = False) -> tuple:
    
    print('training each test generator with a different seed')
    #df_sub has 999 records. 
    datasets = []
    labels = []
    
    assert len(seeds) == n_pos * 2
    
    for i in tqdm(range(n_pos)):
        
        df_w_target = pd.concat([df_sub, target_record], axis=0)

        if shuffle:
            df_w_target = df_w_target.sample(frac=1, random_state=seeds[2*i])
        
        synthetic_w_target = generator.fit_generate(dataset=df_w_target, metadata=meta_data,
                                                   size=n_synth, seed=seeds[2*i])
        datasets.append(synthetic_w_target)
        labels.append(1)
        
    for i in tqdm(range(n_pos)):
        
        df_wo_target = pd.concat([df_sub, df_test.sample(1)], axis=0)

        if shuffle:
            df_wo_target = df_wo_target.sample(frac=1, random_state=seeds[2*i+1])
        
        synthetic_wo_target = generator.fit_generate(dataset=df_wo_target, metadata=meta_data,
                                                    size=n_synth, seed=seeds[2*i+1])
        datasets.append(synthetic_wo_target)
        labels.append(0)

    return datasets, labels