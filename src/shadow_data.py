### Add pipeline to create data for shadow modeling
import asyncio
import pickle as pickle
from random import choice, sample

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.generators import Generator, get_generator
from src.utils import blockPrint, enablePrint

### Parallelized functions for generating shadow and evaluation synthetic datasets

async def generate_evaluation_dataset_parallel(df_target: pd.DataFrame, meta_data: list, target_record: pd.DataFrame, df_eval: pd.DataFrame, in_dataset: bool, generator_name: str,
                                     n_synth: int, seeds: list, idx: int,
                                     synthetic_datasets: list, membership_labels: list, epsilon: float):
    """
    Train a synthetic data generator and generate an evaluation synthetic dataset.

    Parameters
    -----------
    df_target : pandas.DataFrame
        target dataset
    meta_data: list
        list containing metadata concerning the data (feature types, ranges, etc.), necessary for training synthetic data generators
    target_record: pandas.DataFrame
        dataframe containing only the target record
    df_eval: pandas.DataFrame
        evaluation pool to draw the reference record from
    in_dataset: bool
        If True, the target record is in the dataset used to train the synthetic data generator.
        If False, it is replaced by a reference record randomly sampled from df_eval.
    generator_name: str
        name of the generator used, ex 'SYNTHPOP', 'BAYNET', etc. See reprosyn library for more details.
    n_synth: int
        size of each synthetic dataset
    seeds: list
        list of seed values, lenght must be equal to n_datasets
    idx: int
        counter for number of generated synthetic datasets
    synthetic datasets: list
        list containing all generated synthetic datasets. This function generates a single synthetic dataset and places it in the list, selecting the position according to idx.
    membership_labels: list
        list containing membership labels for each synthetic dataset. If the target record is included in the training set, membership label is 1, otherwise it is 0.
        Label at index idx in membership_labels corresponds to the synthetic dataset at index idx in synthetic_dataset.
    epsilon: float
        epsilon when training with differential privacy
    
    Returns
    --------
    None

    Description
    -----------
    This function trains one generator and generates a single dataset. The generator is trained on df_target,
    either with the target record swapped out for a different record sampled from df_eval (in_dataset=False),
    or on the full df_target containing the target record (in_dataset=True). The generated synthetic dataset and
    its membership label are placed in the corresponding lists.
    """
    generator = get_generator(generator_name, epsilon = epsilon)

    if in_dataset:
        df_train = pd.concat([df_target, target_record], axis=0)
    else:
        reference_record = df_eval.sample(1)
        df_train = pd.concat([df_target, reference_record], axis=0)
    blockPrint()
    synthetic_dataset = generator.fit_generate(dataset=df_train, metadata=meta_data,
                                               size=n_synth, seed=seeds[idx])
    enablePrint()

    synthetic_datasets[idx] = synthetic_dataset
    membership_labels[idx] = in_dataset

async def generate_evaluation_datasets_parallel(df_target: pd.DataFrame, meta_data: list, target_record: pd.DataFrame, df_eval: pd.DataFrame, generator_name: str,
                                                n_synth: int, n_datasets: int, seeds: list, epsilon: float):
    """
    Creates and launch tasks to concurrently generate evaluation datasets for a given target record and target dataset

    Parameters
    -----------
    df_target : pandas.DataFrame
        target dataset
    meta_data: list
        list containing metadata concerning the data (feature types, ranges, etc.), necessary for training synthetic data generators
    target_record: pandas.DataFrame
        dataframe containing only the target record
    df_eval: pandas.DataFrame
        evaluation pool to draw the reference record from
    generator_name: str
        name of the generator used, ex 'SYNTHPOP', 'BAYNET', etc. See reprosyn library for more details.
    n_synth: int
        size of each synthetic dataset
    n_datasets: int
        number of synthetic datasets to generate
    seeds: list
        list of seed values, lenght must be equal to n_datasets
    epsilon: float
        epsilon when training with differential privacy
    
    Returns
    --------
    tuple
    list containing all generated synthetic datasets, list containing membership labels for each generated synthetic dataset

    Description
    -----------
    This is a function that allows evaluation datasets to be generated concurrently. This function is not meant to be called directly, 
    but rather from `generate_evaluation_datasets`. Each launched task corresponds to a single synthetic data generator being trained
    and used to generate a single synthetic dataset. Exactly half of the synthetic data generators are trained on data including the target record.
    """
    synthetic_datasets = [None]*n_datasets
    membership_labels = [None]*n_datasets

    tasks = list()

    for i in range(n_datasets):
        in_dataset = i%2==0
        tasks.append(
            asyncio.create_task(generate_evaluation_dataset_parallel(
                df_target=df_target, meta_data=meta_data, target_record=target_record, df_eval=df_eval, in_dataset=in_dataset,
                generator_name=generator_name, n_synth=n_synth, seeds=seeds, idx=i, synthetic_datasets=synthetic_datasets,
                membership_labels=membership_labels, epsilon=epsilon
            ))
        )

    for i in range(n_datasets):
        await tasks[i]
    return synthetic_datasets, membership_labels

def generate_evaluation_datasets(df_target: pd.DataFrame, meta_data: list, target_record_id: int, df_eval: pd.DataFrame, generator_name: str,
                                 n_synth: int = 1000, n_datasets: int = 1000, seeds: list = None, epsilon: float = 0.0):
    """
    Launch the pipeline to generate evaluation synthetic datasets.

    Parameters
    -----------
    df_target : pandas.DataFrame
        target dataset
    meta_data: list
        list containing metadata concerning the data (feature types, ranges, etc.), necessary for training synthetic data generators
    target_record_id: int
        index of the target record
    df_eval: pandas.DataFrame
        evaluation pool to draw the reference record from
    generator_name: str
        name of the generator used, ex 'SYNTHPOP', 'BAYNET', etc. See reprosyn library for more details.
    n_synth: int
        size of each synthetic dataset
    n_datasets: int
        number of synthetic datasets to generate
    seeds: list
        list of seed values, lenght must be equal to n_datasets
    epsilon: float
        epsilon when training with differential privacy
    
    Returns
    --------
    tuple
    list containing all generated synthetic datasets, list containing membership labels for each generated synthetic dataset
    """
    
    if seeds is None:
        seeds = list(range(n_datasets, 2*n_datasets))
    target_record = df_target.loc[[target_record_id]]

    return asyncio.run(generate_evaluation_datasets_parallel(df_target=df_target, meta_data=meta_data, target_record=target_record,
                                                            df_eval=df_eval, generator_name=generator_name, n_synth=n_synth,
                                                            n_datasets=n_datasets, seeds=seeds, epsilon=epsilon))

async def generate_shadow_dataset_parallel(df_aux: pd.DataFrame, meta_data: list, target_record: pd.DataFrame, in_dataset: bool, generator_name: str,
                                     n_original: int, n_synth: int, seeds: list, idx: int,
                                     synthetic_datasets: list, membership_labels: list, epsilon: float = 0.0):
    
    """
    Train a synthetic data generator and generate an evaluation synthetic dataset.

    Parameters
    -----------
    df_aux: pandas.DataFrame
        auxiliary dataset, used when training the membership inference attack
    meta_data: list
        list containing metadata concerning the data (feature types, ranges, etc.), necessary for training synthetic data generators
    target_record: pandas.DataFrame
        dataframe containing only the target record
    in_dataset: bool
        If True, the target record is in the dataset used to train the synthetic data generator.
        If False, it is replaced by a reference record randomly sampled from df_eval.
    generator_name: str
        name of the generator used, ex 'SYNTHPOP', 'BAYNET', etc. See reprosyn library for more details.
    n_original: int
    n_synth: int
        size of each synthetic dataset
    seeds: list
        list of seed values, lenght must be equal to n_datasets
    idx: int
        counter for number of generated synthetic datasets
    synthetic datasets: list
        list containing all generated synthetic datasets. This function generates a single synthetic dataset and places it in the list, selecting the position according to idx.
    membership_labels: list
        list containing membership labels for each synthetic dataset. If the target record is included in the training set, membership label is 1, otherwise it is 0.
        Label at index idx in membership_labels corresponds to the synthetic dataset at index idx in synthetic_dataset.
    epsilon: float
        epsilon when training with differential privacy
    
    Returns
    --------
    None

    Description
    -----------
    This function trains one generator and generates a single dataset. The generator is trained on df_target,
    either with the target record swapped out for a different record sampled from df_eval (in_dataset=False),
    or on the full df_target containing the target record (in_dataset=True). The generated synthetic dataset and
    its membership label are placed in the corresponding lists.
    """
    generator = get_generator(generator_name, epsilon = epsilon)
    
    if in_dataset:
        indices = sample(list(df_aux.index), n_original-1)
        df_train = pd.concat([df_aux.loc[indices], target_record], axis=0)
    else:
        indices = sample(list(df_aux.index), n_original)
        df_train = df_aux.loc[indices]
    
    blockPrint()
    synthetic_dataset = generator.fit_generate(dataset=df_train, metadata=meta_data,
                                               size=n_synth, seed=seeds[idx])
    enablePrint()
    synthetic_datasets[idx] = synthetic_dataset
    membership_labels[idx] = in_dataset


async def generate_shadow_datasets_parallel(df_aux: pd.DataFrame, meta_data: list, generator_name: Generator, target_record: pd.DataFrame,
                                               n_original: int, n_synth: int, n_datasets: int, seeds: list, epsilon: float = 0.0):
    """
    Creates and launch tasks to concurrently generate evaluation datasets for a given target record and target dataset

    Parameters
    -----------
    df_aux: pandas.DataFrame
        auxiliary dataset, used when training the membership inference attack
    meta_data: list
        list containing metadata concerning the data (feature types, ranges, etc.), necessary for training synthetic data generators
    generator_name: str
        name of the generator used, ex 'SYNTHPOP', 'BAYNET', etc. See reprosyn library for more details.
    target_record: pandas.DataFrame
        dataframe containing only the target record
    n_original:int
    n_synth: int
        size of each synthetic dataset
    n_datasets: int
        number of synthetic datasets to generate
    seeds: list
        list of seed values, lenght must be equal to n_datasets
    epsilon: float
        epsilon when training with differential privacy
    
    Returns
    --------
    tuple
    list containing all generated synthetic datasets, list containing membership labels for each generated synthetic dataset

    Description
    -----------
    This is a function that allows evaluation datasets to be generated concurrently. This function is not meant to be called directly, 
    but rather from `generate_evaluation_datasets`. Each launched task corresponds to a single synthetic data generator being trained
    and used to generate a single synthetic dataset. Exactly half of the synthetic data generators are trained on data including the target record.
    """
    
    synthetic_datasets = [None]*n_datasets
    membership_labels = [None]*n_datasets
    tasks = list()

    assert len(seeds) == n_datasets

    for i in range(n_datasets):
        in_dataset = i%2==0
        tasks.append(
            asyncio.create_task(generate_shadow_dataset_parallel(
                df_aux=df_aux, meta_data=meta_data, target_record=target_record, in_dataset=in_dataset,
                generator_name=generator_name, n_original=n_original, n_synth=n_synth, seeds=seeds, idx=i,
                synthetic_datasets=synthetic_datasets, membership_labels=membership_labels, epsilon=epsilon
                ))
        )
    
    for i in range(n_datasets):
        await tasks[i]
    return synthetic_datasets, membership_labels

def generate_shadow_datasets(df_aux:pd.DataFrame, df_target: pd.DataFrame, meta_data: list, target_record_id: int, generator_name: str,
                                n_original: int = 1000, n_synth: int = 1000, n_datasets: int = 1000, seeds: list = None, epsilon: float = 0.0):
    """
    Launch the pipeline to generate shadow synthetic datasets.

    Parameters
    -----------
    df_aux: pandas.DataFrame
        auxiliary dataset, used when training the membership inference attack
    df_target : pandas.DataFrame
        target dataset
    meta_data: list
        list containing metadata concerning the data (feature types, ranges, etc.), necessary for training synthetic data generators
    target_record_id: int
        index of the target record
    generator_name: str
        name of the generator used, ex 'SYNTHPOP', 'BAYNET', etc. See reprosyn library for more details.
    n_original: int
    n_synth: int
        size of each synthetic dataset
    n_datasets: int
        number of synthetic datasets to generate
    seeds: list
        list of seed values, lenght must be equal to n_datasets
    epsilon: float
        value of epsilon when training with differential privacy
    
    Returns
    --------
    tuple
    list containing all generated synthetic datasets, list containing membership labels for each generated synthetic dataset
    """
    
    target_record = df_target.loc[[target_record_id]]
    if seeds is None:
        seeds = list(range(n_datasets))

    return asyncio.run(generate_shadow_datasets_parallel(
        df_aux=df_aux, meta_data=meta_data, generator_name=generator_name, target_record=target_record,
        n_original=n_original, n_synth=n_synth, n_datasets=n_datasets, seeds=seeds, epsilon=epsilon
    ))