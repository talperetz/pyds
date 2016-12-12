# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/8/2016
:TL;DR: this module is responsible for datasets generation
"""
import numpy as np
import pandas as pd
from sklearn import datasets

from pyds import constants


def generate_random_data(n_rows, n_cols, rand_low=0, rand_high=100):
    """
    given desired DataFrame shape and random values bounds returns a pandas DataFrame with random numerical values
    :param n_rows: num of rows for result DataFrame
    :param n_cols: num of cols for result DataFrame
    :param rand_low: upper bound for random values
    :param rand_high: lower bound random values
    :return: pandas DataFrame shape - [n_rows, n_cols] with random numerical values
    """
    gen_df = pd.DataFrame(np.random.randint(rand_low, rand_high, size=(n_rows, n_cols)), columns=np.arange(n_cols))
    return gen_df


def generate_id_cols(n_rows, n_cols, rand_low=0, rand_high=100, col_names=None):
    """
    given desired DataFrame shape and random values bounds returns a pandas DataFrame random id_columns according to
     constants.ID_COLUMN_DIST_RATIO_THRESHOLD
    :param n_rows: num of rows for result DataFrame
    :param n_cols: num of cols for result DataFrame
    :param rand_low: upper bound for random values
    :param rand_high: lower bound random values
    :param col_names: [optional] names for id columns
    :return: pandas DataFrame shape - [n_rows, n_cols] with id columns according to
    constants.ID_COLUMN_DIST_RATIO_THRESHOLD
    """
    df_with_id_cols = pd.DataFrame()
    for i in range(n_cols):
        num_of_constant_rows = int(np.ceil(constants.ID_COLUMN_DIST_RATIO_THRESHOLD * n_rows))
        num_of_rand_rows = n_rows - num_of_constant_rows
        rand_num = np.random.randint(rand_low, rand_high, size=1)[0]
        ser = np.append(np.full((1, num_of_constant_rows), rand_num),
                        generate_random_data(num_of_rand_rows, 1, rand_low, rand_high))
        col_name = col_names[i] if col_names is not None else i
        df_with_id_cols[col_name] = pd.Series(ser, name=col_name)
    return df_with_id_cols


def generate_empty_values(df, frac=0.3):
    """
    given a pandas DataFrame and the desired proportion of empty to filled values
    returns the pandas DataFrame with
    :param df: pandas DataFrame
    :param frac: the desired proportion of empty to filled values
    :return: pandas DataFrame with frac percents empty cells
    """
    nan_idxs = np.random.choice(df.index, size=int(frac * len(df.index)))
    nan_cols = np.random.choice(df.columns, size=int(frac * len(df.index)), replace=True)
    nan_cells = zip(nan_idxs, nan_cols)
    df_with_nans = df.copy()
    for nan_cell in nan_cells:
        df_with_nans = df_with_nans.set_value(nan_cell[0], nan_cell[1], None)
    return df_with_nans


def make_var_density_blobs(n_samples=750, centers=[[0, 0]], cluster_std=[0.5]):
    """
    given number of samples, array of clusters centers and array of cluster's std
    returns pandas DataFrame describing a "clustering problem", true labels of instances to clusters
    :param n_samples: number of samples
    :param centers: array of clusters centers
    :param cluster_std: array of cluster's std
    :return: pandas DataFrame describing a "clustering problem", true labels of instances to clusters
    """
    samples_per_blob = n_samples // len(centers)
    blobs = [datasets.make_blobs(n_samples=samples_per_blob, centers=[c], cluster_std=cluster_std[i])[0]
             for i, c in enumerate(centers)]
    labels = [i * np.ones(samples_per_blob) for i in range(len(centers))]
    return pd.DataFrame(np.vstack(blobs)), np.hstack(labels)
