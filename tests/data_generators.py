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
    gen_df = pd.DataFrame(np.random.randint(rand_low, rand_high, size=(n_rows, n_cols)), columns=np.arange(n_cols))
    return gen_df


def generate_id_cols(n_rows, n_cols, rand_low=0, rand_high=100, col_names=None):
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
    nan_idxs = np.random.choice(df.index, size=int(frac * len(df.index)))
    nan_cols = np.random.choice(df.columns, size=int(frac * len(df.index)), replace=True)
    nan_cells = zip(nan_idxs, nan_cols)
    df_with_nans = df.copy()
    for nan_cell in nan_cells:
        df_with_nans = df_with_nans.set_value(nan_cell[0], nan_cell[1], None)
    return df_with_nans


def make_var_density_blobs(n_samples=750, centers=[[0, 0]], cluster_std=[0.5], random_state=0):
    samples_per_blob = n_samples // len(centers)
    blobs = [datasets.make_blobs(n_samples=samples_per_blob, centers=[c], cluster_std=cluster_std[i])[0]
             for i, c in enumerate(centers)]
    labels = [i * np.ones(samples_per_blob) for i in range(len(centers))]
    return pd.DataFrame(np.vstack(blobs)), np.hstack(labels)
