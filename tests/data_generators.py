# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/8/2016
:TL;DR: this module is responsible for datasets generation
"""
import pandas as pd
import numpy as np
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
        df_with_id_cols[i] = pd.Series(ser, name=col_name)
    return df_with_id_cols

