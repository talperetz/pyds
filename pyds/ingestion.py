""" 
:Authors: Tal Peretz
:Date: 10/14/2016
:TL;DR: this module converts relational data of several forms to a pandas DataFrame, infer and adjust columns types
    and splits the data for train and test
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pyds import constants


def _get_file_extension(file_path):
    """
    given the path of a file
    returns it's extension as string
    :param file_path: the path of an input file
    :return: the extension of the file
    """
    if not file_path:
        raise ValueError('file path is None')
    else:
        file_extension = file_path.split(".")[-1]
        if file_extension == "":
            raise ValueError('expected .[extension] file')
        elif file_extension not in constants.FILE_EXTENSION_TO_READ_ATTRIBUTE.keys():
            raise ValueError('supported file types are \n %s' % constants.FILE_EXTENSION_TO_READ_ATTRIBUTE.keys())
    return file_extension


def read(*args):
    """
    given a collection of file paths representing relational data
    returns a pandas pandas DataFrame of the data
    :param args: collection of file paths representing an input file
    :param kwargs: keyword arguments to pass to pandas read function
    :return: pandas DataFrame
    """
    partial_dfs = []
    for index, file_path in enumerate(args):
        pd_read_function = getattr(pd, constants.FILE_EXTENSION_TO_READ_ATTRIBUTE[_get_file_extension(file_path)])
        partial_df = pd_read_function(file_path)

        # remove wrapper collections if exist
        if not (type(partial_df) in [np.array, pd.DataFrame]):
            partial_df = partial_df[0]
        partial_dfs.append(partial_df)
    return pd.concat(partial_dfs)


def _calc_optimal_chink_size():
    pass


def read_sparse(*args, **kwargs):
    """
    given a collection of file paths representing relational data
    returns a pandas pandas DataFrame of the data
    :param args: collection of file paths representing an input file
    :param kwargs: keyword arguments to pass to pandas read function
    :return: pandas DataFrame
    """
    chunks = read(args, iterator=True, chunksize=_calc_optimal_chink_size(), **kwargs)
    sparse_chunks = []
    for chunk in chunks:
        sparse_chunks.append(chunk.to_sparse())
    return pd.concat(sparse_chunks)


def validate_dataset(df):
    if len(df.index) < 50:
        raise Exception('there are not enough samples to make a decent analysis')


def get_train_test_splits(train_df, test_paths, target_column):
    if (target_column is None) or (target_column not in train_df.columns) or (len(train_df[target_column].index) == 0):
        raise ValueError("target column doesn't exist on train set")
    else:
        y_train = train_df[target_column]
        X_train = train_df.drop(target_column, axis=1)
        is_supervised = True
        split_train = False
        if test_paths:
            test_df = read(test_paths)
            if (target_column in test_df.columns) and (len(test_df[target_column].index) > 0):
                y_test = test_df[target_column]
                X_test = test_df.drop(target_column, axis=1)
            else:
                split_train = True
        else:
            split_train = True
        if split_train:
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=constants.TEST_SPLIT_SIZE)
    return X_train, X_test, y_train, y_test, is_supervised


def infer_columns_statistical_types(X, y=None, col_to_type=None):
    """
    given a pandas DataFrame returns a lists of the dataframe's numerical columns, categorical columns, id columns
    :param col_to_type: dict of column name to type ['numerical', 'categorical', 'id'] for manually setting columns types
    :param y: [pandas Series] target column
    :param X: [pandas DataFrame] predictor columns
    :return: lists of the dataframe's numerical columns, categorical columns, id columns
    """
    numerical_cols, categorical_cols, id_cols = [], [], []
    df = X.copy()
    if y is not None:
        df[y.name] = y

    # manually set columns types
    if col_to_type:
        for col_name, col_type in col_to_type.iteritems():
            assert col_name in df.columns
            if col_type == 'numerical':
                numerical_cols.append(col_name)
            elif col_type == 'categorical':
                categorical_cols.append(col_name)
            elif col_type == 'id':
                id_cols.append(col_name)
            else:
                raise TypeError('supported types are numerical / categorical')
                df.drop(col_name, axis=1, inplace=True)
    unique_df = df.apply(pd.Series.nunique)
    dist_ratios = unique_df / df.apply(pd.Series.count)
    id_cols += dist_ratios.where(dist_ratios == constants.ID_COLUMN_DIST_RATIO_THRESHOLD).dropna().index.tolist()
    suspected_numerical_cols = set(df.select_dtypes(include=constants.NUMERIC_TYPES).columns).difference(id_cols)
    numerical_cols += list(
        set(unique_df.where(unique_df > constants.CATEGORICAL_THRESHOLD).dropna().index.tolist()).intersection(
            suspected_numerical_cols))
    categorical_cols += list(set(df.columns.difference(numerical_cols).drop(id_cols).tolist()))
    return numerical_cols, categorical_cols, id_cols


def adjust_columns_types(cols_to_convert_cat, X_train, X_test, y_train, y_test):
    """
    given train and test dataframes and a list of columns that should be categorical
    returns the initial dataframes with correct types
    :param cols_to_convert_cat: [list of str] columns that should be categorical but aren't
    :param X_train: [pandas DataFrame] train predictor columns
    :param X_test: [pandas DataFrame] test predictor columns
    :param y_train: [pandas Series] train target column
    :param y_test: [pandas Series] train target column
    :return: adjusted_X_train, adjusted_X_test, adjusted_y_train, adjusted_y_test which are the initial dataframes with
    correct types
    """
    adjusted_X_train, adjusted_X_test, adjusted_y_train, adjusted_y_test = \
        X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    if cols_to_convert_cat is not None:

        # adjust Y dataframes
        if y_train.name in cols_to_convert_cat:
            adjusted_y_train = y_train.astype('category')
            cols_to_convert_cat.remove(y_train.name)
            adjusted_y_test = y_test.astype('category')

        # adjust X dataframes
        adjusted_X_train[cols_to_convert_cat] = X_train[cols_to_convert_cat].apply(
            lambda num_col: num_col.astype('category'))
        adjusted_X_test[cols_to_convert_cat] = X_test[cols_to_convert_cat].apply(
            lambda num_col: num_col.astype('category'))
    return adjusted_X_train, adjusted_X_test, adjusted_y_train, adjusted_y_test
