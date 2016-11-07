""" 
:Authors: Tal Peretz
:Date: 10/14/2016
:TL;DR: this module converts relational data of several forms to a pandas DataFrame, infer and adjust columns types
    and splits the data for train and test
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from pyds import constants

# supported file extensions
file_extension_to_read_attribute = {'csv': 'read_csv', 'excel': 'read_excel', 'hdf': 'read_hdf', 'sql': 'read_sql',
                                    'json': 'read_json', 'html': 'read_html', 'stata': 'read_stata', 'sas': 'read_sas',
                                    'pickle': 'read_pickle'}


def _get_file_extension(file_path):
    """
    given the path of a file
    returns it's extension as string
    :param file_path: the path of an input file
    :return: the extension of the file
    """
    if file_path is None:
        raise ValueError('file path is None')
    else:
        file_extension = file_path.split(".")[-1]
        if file_extension == "":
            raise ValueError('expected .[extension] file')
        elif file_extension not in file_extension_to_read_attribute.keys():
            raise ValueError('supported file types are \n %s' % file_extension_to_read_attribute.keys())
    return file_extension


def read(*args):
    """
    given a collection of file paths representing relational data
    returns a pandas pandas DataFrame of the data
    :param args: collection of file paths representing an input file
    :return: pandas pandas DataFrame
    """
    partial_dfs = []
    for index, file_path in enumerate(args):
        pd_read_function = getattr(pd, file_extension_to_read_attribute[_get_file_extension(file_path)])
        partial_dfs.append(pd_read_function(file_path))
    return pd.concat(partial_dfs)


def validate_dataset(df):
    if len(df.index) < 50:
        raise Exception('there are not enough samples to make a decent analysis')


def get_train_test_splits(train_df, test_paths, target_column):
    if (target_column is not None) and (target_column in train_df.columns) and (len(train_df[target_column].index) > 0):
        y_train = train_df[target_column]
        X_train = train_df.drop(target_column, axis=1)
        is_supervised = True
        if test_paths is not None:
            test_df = read(test_paths)
            y_test = test_df[target_column]
            X_test = test_df.drop(target_column, axis=1)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=constants.TEST_SPLIT_SIZE)
    return X_train, X_test, y_train, y_test, is_supervised


def infer_columns_statistical_types(X, y=None):
    """
    given a pandas DataFrame returns a lists of the dataframe's numerical columns, categorical columns, id columns
    :param y: [pandas Series] target column
    :param X: [pandas DataFrame] predictor columns
    :return: lists of the dataframe's numerical columns, categorical columns, id columns
    """
    df = X.copy()
    if y is not None:
        df[y.name] = y
    unique_df = df.apply(pd.Series.nunique)
    dist_ratios = unique_df / df.apply(pd.Series.count)
    id_cols = dist_ratios.where(dist_ratios == 1).dropna().index.tolist()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    suspected_numerical_cols = set(df.select_dtypes(include=numerics).columns.drop(id_cols).tolist())
    numerical_cols = list(
        set(unique_df.where(unique_df > constants.CATEGORICAL_THRESHOLD).dropna().index.tolist()).intersection(
            suspected_numerical_cols))
    categorical_cols = list(set(df.columns.difference(numerical_cols).drop(id_cols).tolist()))
    cols_to_convert_to_categorical = list(suspected_numerical_cols.difference(numerical_cols))
    return numerical_cols, categorical_cols, id_cols, cols_to_convert_to_categorical


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
    adjusted_X_train, adjusted_X_test, adjusted_y_train, adjusted_y_test = X_train, X_test, y_train, y_test
    if (cols_to_convert_cat is not None) and cols_to_convert_cat:

        # adjust Y dataframes
        if y_train.name in cols_to_convert_cat:
            adjusted_y_train = y_train.astype(str)
            adjusted_y_test = y_test.astype(str)
            cols_to_convert_cat.remove(y_train.name)

        # adjust X dataframes
        adjusted_X_train[cols_to_convert_cat] = X_train[cols_to_convert_cat].apply(lambda num_col: num_col.astype(str))
        adjusted_X_test[cols_to_convert_cat] = X_test[cols_to_convert_cat].apply(lambda num_col: num_col.astype(str))
    return adjusted_X_train, adjusted_X_test, adjusted_y_train, adjusted_y_test
