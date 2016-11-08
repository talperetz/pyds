""" 
:Authors: Tal Peretz
:Date: 10/14/2016
:TL;DR: this module is responsible for improving data quality via filling missing values, removing outliers in data and removing id_columns
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import Imputer, RobustScaler, LabelEncoder
from pyds import constants
from pyds import ml


def remove_id_columns(X_train, id_columns):
    """
    given list of columns to drop and a pandas dataframe
    returns a pandas dataframe without the id_columns
    :param X_train: pandas dataframe
    :param id_columns: [list] columns to drop
    :return: pandas dataframe without id_columns
    """
    X_train_without_ids = X_train.copy()
    if (id_columns is not None) and id_columns:
        X_train_without_ids = X_train.drop(id_columns, axis=1)
    return X_train_without_ids


def _knn_imputation(df, pipeline_results):
    """
    given a pandas DataFrame
    returns the dataframe with filled values using K nearest neighbours imputation for each missing value
    before applying knn the data is scaled using sklearn RobustScaler since outliers haven't been removed yet
    :param df: pandas DataFrame
    :return: pandas DataFrame without missing values
    """
    numerical_cols = pipeline_results.Ingestion.numerical_cols
    categorical_cols = pipeline_results.Ingestion.categorical_cols
    filled_df = pd.DataFrame(df)
    for row_index, row in df.iterrows():
        if row.isnull().any():
            filled_cols = row[row.notnull()].index.tolist()
            missed_cols = row[row.isnull()].index.tolist()
            train_df = df[filled_cols]
            for missed_col in missed_cols:
                if missed_col in numerical_cols:
                    neigh = KNeighborsRegressor(n_neighbors=constants.KNN_N_NEIGHBORS, weights='distance')
                else:
                    neigh = KNeighborsClassifier(n_neighbors=constants.KNN_N_NEIGHBORS, weights='distance')
                if len(train_df.loc[categorical_cols, :].index) > 0:
                    le = LabelEncoder()
                    train_df[categorical_cols] = le.fit_transform(train_df[categorical_cols])
                robust_scaler = RobustScaler()
                Y_train = df[missed_col].dropna()
                X_train = robust_scaler.fit_transform(train_df.loc[Y_train.index, :])
                neigh.fit(X_train, Y_train)
                filled_df.loc[row_index, missed_col] = neigh.predict(
                    robust_scaler.transform(row[filled_cols].reshape(1, -1))).reshape(1, -1)

                filled_cols = row[row.notnull()].index.tolist()
                train_df = df[filled_cols]
    return filled_df


def _simple_imputation(df, method):
    """
    given a pandas DataFrame and imputation method
    returns the dataframe with filled values according to sk-learn implementation
    :param df: pandas DataFrame
    :param method: 'mean', 'median', 'most_frequent'
    :return: pandas DataFrame without missing values
    """
    imp = Imputer(missing_values=constants.MISSING_VALUES_REPRESENTATION, strategy=method, axis=0)
    return pd.DataFrame(data=imp.fit_transform(df), columns=df.columns, index=df.index)


# todo: if doesn't work -> replace tuple in a string
def _indicate_missing_values(df):
    """
    given a pandas DataFrame returns pandas Series indicating the presence of each row as tuple of booleans,
    if all values are present in the row it indicates True (instead of tuple)
    :param df: pandas DataFrame
    :return: pandas Series indicating the presence of each row as tuple of booleans, if all values are present in a row
     it indicates True (instead of tuple)
     :links: Data preparation for data_mining [chapter 8] -
     http://www.temida.si/~bojan/MPS/materials/Data_preparation_for_data_mining.pdf
    """
    # init series with True: assume all attributes are present
    presence_series = pd.Series(index=df.index)
    presence_series[presence_series.index] = True
    na_rows = df[df.isnull().any(axis=1)]
    # indicate missing values as a tuple indicating presence for each cell 
    for na_row_idx in na_rows.index:
        presence_series[na_row_idx] = tuple(df.loc[na_row_idx, :].isnull())
    return presence_series


def fill_missing_values(X, pipeline_results, method='knn',
                        drop_above_null_percents=constants.DROP_ABOVE_NULL_THRESHOLD):
    """
    given a pandas DataFrame and imputation method
    returns the dataframe with filled values according to method
    :param X: pandas DataFrame
    :param pipeline_results: class: 'PipelineResults'
    :param method: 'mean', 'median', 'most_frequent', 'knn' - default
    :param drop_above_null_percents: if a row has more than drop_above_null_percents it will be removed
    instead of filled
    :return: pandas DataFrame without missing values, rows containing NaN, filled rows
    """
    df = X.copy()
    null_counts = df.apply(lambda row: row.isnull().sum(), axis=1)
    drop_idxs = null_counts[null_counts > (len(df.columns) * drop_above_null_percents)]
    df_to_fill = df.drop(drop_idxs)
    na_rows = df[df.isnull().any(axis=1)]
    presence_series = _indicate_missing_values(df)
    if method == 'knn':
        filled_df = _knn_imputation(df_to_fill, pipeline_results)
    else:
        filled_df = _simple_imputation(df_to_fill, method)
    filled_df['presence_series'] = presence_series
    return filled_df, na_rows, filled_df.loc[na_rows.index, :]


def _remove_column_outliers(column, m=3):
    """
    given a pandas Series representing a column in a dataframe
    returns pandas Series without the values which are further than m*std
    :param column: pandas Series representing a column in a dataframe
    :param m: num of std as of to remove outliers
    :return: pandas Series without the values which are further than m*std
    """
    outliers = column[abs(column - np.mean(column)) > m * np.std(column)].index
    return outliers


def detect_outliers(X, y=None, contamination=0.1, method='IsolationForest', m=3):
    """
    given a pandas DataFrame returns dataframe with contamination*num of instances
    rows indexes indicating outliers using isolation forest or m*std per column to detect outliers
    :param m: num of std as of to remove outliers
    :param y: [pandas series] target column
    :param X: [pandas DataFrame] raw features
    :param contamination:  the proportion of outliers in the data set
    :param method:
    :return: outliers indexes
    """
    if method == 'IsolationForest':
        outliers = ml.detect_anomalies(X, y=y, contamination=contamination)
    else:
        # find columns that are m*std further than the mean
        outliers = X.apply(lambda col: _remove_column_outliers(col, m=m), axis=1).dropna().index.tolist()
    return outliers
