""" 
@author: Tal Peretz
@date: 10/14/2016
@TL;DR: this module is responsible for improving data quality via filling missing values and removing outliers in data
and removing id_columns
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import Imputer, RobustScaler, LabelEncoder

from pyds import constants


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
    imp = Imputer(missing_values='NaN', strategy=method, axis=0)
    return pd.DataFrame(data=imp.fit_transform(df), columns=df.columns, index=df.index)


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
    if method == 'knn':
        filled_df = _knn_imputation(df_to_fill, pipeline_results)
    else:
        filled_df = _simple_imputation(df_to_fill, method)
    return filled_df, na_rows, filled_df.loc[na_rows.index, :]


def remove_column_outliers(column, m=3):
    """
    given a pandas Series representing a column in a dataframe
    returns pandas Series without the values which are further than m*std
    :param column: pandas Series representing a column in a dataframe
    :param m: num of std as of to remove outliers
    :return: pandas Series without the values which are further than m*std
    """
    return column[abs(column - np.mean(column)) < m * np.std(column)]


def detect_outliers(X, y=None, contamination=0.1):
    """
    given a pandas DataFrame returns dataframe with contamination*num of instances
    rows dropped using isolation forest to detect outliers
    :param y: [pandas series] target column
    :param X: [pandas DataFrame] raw features
    :param contamination:  the proportion of outliers in the data set
    :return: outliers indexes
    """
    df = X.copy()
    if y is not None:
        df[y.name] = y
    clf = IsolationForest(max_samples=len(df.index), n_estimators=constants.ISOLATION_FOREST_N_ESTIMATORS,
                          contamination=contamination)
    clf.fit(df)
    Y = clf.predict(df)
    outliers = []
    for i, is_outlier in enumerate(Y):
        if is_outlier == -1:
            outliers.append(i)
    return df.index[outliers]
