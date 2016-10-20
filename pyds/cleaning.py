""" 
@author: Tal Peretz
@date: 10/14/2016
@TL;DR: this module is responsible for improving data quality via filling missing values and removing outliers in data
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer


def knn_imputation(df):
    """
    given a pandas DataFrame
    returns the dataframe with filled values using K nearest neighbours imputation for each column
    :param df: pandas DataFrame
    :return: pandas DataFrame without missing values
    """
    filled_df = pd.DataFrame()
    for col in df.columns:
        neigh = KNeighborsClassifier(n_neighbors=3)
        train_df = df[np.isfinite(df[col])]
        X_train = train_df.drop(col)
        Y_train = train_df[col]
        neigh.fit(X_train, Y_train)
        filled_col = df.apply(lambda row: neigh.predict(row) if not np.isfinite(row[col]) else row, axis=1)[col]
        filled_df[col] = filled_col
    return filled_df


def simple_imputation(df, method):
    """
    given a pandas DataFrame and imputation method
    returns the dataframe with filled values according to sk-learn implementation
    :param df: pandas DataFrame
    :param method: 'mean', 'median', 'most_frequent'
    :return: pandas DataFrame without missing values
    """
    imp = Imputer(missing_values='NaN', strategy=method, axis=0)
    imp.fit(df)
    Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
    return imp.transform(df)


def fill_missing_values(df, method='knn', drop_above_null_percents=0.6):
    """
    given a pandas DataFrame and imputation method
    returns the dataframe with filled values according to method
    :param df: pandas DataFrame
    :param method: 'mean', 'median', 'most_frequent', 'knn' - default
    :param drop_above_null_percents: if a row has more than drop_above_null_percents it will be removed
    instead of filled
    :return: pandas DataFrame without missing values
    """
    null_counts = df.apply(lambda row: row.isnull().sum(), axis=1)
    drop_idxs = null_counts[null_counts > (len(df.columns) * drop_above_null_percents)]
    df_to_fill = df.drop(drop_idxs)
    if method == 'knn':
        return knn_imputation(df_to_fill)
    else:
        return simple_imputation(df_to_fill, method)


def remove_column_outliers(column, m=3):
    """
    given a pandas Series representing a column in a dataframe
    returns pandas Series without the values which are further than m*std
    :param column: pandas Series representing a column in a dataframe
    :param m: num of std as of to remove outliers
    :return: pandas Series without the values which are further than m*std
    """
    return column[abs(column - np.mean(column)) < m * np.std(column)]


def remove_outliers(df, contamination=0.1):
    """
    given a pandas DataFrame returns dataframe with contamination*num of instances
    rows dropped using isolation forest to detect outliers
    :param df: pandas DataFrame
    :param contamination:  the proportion of outliers in the data set
    :return: dataframe with contamination*num of instances rows dropped using isolation forest to detect outliers
    """
    X = df
    clf = IsolationForest(max_samples=len(df.index), n_estimators=100, contamination=contamination)
    clf.fit(X)
    Y = clf.predict(X)
    outliers = []
    for i, is_outlier in enumerate(Y):
        print(is_outlier)
        if is_outlier == -1:
            outliers.append(i)
    return X.drop(X.index[outliers])
