""" 
:Authors: Tal Peretz
:Date: 10/14/2016
:TL;DR: this module is responsible for improving data quality via filling missing values, removing outliers in data and removing id_columns
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import Imputer, RobustScaler, MinMaxScaler

from pyds import constants, ml, transformations


def remove_id_columns(X_train, id_columns):
    """
    given list of columns to drop and a pandas dataframe
    returns a pandas dataframe without the id_columns
    :param X_train: pandas dataframe
    :param id_columns: [list] columns to drop
    :return: pandas dataframe without id_columns
    """
    assert (isinstance(X_train, pd.DataFrame)) and (not X_train.empty), 'X should be a valid pandas DataFrame'
    assert id_columns is not None
    return X_train.drop(id_columns, axis=1)


def _knn_imputation(df):
    """
    given a pandas DataFrame
    returns the dataframe with filled values using K nearest neighbours imputation for each missing value
    before applying knn the data is scaled using sklearn RobustScaler since outliers haven't been removed yet
    :param df: pandas DataFrame
    :return: pandas DataFrame without missing values
    """
    numerical_cols = df.select_dtypes(include=['float', 'int']).columns
    filled_df = df.copy()  # the dataset we are filling and returning in the end
    nan_df = df[df.isnull().any(axis=1)]
    na_idxs_to_fill = set(nan_df.index)
    #  iterate through all na_rows
    while bool(na_idxs_to_fill):
        nan_row_index = na_idxs_to_fill.pop()
        nan_row = nan_df.loc[nan_row_index, :]
        filled_cols = nan_row[nan_row.notnull()].index.tolist()
        missed_cols = nan_row[nan_row.isnull()].index.tolist()

        # test set = rows with same nan mask (same columns has nan)
        test_df = nan_df[nan_df.isnull().apply(lambda row: row.equals(nan_row.isnull()), axis=1)]
        # remove test set indexes from iterations because we are filling all of it in one iteration
        na_idxs_to_fill = na_idxs_to_fill.difference(set(test_df.index))
        # train set = all rows where the test set's missing columns are filled, in order to learn from
        train_df = df.loc[:, filled_cols].dropna()
        # preprocess dataset -> encode and dummify categorical columns, scale and group numerical data
        transformed_train_df, train_transformations = \
            transformations.preprocess_train_columns(train_df, col_to_scaler=defaultdict(RobustScaler))

        for missed_col in missed_cols:
            # scaling before applying KNN so the distance would be meaningful, using robust because the
            # data is before outliers removal
            Y_train = df.loc[transformed_train_df.index, missed_col].dropna()
            X_train = transformed_train_df.loc[Y_train.index, :]

            # using regressor if missed column is numerical and else classifier
            knn_regressor = KNeighborsRegressor(n_neighbors=constants.KNN_N_NEIGHBORS, weights='distance')
            knn_classifier = KNeighborsClassifier(n_neighbors=constants.KNN_N_NEIGHBORS, weights='distance')
            neigh = knn_regressor if missed_col in numerical_cols else knn_classifier
            transformed_test_df = transformations.preprocess_test_columns(test_df.loc[:, filled_cols],
                                                                          train_transformations)
            train_test_columns = set(transformed_test_df.columns).intersection(X_train.columns)
            neigh.fit(X_train.loc[:, train_test_columns], Y_train)
            filled_df.loc[test_df.index, missed_col] = neigh.predict(transformed_test_df.loc[:, train_test_columns])
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
    if isinstance(df, pd.Series):
        return df.isnull

    else:
        # init series with True: assume all attributes are present
        presence_series = pd.Series(index=df.index, data=(True for _ in df.index))
        na_rows = df[df.isnull().any(axis=1)]

        # indicate missing values as a tuple indicating presence for each cell
        for na_row_idx, na_row in na_rows.iterrows():
            presence_series.loc[na_row_idx] = tuple(na_row.isnull())
    return presence_series


def fill_missing_values(X, method='knn', drop_above_null_percents=constants.DROP_ABOVE_NULL_THRESHOLD):
    """
    given a pandas DataFrame and imputation method
    returns the dataframe with filled values according to method
    :param X: pandas DataFrame
    :param method: 'mean', 'median', 'most_frequent', 'knn' - default
    :param drop_above_null_percents: if a row has more than drop_above_null_percents it will be removed
    instead of filled
    :return: pandas DataFrame without missing values, rows containing NaN, filled rows
    """
    assert (isinstance(X, pd.DataFrame)) and (not X.empty), 'X should be a valid pandas DataFrame'
    df = X.copy()
    null_counts = df.apply(lambda row: row.isnull().sum(), axis=1)
    drop_idxs = null_counts[null_counts > (len(df.columns) * drop_above_null_percents)]
    df_to_fill = df.drop(drop_idxs)
    na_rows = df[df.isnull().any(axis=1)]
    if na_rows.empty:
        return df.copy(), na_rows, na_rows
    else:
        presence_series = _indicate_missing_values(df)
        if method == 'knn':
            filled_df = _knn_imputation(df_to_fill)
        else:
            filled_df = _simple_imputation(df_to_fill, method)
        filled_df['presence_series'] = presence_series
    return filled_df, na_rows, filled_df.loc[na_rows.index, :]


def _get_column_outliers_std(column, m=3):
    """
    given a pandas Series representing a column in a dataframe
    returns pandas Series without the values which are further than m*std
    :param column: pandas Series representing a column in a dataframe
    :param m: num of std as of to remove outliers
    :return: pandas Series with the values which exceeds m*std
    """
    outliers = column[abs(column - np.mean(column)) > m * np.std(column)].index
    return outliers


def detect_outliers(X, y=None, contamination=0.1, method='hdbscan', m=3,
                    numerical_scaler=defaultdict(MinMaxScaler)):
    """
    given a pandas DataFrame returns dataframe with contamination*num of instances
    rows indexes indicating outliers using isolation forest or m*std per column to detect outliers
    :param numerical_scaler: numerical scaler to apply on each of the numerical columns
    :param pipeline_results: class: 'PipelineResults'
    :param m: num of std as of to remove outliers
    :param y: [pandas series] target column
    :param X: [pandas DataFrame] raw features
    :param contamination:  the proportion of outliers in the data set
    :param method:
    :return: outliers indexes
    """
    assert (isinstance(X, pd.DataFrame)) and (not X.empty), 'X should be a valid pandas DataFrame'
    transformed_X = transformations.preprocess_train_columns(X, col_to_scaler=numerical_scaler)[0]
    if method == 'IsolationForest':
        outliers = ml.detect_anomalies_with_isolation_forest(transformed_X, y=y, contamination=contamination)
    elif method == 'hdbscan':
        outliers = ml.detect_anomalies_with_hdbscan(transformed_X, y=y, contamination=contamination)
    elif method == 'std':
        # find columns that are m*std further than the mean
        outliers = list(transformed_X.apply(lambda col: _get_column_outliers_std(col, m=m), axis=1).as_matrix().ravel())
    else:
        raise ValueError('supporting methods are: IsolationForest, hdbscan, std')
    return outliers
