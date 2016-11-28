"""
:Authors: Tal Peretz
:Date: 11/24/2016
:TL;DR: this module is responsible for categorical and numerical columns transformations
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def _calc_optimal_num_of_bins(col):
    iqr = np.subtract(*np.percentile(col, [75, 25]))
    h = int(np.ceil((2 * iqr) / (len(col) ** (1 / 3))))
    return h


def _pct_rank_qcut(series, n, edges=None):
    if edges is None:
        edges = pd.Series([float(i) / n for i in range(n + 1)])
    return series.rank(pct=1).apply(lambda x: (edges >= x).argmax()), edges


def _encode_categorical_columns(encode_df, expand_fit_df=None, col_to_encoder=None):
    if expand_fit_df:
        assert set(encode_df.columns).issubset(expand_fit_df.columns)
        expand_fit_df = pd.concat([encode_df, expand_fit_df], ignore_index=True)
    else:
        expand_fit_df = encode_df
    ordered_fit_df = expand_fit_df.apply(lambda col: col.astype('category', ordered=True))
    if not col_to_encoder:
        col_to_encoder = defaultdict(LabelEncoder)
        ordered_fit_df.apply(
            lambda col: col_to_encoder[col.name].fit(col.sort_values()))
    label_encoded_df = encode_df.apply(
        lambda col: col_to_encoder[col.name].transform(col.sort_values().values))
    label_encoded_df.columns = ['ordered_%s' % col for col in label_encoded_df.columns]
    return label_encoded_df, col_to_encoder


def _transform_categorical_columns(train_categorical_df, test_categorical_df=None, col_to_encoder=None):
    # assume there's an order - encode according to sort values
    label_encoded_df, col_to_encoder = _encode_categorical_columns(encode_df=train_categorical_df,
                                                                   expand_fit_df=test_categorical_df,
                                                                   col_to_encoder=col_to_encoder)

    # assume there is no order - dummify categorical data
    train_categorical_df = train_categorical_df.apply(lambda col: col.astype('category', ordered=False))
    dummiefied_categorical_df = pd.get_dummies(train_categorical_df,
                                               prefix=train_categorical_df.columns.tolist())
    return label_encoded_df, dummiefied_categorical_df, col_to_encoder


def _transform_numerical_columns(train_numerical_df, col_to_scaler=defaultdict(MinMaxScaler)):
    transformed_numerical_df = train_numerical_df.apply(
        lambda col: col_to_scaler[col.name].fit_transform(col))
    transformed_numerical_df = pd.DataFrame(data=transformed_numerical_df, index=train_numerical_df.index,
                                            columns=train_numerical_df.columns)
    return transformed_numerical_df, col_to_scaler


def _discretize(numerical_df, col_to_width_edges=None, col_to_depth_edges=None):
    is_edges_recieved = True
    if (not col_to_width_edges) and (not col_to_depth_edges):
        col_to_width_edges, col_to_depth_edges = {}, {}
        is_edges_recieved = False
    equal_width_num_df, equal_depth_num_df = pd.DataFrame(), pd.DataFrame()
    for col_name, col in numerical_df.iteritems():
        num_of_bins = _calc_optimal_num_of_bins(col)
        if is_edges_recieved and (col_name in col_to_width_edges.keys()) and (col_name in col_to_depth_edges.keys()):
            equal_width_col = pd.cut(col, bins=col_to_width_edges[col_name])
            equal_width_col.name = 'equal_w_%s' % col_name
            equal_width_num_df.loc[:, equal_width_col.name] = equal_width_col
            equal_depth_col, _ = _pct_rank_qcut(col, num_of_bins, edges=col_to_depth_edges[col_name])
            equal_depth_col.name = 'equal_d_%s' % col_name
            equal_depth_num_df.loc[:, equal_depth_col.name] = equal_depth_col
        else:
            if num_of_bins > 1:
                equal_width_col, col_to_width_edges[col_name] = pd.cut(col, num_of_bins, labels=False, retbins=True)
                equal_width_col.name = 'equal_w_%s' % col_name
                equal_width_num_df.loc[:, equal_width_col.name] = equal_width_col
                equal_depth_col, col_to_depth_edges[col_name] = _pct_rank_qcut(col, num_of_bins)
                equal_depth_col.name = 'equal_d_%s' % col_name
                equal_depth_num_df.loc[:, equal_depth_col.name] = equal_depth_col
    return equal_width_num_df, col_to_width_edges, equal_depth_num_df, col_to_depth_edges


def preprocess_train_columns(X_train, pipeline_results, col_to_scaler=defaultdict(MinMaxScaler), X_test=None):
    """
    given a pandas DataFrame and a PipelineResults object
    returns a dataframe with columns ready for an ML model , categorical transformations list,
    numerical transformations list
    :param col_to_scaler: numerical scaler to apply on each of the numerical columns
    :param X_train: pandas DataFrame
    :param pipeline_results: class: 'PipelineResults'
    :return: dataframe with columns ready for an ML model, categorical transformations list,
    numerical transformations list
    """
    numerical_cols = list(set(pipeline_results.ingestion_results.numerical_cols).intersection(X_train.columns.tolist()))
    categorical_cols = list(
        set(pipeline_results.ingestion_results.categorical_cols).intersection(X_train.columns.tolist()))
    is_numerical = len(numerical_cols) > 0

    # discretization of numerical columns
    if is_numerical:
        numerical_df = X_train.loc[:, numerical_cols]
        equal_width_num_df, col_to_width_edges, equal_depth_num_df, col_to_depth_edges = _discretize(numerical_df)

    # add the discretized columns to categorical columns
    categorical_df = pd.concat(
        [df for df in [X_train.loc[:, categorical_cols], equal_width_num_df, equal_depth_num_df] if df is not None],
        axis=1) if is_numerical else X_train.loc[:, categorical_cols]
    if X_test:
        assert set(categorical_df.columns).issubset(X_test.columns)
        label_encoded_df, dummiefied_categorical_df, col_to_encoder = \
            _transform_categorical_columns(categorical_df, X_test.loc[:, categorical_cols])
    else:
        label_encoded_df, dummiefied_categorical_df, col_to_encoder = _transform_categorical_columns(categorical_df)

    # add the encoded categorical columns to numerical columns
    numerical_df = pd.concat([X_train.loc[:, numerical_cols], label_encoded_df], axis=1)
    updated_categorical_cols = categorical_df.columns.tolist() + dummiefied_categorical_df.columns.tolist()
    scaled_numerical_df, col_to_scaler = _transform_numerical_columns(numerical_df, col_to_scaler)
    transformed_df = pd.concat([scaled_numerical_df, dummiefied_categorical_df], axis=1)
    return transformed_df, [col_to_scaler], [
        col_to_encoder], numerical_cols, updated_categorical_cols, col_to_width_edges, col_to_depth_edges


def transform_test_columns(X_test, pipeline_results):
    """
    given a pandas dataframe this function returns it after passing through the same transformations as the train set
    :param X_test: pandas dataframe where we should apply what we've learned
    :param pipeline_results: class: 'PipelineResults'
    :return: dataframe transformed exactly the same way the train set have transformed
    """
    numerical_cols = list(set(pipeline_results.ingestion_results.numerical_cols).intersection(X_test.columns.tolist()))
    categorical_cols = list(
        set(pipeline_results.ingestion_results.categorical_cols).intersection(X_test.columns.tolist()))
    is_numerical = len(numerical_cols) > 0

    # discretization of numerical columns
    if is_numerical:
        numerical_df = X_test.loc[:, numerical_cols]
        equal_width_num_df, _, equal_depth_num_df, _ = _discretize(numerical_df,
                                                                   col_to_width_edges=pipeline_results.transformations_results.col_to_width_edges,
                                                                   col_to_depth_edges=pipeline_results.transformations_results.col_to_depth_edges)

    # add the discretized columns to categorical columns
    categorical_df = pd.concat(
        [df for df in [X_test.loc[:, categorical_cols], equal_width_num_df, equal_depth_num_df] if df is not None],
        axis=1) if is_numerical else X_test.loc[:, categorical_cols]
    num_transformations = pipeline_results.transformations_results.num_transformations
    cat_transformations = pipeline_results.transformations_results.cat_transformations
    label_encoded_df, dummiefied_categorical_df, col_to_encoder = \
        _transform_categorical_columns(categorical_df, col_to_encoder=cat_transformations[0])

    # add the encoded categorical columns to numerical columns
    numerical_df = pd.concat([X_test.loc[:, numerical_cols], label_encoded_df], axis=1)
    scaled_numerical_df, scaler = _transform_numerical_columns(numerical_df, col_to_scaler=num_transformations[0])
    transformed_df = pd.concat([scaled_numerical_df, dummiefied_categorical_df], axis=1)
    return transformed_df


@DeprecationWarning
def inverse_transform_columns(transformed_df, cat_transformations, num_transformations, pipeline_results):
    """
    given dataframe with columns ready for an ML model, categorical transformations list and numerical
    transformations list, returns the original dataframe before the transformations
    :param transformed_df: dataframe with columns ready for an ML model
    :param cat_transformations: categorical transformations list
    :param num_transformations: numerical transformations list
    :param pipeline_results: class: 'PipelineResults'
    :return: original dataframe before the transformations
    """
    numerical_cols = pipeline_results.ingestion_results.numerical_cols
    categorical_cols = pipeline_results.ingestion_results.categorical_cols
    X_cat = transformed_df[categorical_cols].copy()
    X_num = transformed_df[numerical_cols].copy()

    for transformation in cat_transformations:
        X_cat = transformation.inverse_transform(X_cat)
    for transformation in num_transformations:
        X_num = transformation.inverse_transform(X_num)

    return pd.concat([X_cat, X_num], axis=1)
