"""
:Authors: Tal Peretz
:Date: 11/24/2016
:TL;DR: this module is responsible for categorical and numerical columns transformations
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def _calc_optimal_bin_size(col):
    iqr = np.subtract(*np.percentile(col, [75, 25]))
    h = int(np.ceil((2 * iqr) / (len(col) ** (1 / 3))))
    return h


def _pct_rank_qcut(series, n):
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    return series.rank(pct=1).apply(lambda x: (edges >= x).argmax())


def _encode_categorical_columns(encode_df, expand_fit_df=None):
    if expand_fit_df:
        assert set(encode_df.columns).issubset(expand_fit_df.columns)
        expand_fit_df = pd.concat([encode_df, expand_fit_df], ignore_index=True)
    else:
        expand_fit_df = encode_df
    ordered_fit_df = expand_fit_df.apply(lambda col: col.astype('category', ordered=True))
    col_to_encoder = defaultdict(LabelEncoder)
    ordered_fit_df.apply(
        lambda col: col_to_encoder[col.name].fit(col.sort_values()))
    label_encoded_df = encode_df.apply(
        lambda col: col_to_encoder[col.name].transform(col.sort_values()))
    label_encoded_df.columns = ['ordered_%s' % col for col in label_encoded_df.columns]
    return label_encoded_df, col_to_encoder


def _transform_categorical_columns(train_categorical_df, test_categorical_df=None):
    # assume there's an order - encode according to sort values
    label_encoded_df, col_to_encoder = _encode_categorical_columns(encode_df=train_categorical_df,
                                                                   expand_fit_df=test_categorical_df)

    # assume there is no order - dummify categorical data
    train_categorical_df = train_categorical_df.apply(lambda col: col.astype('category', ordered=False))
    dummiefied_categorical_df = pd.get_dummies(train_categorical_df,
                                               prefix=train_categorical_df.columns.tolist())
    return label_encoded_df, dummiefied_categorical_df, col_to_encoder


def _transform_numerical_columns(train_numerical_df, numerical_scaler):
    if numerical_scaler:
        col_to_scaler = defaultdict(numerical_scaler)
        # todo: take care of deprecation warning
        transformed_numerical_df = train_numerical_df.apply(
            lambda col: col_to_scaler[col.name].fit_transform(col))
        transformed_numerical_df = pd.DataFrame(data=transformed_numerical_df, index=train_numerical_df.index,
                                                columns=train_numerical_df.columns)
        return transformed_numerical_df, col_to_scaler
    else:
        return train_numerical_df, None


def _discretize(numerical_df):
    if isinstance(numerical_df, pd.DataFrame):
        equal_width_num_df = numerical_df.apply(
            lambda col: pd.cut(col, _calc_optimal_bin_size(col), labels=False) if _calc_optimal_bin_size(
                col) > 0 else None).dropna(axis=1)
        equal_width_num_df.columns = ['equal_w_%s' % col for col in equal_width_num_df.columns]
        # and equal depth
        equal_depth_num_df = numerical_df.apply(
            lambda col: _pct_rank_qcut(col, _calc_optimal_bin_size(col)) if _calc_optimal_bin_size(
                col) > 0 else None).dropna(axis=1)
        equal_depth_num_df.columns = ['equal_d_%s' % col for col in equal_depth_num_df.columns]

    elif isinstance(numerical_df, pd.Series):
        equal_width_num_df = pd.cut(numerical_df.values, _calc_optimal_bin_size(numerical_df),
                                    labels=False) if _calc_optimal_bin_size(numerical_df) > 0 else None
        equal_depth_num_df = _pct_rank_qcut(numerical_df,
                                            _calc_optimal_bin_size(numerical_df)) if _calc_optimal_bin_size(
            numerical_df) > 0 else None
    return equal_width_num_df, equal_depth_num_df


def preprocess_train_columns(X_train, pipeline_results, col_to_scaler=MinMaxScaler, X_test=None,
                             update_columns_types=False):
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
    numerical_cols = list(set(pipeline_results.Ingestion.numerical_cols).intersection(X_train.columns.tolist()))
    categorical_cols = list(set(pipeline_results.Ingestion.categorical_cols).intersection(X_train.columns.tolist()))
    is_numerical = len(numerical_cols) > 0

    # discretization of numerical columns
    if is_numerical:
        numerical_df = X_train.loc[:, numerical_cols]
        equal_width_num_df, equal_depth_num_df = _discretize(numerical_df)

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
    if update_columns_types:
        pipeline_results.update_categorical_cols(
            categorical_df.columns.tolist() + dummiefied_categorical_df.columns.tolist())
        pipeline_results.update_numerical_cols(numerical_df.columns.tolist())
    scaled_numerical_df, col_to_scaler = _transform_numerical_columns(numerical_df, col_to_scaler)
    transformed_df = pd.concat([scaled_numerical_df, dummiefied_categorical_df], axis=1)
    return transformed_df, [col_to_scaler], [col_to_encoder]


def transform_test_columns(X_test, pipeline_results):
    """
    given a pandas dataframe this function returns it after passing through the same transformations as the train set
    :param X_test: pandas dataframe where we should apply what we've learned
    :param pipeline_results: class: 'PipelineResults'
    :return: dataframe transformed exactly the same way the train set have transformed
    """
    # todo: get raw numerical and categorical columns
    # todo: _discretize numerical columns and add to categorical
    # todo: transform cat according to the fitted transformers add dummified columns
    # todo: scale numerical data + label_encoded data
    num_transformations = pipeline_results.Features.num_transformations
    cat_transformations = pipeline_results.Features.cat_transformations
    transformed_df = X_test.copy()
    if num_transformations:
        for col_to_transformer in num_transformations:
            for col_name in col_to_transformer:
                if col_name in transformed_df.columns:
                    transformer = col_to_transformer[col_name]
                    transformed_df.loc[:, col_name] = transformer.transform(transformed_df[col_name])
    if cat_transformations:
        for col_to_transformer in cat_transformations:
            for col_name in col_to_transformer:
                if col_name in transformed_df.columns:
                    transformer = col_to_transformer[col_name]
                    transformed_df.loc[:, col_name] = transformer.transform(transformed_df[col_name])
    return transformed_df


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
    numerical_cols = pipeline_results.Exploration.numerical_cols
    categorical_cols = pipeline_results.Exploration.categorical_cols
    X_cat = transformed_df[categorical_cols].copy()
    X_num = transformed_df[numerical_cols].copy()

    for transformation in cat_transformations:
        X_cat = transformation.inverse_transform(X_cat)
    for transformation in num_transformations:
        X_num = transformation.inverse_transform(X_num)

    return pd.concat([X_cat, X_num], axis=1)
