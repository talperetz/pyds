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
        transformed_numerical_df = numerical_scaler.fit_transform(train_numerical_df)
        transformed_numerical_df = pd.DataFrame(data=transformed_numerical_df, index=train_numerical_df.index,
                                                columns=train_numerical_df.columns)
        return transformed_numerical_df, numerical_scaler
    else:
        return train_numerical_df, numerical_scaler


def preprocess_train_columns(X_train, pipeline_results, numerical_scaler=MinMaxScaler(), X_test=None):
    """
    given a pandas DataFrame and a PipelineResults object
    returns a dataframe with columns ready for an ML model , categorical transformations list,
    numerical transformations list
    :param numerical_scaler: numerical scaler to apply on each of the numerical columns
    :param X_train: pandas DataFrame
    :param pipeline_results: class: 'PipelineResults'
    :return: dataframe with columns ready for an ML model, categorical transformations list,
    numerical transformations list
    """
    numerical_cols = list(set(pipeline_results.Ingestion.numerical_cols).intersection(X_train.columns.tolist()))
    categorical_cols = list(set(pipeline_results.Ingestion.categorical_cols).intersection(X_train.columns.tolist()))
    is_numerical = len(numerical_cols) > 0

    # if there are columns with high variance of values - group to categories according to equal width
    if is_numerical:
        numerical_df = X_train.loc[:, numerical_cols]
        equal_width_num_df = numerical_df.apply(lambda col: pd.cut(col, _calc_optimal_bin_size(col) + 1, labels=False))
        equal_width_num_df.columns = ['equal_w_%s' % col for col in equal_width_num_df.columns]
        # and equal depth
        equal_depth_num_df = numerical_df.apply(
            lambda col: _pct_rank_qcut(col, _calc_optimal_bin_size(col)))
        equal_depth_num_df.columns = ['equal_d_%s' % col for col in equal_depth_num_df.columns]

    # encode, dummify and add the encoded to numerical columns
    categorical_df = pd.concat([X_train.loc[:, categorical_cols], equal_width_num_df, equal_depth_num_df],
                               axis=1) if is_numerical else X_train.loc[:, categorical_cols]

    if X_test:
        assert set(categorical_df.columns).issubset(X_test.columns)
        label_encoded_df, dummiefied_categorical_df, col_to_encoder = \
            _transform_categorical_columns(categorical_df, X_test.loc[:, categorical_cols])
    else:
        label_encoded_df, dummiefied_categorical_df, col_to_encoder = _transform_categorical_columns(categorical_df)
    numerical_df = pd.concat([X_train.loc[:, numerical_cols], label_encoded_df], axis=1)
    scaled_numerical_df, numerical_scaler = _transform_numerical_columns(numerical_df, numerical_scaler)
    transformed_df = pd.concat([scaled_numerical_df, dummiefied_categorical_df], axis=1)
    return transformed_df, numerical_scaler, col_to_encoder


def transform_train_columns2(X_train, pipeline_results, numerical_scaler=MinMaxScaler(), X_test=None):
    """
    given a pandas DataFrame and a PipelineResults object
    returns a dataframe with columns ready for an ML model , categorical transformations list,
    numerical transformations list
    :param numerical_scaler: numerical scaler to apply on each of the numerical columns
    :param X_train: pandas DataFrame
    :param pipeline_results: class: 'PipelineResults'
    :return: dataframe with columns ready for an ML model, categorical transformations list,
    numerical transformations list
    """
    numerical_cols = list(set(pipeline_results.Ingestion.numerical_cols).intersection(X_train.columns.tolist()))
    categorical_cols = list(set(pipeline_results.Ingestion.categorical_cols).intersection(X_train.columns.tolist()))
    cat_encoder, num_transformations, cat_transformations = (None for _ in range(3))
    transformed_df = X_train.copy()

    # todo:
    # if there are columns with high variance of values - group to categories according to equal width
    # and equal depth

    if len(categorical_cols) > 0:
        categorical_df = X_train.loc[:, categorical_cols]
        full_cat_df = categorical_df
        # assume there's a natural order in the categories, encode them accordingly and add to numerical columns
        # because we want to scale the encoded columns as we scale the numerical columns
        if X_test:
            assert set(categorical_df.columns).issubset(X_test.columns)
            full_cat_df = pd.concat([categorical_df, X_test.loc[:, categorical_cols]], ignore_index=True)
        ordered_full_categorical_df = full_cat_df.apply(lambda col: col.astype('category', ordered=True))
        col_to_encoder = defaultdict(LabelEncoder)
        ordered_full_categorical_df.apply(
            lambda col: col_to_encoder[col.name].fit(col.sort_values()))
        label_encoded_categorical_df = categorical_df.apply(
            lambda col: col_to_encoder[col.name].transform(col.sort_values()))
        label_encoded_categorical_df.columns = ['ordered_%s' % col for col in label_encoded_categorical_df.columns]
        numerical_cols.extend(label_encoded_categorical_df.columns)

        # assume there is no order - dummify categorical data
        dummiefied_categorical_df = pd.get_dummies(X_train[categorical_cols],
                                                   prefix={col: col for col in categorical_df.columns})
        transformed_categorical_df = pd.concat([label_encoded_categorical_df, dummiefied_categorical_df], axis=1)
        transformed_df = transformed_categorical_df
        cat_transformations = [col_to_encoder]

    if len(numerical_cols) > 0:
        X_train_num = X_train.loc[:, set(X_train.columns).intersection(numerical_cols)]
        transformed_cat = transformed_df.loc[:,
                          set(transformed_df.columns).intersection(numerical_cols).difference(X_train_num.columns)]
        numerical_df = pd.concat([X_train_num, transformed_cat], axis=1).loc[:, numerical_cols]
        if numerical_scaler:
            transformed_numerical_df = numerical_scaler.fit_transform(numerical_df)
            transformed_numerical_df = pd.DataFrame(data=transformed_numerical_df, index=X_train.index,
                                                    columns=numerical_cols)
            transformed_df = transformed_numerical_df
            num_transformations = [numerical_scaler]
        else:
            transformed_numerical_df = numerical_df
            transformed_df = transformed_numerical_df

        if (len(numerical_cols) > 0) and (len(categorical_cols) > 0):
            transformed_df = pd.concat([transformed_categorical_df, transformed_numerical_df], axis=1)
    return transformed_df, num_transformations, cat_transformations


def transform_test_columns(X_test, pipeline_results):
    """
    given a pandas dataframe this function returns it after passing through the same transformations as the train set
    :param X_test: pandas dataframe where we should apply what we've learned
    :param pipeline_results: class: 'PipelineResults'
    :return: dataframe transformed exactly the same way the train set have transformed
    """
    numerical_cols = pipeline_results.Ingestion.numerical_cols
    num_transformations = pipeline_results.Features.num_transformations
    cat_transformations = pipeline_results.Features.cat_transformations
    transformed_df = X_test.copy()
    if (num_transformations is not None) and num_transformations:
        x_test_numerical_cols = list(set(numerical_cols).intersection(transformed_df.columns.tolist()))
        transformed_cols = transformed_df[x_test_numerical_cols]
        for transformer in num_transformations:
            transformed_cols = transformer.transform(transformed_cols)
        transformed_df[x_test_numerical_cols] = transformed_cols
    if (cat_transformations is not None) and any(cat_transformations):
        for col_to_transformer in cat_transformations:
            for col_name in col_to_transformer:
                transformer = col_to_transformer[col_name]
                transformed_df[col_name] = transformer.transform(transformed_df[col_name])
                # if there's a new value in the test set, get the closest value the transformer knows
                # except ValueError:
                #     new_val_to_closest_old_val = {}
                #     for index, new_val in transformed_df[col_name].iteritems():
                #         if new_val not in transformer.classes_:
                #             closest_old_val = get_close_matches(new_val, transformer.classes_, n=1, cutoff=0)[0]
                #             new_val_to_closest_old_val[new_val] = closest_old_val
                #     transformed_df[col_name] = transformer.transform(
                #         transformed_df[col_name].replace(new_val_to_closest_old_val))
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
