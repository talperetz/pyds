""" 
:Authors: Tal Peretz
:Date: 10/14/2016
:TL;DR: this module is responsible for the transformation and creation and selection of features
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RandomizedLasso
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, LabelEncoder, OneHotEncoder, MinMaxScaler

from pyds import constants


def transform_train_columns(X_train, pipeline_results):
    """
    given a pandas DataFrame and a PipelineResults object
    returns a dataframe with columns ready for an ML model , categorical transformations list,
    numerical transformations list
    :param X_train: pandas DataFrame
    :param pipeline_results: class: 'PipelineResults'
    :return: dataframe with columns ready for an ML model, categorical transformations list,
    numerical transformations list
    """
    numerical_cols = list(set(pipeline_results.Ingestion.numerical_cols).intersection(X_train.columns.tolist()))
    categorical_cols = list(set(pipeline_results.Ingestion.categorical_cols).intersection(X_train.columns.tolist()))
    cat_encoder, num_scaler, num_transformations, cat_transformations = (None for i in range(4))
    transformed_nd = X_train.copy()
    if len(numerical_cols) > 0:
        num_scaler = MinMaxScaler()
        transformed_num_cols = num_scaler.fit_transform(X_train[numerical_cols])
        transformed_nd = transformed_num_cols
        num_transformations = [num_scaler]

    if len(categorical_cols) > 0:
        cat_encoder = LabelEncoder()
        transformed_cat_cols = X_train[categorical_cols].apply(lambda col: cat_encoder.fit_transform(col), axis=1)
        transformed_nd = transformed_cat_cols
        cat_transformations = [cat_encoder]

    if (len(numerical_cols) > 0) and (len(categorical_cols) > 0):
        transformed_cat_df = pd.DataFrame(data=transformed_cat_cols, columns=X_train[categorical_cols].columns)
        transformed_num_df = pd.DataFrame(data=transformed_num_cols, columns=X_train[numerical_cols].columns)
        transformed_nd = pd.concat([transformed_cat_df, transformed_num_df], axis=1)
    transformed_df = pd.DataFrame(data=transformed_nd, columns=X_train.columns)
    return transformed_df, num_transformations, cat_transformations


def transform_test_columns(X_test, pipeline_results):
    """
    given a pandas dataframe this function returns it after passing through the same transformations as the train set
    :param X_test: pandas dataframe where we should apply what we've learned
    :param pipeline_results: class: 'PipelineResults'
    :return: dataframe transformed exactly the same way the train set have transformed
    """
    numerical_cols = pipeline_results.Ingestion.numerical_cols
    categorical_cols = pipeline_results.Ingestion.categorical_cols
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
        x_test_categorical_cols = list(set(categorical_cols).intersection(transformed_df.columns.tolist()))
        transformed_cols = transformed_df[x_test_categorical_cols]
        for transformer in cat_transformations:
            transformed_cols = transformer.transform(transformed_cols)
        transformed_df[x_test_categorical_cols] = transformed_cols
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


def create_features(X, y, pipeline_results):
    """
    given an encoded and scaled pandas DataFrame returns dataframe with new features columns based on
    polynomial features combination, log transformation and one hot encoding
    :param y: [pandas Series] target column
    :param X: encoded and scaled pandas DataFrame
    :param pipeline_results: class: 'PipelineResults'
    :return: dataframe with new features columns based on polynomial features combination, log transformation
    and one hot encoding
    links: `feature engineering - getting good at it <http://machinelearningmastery.com/
    discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/>`_
    `Quora - feature engineering <https://www.quora.com/What-are-some-best-practices-in-Feature-Engineering>`_
    """

    numerical_cols = [col_name for col_name in pipeline_results.Ingestion.numerical_cols if col_name != y.name]
    categorical_cols = [col_name for col_name in pipeline_results.Ingestion.categorical_cols if
                        col_name != y.name]

    X_num = X[numerical_cols].copy()
    X_cat = X[categorical_cols].copy()
    created_features = set()
    poly_features, log_features, one_hot_features = None, None, None
    if len(X_num.columns) > 0:
        poly = PolynomialFeatures()
        log_transformer = FunctionTransformer(func=np.log)

        poly_features = pd.DataFrame(data=poly.fit_transform(X_num),
                                     columns=poly.get_feature_names(X_num.columns.tolist()), index=X_num.index)
        log_features = pd.DataFrame(data=log_transformer.fit_transform(X_num),
                                    columns=['log_%s' % feature for feature in X_num.columns.tolist()],
                                    index=X_num.index)
        # replace inf values with largest non-inf value * NEG_INF_REPRESENTATION
        replacements = {
            -np.inf: constants.NEG_INF_REPRESENTATION * abs(
                max([log_features.replace([np.inf, -np.inf, np.nan], 0).as_matrix().min(),
                     log_features.replace([np.inf, -np.inf, np.nan], 0).as_matrix().max()],
                    key=abs)), np.nan: constants.NEG_INF_REPRESENTATION * abs(
                max([log_features.replace([np.inf, -np.inf, np.nan], 0).as_matrix().min(),
                     log_features.replace([np.inf, -np.inf, np.nan], 0).as_matrix().max()],
                    key=abs))}
        column_to_replacements = {col: replacements for col in log_features.columns.tolist()}
        log_features = log_features.replace(column_to_replacements)

        created_features.update(poly_features.columns.tolist())
        created_features.update(log_features.columns.tolist())

    if len(X_cat.columns) > 0:
        one_hot = OneHotEncoder()
        one_hot_features = pd.DataFrame(data=one_hot.fit_transform(X_cat), index=X_cat.index)
        created_features.update(one_hot_features.columns.tolist())

    return pd.concat([df for df in [poly_features, log_features, one_hot_features] if df is not None],
                     axis=1), created_features


def _rank_to_dict(ranks, names, order=1):
    scaler = MinMaxScaler()
    ranks = scaler.fit_transform(order * np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))


def select_features(X, y):
    """
    given a pandas DataFrame and the column name of target variable returns dataframe after dropping irrelevant features
    according to RandomizedLasso and RandomForestRegressor feature selection
    :param y: [pandas Series] target column
    :param X: [pandas DataFrame] predictor columns
    :return: dataframe without meaningless features according to RandomizedLasso and RandomForestRegressor
    feature selection
    links: `Quora - feature selection <https://www.quora.com/How-do-I-perform-feature-selection>`_
    `selecting good features
    <http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/>`_
    """
    reduced_df = X.copy()
    names = X.columns.tolist()
    ranks = {}

    # ranking
    rf = RandomForestRegressor()
    rf.fit(X, y)
    ranks["RF"] = _rank_to_dict(rf.feature_importances_, names)

    rlasso = RandomizedLasso(alpha=0.04)
    rlasso.fit(X, y)
    ranks["Stability"] = _rank_to_dict(np.abs(rlasso.scores_), names)

    # dropping irrelevant features by two models
    rf_zeros = {i for i, score in enumerate(ranks["RF"]) if score == 0}
    rlasso_zeros = {i for i, score in enumerate(ranks["Stability"]) if score == 0}
    columns_to_drop = list(rf_zeros.intersection(rlasso_zeros))
    if columns_to_drop:
        reduced_df = X.drop(X.columns[columns_to_drop], axis=1)
    return reduced_df, reduced_df.columns
