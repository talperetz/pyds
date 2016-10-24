""" 
@author: Tal Peretz
@date: 10/14/2016
@TL;DR: this module is responsible for the transformation and creation and selection of features
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RandomizedLasso
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, LabelEncoder, StandardScaler, \
    OneHotEncoder, MinMaxScaler


def transform_variables(df, pipeline_results):
    """
    given a pandas DataFrame and a PipelineResults object
    returns a dataframe with columns ready for an ML model , categorical transformations list,
    numerical transformations list
    :param df: pandas DataFrame
    :param pipeline_results: class: 'PipelineResults'
    :return: dataframe with columns ready for an ML model, categorical transformations list,
    numerical transformations list
    """
    numerical_cols = pipeline_results.Exploration.numerical_cols
    categorical_cols = pipeline_results.Exploration.categorical_cols
    cat_encoder, cat_scaler, num_scaler = None, None, None
    transformed_nd = df
    if len(numerical_cols) > 0:
        num_scaler = StandardScaler()
        transformed_num_cols = num_scaler.fit_transform(df[numerical_cols])
        transformed_nd = transformed_num_cols

    if len(categorical_cols) > 0:
        cat_encoder = LabelEncoder()
        cat_scaler = StandardScaler()
        transformed_cat_cols = cat_scaler.fit_transform(cat_encoder.fit_transform(df[categorical_cols]))
        transformed_nd = transformed_cat_cols

    if (len(numerical_cols) > 0) & (len(categorical_cols) > 0):
        transformed_nd = pd.concat([transformed_cat_cols, transformed_num_cols], axis=1)
    transformed_df = pd.DataFrame(data=transformed_nd, columns=df.columns)
    return transformed_df, [num_scaler], [cat_encoder, cat_scaler]


def restore_variables(transformed_df, cat_transformations, num_transformations, pipeline_results):
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
    X_cat = transformed_df[categorical_cols]
    X_num = transformed_df[numerical_cols]

    for transformation in cat_transformations:
        X_cat = transformation.inverse_transform(X_cat)
    for transformation in num_transformations:
        X_num = transformation.inverse_transform(X_num)

    return pd.concat([X_cat, X_num], axis=1)


def create_features(transformed_df, target_column, pipeline_results):
    """
    given an encoded and scaled pandas DataFrame returns dataframe with new features columns based on
    polynomial features combination, log transformation and one hot encoding
    :param target_column: the column name of which variable we want to predict
    :param transformed_df: encoded and scaled pandas DataFrame
    :param pipeline_results: class: 'PipelineResults'
    :return: dataframe with new features columns based on polynomial features combination, log transformation
    and one hot encoding
    todo: create more complex ad-hoc features
    links: `feature engineering - getting good at it <http://machinelearningmastery.com/
    discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/>`_
    `Quora - feature engineering <https://www.quora.com/What-are-some-best-practices-in-Feature-Engineering>`_
    """

    numerical_cols = [col_name for col_name in pipeline_results.Exploration.numerical_cols if col_name != target_column]
    categorical_cols = [col_name for col_name in pipeline_results.Exploration.categorical_cols if
                        col_name != target_column]
    Y = transformed_df[target_column]
    X = transformed_df.drop(target_column, axis=1)
    X_num = X[numerical_cols]
    X_cat = X[categorical_cols]
    created_features = set()

    poly_features, log_features, one_hot_features = None, None, None

    if len(X_num.columns) > 0:
        poly = PolynomialFeatures()
        log_transformer = FunctionTransformer(func=np.log)
        poly_features = pd.DataFrame(data=poly.fit_transform(X_num),
                                     columns=poly.get_feature_names(X_num.columns.tolist()))
        log_features = pd.DataFrame(data=log_transformer.fit_transform(X_num),
                                    columns=['log_%s' % feature for feature in X_num.columns.tolist()])
        created_features.update(poly_features.columns.tolist())
        created_features.update(log_features.columns.tolist())

    if len(X_cat.columns) > 0:
        one_hot = OneHotEncoder()
        one_hot_features = one_hot.fit_transform(X_cat)
        created_features.update(one_hot_features.columns.tolist())
    return pd.concat(
        [df for df in [X, poly_features, log_features, one_hot_features, Y] if df is not None]), created_features


def _rank_to_dict(ranks, names, order=1):
    scaler = MinMaxScaler()
    ranks = scaler.fit_transform(order * np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))


def select_features(df, target_column, pipeline_results):
    """
    given a pandas DataFrame and the column name of target variable returns dataframe after dropping irrelevant features
     according to RandomizedLasso and RandomForestRegressor feature selection
    :param df: pandas DataFrame
    :param target_column: the column name of which variable we want to predict
    :return: dataframe without meaningless features according to RandomizedLasso and RandomForestRegressor
    feature selection
    todo: different approach for regression and classification tasks(?)
    links: `Quora - feature selection <https://www.quora.com/How-do-I-perform-feature-selection>`_
    `selecting good features
    <http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/>`_
    """

    Y = df[target_column]
    X = df.drop(target_column)
    names = X.columns.tolist()
    ranks = {}

    # ranking
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    ranks["RF"] = _rank_to_dict(rf.feature_importances_, names)

    rlasso = RandomizedLasso(alpha=0.04)
    rlasso.fit(X, Y)
    ranks["Stability"] = _rank_to_dict(np.abs(rlasso.scores_), names)

    # dropping irrelevant features by two models
    rf_zeros = {i for i, score in enumerate(ranks["RF"]) if score == 0}
    rlasso_zeros = {i for i, score in enumerate(ranks["Stability"]) if score == 0}
    columns_to_drop = rf_zeros.intersection(rlasso_zeros)
    reduced_df = df.drop(df.columns[columns_to_drop], axis=1)
    return reduced_df
