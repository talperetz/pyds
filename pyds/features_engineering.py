""" 
:Authors: Tal Peretz
:Date: 10/14/2016
:TL;DR: this module is responsible for the transformation and creation and selection of features
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RandomizedLasso
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, MinMaxScaler

from pyds import constants


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
    numerical_cols = list(set(pipeline_results.transformations_results.numerical_cols).difference([y.name]))
    X_num = X.loc[:, numerical_cols].copy()
    created_features = set()
    poly_features, log_features, one_hot_features = None, None, None
    if len(X_num.columns) > 0:
        poly = PolynomialFeatures()
        log_transformer = FunctionTransformer(func=np.log)
        poly_features = pd.DataFrame(data=poly.fit_transform(X_num),
                                     columns=poly.get_feature_names(X_num.columns.tolist()), index=X_num.index)
        log_features = pd.DataFrame(data=log_transformer.fit_transform(X_num),
                                    columns=('log_%s' % col_name for col_name in numerical_cols),
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
