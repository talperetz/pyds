"""
:Authors: Tal Peretz
:Date: 11/24/2016
:TL;DR: this module is responsible for categorical and numerical columns transformations
"""

from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def transform_train_columns(X_train, pipeline_results, numerical_scaler=MinMaxScaler()):
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
    if numerical_scaler:
        if len(numerical_cols) > 0:
            transformed_num_cols = numerical_scaler.fit_transform(X_train[numerical_cols])
            transformed_num_cols = pd.DataFrame(data=transformed_num_cols, index=X_train.index, columns=numerical_cols)
            transformed_df = transformed_num_cols
            num_transformations = [numerical_scaler]
    else:
        transformed_num_cols = X_train[numerical_cols]

    if len(categorical_cols) > 0:
        col_to_encoder = defaultdict(LabelEncoder)

        # Encoding the variable
        transformed_cat_cols = X_train[categorical_cols].apply(lambda col: col_to_encoder[col.name].fit_transform(col))

        transformed_df = transformed_cat_cols
        cat_transformations = [col_to_encoder]

    if (len(numerical_cols) > 0) and (len(categorical_cols) > 0):
        transformed_df = pd.concat([transformed_cat_cols, transformed_num_cols], axis=1)
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
