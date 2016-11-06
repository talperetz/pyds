""" 
:Authors: Tal Peretz
:Date: 10/14/2016
:TL;DR: this module is responsible for univariate and bi-variate analysis
"""

import pandas as pd
from matplotlib import pyplot as plt

from pyds import constants


def describe(X, pipeline_results, y=None, **kwargs):
    """
    given a pandas pandas DataFrame returns a pandas pandas DataFrame describing basic statistics about numeric columns,
     dropping id-columns
    :param y: [pandas Series] target column
    :param pipeline_results: class: 'PipelineResults'
    :param X: [pandas DataFrame] predictor columns
    :return: pandas pandas DataFrame describing basic statistics about numeric columns
    """
    df = X.copy()
    numerical_cols = pipeline_results.Ingestion.numerical_cols
    categorical_cols = pipeline_results.Ingestion.categorical_cols
    num_description, cat_description = None, None
    if y is not None:
        df[y.name] = y
    if len(numerical_cols) > 0:
        num_description = df[numerical_cols].describe(**kwargs)
    if len(categorical_cols) > 0:
        cat_description = df[categorical_cols].describe(**kwargs)
    return num_description, cat_description


def hist(X, pipeline_results, y=None, **kwargs):
    """
    given a pandas DataFrame plots a histogram for each numeric columns
    if a by=column keyword is passed splits the groups according to other column
    :param X: [pandas DataFrame] predictor columns
    :param pipeline_results: class: 'PipelineResults'
    :param y: [pandas Series] target column
    :param kwargs: passed to pandas.Series.hist
    """
    numerical_cols = pipeline_results.Ingestion.numerical_cols
    categorical_cols = pipeline_results.Ingestion.categorical_cols
    numerical_figures, categorical_figures = [], []
    df = X.copy()
    if y is not None:
        df[y.name] = y

    # numerical columns histogram plotting
    for i, col in enumerate(numerical_cols):
        numerical_figures.append(plt.figure(i))
        plt.suptitle(col)
        df[col].hist(**kwargs)

    # categorical columns histogram plotting
    for i, col in enumerate(categorical_cols):
        categorical_figures.append(plt.figure(i))
        plt.suptitle(col)
        df[col].value_counts().dropna().plot(kind='bar')

    return numerical_figures, categorical_figures


def box_plot(X, pipeline_results, y=None, **kwargs):
    """
    given a pandas DataFrame plots a boxplot for each numeric columns
    if a by=column keyword is passed splits the groups according to other column
    :param y: [pandas Series] target column
    :param pipeline_results: class: 'PipelineResults'
    :param X: [pandas DataFrame] predictor columns
    :param kwargs: passed to pandas.Series.hist
    """
    numerical_cols = pipeline_results.Ingestion.numerical_cols
    categorical_cols = pipeline_results.Ingestion.categorical_cols
    numerical_figures, categorical_figures = [], []
    df = X.copy()
    df[y.name] = y

    # numerical columns box_plot plotting
    for i, col in enumerate(numerical_cols):
        numerical_figures.append(plt.figure(i))
        plt.suptitle(col)
        df[col].plot(kind='box', **kwargs)

    # categorical columns box_plot plotting
    for i, col in enumerate(categorical_cols):
        categorical_figures.append(plt.figure(i))
        plt.suptitle(col)
        df[col].value_counts().dropna().plot(kind='box')

    return numerical_figures, categorical_figures


def contingency_table(X, pipeline_results, y=None):
    """
    given a pandas DataFrame and target_column
    returns a list of contingency tables per categorical column
    :param y: [pandas Series] target column
    :param pipeline_results: class: 'PipelineResults'
    :param X: [pandas DataFrame] predictor columns
    :return: list of contingency tables per categorical column
    """
    df = X.copy()
    contingency_tables = []
    categorical_cols = list(pipeline_results.Ingestion.categorical_cols)
    cross_col = "count"
    if y is not None:
        cross_col = y
    if y.name in categorical_cols:
        categorical_cols.remove(y.name)
    for col in categorical_cols:
        contingency_tables.append(
            pd.crosstab(df[col], cross_col, margins=True).apply(lambda column: column / column[-1], axis=1))
    return contingency_tables


def correlations(X, y=None, size=constants.FIGURE_SIZE):
    """
    given a pandas DataFrame returns correlation matrix and figure representing the correlations
    :param y: [pandas Series] target column
    :param X: [pandas DataFrame] predictor columns
    :param size: matplotlib figure size
    :return: correlation matrix and figure representing the correlations
    """
    df = X.copy()
    if y is not None:
        df[y.name] = y
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    return corr, fig
