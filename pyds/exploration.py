""" 
:Authors: Tal Peretz
:Date: 10/14/2016
:TL;DR: this module is responsible for univariate and bi-variate analysis
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
import numpy as np

from pyds import constants
from pyds import ml


def describe(X, y=None, **kwargs):
    """
    given a pandas pandas DataFrame returns a pandas DataFrame describing basic statistics about numeric columns
    :param y: [pandas Series] target column
    :param pipeline_results: class: 'PipelineResults'
    :param X: [pandas DataFrame] predictor columns
    :return: pandas DataFrame describing basic statistics about numeric columns
    """
    assert (isinstance(X, pd.DataFrame)) and (not X.empty), 'X should be a valid pandas DataFrame'
    df = X.copy()
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['category']).columns
    num_description, cat_description = None, None
    if y is not None:
        df[y.name] = y
    if len(numerical_cols) > 0:
        num_description = df[numerical_cols].describe(**kwargs)
    if len(categorical_cols) > 0:
        cat_description = df[categorical_cols].describe(**kwargs)
    return num_description, cat_description


def hist(X, y=None, **kwargs):
    """
    given a pandas DataFrame plots a histogram for each numeric columns
    if a by=column keyword is passed splits the groups according to other column
    :param X: [pandas DataFrame] predictor columns
    :param pipeline_results: class: 'PipelineResults'
    :param y: [pandas Series] target column
    :param kwargs: passed to pandas.Series.hist
    """
    assert (isinstance(X, pd.DataFrame)) and (not X.empty), 'X should be a valid pandas DataFrame'
    numerical_figures, categorical_figures = [], []
    df = X.copy()
    if y is not None:
        df[y.name] = y
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['category']).columns

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


def box_plot(X, y=None):
    """
    given a pandas DataFrame plots a boxplot for each numeric columns
    if a by=column keyword is passed splits the groups according to other column
    :param y: [pandas Series] target column
    :param pipeline_results: class: 'PipelineResults'
    :param X: [pandas DataFrame] predictor columns
    :param kwargs: passed to pandas.Series.hist
    """
    assert (isinstance(X, pd.DataFrame)) and (not X.empty), 'X should be a valid pandas DataFrame'
    df = X.copy()
    if y is not None:
        df[y.name] = y
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['category']).columns
    numerical_figure = plt.figure()
    sns.boxplot(x="distance", y="method", data=df[numerical_cols],
                whis=np.inf, color="c")
    sns.stripplot(x="distance", y="method", data=df[numerical_cols],
                  jitter=True, size=3, color=".3", linewidth=0)
    categorical_figure = plt.figure()
    sns.boxplot(x="distance", y="method", data=df[categorical_cols],
                whis=np.inf, color="c")
    sns.stripplot(x="distance", y="method", data=df[categorical_cols],
                  jitter=True, size=3, color=".3", linewidth=0)
    return numerical_figure, categorical_figure


def scatter_plot(X, y, **kwargs):
    """
    given a pandas DataFrame without categorical data plots scatterplot of the data for each
    reducer in ml.reduce_dimensions()
    :param X: [pandas DataFrame] predictor columns without categorical data
    :param kwargs: passed to pandas.Series.plot
    """
    assert (isinstance(X, pd.DataFrame)) and (not X.empty), 'X should be a valid pandas DataFrame'
    assert (isinstance(y, pd.Series)) and (not y.empty), 'X should be a valid pandas Series'
    df = X.copy()
    df[y.name] = y
    scatter_mat = sns.pairplot(df, hue=y.name, diag_kind="kde", plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                               diag_kws=dict(shade=True), **kwargs)
    return scatter_mat


def contingency_table(X, y=None):
    """
    given a pandas DataFrame and target_column
    returns a list of contingency tables per categorical column
    :param y: [pandas Series] target column
    :param pipeline_results: class: 'PipelineResults'
    :param X: [pandas DataFrame] predictor columns
    :return: list of contingency tables per categorical column
    """
    assert (isinstance(X, pd.DataFrame)) and (not X.empty), 'X should be a valid pandas DataFrame'
    df = X.copy()
    contingency_tables = []
    categorical_cols = df.select_dtypes(include=['category']).columns
    cross_col = "count"
    if y is not None:
        cross_col = y
        if y.name in categorical_cols:
            categorical_cols.remove(y.name)
    if len(categorical_cols) > 1:
        for col in categorical_cols:
            contingency_tables.append(
                pd.crosstab(df[col].values, cross_col, margins=True, normalize=True))
    elif len(categorical_cols) == 1:
        contingency_tables.append(pd.crosstab(df[categorical_cols], cross_col, margins=True, normalize=True))
    return contingency_tables


def correlations(X, y=None, size=constants.FIGURE_SIZE):
    """
    given a pandas DataFrame returns correlation matrix and figure representing the correlations
    :param y: [pandas Series] target column
    :param X: [pandas DataFrame] predictor columns
    :param size: matplotlib figure size
    :return: correlation matrix and figure representing the correlations
    """
    assert (isinstance(X, pd.DataFrame)) and (not X.empty), 'X should be a valid pandas DataFrame'
    df = X.copy()
    if y is not None:
        df[y.name] = y
    fig = plt.figure()
    corr = df.corr()
    sns.clustermap(corr, linewidths=.5, figsize=(13, 13))
    return corr, fig
