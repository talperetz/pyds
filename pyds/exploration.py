""" 
@author: Tal Peretz
@date: 10/14/2016
@TL;DR: this module is responsible for univariate and bi-variate analysis
"""

import pandas as pd
from matplotlib import pyplot as plt


# def _run_once(f):
#     """
#     internal decorator for running infer_columns_statistical_types only once, yet using results multiple times
#     :param f: function to decorate
#     :return: f which can only run once
#     """
#     def wrapper(*args, **kwargs):
#         if not wrapper.has_run:
#             wrapper.has_run = True
#             return f(*args, **kwargs)
#         else:
#             return numerical_cols, categorical_cols, id_cols
#
#     wrapper.has_run = False
#     return wrapper


# @_run_once
def infer_columns_statistical_types(df):
    """
    given a pandas DataFrame returns a lists of the dataframe's numerical columns, categorical columns, id columns
    :param df: pandas DataFrame
    :return: lists of the dataframe's numerical columns, categorical columns, id columns
    """
    dist_ratios = df.apply(pd.Series.nunique) / df.apply(pd.Series.count)
    id_cols = dist_ratios.where(dist_ratios == 1).dropna().index.tolist()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_cols = df.select_dtypes(include=numerics).columns.drop(id_cols).tolist()
    categorical_cols = df.columns.difference(numerical_cols).drop(id_cols).tolist()
    return numerical_cols, categorical_cols, id_cols


def describe(df, **kwargs):
    """
    given a pandas pandas DataFrame returns a pandas pandas DataFrame describing basic statistics about numeric columns,
     dropping id-columns
    :param df: pandas pandas DataFrame
    :return: pandas pandas DataFrame describing basic statistics about numeric columns
    """
    numerical_columns, categorical_columns, _ = infer_columns_statistical_types(df)
    num_description = None
    cat_description = None
    if len(numerical_columns) > 0:
        num_description = df[numerical_columns].describe(**kwargs)
        print(num_description)
    if len(categorical_columns) > 0:
        cat_description = df[categorical_columns].describe(**kwargs)
    return num_description, cat_description


def hist(df, **kwargs):
    """
    given a pandas DataFrame plots a histogram for each numeric columns
    if a by=column keyword is passed splits the groups according to other column
    :param df: pandas DataFrame
    :param kwargs: passed to pandas.Series.hist
    """
    numerical_columns, categorical_columns, _ = infer_columns_statistical_types(df)
    numerical_figures, categorical_figures = [], []

    # numerical columns histogram plotting
    for i, col in enumerate(numerical_columns):
        numerical_figures.append(plt.figure(i))
        plt.suptitle(col)
        df[col].hist(**kwargs)

    # categorical columns histogram plotting
    for i, col in enumerate(categorical_columns):
        categorical_figures.append(plt.figure(i))
        plt.suptitle(col)
        df[col].value_counts().dropna().plot(kind='bar')

    return numerical_figures, categorical_figures


def box_plot(df, **kwargs):
    """
    given a pandas DataFrame plots a boxplot for each numeric columns
    if a by=column keyword is passed splits the groups according to other column
    :param df: pandas DataFrame
    :param kwargs: passed to pandas.Series.hist
    """
    numerical_columns, categorical_columns, _ = infer_columns_statistical_types(df)
    numerical_figures, categorical_figures = [], []

    # numerical columns box_plot plotting
    for i, col in enumerate(numerical_columns):
        numerical_figures.append(plt.figure(i))
        plt.suptitle(col)
        df[col].plot(kind='box', **kwargs)

    # categorical columns box_plot plotting
    for i, col in enumerate(categorical_columns):
        categorical_figures.append(plt.figure(i))
        plt.suptitle(col)
        df[col].value_counts().dropna().plot(kind='box')

    return numerical_figures, categorical_figures


def contingency_table(df, target_column):
    """
    given a pandas DataFrame and target_column
    returns a list of contingency tables per categorical column
    :param df: pandas DataFrame
    :param target_column: column name of the target variable
    :return: list of contingency tables per categorical column
    """
    dfs_list = []
    numerical_columns, categorical_columns, _ = infer_columns_statistical_types(df)

    if target_column in categorical_columns:
        categorical_columns = categorical_columns.drop(target_column)

    for col in categorical_columns:
        dfs_list.append(
            pd.crosstab(df[col], df[target_column], margins=True).apply(lambda column: column / column[-1], axis=1))
    return dfs_list


def correlations(df, size=8):
    """
    given a pandas DataFrame returns correlation matrix and figure representing the correlations
    :param df: pandas DataFrame 
    :param size: matplotlib figure size
    :return: correlation matrix and figure representing the correlations
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    return corr, fig

# numerical_cols = []
# categorical_cols = []
# id_cols = []
