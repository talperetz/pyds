"""
:Authors: Tal Peretz
:Date: 15/12/2016
:TL;DR: this module is responsible for the ML model evaluation
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter, defaultdict
from pyds import constants
from sklearn import metrics

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

import seaborn as sns
sns.set_style("whitegrid")


def evaluate_classification(y_true, y_pred, target_names=None):
    """
    given array of correct target values, array of estimated targets and classes names
    returns confusion matrix, classification report and roc curve
    :param y_true: array of ground truth (correct) target values
    :param y_pred: array of estimated targets as returned by a classifier
    :param target_names: Optional display names matching the labels (same order)
    :return: classification report, confusion matrix figure, roc figure
    """
    if target_names is None:
        target_names = [str(i) for i in range(len(y_true))]
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=target_names)

    # plot roc
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=2)
    roc_auc = metrics.auc(fpr, tpr)
    roc_fig = plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # plot confusion matrix
    plt.figure()
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    cm_fig = sns.heatmap(cm, annot=True, xticklabels=target_names, yticklabels=target_names)
    return cr, cm_fig, roc_fig


def evaluate_regression(y_true, y_pred):
    """
    given array of correct target values and array of estimated targets
    returns mean_absolute_error, mean_squared_error, median_absolute_error and r_squared_score
    :param y_true: array of ground truth (correct) target values
    :param y_pred: array of estimated targets as returned by a regressor
    :param target_names: Optional display names matching the labels (same order)
    :return: mean_absolute_error, mean_squared_error, median_absolute_error and r_squared_score
    """
    mean_abs_err = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    med_abs_err = median_absolute_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)

    return mean_abs_err, mse, med_abs_err, r_squared


def evaluate_clusters(X, labels_pred, algorithm_name, labels_true=None):
    """
    given pandas DataFrame and labels of each point returns dictionary of cluster_num to list of cluster items.
    :param X: pandas DataFrame
    :param labels_pred: numpy.ndarray with clustering labels of each point
    :param labels_true: numpy.ndarray with real labels of each point
    :return: dictionary {cluster_num: [cluster_item_1, ..., cluster_item_n]}
    """
    n_clusters_ = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
    clustering_metrics_df = pd.Series(data=None,
                                      index=['items', 'size', 'real_label_to_frequency'].extend(
                                          constants.CLUSTERING_METRICS),
                                      name=algorithm_name,
                                      columns=['cluster %s' % i for i in n_clusters_])

    # build dictionary of cluster_label to cluster_items
    cluster_num_to_items_in_cluster = defaultdict(list)
    cluster_num_to_real_labels_in_cluster = defaultdict(list)
    for i, cluster_label in enumerate(labels_pred):
        cluster_num_to_items_in_cluster[cluster_label].append(X[i])
        if labels_true:
            cluster_num_to_real_labels_in_cluster[cluster_label].append(labels_true[i])

    # fill clustering_metrics values
    for cluster_label in cluster_num_to_items_in_cluster:

        # add meta data
        cluster_size = len(cluster_items)
        cluster_items = cluster_num_to_items_in_cluster[cluster_label]
        clustering_metrics_df.loc['items', 'cluster %s' % cluster_label] = cluster_items
        clustering_metrics_df.loc['size', 'cluster %s' % cluster_label] = cluster_size
        if labels_true:

            # add {label_true: label_frequency}
            clustering_metrics_df.loc['real_label_to_frequency', 'cluster %s' % cluster_label] = {
                cluster_num: real_labels_count / float(cluster_size) for cluster_num, real_labels_count in
                dict(Counter(cluster_num_to_real_labels_in_cluster[cluster_label])).iteritems()}

            # add sklearn clustering quality metrics
            for metric in constants.CLUSTERING_METRICS:
                clustering_metrics_df.loc[metric, 'cluster %s' % cluster_label] = getattr(metrics, metric)(labels_true,
                                                                                                           labels_pred)
    return clustering_metrics_df
