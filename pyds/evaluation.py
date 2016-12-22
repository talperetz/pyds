"""
:Authors: Tal Peretz
:Date: 15/12/2016
:TL;DR: this module is responsible for the ML model evaluation
"""

from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, mean_absolute_error, \
    mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import label_binarize

from pyds import constants

sns.set_style("whitegrid")


def evaluate_classification(y_true, y_pred, target_values, y_scores=None):
    """
    given array of correct target values, array of estimated targets and classes names
    returns confusion matrix, classification report and roc curve
    :param y_true: array of ground truth (correct) target values
    :param y_pred: array of estimated targets as returned by a classifier
    :param target_values: Optional display names matching the labels (same order)
    :return: classification report, confusion matrix figure, roc figure
    """
    roc_fig = None
    target_names = [str(val) for val in target_values]
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=target_names)

    # build multiclass ROC
    if y_scores is not None:
        # plot roc
        y = label_binarize(y_true, classes=target_values)

        # Compute ROC curve and ROC area for each class
        for i, class_name in enumerate(target_names):
            fpr, tpr, _ = roc_curve(y[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=4, label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(class_name, roc_auc))
        roc_fig = plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Model')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC', fontsize=20)
        plt.legend(loc="lower right")

    # plot confusion matrix
    plt.figure()
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.suptitle('Confusion Matrix', fontsize=20)
    cm_fig = sns.heatmap(cm, annot=True, xticklabels=target_names, yticklabels=target_names)
    return cr, cm_fig, roc_fig


def evaluate_regression(y_true, y_pred):
    """
    given array of correct target values and array of estimated targets
    returns mean_absolute_error, mean_squared_error, median_absolute_error and r_squared_score
    :param y_true: array of ground truth (correct) target values
    :param y_pred: array of estimated targets as returned by a regressor
    :return: mean_absolute_error, mean_squared_error, median_absolute_error and r_squared_score
    """
    mean_abs_err = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    med_abs_err = median_absolute_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)
    return mean_abs_err, mse, med_abs_err, r_squared


def evaluate_clusters(labels_pred, algorithm_name, labels_true=None):
    """
    given pandas DataFrame and labels of each point returns dictionary of cluster_num to list of cluster items.
    :param algorithm_name: string for clustering algorithm name
    :param labels_pred: numpy.ndarray with clustering labels of each point
    :param labels_true: numpy.ndarray with real labels of each point
    :return: dictionary {cluster_num: [cluster_item_1, ..., cluster_item_n]}
    """
    n_clusters_ = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
    clustering_metrics_df = pd.Series(data=None,
                                      index=['items', 'size', 'real_label_to_frequency'].extend(
                                          constants.CLUSTERING_METRICS),
                                      name=algorithm_name,
                                      columns=['cluster %s' % i for i in range(n_clusters_)])

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
        cluster_items = cluster_num_to_items_in_cluster[cluster_label]
        cluster_size = len(cluster_items)
        clustering_metrics_df.loc['items', 'cluster %s' % cluster_label] = cluster_items
        clustering_metrics_df.loc['size', 'cluster %s' % cluster_label] = cluster_size
        if labels_true:

            # add {label_true: label_frequency}
            clustering_metrics_df.loc['real_label_to_frequency', 'cluster %s' % cluster_label] = {
                cluster_num: real_labels_count / float(cluster_size) for cluster_num, real_labels_count in
                dict(Counter(cluster_num_to_real_labels_in_cluster[cluster_label])).items()}

            # add sklearn clustering quality metrics
            for metric in constants.CLUSTERING_METRICS:
                clustering_metrics_df.loc[metric, 'cluster %s' % cluster_label] = getattr(metrics, metric)(labels_true,
                                                                                                           labels_pred)
    return clustering_metrics_df
