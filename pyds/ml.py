""" 
:Authors: Tal Peretz
:Date: 10/21/2016
:TL;DR: this module is responsible for the data science classic model-based solutions implementation
:Links: https://www.analyticsvidhya.com/blog/2015/01/scikit-learn-python-machine-learning-tool/
"""

from collections import defaultdict, Counter

import pandas as pd
from orangecontrib.associate.fpgrowth import association_rules, frequent_itemsets, rules_stats
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, DBSCAN, AffinityPropagation
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, IsolationForest
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import SGDRegressor, Lasso, ElasticNet, Ridge
from sklearn.manifold import Isomap, SpectralEmbedding, LocallyLinearEmbedding
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

from pyds import constants


def classify(X_train, X_test, y_train, y_test):
    """
    given the train and test set with their labels returns the best_classifier according to the metric,
    it's predictions on the test set and it's metric score
    :param X_train: training dataframe
    :param y_train: training true labels (target var)
    :param X_test: test dataframe
    :param y_test: test true labels (target var)
    :return: the best_classifier according to the metric, it's predictions on the test set and it's metric score
    """
    classifiers_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
                         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                         "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    clf_to_score = dict()
    # iterate over classifiers
    for name, clf in zip(classifiers_names, classifiers):
        clf.fit(X_train, y_train)
        clf_to_score[clf] = clf.score(X_test, y_test)

    best_clf = max(clf_to_score, key=clf_to_score.get)
    return best_clf, best_clf.predict(X_test), clf_to_score[best_clf]


def regress(X_train, X_test, y_train, y_test):
    """
    given the train and test set with their labels returns the best_classifier according to the metric,
    it's predictions on the test set and it's metric score
    :param X_train: training dataframe
    :param y_train: training true labels (target var)
    :param X_test: test dataframe
    :param y_test: test true labels (target var)
    :return: the best_regressor according to the metric, it's predictions on the test set and it's metric score
    """
    regressors_names, regressors, regressor_to_score = set(), set(), dict()
    if len(X_train.index) > 100000:
        sgd = SGDRegressor(alpha=0.0001, average=False, epsilon=0.1, eta0=0.01,
                           fit_intercept=True, l1_ratio=0.15, learning_rate='invscaling',
                           loss='squared_loss', n_iter=5, penalty='l2', power_t=0.25,
                           random_state=None, shuffle=True, verbose=0, warm_start=False).fit(X_train, y_train)
        regressors_names.add("SGDRegressor")
        regressors.add(sgd)
    else:
        lasso = Lasso(alpha=0.1).fit(X_train, y_train)
        enet = ElasticNet().fit(X_train, y_train)
        ridge = Ridge().fit(X_train, y_train)
        linear_svr = SVR(kernel='linear').fit(X_train, y_train)
        rbf_svr = SVR(kernel='rbf').fit(X_train, y_train)
        gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                        max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
        rf_regressor = RandomForestRegressor().fit(X_train, y_train)
        regressors_names.update(["Lasso", "ElasticNet", "Ridge", "linear SVR", "rbf SVR", "GradientBoostingRegressor",
                                 "RandomForestRegressor"])
        regressors.update([lasso, enet, ridge, linear_svr, rbf_svr, gbr, rf_regressor])
    for regressor in regressors:
        regressor_to_score[regressor] = regressor.score(X_test, y_test)

    best_regressor = max(regressor_to_score, key=regressor_to_score.get)
    return best_regressor, best_regressor.predict(X_test), regressor_to_score[best_regressor]


def _analyze_clusters(X, labels_pred, algorithm_name, labels_true=None):
    """
    given pandas DataFrame and labels of each point returns dictionary of cluster_num to list of cluster items
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


def create_clusters(df, cluster_cols, n_clusters=None, labels_true=None):
    """
    given a dataframe, relevant columns for clustering and num of clusters [optional]
    returns a dictionary of clustering algorithm to it's name, labels and metrics
    :param df: the dataframe upon we want to perform clustering
    :param cluster_cols: relevant columns for clustering
    :param n_clusters: num of clusters if known
    :return: dictionary of clustering algorithm to it's name, labels and metrics
    """
    X = df[cluster_cols]
    clustering_names, clustering_algorithms, clusterer_to_results = set(), set(), dict()
    if n_clusters is not None:
        if len(df.index) > 10000:
            k_means = KMeans(n_clusters=n_clusters).fit(X)
            spectral = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack',
                                          affinity="nearest_neighbors").fit(X)
            mixture = GaussianMixture(n_components=n_clusters, covariance_type='full').fit(X)
            clustering_names.update(['K-Means', 'SpectralClustering', 'GaussianMixture'])
            clustering_algorithms.update([k_means, spectral, mixture])
        else:
            mini_k_means = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters,
                                           n_init=10, max_no_improvement=10, verbose=0).fit(X)
            clustering_names.add("MiniBatchKMeans")
            clustering_algorithms.add(mini_k_means)
    else:
        dbs = DBSCAN(eps=0.3, min_samples=10).fit(X)
        af = AffinityPropagation().fit(X)
        clustering_names.update(["DBSCAN", "AffinityPropagation"])
        clustering_algorithms.update([dbs, af])
    for name, clusterer in zip(clustering_names, clustering_algorithms):
        labels_pred = clusterer.labels_
        metrics_ = _analyze_clusters(X, labels_pred, name, labels_true)
        clusterer_to_results[clusterer] = name, labels_pred, metrics_
    return clusterer_to_results


def reduce_dimensions(df, reduce_cols=None, n_components=None):
    """
    given a dataframe, columns to reduce and number of components for dimensionality reduction algorithm
    returns a dictionary of reduction algorithm to it's name and reduced df
    :param df: pandas dataframe
    :param reduce_cols: columns to perform dimensionality reduction on
    :param n_components: number of components for dimensionality reduction algorithm
    :return: dictionary of reduction algorithm to it's name and reduced df
    """
    if reduce_cols and (set(reduce_cols).issubset(set(df.columns.tolist()))) and (
                len(df[reduce_cols].index) > 0):
        X = df[reduce_cols].copy()
    else:
        X = df.copy()
    if not n_components:
        n_components = 2
    reductions_names, reductions_algorithms = set(), set()

    pca = PCA(n_components=n_components, svd_solver='randomized')
    reductions_names.add("PCA")
    reductions_algorithms.add(pca)
    if len(X.index) > 10000:
        k_pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
        reductions_names.add("KernelPCA")
        reductions_algorithms.add(k_pca)
    else:
        n_neighbors = 10
        isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        se = SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors)
        lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, method='standard')
        reductions_names.update(["Isomap", "SpectralEmbedding", "LocallyLinearEmbedding"])
        reductions_algorithms.update([isomap, se, lle])
    reducer_to_results = {}
    for name, reducer in zip(reductions_names, reductions_algorithms):
        reduced_df = reducer.fit_transform(X)
        reducer_to_results[reducer] = name, reduced_df
    return reducer_to_results


# http://aimotion.blogspot.co.il/2013/01/machine-learning-and-data-mining.html
# https://github.com/biolab/orange3-associate
"""
Association analysis is the task of finding interesting relationships in large data sets.
There hidden relationships are then expressed as a collection of association rules and frequent item sets.
Frequent item sets are simply a collection of items that frequently occur together.
And association rules suggest a strong relationship that exists between two items.
"""


def associate_rules(df, min_support, min_confidence):
    """
    given a pandas DataFrame, minimum support level and minimum confidence level
    returns dataframe with rules and statistics
    :param df: pandas DataFrame
    :param min_support: (float or int) – If float in range (0, 1), percent of minimal support for itemset to be considered
    frequent. If int > 1, the absolute number of instances. For example, general iterators don’t have defined length,
    so you need to pass the absolute minimal support as int.
    :param min_confidence: (float) – Confidence percent. Defined as itemset_support / antecedent_support.
    :return: pandas DataFrame with columns 'antecedent', 'consequent', 'support', 'confidence', 'coverage', 'strength',
    'lift', 'leverage'
    """
    matrix = df.to_matrix()
    itemsets = frequent_itemsets(matrix, min_support)
    rules = list(association_rules(itemsets, min_confidence))
    rstats = list(rules_stats(rules, itemsets, df.shape[0]))
    rules_df = pd.DataFrame(data=rstats, columns=['antecedent', 'consequent', 'support', 'confidence', 'coverage',
                                                  'strength', 'lift', 'leverage'])
    return rules_df


def detect_anomalies(X, y=None, contamination=0.1):
    """
    given a pandas DataFrame returns dataframe with contamination*num of instances
    rows dropped using isolation forest to detect outliers
    :param y: [pandas series] target column
    :param X: [pandas DataFrame] raw features
    :param contamination:  the proportion of outliers in the data set
    :return: outliers indexes
    """
    df = X.copy()
    if y is not None:
        df[y.name] = y
    clf = IsolationForest(max_samples=len(df.index), n_estimators=constants.ISOLATION_FOREST_N_ESTIMATORS,
                          contamination=contamination)
    clf.fit(df)
    Y = clf.predict(df)
    outliers = []
    for i, is_outlier in enumerate(Y):
        if is_outlier == -1:
            outliers.append(i)
    return df.index[outliers]
