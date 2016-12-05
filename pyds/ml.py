""" 
:Authors: Tal Peretz
:Date: 10/21/2016
:TL;DR: this module is responsible for the data science classic model-based solutions implementation
:Links: https://www.analyticsvidhya.com/blog/2015/01/scikit-learn-python-machine-learning-tool/
"""

from collections import defaultdict, Counter

import pandas as pd
from hyperopt import fmin, tpe, hp
from orangecontrib.associate.fpgrowth import association_rules, frequent_itemsets, rules_stats
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, DBSCAN, AffinityPropagation
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, IsolationForest
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDRegressor, Lasso, ElasticNet, Ridge
from sklearn.manifold import Isomap, SpectralEmbedding, LocallyLinearEmbedding
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

from pyds import constants


class MLModel:
    """
    machine learning model object
    """
    name, implementation, param_space = (None for _ in range(3))
    X, y, best_params, best_score, predictions = (None for _ in range(5))
    scoring = 'accuracy'

    def __init__(self, name, implementation, param_space):
        self.name = name
        self.implementation = implementation
        self.param_space = param_space

    def _hyperopt_function(self, params):
        model = self.implementation(**params)
        return cross_val_score(model, self.X, self.y, scoring=self.scoring).mean()

    def optimize(self, X_train, y_train, scoring, max_evals=500):
        """
        given a loss function find the params [in the param_space] that minimize it
        and update the best_params, best_score and implementation to use best_params
        :param y_train:
        :param X_train:
        :param scoring: string that controls what metric would the model use for evaluation
        :param max_evals: limit to the num of evaluation
        :param loss_func: function to minimize (for maximization negate function)
        """
        self.X = X_train
        self.y = y_train
        self.scoring = scoring
        self.best_params = fmin(self._hyperopt_function, self.param_space,
                                algo=tpe.suggest, max_evals=max_evals)
        self.best_score = self._hyperopt_function(self.best_params)
        self.implementation = self.implementation(self.best_params)
        return self

    def predict(self, X_test):
        """
        given a pandas dataframe returns the model's predictions as np.array
        :param X_test: test dataframe
        :return: predictions as np.array
        """
        return self.implementation.predict(X_test)


def classify(X_train, X_test, y_train, scoring='accuracy'):
    """
    given the train and test set with their labels returns the best_classifier according to the metric,
    it's predictions on the test set and it's metric score.
    classification is the problem of identifying to which of a set of categories (sub-populations) a new observation
    belongs, on the basis of a training set of data containing observations (or instances) whose category membership
    is known.
    :param scoring: model evaluation definition (from http://scikit-learn.org/stable/modules/model_evaluation.html)
    :param X_train: training dataframe
    :param y_train: training true labels (target var)
    :param X_test: test dataframe
    :param y_test: test true labels (target var)
    :return: the best_classifier according to the metric, it's predictions on the test set and it's metric score
    """
    # models configurations
    knn = MLModel('KNeighborsClassifier', KNeighborsClassifier, {
        'n_neighbors': hp.choice('n_neighbors', range(1, 50)),
        'scale': hp.choice('scale', [0, 1]),
        'normalize': hp.choice('normalize', [0, 1])
    })
    svc = MLModel('SVC', SVC, {
        'C': hp.uniform('C', 0, 20),
        'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
        'gamma': hp.uniform('gamma', 0, 20),
        'scale': hp.choice('scale', [0, 1]),
        'normalize': hp.choice('normalize', [0, 1])
    })
    gp = MLModel('GaussianProcessClassifier', GaussianProcessClassifier, {
        'warm_start': hp.choice('warm_start', [True, False])
    })
    rf = MLModel('RandomForestClassifier', RandomForestClassifier, {
        'max_depth': hp.choice('max_depth', range(1, 20)),
        'max_features': hp.choice('max_features', range(1, 5)),
        'n_estimators': hp.choice('n_estimators', range(1, 20)),
        'criterion': hp.choice('criterion', ["gini", "entropy"]),
        'scale': hp.choice('scale', [0, 1]),
        'normalize': hp.choice('normalize', [0, 1])
    })
    tree = MLModel('DecisionTreeClassifier', DecisionTreeClassifier, {
    })
    mlp = MLModel('MLPClassifier', MLPClassifier, {
        'alpha': hp.choice('alpha', range(0, 10))
    })
    ada = MLModel('AdaBoostClassifier', AdaBoostClassifier, {
    })
    gnb = MLModel('GaussianNB', GaussianNB, {
    })
    qda = MLModel('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis, {
    })

    # models competition
    competing_models = [knn, svc, gp, tree, rf, mlp, ada, gnb, qda]
    model_to_best_score = {model: model.optimize(X_train=X_train, y_train=y_train, scoring=scoring).best_score for model
                           in
                           competing_models}
    best_model = max(model_to_best_score, key=model_to_best_score.get)
    return best_model, best_model.predict(X_test), best_model.best_score


def regress(X_train, X_test, y_train, scoring='neg_mean_squared_error'):
    """
    given the train and test set with their labels returns the best_classifier according to the metric,
    it's predictions on the test set and it's metric score.
    regression analysis is a statistical process for estimating the relationships among variables.
    It includes many techniques for modeling and analyzing several variables, when the focus is on the relationship
    between a dependent variable and one or more independent variables
    :param scoring: model evaluation definition (from http://scikit-learn.org/stable/modules/model_evaluation.html)
    :param X_train: training dataframe
    :param y_train: training true labels (target var)
    :param X_test: test dataframe
    :param y_test: test true labels (target var)
    :return: the best_regressor according to the metric, it's predictions on the test set and it's metric score
    """
    # models configurations
    sgd = MLModel('SGDRegressor', SGDRegressor, {
        'loss': hp.choice('loss', ['squared_loss', 'huber']),
        'penalty': hp.choice('penalty ', ['none', 'l2', 'l1', 'elasticnet'])
    })
    lasso = MLModel('Lasso', Lasso, {
    })
    enet = MLModel('ElasticNet', ElasticNet, {
    })
    ridge = MLModel('Ridge', Ridge, {
    })
    svr = MLModel('SVR', SVR, {
        'kernel': hp.choice('kernel', ['linear', 'rbf']),
    })
    gbr = MLModel('GradientBoostingRegressor', GradientBoostingRegressor, {
    })
    rf_regressor = MLModel('RandomForestRegressor', RandomForestRegressor, {
    })

    # models competition
    competing_models = [sgd, lasso, enet, ridge, svr, gbr, rf_regressor]
    model_to_best_score = {model: model.optimize(X_train=X_train, y_train=y_train, scoring=scoring).best_score for model
                           in competing_models}
    best_model = max(model_to_best_score, key=model_to_best_score.get)
    return best_model, best_model.predict(X_test), best_model.best_score


def _analyze_clusters(X, labels_pred, algorithm_name, labels_true=None):
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


def create_clusters(df, cluster_cols, n_clusters=None, labels_true=None):
    """
    given a dataframe, relevant columns for clustering and num of clusters [optional]
    returns a dictionary of clustering algorithm to it's name, labels and metrics.

    Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same
    group (called a cluster) are more similar (in some sense or another) to each other than to those
     in other groups (clusters).

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
    returns a dictionary of reduction algorithm to it's name and reduced df.

    dimensionality reduction or dimension reduction is the process of reducing the number of random variables under
     consideration, via obtaining a set of principal variables.

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


def associate_rules(df, min_support, min_confidence):
    """
    given a pandas DataFrame, minimum support level and minimum confidence level
    returns dataframe with rules and statistics.

    Association analysis is the task of finding interesting relationships in large data sets.
    There hidden relationships are then expressed as a collection of association rules and frequent item sets.
    Frequent item sets are simply a collection of items that frequently occur together.
    And association rules suggest a strong relationship that exists between two items.

    :param df: pandas DataFrame
    :param min_support: (float or int) – If float in range (0, 1), percent of minimal support for itemset to be considered
    frequent. If int > 1, the absolute number of instances. For example, general iterators don’t have defined length,
    so you need to pass the absolute minimal support as int.
    :param min_confidence: (float) – Confidence percent. Defined as itemset_support / antecedent_support.
    :return: pandas DataFrame with columns 'antecedent', 'consequent', 'support', 'confidence', 'coverage', 'strength',
    'lift', 'leverage'
    :links: http://aimotion.blogspot.co.il/2013/01/machine-learning-and-data-mining.html, https://github.com/biolab/orange3-associate
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
    rows dropped using isolation forest to detect outliers.

    In data mining, anomaly detection (also outlier detection) is the identification of items,
    events or observations which do not conform to an expected pattern or other items in a dataset.

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
