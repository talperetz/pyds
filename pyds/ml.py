""" 
:Authors: Tal Peretz
:Date: 10/21/2016
:TL;DR: this module is responsible for the data science classic model-based solutions implementation
:Links: https://www.analyticsvidhya.com/blog/2015/01/scikit-learn-python-machine-learning-tool/
"""

import math

import pandas as pd
from orangecontrib.associate.fpgrowth import association_rules, frequent_itemsets, rules_stats
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
        score_to_max = cross_val_score(model, self.X, self.y, scoring=self.scoring).mean()
        score_to_min = -1 * score_to_max
        return score_to_min

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
                                algo=rand.suggest, max_evals=max_evals)
        self.best_score = self._hyperopt_function(self.best_params)
        self.implementation = self.implementation(**self.best_params)
        return self

    def predict(self, X_test):
        """
        given a pandas dataframe returns the model's predictions as np.array
        :param X_test: test dataframe
        :return: predictions as np.array
        """
        assert (self.X is not None) and (self.implementation is not None), "you should run optimize before predict"
        model = self.implementation.fit(self.X, self.y)
        return model.predict(X_test)


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
    assert (isinstance(X_train, pd.DataFrame)) and (not X_train.empty), 'X_train should be a valid pandas DataFrame'
    assert (isinstance(X_test, pd.DataFrame)) and (not X_test.empty), 'X_test should be a valid pandas DataFrame'
    assert (isinstance(y_train, pd.Series)) and (not y_train.empty), 'y_train should be a valid pandas Series'
    # models configurations
    num_of_features = len(X_train.columns)
    knn = MLModel('KNeighborsClassifier', KNeighborsClassifier, param_space={
        'n_neighbors': hp.choice('n_neighbors', range(1, 50)),
        'weights': hp.choice('weights', ['uniform', 'distance']),
        'leaf_size': hp.choice('leaf_size', range(1, 50)),
    })
    svc = MLModel('SVC', SVC, param_space={
        'C': hp.uniform('C', 0, 20),
        'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
        'gamma': hp.uniform('gamma', 0, 20),
        'degree': hp.choice('degree', range(1, 5))
    })
    gp = MLModel('GaussianProcessClassifier', GaussianProcessClassifier, param_space={
        'warm_start': hp.choice('warm_start', [True, False])
    })
    rf = MLModel('RandomForestClassifier', RandomForestClassifier, param_space={
        'max_depth': hp.choice('max_depth', range(1, 20)),
        'max_features': hp.choice('max_features', [int(math.sqrt(num_of_features) / 2.0),
                                                   int(math.sqrt(num_of_features)),
                                                   int(2 * math.sqrt(num_of_features))]),
        'n_estimators': hp.choice('n_estimators', [100, 500, 1000, 2000]),
        'criterion': hp.choice('criterion', ["gini", "entropy"])
    })
    tree = MLModel('DecisionTreeClassifier', DecisionTreeClassifier, param_space={
        'max_depth': hp.choice('max_depth', range(3, 5)),
        'min_samples_split': hp.choice('min_samples_split', [10, 50]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 5, 10])
    })
    mlp = MLModel('MLPClassifier', MLPClassifier, param_space={
        'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [50, 100, 500]),
        'alpha': hp.choice('alpha', [0.0001, 0.1])
    })
    ada = MLModel('AdaBoostClassifier', AdaBoostClassifier, param_space={
        'n_estimators': hp.choice('n_estimators', [100, 500, 1000])
    })
    gnb = MLModel('GaussianNB', GaussianNB, param_space={
        'priors': None
    })
    qda = MLModel('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis, param_space={
        'priors': None
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
    assert (isinstance(X_train, pd.DataFrame)) and (not X_train.empty), 'X_train should be a valid pandas DataFrame'
    assert (isinstance(X_test, pd.DataFrame)) and (not X_test.empty), 'X_test should be a valid pandas DataFrame'
    assert (isinstance(y_train, pd.Series)) and (not y_train.empty), 'y_train should be a valid pandas Series'
    # models configurations
    num_of_features = len(X_train.columns)
    sgd = MLModel('SGDRegressor', SGDRegressor, {
        'loss': hp.choice('loss', ['squared_epsilon_insensitive', 'huber']),
        'warm_start': hp.choice('warm_start', [False, True])
    })
    lasso = MLModel('Lasso', Lasso, {
        'alpha': hp.uniform('alpha', 0.2, 2.0)
    })
    enet = MLModel('ElasticNet', ElasticNet, {
        'alpha': hp.uniform('alpha', 0.2, 2.5),
        'normalize': hp.choice('normalize', [False, True])
    })
    ridge = MLModel('Ridge', Ridge, {
        'alpha': hp.uniform('alpha', 0.2, 2.5),
        'normalize': hp.choice('normalize', [False, True])
    })
    svr = MLModel('SVR', SVR, {
        'C': hp.uniform('C', 0, 20),
        'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
        'gamma': hp.uniform('gamma', 0, 20),
        'degree': hp.choice('degree', range(1, 5))
    })
    gbr = MLModel('GradientBoostingRegressor', GradientBoostingRegressor, {
        'n_estimators': hp.choice('n_estimators', [100, 500, 1000, 2000])
    })
    rf_regressor = MLModel('RandomForestRegressor', RandomForestRegressor, {
        'max_depth': hp.choice('max_depth', [1, 5, 10, 20]),
        'max_features': hp.choice('max_features', [int(math.sqrt(num_of_features) / 2.0),
                                                   int(math.sqrt(num_of_features)),
                                                   int(2 * math.sqrt(num_of_features))]),
        'n_estimators': hp.choice('n_estimators', [100, 500, 1000, 2000]),
        'criterion': hp.choice('criterion', ["gini", "entropy"])
    })

    # models competition
    competing_models = [lasso, enet, ridge, svr, gbr]
    model_to_best_score = {model: model.optimize(X_train=X_train, y_train=y_train, scoring=scoring).best_score for model
                           in competing_models}
    best_model = max(model_to_best_score, key=model_to_best_score.get)
    return best_model, best_model.predict(X_test), best_model.best_score


def create_clusters(df, cluster_cols, n_clusters=None):
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
    assert (isinstance(df, pd.DataFrame)) and (not df.empty), 'df should be a valid pandas DataFrame'
    X = df[cluster_cols]
    clustering_algorithms = set()
    if n_clusters is not None:
        if len(df.index) > 10000:
            k_means = KMeans(n_clusters=n_clusters).fit(X)
            spectral = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack',
                                          affinity="nearest_neighbors").fit(X)
            mixture = GaussianMixture(n_components=n_clusters, covariance_type='full').fit(X)
            clustering_algorithms.update([k_means, spectral, mixture])
        else:
            mini_k_means = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters,
                                           n_init=10, max_no_improvement=10, verbose=0).fit(X)
            clustering_algorithms.add(mini_k_means)
    else:
        dbs = DBSCAN(eps=0.3, min_samples=10).fit(X)
        af = AffinityPropagation().fit(X)
        clustering_algorithms.update([dbs, af])
    return clustering_algorithms


def reduce_dimensions(df, reduce_cols=None, n_components=2):
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
    assert (isinstance(df, pd.DataFrame)) and (not df.empty), 'df should be a valid pandas DataFrame'
    if reduce_cols:
        assert (set(reduce_cols).issubset(set(df.columns.tolist()))) and (
            len(df[reduce_cols].index) > 0), "reduce_cols must be a subset of df columns"
        X = df[reduce_cols].copy()
    else:
        X = df.copy()
    reductions_algorithms, reducer_to_results = set(), dict()
    pca = PCA(n_components=n_components, svd_solver='randomized')
    reductions_algorithms.add(pca)
    if len(X.index) > 10000:
        k_pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
        reductions_algorithms.add(k_pca)
    else:
        n_neighbors = 10
        isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        se = SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors)
        lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, method='standard')
        reductions_algorithms.update([isomap, se, lle])
    for reducer in reductions_algorithms:
        reduced_df = pd.DataFrame(reducer.fit_transform(X))
        reducer_to_results[reducer.__class__.__name__] = reduced_df
    return reducer_to_results


def associate_rules(df, min_support, min_confidence, rule_max_len=None, rules_consequents=None):
    """
    given a pandas DataFrame, minimum support level and minimum confidence level
    returns dataframe with rules and statistics.

    Association analysis is the task of finding interesting relationships in large data sets.
    There hidden relationships are then expressed as a collection of association rules and frequent item sets.
    Frequent item sets are simply a collection of items that frequently occur together.
    And association rules suggest a strong relationship that exists between two items.

    :param rules_consequents: collection that holds the desired original columns that would serve as rule consequent
    :param rule_max_len: int maximum length for antecedent (num of participating values)
    :param df: pandas DataFrame
    :param min_support: (float or int) – If float in range (0, 1), percent of minimal support for itemset to be considered
    frequent. If int > 1, the absolute number of instances. For example, general iterators don’t have defined length,
    so you need to pass the absolute minimal support as int.
    :param min_confidence: (float) – Confidence percent. Defined as itemset_support / antecedent_support.
    :return: pandas DataFrame with columns 'antecedent', 'consequent', 'support', 'confidence', 'coverage', 'strength',
    'lift', 'leverage'
    :links: http://aimotion.blogspot.co.il/2013/01/machine-learning-and-data-mining.html, https://github.com/biolab/orange3-associate
    """
    assert (isinstance(df, pd.DataFrame)) and (not df.empty), 'df should be a valid pandas DataFrame'
    matrix = df.as_matrix()
    matrix_col_to_df_col = dict(zip(range(matrix.shape[1]), df.columns.tolist()))
    itemsets = dict(frequent_itemsets(matrix, min_support))
    rules = list(association_rules(itemsets, min_confidence))

    # filter rules by rules length
    if rule_max_len:
        rules = [(P, Q, supp, conf)
                 for P, Q, supp, conf in association_rules(itemsets, .8)
                 if len(Q) == rule_max_len]

    rstats = list(rules_stats(rules, itemsets, df.shape[0]))

    # replace columns positions with the columns names
    rstats = (
        (' & '.join(matrix_col_to_df_col[i] for i in rule[0]), matrix_col_to_df_col[next(iter(rule[1]))], rule[2],
         rule[3], rule[4], rule[5], rule[6], rule[7])
        for rule in rstats)

    # filter rules where consequent is in rules_consequents
    if rules_consequents:
        rstats = (rule for rule in rstats if any(
            consequent in matrix_col_to_df_col[next(iter(rule[1]))] for consequent in
            rules_consequents))

    rules_df = pd.DataFrame(data=rstats, columns=['antecedent', 'consequent', 'support', 'confidence', 'coverage',
                                                  'strength', 'lift', 'leverage'])
    return rules_df


def detect_anomalies_with_isolation_forest(X, y=None, contamination=0.1):
    """
    given a pandas DataFrame returns outliers indexes using isolation forest to detect outliers.

    In data mining, anomaly detection (also outlier detection) is the identification of items,
    events or observations which do not conform to an expected pattern or other items in a dataset.

    :param y: [pandas series] target column
    :param X: [pandas DataFrame] raw features
    :param contamination:  the proportion of outliers in the data set
    :return: outliers indexes
    """
    assert (isinstance(X, pd.DataFrame)) and (not X.empty), 'X should be a valid pandas DataFrame'
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


def detect_anomalies_with_hdbscan(X, y=None, contamination=0.1, min_cluster_size=15):
    """
    given a pandas DataFrame returns outliers indexes using hdbscan to detect outliers.

    In data mining, anomaly detection (also outlier detection) is the identification of items,
    events or observations which do not conform to an expected pattern or other items in a dataset.

    :param min_cluster_size: the minimum size of clusters
    :param y: [pandas series] target column
    :param X: [pandas DataFrame] raw features
    :param contamination:  the proportion of outliers in the data set
    :return: outliers indexes
    """
    assert (isinstance(X, pd.DataFrame)) and (not X.empty), 'X should be a valid pandas DataFrame'
    df = X.copy()
    if y is not None:
        df[y.name] = y
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(df)
    outliers_scores = pd.Series(clusterer.outlier_scores_, index=X.index)
    threshold = pd.Series(outliers_scores).quantile(1.0 - contamination)
    outliers = outliers_scores[outliers_scores > threshold].index
    return outliers
