""" 
:Authors: Tal Peretz
:Date: 10/21/2016
:TL;DR: this module
:Links: https://www.analyticsvidhya.com/blog/2015/01/scikit-learn-python-machine-learning-tool/
"""

# classification models
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, DBSCAN, AffinityPropagation
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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

    clf_to_score = {}
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
    regressors_names = set()
    regressors = set()
    regressor_to_score = {}
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


def create_clusters(df, cluster_cols, n_clusters=None):
    """
    given a dataframe, relevant columns for clustering and num of clusters [optional]
    returns a dictionary of clustering algorithm to it's name, labels and metrics
    :param df: the dataframe upon we want to perform clustering
    :param cluster_cols: relevant columns for clustering
    :param n_clusters: num of clusters if known
    :return: dictionary of clustering algorithm to it's name, labels and metrics
    """
    X = df[cluster_cols]
    clustering_names = set()
    clustering_algorithms = set()
    metrics_ = set()
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
    clusterer_to_results = {}
    for name, clusterer in zip(clustering_names, clustering_algorithms):
        labels = clusterer.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        metrics_.update([n_clusters_, metrics.silhouette_score(X, labels)])
        clusterer_to_results[clusterer] = name, labels, metrics_
    return clusterer_to_results


def reduce_dimensions(df, reduce_cols, n_components):
    """
    given a dataframe, columns to reduce and number of components for dimensionality reduction algorithm
    returns a dictionary of reduction algorithm to it's name and reduced df
    :param df: pandas dataframe
    :param reduce_cols: columns to perform dimensionality reduction on
    :param n_components: number of components for dimensionality reduction algorithm
    :return: dictionary of reduction algorithm to it's name and reduced df
    """
    if n_components is None:
        n_components = 2
    reductions_names = set()
    reductions_algorithms = set()
    X = df
    if (reduce_cols is not None) and (set(reduce_cols).issubset(set(df.columns.tolist()))) and (
                len(df[reduce_cols].index) > 0):
        X = df[reduce_cols]
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


def associate_rules():
    pass


def detect_anomalies():
    pass
