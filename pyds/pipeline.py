""" 
:Authors: Tal Peretz
:Date: 10/14/2016
:TL;DR: this module responsible for executing the data science pipelilne and holding it's results
"""
import logging

from pyds import ingestion, exploration, transformations, cleaning, features_engineering, ml, evaluation

logger = logging.getLogger(__name__)


class ExplorationResults:
    """
    this class holds the results for the exploration stage
    """
    num_description, cat_description, hist, box_plot, contingency_table, scatter_plot, correlations = (None for _ in
                                                                                                       range(7))

    def save_exploration_results(self, num_description, cat_description, hist, box_plot, scatter_plot,
                                 contingency_table, correlations):
        self.exploration_results.num_description = num_description
        self.exploration_results.cat_description = cat_description
        self.exploration_results.hist = hist
        self.exploration_results.box_plot = box_plot
        self.exploration_results.scatter_plot = scatter_plot
        self.exploration_results.contingency_table = contingency_table
        self.exploration_results.correlations = correlations


class PipelineResults:
    """
    this class holds the results for each stage of the data science pipeline
    """

    initial_X_train, initial_X_test, initial_y_train, initial_y_test, numerical_cols, categorical_cols, id_cols = (
        None for _ in range(7))
    na_rows, imputation, outliers = (None for _ in range(3))
    num_transformations, cat_transformations, numerical_cols, categorical_cols, col_to_width_edges, col_to_depth_edges = \
        (None for _ in range(6))
    created_features, dropped_features = (None for _ in range(2))
    best_model, predictions_df, scores, clusterer_to_results, scatter_plots, evaluation = (None for _ in range(6))

    def save_ingestion_results(self, X_train, X_test, y_train, y_test, numerical_cols, categorical_cols, id_cols):
        self.initial_X_train = X_train
        self.initial_X_test = X_test
        self.initial_y_train = y_train
        self.initial_y_test = y_test
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.id_cols = id_cols

    def save_cleaning_results(self, na_rows, imputation, outliers):
        self.na_rows = na_rows
        self.imputation = imputation
        self.outliers = outliers

    def save_transformations(self, train_transformations):
        self.num_transformations = train_transformations.col_to_scaler
        self.cat_transformations = train_transformations.col_to_encoder
        self.col_to_width_edges = train_transformations.col_to_width_edges
        self.col_to_depth_edges = train_transformations.col_to_depth_edges

    def save_features(self, created_features, dropped_features):
        self.created_features = created_features
        self.dropped_features = dropped_features

    def save_models_results(self, best_model, predictions_df, scores, clusterer_to_results, scatter_plots, evaluation):
        self.best_model = best_model
        self.predictions_df = predictions_df
        self.scores = scores
        self.clusterer_to_results = clusterer_to_results
        self.scatter_plots = scatter_plots
        self.evaluation = evaluation


def explore(train_paths):
    # load data, validate, infer and adjust columns types
    train_df = ingestion.read(train_paths)
    logger.info('loaded train data from %s successfully' % train_paths)
    ingestion.validate_dataset(train_df)
    X_train, X_test, y_train, y_test, is_supervised = ingestion.get_train_test_splits(train_df)
    logger.info(
        'split data to train and test sets: \nX_train shape - %s \nX_test shape - %s' % (
            X_train.shape, X_test.shape))
    numerical_columns, categorical_columns, id_columns = \
        ingestion.infer_columns_statistical_types(X_train, y_train)
    X_train, X_test, y_train, y_test = ingestion.adjust_columns_types(categorical_columns, X_train, X_test, y_train,
                                                                      y_test)
    logger.info('columns types inferred: \nid - %s \n categorical - %s \n numerical - %s' % (
        id_columns, categorical_columns, numerical_columns))

    # exploration
    num_description, cat_description = exploration.describe(X=X_train, y=y_train)
    logger.info(
        'exploration results are ready: \n numerical columns description - \n%s \n categorical columns description -\n%s'
        % (num_description, cat_description))
    results = ExplorationResults()
    results.save_exploration_results(num_description, cat_description,
                                     exploration.dist_plot(X=X_train, y=y_train),
                                     exploration.box_plot(X=X_train, y=y_train),
                                     exploration.scatter_plot(X=X_train, y=y_train),
                                     exploration.contingency_table(X=X_train, y=y_train),
                                     exploration.correlations(X=X_train, y=y_train))
    return results


def exec_offline_pipeline(train_paths, test_paths=None, target_column=None, columns_to_clusterize=None,
                          n_clusters=None):
    """
    given data and arguments describing the data
    this function runs a full pipeline and returns it's result in a PipelineResults object
    :param train_paths: [list of str] full paths for training data
    :param test_paths: [list of str] full paths for test data
    :param target_column: [str] the name of the target variable column
    :param columns_to_clusterize: [list of str] names of columns to cluster the data upon
    :param columns_to_reduce_d: [list of str] names of columns to reduce dimensions upon
    :param n_clusters: [int] number of clusters in data if known
    :param n_components: [int] number of desired dimensions
    :return: pipeline results object holding all results from the process
    """
    pipeline_results = PipelineResults()

    # load data, validate, infer and adjust columns types
    train_df = ingestion.read(train_paths)
    logger.info('loaded train data from %s successfully' % train_paths)
    ingestion.validate_dataset(train_df)
    X_train, X_test, y_train, y_test, is_supervised = ingestion.get_train_test_splits(train_df, target_column,
                                                                                      test_paths)
    logger.info(
        'split data to train and test sets: \nX_train shape - %s \nX_test shape - %s' % (
            X_train.shape, X_test.shape))
    numerical_columns, categorical_columns, id_columns = \
        ingestion.infer_columns_statistical_types(X_train, y_train)
    X_train, X_test, y_train, y_test = ingestion.adjust_columns_types(categorical_columns, X_train, X_test, y_train,
                                                                      y_test)
    logger.info('columns types inferred: \nid - %s \n categorical - %s \n numerical - %s' % (
        id_columns, categorical_columns, numerical_columns))
    pipeline_results.save_ingestion_results(X_train, X_test, y_train, y_test, numerical_columns, categorical_columns,
                                            id_columns)

    # cleaning
    X_train_without_ids = cleaning.remove_id_columns(X_train, id_columns)
    logger.info('id columns removed')
    filled_X_train, na_rows, imputation = cleaning.fill_missing_values(X_train_without_ids)
    logger.info('filled %s missing values on train set \n %s' % (len(na_rows.index), imputation.head()))
    outliers = cleaning.detect_outliers(filled_X_train, y=y_train)
    cleaned_X_train = filled_X_train.drop(outliers, axis=0)
    logger.info('removed %s outliers on train set' % len(outliers))
    ml_ready_y_train = y_train.drop(outliers, axis=0)
    pipeline_results.save_cleaning_results(na_rows, imputation, outliers)
    # clean X_test as X_train
    X_test_cols_to_drop = list(set(id_columns).intersection(X_test.columns.tolist()))
    if X_test_cols_to_drop:
        X_test.drop(X_test_cols_to_drop, axis=1, inplace=True)
    filled_X_test = X_test
    if X_test.isnull().values.any():
        filled_X_test = cleaning.fill_missing_values(X_test)[0]

    # preprocessing
    transformed_X_train, train_transformations = transformations.preprocess_train_columns(
        X_train=cleaned_X_train, X_test=filled_X_test)
    logger.info('categorical and numerical columns transformed')
    pipeline_results.save_transformations(train_transformations)

    # features engineering
    X_train_with_new_features, created_features = features_engineering.create_features(transformed_X_train)
    logger.info('created new simple features, overall %s:\n%s' % (
        len(X_train_with_new_features.columns.tolist()), X_train_with_new_features.columns.tolist()))
    ml_ready_X_train, dropped_columns = features_engineering.select_features(X_train_with_new_features,
                                                                             ml_ready_y_train)
    logger.info(
        'dropped %s features:\n%s' % (len(dropped_columns), dropped_columns))
    pipeline_results.save_features(created_features, dropped_columns)

    # transform X_test as X_train
    transformed_X_test = transformations.preprocess_test_columns(filled_X_test, train_transformations)
    X_test_with_new_features, _ = features_engineering.create_features(transformed_X_test)
    ml_ready_X_test = X_test_with_new_features.loc[:,
                      list(set(ml_ready_X_train.columns).intersection(X_test_with_new_features.columns.tolist()))]
    ml_ready_X_train = ml_ready_X_train.loc[:,
                       list(set(ml_ready_X_train.columns).intersection(ml_ready_X_test.columns.tolist()))]
    logger.info('test set transformed successfully\n applying ML models')

    # ML
    best_model, predictions_df, score, clustering_algorithms, model_evaluation = (None for _ in range(5))
    # supervised problem
    if is_supervised:
        # classification problem
        if target_column in pipeline_results.categorical_cols:
            best_model, predictions, score, models_comparison_df = ml.classify(ml_ready_X_train, ml_ready_X_test,
                                                                               ml_ready_y_train)
            model_evaluation = evaluation.evaluate_classification(y_test, predictions)
            logger.info('finished classification')
        # regression problem
        else:
            best_model, predictions, score, models_comparison_df = ml.regress(ml_ready_X_train, ml_ready_X_test,
                                                                              ml_ready_y_train)
            model_evaluation = evaluation.evaluate_regression(y_test, predictions)
            logger.info('finished regression')
        logger.info("best model : \n %s" % best_model)
        logger.info("predictions : \n %s" % predictions)
        logger.info("best score : \n %s" % score)
        logger.info("models comparison : \n %s" % models_comparison_df)
    # unsupervised problem
    else:
        clustering_algorithms = ml.create_clusters(ml_ready_X_train, columns_to_clusterize, n_clusters)
        logger.info('finished clustering')
    reducer_to_reduced_df = ml.reduce_dimensions(ml_ready_X_train)
    scatter_plots = []
    for reducer, reduced_df in reducer_to_reduced_df.items():
        scatter_plots = exploration.scatter_plot(reduced_df, y=ml_ready_y_train,
                                                 figure_title='post processing after %s' % reducer)
    logger.info('finished scatter plotting')
    pipeline_results.save_models_results(best_model, predictions_df, score, clustering_algorithms, scatter_plots,
                                         model_evaluation)
    return pipeline_results


def associate_rules_pipeline(df, min_support, min_confidence, rule_max_len=None, rules_consequents=None):
    """
    given data and arguments describing the data
    this function runs a full pipeline and returns it's result in a PipelineResults object
    :param train_paths: [list of str] full paths for training data
    :param test_paths: [list of str] full paths for test data
    :param target_column: [str] the name of the target variable column
    :param columns_to_clusterize: [list of str] names of columns to cluster the data upon
    :param columns_to_reduce_d: [list of str] names of columns to reduce dimensions upon
    :param n_clusters: [int] number of clusters in data if known
    :param n_components: [int] number of desired dimensions
    :return: pipeline results object holding all results from the process
    """

    # load data, validate, infer and adjust columns types
    if rules_consequents:
        assert all(consequent in df.columns for consequent in
                   rules_consequents), 'consequents should be names of desired original columns'
    ingestion.validate_dataset(df)
    numerical_columns, categorical_columns, id_columns = \
        ingestion.infer_columns_statistical_types(df)
    df[categorical_columns] = df[categorical_columns].apply(lambda num_col: num_col.astype('category'))
    logger.info('columns types inferred: \nid - %s \n categorical - %s \n numerical - %s' % (
        id_columns, categorical_columns, numerical_columns))

    # cleaning
    df_without_ids = cleaning.remove_id_columns(df, id_columns)
    logger.info('id columns removed')
    filled_df, na_rows, imputation = cleaning.fill_missing_values(df_without_ids)
    logger.info('filled %s missing values on train set \n %s' % (len(na_rows.index), imputation.head()))

    # preprocessing
    transformed_df = transformations.preprocess_for_association_rules(filled_df)
    logger.info('categorical and numerical columns transformed')

    # run association rules
    result_df = ml.associate_rules(transformed_df, min_support, min_confidence, rule_max_len=rule_max_len,
                                   rules_consequents=rules_consequents)
    return result_df
