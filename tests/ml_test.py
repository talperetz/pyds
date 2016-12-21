# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/8/2016
:TL;DR: this module is responsible for testing ml module
"""

import os
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pyds import ml, constants
from tests import data_generators, tests_constants


class MLTestCase(unittest.TestCase):
    logger = None
    is_warnings_traced = False

    def setUp(self):
        import traceback
        import warnings
        import sys
        import logging.config

        # setting log configuration
        log_conf_path = os.path.abspath(constants.LOGGER_CONFIGURATION_RELATIVE_PATH)
        logging.config.fileConfig(log_conf_path)
        self.logger = logging.getLogger(__name__)

        def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
            traceback.print_stack()

            log = file if hasattr(file, 'write') else sys.stderr
            log.write(warnings.formatwarning(message, category, filename, lineno, line))

        if self.is_warnings_traced:
            warnings.showwarning = warn_with_traceback

    def test_classify(self):
        # todo: generate dataframes
        best_model, predictions, best_score = ml.classify()
        # todo: check the returning values
        pass

    def test_regress(self):
        X = data_generators.generate_random_data(1000, 15)
        y = (0.3 * X[0] + X[1] + X[2] - 2 * X[3]) * 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=constants.TEST_SPLIT_SIZE)
        best_model, predictions, best_score = ml.regress(X_train, X_test, y_train)
        self.assertIsNotNone(best_model)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertGreater(best_score, tests_constants.REGRESSION_SCORE_THRESHOLD)

    def test_create_clusters(self):
        centers = [[1, 1], [-1, -1], [1, -1]]
        densities = [0.2, 0.35, 0.5]
        X, labels_true = data_generators.make_var_density_blobs(n_samples=750, centers=centers, cluster_std=densities)
        clustering_algorithms = ml.create_clusters(X, X.columns.tolist(), n_clusters=3)
        self.assertTrue(clustering_algorithms != set())
        clustering_algorithms = ml.create_clusters(X, X.columns.tolist())
        self.assertTrue(clustering_algorithms != set())

        print(clustering_algorithms)
        pass

    def test_reduce_dimensions(self):
        df = data_generators.generate_random_data(100, 5)
        reducer_to_results = ml.reduce_dimensions(df)
        self.assertTrue(reducer_to_results != {})
        for reducer, results in reducer_to_results.items():
            self.assertIsInstance(results, pd.DataFrame)
            self.assertGreaterEqual(len(df.columns), len(results.columns))

    def test_detect_anomalies(self):
        outliers = ml.detect_anomalies_with_isolation_forest()
        # todo: check the returning values
        pass


if __name__ == '__main__':
    unittest.main()
