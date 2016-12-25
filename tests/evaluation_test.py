# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/22/2016
:TL;DR: this module is responsible for testing evaluation module
"""

import os
import unittest

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.ensemble import RandomForestClassifier

from pyds import evaluation, constants
from tests import data_generators


class EvaluationTestCase(unittest.TestCase):
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

    def test_evaluate_classification(self):
        hr_df = data_generators.get_hr_dataset()
        y = hr_df['salary']
        y = y.cat.add_categories(['high'])
        y.loc[6:8] = 'high'
        X = hr_df.drop('salary', axis=1).select_dtypes([np.number])
        clf = RandomForestClassifier()
        clf.fit(X, y)
        y_pred = clf.predict(X)
        y_true = y
        y_scores = clf.predict_proba(X)
        cr, cm_fig, roc_fig = evaluation.evaluate_classification(y_true, y_pred, y.unique(),
                                                                 y_scores=y_scores)
        self.assertIsInstance(cr, str)
        self.assertIsInstance(cm_fig, matplotlib.axes._axes.Axes)
        self.assertIsInstance(roc_fig, list)

    def evaluate_regression(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 0, 1]
        mean_abs_err, mse, med_abs_err, r_squared = evaluation.evaluate_regression(y_true, y_pred)
        self.assertGreater(mean_abs_err, 0)
        self.assertGreater(mse, 0)
        self.assertGreater(med_abs_err, 0)
        self.assertGreater(r_squared, 0)

    def evaluate_clusters(self):
        centers = [[1, 1], [-1, -1], [1, -1]]
        X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
                                    random_state=0)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
        clustering_metrics_df = evaluation.evaluate_clusters(kmeans.labels_, kmeans.__class__.__name__,
                                                             labels_true=None)
        self.assertIsInstance(clustering_metrics_df, pd.DataFrame)
        self.assertTrue(clustering_metrics_df.empty)


if __name__ == '__main__':
    unittest.main()
