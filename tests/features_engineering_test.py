# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/8/2016
:TL;DR: this module is responsible for testing feature engineering module
"""

import os
import unittest

import pandas as pd

from pyds import features_engineering, constants
from tests import data_generators


class FeatureEngineeringTestCase(unittest.TestCase):
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

    def test_create_features(self):
        X = data_generators.generate_random_data(100, 15)
        df_with_new_features, features_created = features_engineering.create_features(X)
        self.assertGreater(df_with_new_features.shape[1], X.shape[1])
        self.assertTrue(set(df_with_new_features.columns).intersection(X.columns) == set(X.columns))
        self.assertTrue(set(df_with_new_features.columns).difference(X.columns) == features_created)

    def test_select_features(self):
        X = data_generators.generate_random_data(1000, 15)
        y = (0.3 * X[0] + X[1] + X[2] - 2 * X[3]) * 0.2
        X = pd.concat([X, X[[1, 2, 3]].rename(columns=lambda name: name + 15)], axis=1)
        reduced_df, dropped_columns = features_engineering.select_features(X, y)
        self.assertGreaterEqual(len(X.columns), len(reduced_df.columns))
        if len(dropped_columns) > 0:
            self.assertTrue(set(X.columns).difference(reduced_df.columns) == set(dropped_columns))


if __name__ == '__main__':
    unittest.main()
