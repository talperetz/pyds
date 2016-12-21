# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/8/2016
:TL;DR: this module is responsible for testing transformations module
"""

import os
import unittest

import numpy as np

from pyds import transformations, constants
from tests import data_generators


class TransformationsTestCase(unittest.TestCase):
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

    def test_discretize(self):
        gen_df = data_generators.generate_random_data(100, 5)
        equal_width_num_df, col_to_width_edges, equal_depth_num_df, col_to_depth_edges = transformations.discretize(
            gen_df)
        equal_width_num_df_with_predefined_bins, _, equal_depth_num_df_with_predefined_bins, _ = transformations.discretize(
            gen_df, col_to_width_edges, col_to_depth_edges)

        # check that equal_width_num_df and equal_depth_num_df has optimal num of bins
        for col_name in gen_df:
            self.assertEqual(len(equal_width_num_df["equal_w_%s" % col_name].unique()),
                             transformations._calc_optimal_num_of_bins(gen_df[col_name]))
            self.assertEqual(len(equal_depth_num_df["equal_d_%s" % col_name].unique()),
                             transformations._calc_optimal_num_of_bins(gen_df[col_name]))
        self.assertTrue(equal_width_num_df_with_predefined_bins.equals(equal_width_num_df))
        self.assertTrue(equal_depth_num_df_with_predefined_bins.equals(equal_depth_num_df))

    def test_preprocess_train_columns(self):
        hr_df = data_generators.get_hr_dataset()
        transformed_df, train_transformations = transformations.preprocess_train_columns(hr_df)
        self.assertEqual(len(hr_df.select_dtypes(include=[np.number])),
                         len(transformed_df.select_dtypes(include=[np.number])))  # num of num cols hasn't changed
        self.assertGreaterEqual(len(hr_df.select_dtypes(include=['category'])),
                         len(transformed_df.select_dtypes(include=['category'])))  # num of cat cols is greater or equal

    def test_preprocess_test_columns(self):
        # run preprocess_train_columns and preprocess_test_columns on same raw data and check if the returned dataframes
        # are equal
        hr_df = data_generators.get_hr_dataset()
        transformed_train_df, train_transformations = transformations.preprocess_train_columns(hr_df)
        transformed_test_df = transformations.preprocess_test_columns(hr_df, train_transformations)
        self.assertTrue(transformed_train_df.equals(transformed_test_df))

if __name__ == '__main__':
    unittest.main()
