# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/8/2016
:TL;DR: this module is responsible for testing transformations module
"""

import os
import unittest

from pyds import transformations, constants


class PipelineTestCase(unittest.TestCase):
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
        # todo: generate dataframe with numerical values
        equal_width_num_df, col_to_width_edges, equal_depth_num_df, col_to_depth_edges = transformations.discretize()
        # todo: check the returning values
        pass

    def test_preprocess_train_columns(self):
        # todo: generate dataframes
        transformed_df, col_to_scaler, col_to_encoder, updated_numerical_cols, updated_categorical_cols, \
        col_to_width_edges, col_to_depth_edges = transformations.preprocess_train_columns()
        # todo: check the returning values
        pass

    def test_preprocess_test_columns(self):
        # todo: generate dataframes
        transformed_df = transformations.preprocess_test_columns()
        # todo: check the returning values
        pass


if __name__ == '__main__':
    unittest.main()
