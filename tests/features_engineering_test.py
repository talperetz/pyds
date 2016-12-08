# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/8/2016
:TL;DR: this module is responsible for testing feature engineering module
"""

import os
import unittest

from pyds import features_engineering, constants


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

    def test_create_features(self):
        # todo: generate dataframe with numerical values
        df_with_new_features = features_engineering.create_features()
        # todo: check the returning values
        pass

    def test_select_features(self):
        # todo: generate dataframes
        reduced_df, reduced_cols = features_engineering.select_features()
        # todo: check the returning values
        pass


if __name__ == '__main__':
    unittest.main()
