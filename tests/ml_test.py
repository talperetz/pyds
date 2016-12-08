# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/8/2016
:TL;DR: this module is responsible for testing ml module
"""

import os
import unittest

from pyds import ml, constants


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

    def test_classify(self):
        # todo: generate dataframe with numerical values
        best_model, predictions, best_score = ml.classify()
        # todo: check the returning values
        pass

    def test_regress(self):
        # todo: generate dataframes
        best_model, predictions, best_score = ml.regress()
        # todo: check the returning values
        pass

    def test_create_clusters(self):
        # todo: generate dataframe with numerical values
        clusterer_to_results = ml.create_clusters()
        # todo: check the returning values
        pass

    def test_reduce_dimensions(self):
        # todo: generate dataframes
        reducer_to_results = ml.reduce_dimensions()
        # todo: check the returning values
        pass

    def test_associate_rules(self):
        # todo: generate dataframes
        rules_df = ml.associate_rules()
        # todo: check the returning values
        pass

    def test_detect_anomalies(self):
        # todo: generate dataframes
        outliers = ml.detect_anomalies()
        # todo: check the returning values
        pass


if __name__ == '__main__':
    unittest.main()
