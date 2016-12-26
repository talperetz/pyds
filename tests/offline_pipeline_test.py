# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 10/16/2016
:TL;DR: this module is responsible for testing the pipeline functionality
"""

import os
import unittest

from pyds import constants


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

    def test_pipeline(self):
        """
        this function walks on all files in each subdirectory in resources/datasets directory
        finds the paths of train set and test set files
        and asserts the whole pipeline to check basic functionality
        """
        from pyds import pipeline
        root_dir = os.path.abspath("/resources/datasets/")
        for subdir, dirs, files in os.walk(root_dir):
            train_path, test_path = None, None
            for file in files:
                if (file is not None) and (not file.endswith('.py')):
                    if 'train' in file:
                        train_path = os.path.join(subdir, file)

                    elif 'test' in file:
                        test_path = os.path.join(subdir, file)
            if train_path:
                self.logger.debug('train_set: %s \n test_set: %s' % (train_path, test_path))
                pipeline_results = pipeline.exec_offline_pipeline(train_paths=train_path, test_paths=test_path,
                                                                  target_column='target')
                try:
                    self.assert_ingestion(pipeline_results)
                    self.assert_cleaning(pipeline_results)
                    self.assert_features(pipeline_results)
                    self.assert_ml(pipeline_results)
                except AssertionError as e:
                    self.logger.error(e)

    def assert_ingestion(self, pipeline_results):
        self.assertIsNotNone(pipeline_results.initial_X_train)
        self.assertIsNotNone(pipeline_results.initial_X_test)
        self.assertIsNotNone(pipeline_results.initial_y_train)
        self.assertIsNotNone(pipeline_results.initial_y_test)
        self.assertTrue((pipeline_results.numerical_cols is not None) or (
            pipeline_results.categorical_cols is not None) or (
                            pipeline_results.id_cols is not None))

    def assert_cleaning(self, pipeline_results):
        pass

    def assert_features(self, pipeline_results):
        self.assertIsNotNone(pipeline_results.created_features)

    def assert_ml(self, pipeline_results):
        self.assertTrue(
            (pipeline_results.best_model is not None) or (
                pipeline_results.clusterer_to_results is not None))
        self.assertIsNotNone(pipeline_results.scatter_plots)

    def assert_ui(self, pipeline_results):
        pass


if __name__ == '__main__':
    unittest.main()
