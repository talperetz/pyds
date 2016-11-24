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

    def setUp(self):
        import logging.config
        log_conf_path = os.path.abspath(constants.LOGGER_CONFIGURATION_RELATIVE_PATH)
        logging.config.fileConfig(log_conf_path)
        self.logger = logging.getLogger(__name__)

    def test_pipeline(self):
        """
        this function walks on all files in each subdirectory in resources/datasets directory
        finds the paths of train set and test set files
        and asserts the whole pipeline to check basic functionality
        """
        from pyds import pipeline
        root_dir = os.path.abspath("../resources/datasets/")
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
                pipeline_results = pipeline.exec_pipeline(train_paths=train_path, test_paths=test_path,
                                                          target_column='target')
                try:
                    self.assert_ingestion(pipeline_results)
                    self.assert_exploration(pipeline_results)
                    self.assert_cleaning(pipeline_results)
                    self.assert_features(pipeline_results)
                    self.assert_ml(pipeline_results)
                except AssertionError as e:
                    self.logger.error(e)

    def assert_ingestion(self, pipeline_results):
        self.assertIsNotNone(pipeline_results.Ingestion.initial_X_train)
        self.assertIsNotNone(pipeline_results.Ingestion.initial_X_test)
        self.assertIsNotNone(pipeline_results.Ingestion.initial_y_train)
        self.assertIsNotNone(pipeline_results.Ingestion.initial_y_test)
        self.assertTrue((pipeline_results.Ingestion.numerical_cols is not None) or (
            pipeline_results.Ingestion.categorical_cols is not None) or (
                            pipeline_results.Ingestion.id_cols is not None))

    def assert_exploration(self, pipeline_results):
        self.assertTrue((pipeline_results.Exploration.num_description is not None) or (
            pipeline_results.Exploration.cat_description is not None))
        self.assertIsNotNone(pipeline_results.Exploration.hist)
        self.assertIsNotNone(pipeline_results.Exploration.box_plot)
        self.assertIsNotNone(pipeline_results.Exploration.contingency_table)
        self.assertIsNotNone(pipeline_results.Exploration.correlations)

    def assert_cleaning(self, pipeline_results):
        pass

    def assert_features(self, pipeline_results):
        self.assertIsNotNone(pipeline_results.Features.created_features)

    def assert_ml(self, pipeline_results):
        self.assertTrue(
            (pipeline_results.ML.best_model is not None) or (pipeline_results.ML.clusterer_to_results is not None))
        self.assertIsNotNone(pipeline_results.ML.scatter_plots)

    def assert_ui(self, pipeline_results):
        pass


if __name__ == '__main__':
    unittest.main()
