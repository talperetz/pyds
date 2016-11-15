# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 10/16/2016
:TL;DR: this module is responsible for testing the pipeline functionality
"""

import logging
import os
import unittest

from pyds import pipeline


class PipelineTestCase(unittest.TestCase):
    def test_pipeline(self):
        """
        this function walks on all files in each subdirectory in resources/datasets directory
        finds the paths of trainset and testset files
        and asserts the whole pipeline to check basic functionality
        """
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
                logging.debug('train_set: %s \n test_set: %s' % (train_path, test_path))
                pipeline_results = pipeline.exec_pipeline(train_paths=train_path, test_paths=test_path,
                                                          target_column='target')
                try:
                    self.assert_ingestion(pipeline_results)
                    self.assert_exploration(pipeline_results)
                    self.assert_cleaning(pipeline_results)
                    self.assert_features(pipeline_results)
                    self.assert_ml(pipeline_results)
                except AssertionError as e:
                    logging.error(e)

    def assert_ingestion(self, pipeline_results):
        self.assertIsNotNone(pipeline_results.Ingestion.initial_train_df)

    def assert_exploration(self, pipeline_results):
        self.assertIsNotNone(pipeline_results.Exploration.numerical_cols)
        self.assertIsNotNone(pipeline_results.Exploration.categorical_cols)
        self.assertIsNotNone(pipeline_results.Exploration.id_cols)
        self.assertTrue(pipeline_results.Exploration.num_description & (
            ~pipeline_results.Exploration.cat_description))
        self.assertIsNotNone(pipeline_results.Exploration.hist)
        self.assertIsNotNone(pipeline_results.Exploration.box_plot)
        self.assertIsNotNone(pipeline_results.Exploration.contingency_table)
        self.assertIsNotNone(pipeline_results.Exploration.correlations)

    def assert_cleaning(self, pipeline_results):
        df = pipeline_results.Cleaning.cleaned_df
        self.assertFalse(df.isnull().values.any())

    def assert_features(self, pipeline_results):
        # todo: test features transformations and selection
        self.assertIsNotNone(pipeline_results.Features.created_features)

    def assert_ml(self, pipeline_results):
        self.assertTrue(pipeline_results.ML.best_model & (
            ~pipeline_results.ML.clusterer_to_results))

        self.assertIsNotNone(pipeline_results.ML.reducer_to_results)

    def assert_ui(self, pipeline_results):
        pass


if __name__ == '__main__':
    logging.basicConfig(filename='pipeline_test.log', level=logging.DEBUG)
    unittest.main()
    # PipelineTestCase('test_%s' % file).debug()
