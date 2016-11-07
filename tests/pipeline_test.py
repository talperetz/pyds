# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 10/16/2016
:TL;DR: this module is responsible for testing the pipeline functionality
"""

import unittest

from pyds import pipeline


class PipelineTestCase(unittest.TestCase):
    PATH = 'D:/Devl/WorkSpace/pyds/tests/dataset.csv'
    pipeline_results = None

    def setUp(self):
        self.pipeline_results = pipeline.exec_pipeline(PipelineTestCase.PATH, target_column='target')

    def tearDown(self):
        del self.pipeline_results

    def test_ingestion(self):
        self.assertIsNotNone(self.pipeline_results.Ingestion.initial_train_df)

    def test_exploration(self):
        self.assertIsNotNone(self.pipeline_results.Exploration.numerical_cols)
        self.assertIsNotNone(self.pipeline_results.Exploration.categorical_cols)
        self.assertIsNotNone(self.pipeline_results.Exploration.id_cols)
        self.assertTrue(~(self.pipeline_results.Exploration.num_description is None) & (
            self.pipeline_results.Exploration.cat_description is None))
        self.assertIsNotNone(self.pipeline_results.Exploration.hist)
        self.assertIsNotNone(self.pipeline_results.Exploration.box_plot)
        self.assertIsNotNone(self.pipeline_results.Exploration.contingency_table)
        self.assertIsNotNone(self.pipeline_results.Exploration.correlations)

    def test_cleaning(self):
        df = self.pipeline_results.Cleaning.cleaned_df
        self.assertFalse(df.isnull().values.any())

    def test_features(self):
        # todo: test features transformations and selection
        self.assertIsNotNone(self.pipeline_results.Features.created_features)

    def test_ml(self):
        self.assertTrue(~(self.pipeline_results.ML.best_model is None) & (
            self.pipeline_results.ML.clusterer_to_results is None))

        self.assertIsNotNone(self.pipeline_results.ML.reducer_to_results)

    def test_ui(self):
        pass


suite = unittest.TestLoader().loadTestsFromTestCase(PipelineTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
