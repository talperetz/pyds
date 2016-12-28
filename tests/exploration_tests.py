# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/8/2016
:TL;DR: this module is responsible for testing exploration module
"""

import os
import unittest

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes

from pyds import exploration, constants
from tests import data_generators


class ExplorationTestCase(unittest.TestCase):
    logger = None
    is_warnings_traced = False
    cat_and_num_X = None
    cat_and_num_y = None
    num_X = None
    num_y = None
    cat_X = None
    cat_y = None

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

        cat_and_num_df = pd.DataFrame(load_diabetes()['data'])
        cat_and_num_df[[0, 1, 2]] = cat_and_num_df[[0, 1, 2]].apply(lambda col: col.astype('category'))
        self.cat_and_num_X = cat_and_num_df
        self.cat_and_num_y = pd.Series(load_diabetes()['target'])
        self.num_X = data_generators.generate_random_data(100, 15)
        self.num_y = data_generators.generate_random_data(100, 1)
        self.cat_X = cat_and_num_df.select_dtypes(exclude=[np.number])
        self.cat_y = self.cat_and_num_y

    def test_describe(self):
        diabetes_num_description, diabetes_cat_description = exploration.describe(self.cat_and_num_X)
        self.assertIsInstance(diabetes_num_description, pd.DataFrame)
        self.assertIsInstance(diabetes_cat_description, pd.DataFrame)
        self.assertTrue((not diabetes_num_description.empty) and (not diabetes_cat_description.empty))
        rand_df_num_description, rand_df_cat_description = exploration.describe(self.num_X)
        self.assertIsInstance(rand_df_num_description, pd.DataFrame)
        self.assertTrue(not rand_df_num_description.empty)
        self.assertIsNone(rand_df_cat_description)
        cat_df_num_description, cat_df_cat_description = exploration.describe(self.cat_X)
        self.assertIsInstance(cat_df_cat_description, pd.DataFrame)
        self.assertTrue(not cat_df_cat_description.empty)
        self.assertIsNone(cat_df_num_description)

    def _assert_figures(self, figures):
        for figure in figures:
            self.assertIsInstance(figure, matplotlib.figure.Figure)

    def test_dist_plot(self):
        diabetes_num_figure, diabetes_cat_figure = exploration.dist_plot(self.cat_and_num_X)
        self._assert_figures(diabetes_num_figure)
        self._assert_figures(diabetes_cat_figure)
        rand_df_num_figures, rand_df_cat_figures = exploration.dist_plot(self.num_X)
        self._assert_figures(rand_df_num_figures)
        self.assertEqual(rand_df_cat_figures, [])
        cat_df_num_figures, cat_df_cat_figures = exploration.dist_plot(self.cat_X)
        self._assert_figures(cat_df_cat_figures)
        self.assertEqual(cat_df_num_figures, [])

    def test_box_plot(self):
        diabetes_num_figure, diabetes_cat_figure = exploration.box_plot(self.cat_and_num_X)
        self.assertIsInstance(diabetes_num_figure, matplotlib.figure.Figure)
        self.assertIsInstance(diabetes_cat_figure, matplotlib.figure.Figure)
        rand_df_num_figures, rand_df_cat_figures = exploration.box_plot(self.num_X)
        self.assertIsInstance(rand_df_num_figures, matplotlib.figure.Figure)
        self.assertEqual(rand_df_cat_figures, None)
        cat_df_num_figures, cat_df_cat_figures = exploration.box_plot(self.cat_X)
        self.assertIsInstance(cat_df_cat_figures, matplotlib.figure.Figure)
        self.assertEqual(cat_df_num_figures, None)

    def test_scatter_plot(self):
        diabetes_figure = exploration.scatter_plot(self.cat_and_num_X, self.cat_and_num_y)
        self.assertIsNone(diabetes_figure)
        rand_df_figure = exploration.scatter_plot(data_generators.generate_random_data(100, 5),
                                                  data_generators.generate_random_data(100, 1))
        self.assertIsInstance(rand_df_figure, sns.axisgrid.PairGrid)
        cat_df_figures = exploration.scatter_plot(self.cat_X, self.cat_y)
        self.assertIsInstance(cat_df_figures, sns.axisgrid.PairGrid)

    def _assert_contingency_tables(self, tables):
        for table in tables:
            self.assertIsInstance(table, pd.DataFrame)
            self.assertTrue(not table.empty)

    def test_contingency_table(self):
        diabetes_contingency_tables = exploration.contingency_table(self.cat_and_num_X, self.cat_and_num_y)
        self._assert_contingency_tables(diabetes_contingency_tables)
        rand_df_contingency_tables = exploration.contingency_table(self.num_X)
        self.assertEqual(rand_df_contingency_tables, [])
        cat_df_contingency_tables = exploration.contingency_table(self.cat_X)
        self._assert_contingency_tables(cat_df_contingency_tables)

    def test_correlations(self):
        diabetes_corr_matrix, diabetes_corr_figure = exploration.correlations(self.cat_and_num_X)
        self.assertIsInstance(diabetes_corr_matrix, pd.DataFrame)
        self.assertIsInstance(diabetes_corr_figure, sns.matrix.ClusterGrid)
        self.assertTrue(not diabetes_corr_matrix.empty)
        rand_df_corr_matrix, rand_df_corr_figure = exploration.correlations(self.num_X)
        self.assertIsInstance(rand_df_corr_matrix, pd.DataFrame)
        self.assertTrue(not rand_df_corr_matrix.empty)
        self.assertIsInstance(rand_df_corr_figure, sns.matrix.ClusterGrid)
        cat_df_corr_matrix, cat_df_corr_figure = exploration.correlations(self.cat_X)
        self.assertIsNone(cat_df_corr_matrix)
        self.assertIsNone(cat_df_corr_figure)


if __name__ == '__main__':
    unittest.main()
