# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/8/2016
:TL;DR: this module is responsible for testing exploration module
"""

import os
import unittest

from pyds import exploration, constants


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

    def test_describe(self):
        # todo: generate dataframes
        exploration.describe()
        # todo: test received dataframe
        pass

    def test_hist(self):
        # todo: generate dataframes
        numerical_figures, categorical_figures = exploration.hist()
        # todo: test received numerical_figures, categorical_figures
        pass

    def test_box_plot(self):
        # todo: generate dataframes
        numerical_figures, categorical_figures = exploration.box_plot()
        # todo: test received numerical_figures, categorical_figures
        pass

    def test_scatter_plot(self):
        # todo: generate dataframes
        numerical_figures, categorical_figures = exploration.scatter_plot()
        # todo: test received numerical_figures, categorical_figures
        pass

    def test_contingency_table(self):
        # todo: generate dataframes
        contingency_tables = exploration.contingency_table()
        # todo: test received contingency_tables
        pass

    def test_correlations(self):
        # todo: generate dataframes
        corr, fig = exploration.correlations()
        # todo: test received corr, fig
        pass


if __name__ == '__main__':
    unittest.main()
