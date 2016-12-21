# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/21/2016
:TL;DR: this module is responsible for testing the associate rules pipeline functionality
"""

import os
import unittest

import pandas as pd

from pyds import constants, ingestion


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
        root_dir = os.path.abspath("../resources/datasets/")
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if (file is not None) and (not file.endswith('.py')):
                    if 'train' in file:
                        train_path = os.path.join(subdir, file)
                        self.logger.debug('train_set: %s ' % train_path)
                        df = ingestion.read(train_path)
                        rules_df = pipeline.associate_rules_pipeline(df, 0.3, 0.8)
                        self.assertIsInstance(rules_df, pd.DataFrame)
                        self.assertFalse(rules_df.empty)
                        self.logger.info('association rules: \n %s' % rules_df)


if __name__ == '__main__':
    unittest.main()
