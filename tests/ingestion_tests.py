# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/8/2016
:TL;DR: this module is responsible for testing ingestion module
"""

import os
import unittest

import pandas as pd

from pyds import ingestion, constants


class IngestionTestCase(unittest.TestCase):
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

    def test_get_extensions(self):
        good_examples = {'C:/dev/talos.%s' % extension: extension for extension in
                         constants.FILE_EXTENSION_TO_READ_ATTRIBUTE.values()}
        for path, extension in good_examples.items():
            self.assertEquals(ingestion._get_file_extension(path), extension)
        error_examples = ['C:/dev/talos', None, 'C:/dev/talos.wrong']
        for path in error_examples:
            self.assertRaises(ValueError, ingestion._get_file_extension(path))

    def test_read(self):
        root_dir = os.path.abspath("/resources/iris different extensions/")
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if (file is not None) and (not file.endswith('.py')):
                    if 'train' in file:
                        train_path = os.path.join(subdir, file)
                        df = ingestion.read(train_path)
                        self.assertIsInstance(df, pd.DataFrame)
                        self.assertFalse(df.empty)

    def test_infer_columns_statistical_types(self):
        # todo: create dataframes (with different statistical types)
        ingestion.infer_columns_statistical_types()
        # todo: check statistical types
        pass

    def test_adjust_columns_types(self):
        # todo: create dataframes (with different statistical types)
        ingestion.adjust_columns_types()
        # todo: check statistical types were adjusted
        pass

    def test_get_file_extension(self):
        pass


if __name__ == '__main__':
    unittest.main()
