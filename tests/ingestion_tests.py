# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/8/2016
:TL;DR: this module is responsible for testing ingestion module
"""

import os
import unittest

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

    def test_read_extensions(self):
        # todo: generate dataframes with different extensions
        ingestion.read()
        # todo: test recieved dataframe
        pass

    def test_read(self):
        # todo: generate dataframes (normal, with line breaks, delimiters...)
        ingestion.read()
        # todo: test recieved dataframe
        pass

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
