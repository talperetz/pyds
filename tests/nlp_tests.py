# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/8/2016
:TL;DR: this module is responsible for testing nlp module
"""

import os
import unittest

import pandas as pd

from pyds import nlp, constants
from tests import data_generators


class NLPTestCase(unittest.TestCase):
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

    def test_decode_cols(self):
        df = data_generators.get_hebrew_mixed_hr_dataset()
        decoded_df = nlp.decode_cols(df, language='he', decode_columns='משכורת')
        self.assertIsInstance(decoded_df, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
