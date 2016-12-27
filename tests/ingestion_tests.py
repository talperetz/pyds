# -*- coding: utf-8 -*-

"""
:Authors: Tal Peretz
:Date: 12/8/2016
:TL;DR: this module is responsible for testing ingestion module
"""

import os
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pyds import ingestion, constants
from tests import data_generators


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
                         constants.FILE_EXTENSION_TO_READ_ATTRIBUTE.keys()}
        for path, extension in good_examples.items():
            self.assertEquals(ingestion._get_file_extension(path), extension)
        error_examples = ['C:/dev/talos', None, 'C:/dev/talos.wrong']
        for path in error_examples:
            self.assertRaises(ValueError, ingestion._get_file_extension, path)

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
        df = data_generators.generate_random_data(10, 5)
        df['str'] = ['high', 'high', 'high', 'low', 'medium', 'high', 'high', 'high', 'low', 'medium']
        df['category'] = ['high', 'high', 'high', 'high', 'low', 'high', 'high', 'high', 'high', 'low']
        df['category'] = df['category'].astype('category')
        df['id'] = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        df['ordered_category'] = ['2', '2', '3', '4', '4', '2', '2', '3', '4', '4']
        df['numeric_cat'] = [2, 2, 3, 4, 4, 2, 2, 3, 4, 4]
        df['ordered_category'] = df['ordered_category'].astype('category', ordered=True)
        df['float'] = [float(x) for x in df['id']]
        numerical_cols = [0, 1, 2, 3, 4, 'float']
        cat_cols = ['str', 'category', 'ordered_category', 'numeric_cat']
        id_cols = ['id']
        inferred_numerical_cols, inferred_categorical_cols, inferred_id_cols = \
            ingestion.infer_columns_statistical_types(df)
        for member in numerical_cols:
            self.assertIn(member, inferred_numerical_cols)
            self.assertNotIn(member, inferred_categorical_cols)
            self.assertNotIn(member, inferred_id_cols)
        for member in cat_cols:
            self.assertIn(member, inferred_categorical_cols)
            self.assertNotIn(member, inferred_numerical_cols)
            self.assertNotIn(member, inferred_id_cols)
        for member in id_cols:
            self.assertIn(member, inferred_id_cols)
            self.assertNotIn(member, inferred_numerical_cols)
            self.assertNotIn(member, inferred_categorical_cols)

    def test_adjust_columns_types(self):
        df = data_generators.generate_random_data(10, 5)
        df['str'] = ['high', 'high', 'high', 'low', 'medium', 'high', 'high', 'high', 'low', 'medium']
        df['category'] = ['high', 'high', 'high', 'high', 'low', 'high', 'high', 'high', 'high', 'low']
        df['category'] = df['category'].astype('category')
        df['id'] = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        df['numeric_cat'] = [2, 2, 3, 4, 4, 2, 2, 3, 4, 4]
        df['ordered_category'] = ['2', '2', '3', '4', '4', '2', '2', '3', '4', '4']
        df['ordered_category'] = df['ordered_category'].astype('category', ordered=True)
        df['float'] = [float(x) for x in df['id']]
        cols_to_convert_cat = ['str', 'category', 'ordered_category', 'numeric_cat', 'id']
        y = df[2]
        df.drop(2, axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=constants.TEST_SPLIT_SIZE)
        adjusted_X_train, adjusted_X_test, adjusted_y_train, adjusted_y_test = ingestion.adjust_columns_types(
            cols_to_convert_cat, X_train, X_test, y_train, y_test)

        for col_name, col_dtype in adjusted_X_train.dtypes.iteritems():
            col_type = col_dtype.type
            if col_type != pd.types.dtypes.CategoricalDtypeType:
                self.assertTrue(np.issubdtype(col_dtype, np.number),
                                msg="expected np.number or 'category' got type: %s" % col_dtype)
        for col_name, col_dtype in adjusted_X_test.dtypes.iteritems():
            col_type = col_dtype.type
            if col_type != pd.types.dtypes.CategoricalDtypeType:
                self.assertTrue(np.issubdtype(col_dtype, np.number),
                                msg="expected np.number or 'category' got type: %s" % col_dtype)
        self.assertTrue((np.issubdtype(y_train.dtype, np.number)) or (y_train.dtype == 'category'))
        self.assertTrue((np.issubdtype(y_test.dtype, np.number)) or (y_test.dtype == 'category'))


if __name__ == '__main__':
    unittest.main()
