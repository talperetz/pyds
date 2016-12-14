"""
:Authors: Tal Peretz
:Date: 11/7/2016
:TL;DR: this module is responsible for all natural language processing aspects
"""

import pandas as pd
from pyds import constants


def decode_cols(X, language, decode_columns=None):
    """
    given pandas DataFrame and the desired language of the text in columns
    returns the dataframe where columns are decoded according to desired language
    :param X: pandas DataFrame
    :param language: the desired language of the text in columns
    :param decode_columns: [list] columns to decode in X
    :return: X where columns are decoded according to desired language
    """
    assert (isinstance(X, pd.DataFrame)) and (not X.empty), 'X should be a valid pandas DataFrame'
    assert language in constants.LANGUAGE_TO_ENCODING.keys(), \
        'supported languages are %s' % constants.LANGUAGE_TO_ENCODING.keys()
    df = X[decode_columns].copy() if decode_columns else X.copy()
    return df.apply(lambda col: df[col].str.decode(constants.LANGUAGE_TO_ENCODING[language.lower()]))
