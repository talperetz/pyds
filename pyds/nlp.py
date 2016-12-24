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
    assert language.lower() in constants.LANGUAGE_TO_ENCODING.keys(), \
        'supported languages are %s' % constants.LANGUAGE_TO_ENCODING.keys()
    df = X.copy()
    if decode_columns is not None:
        if isinstance(df[decode_columns], pd.Series):
            df[decode_columns] = df[decode_columns].str.decode(constants.LANGUAGE_TO_ENCODING[language.lower()])
        else:
            df[decode_columns] = df.apply(
                lambda col: df[col].str.decode(constants.LANGUAGE_TO_ENCODING[language.lower()]))
    else:
        df = df.apply(lambda col: df[col].str.decode(constants.LANGUAGE_TO_ENCODING[language.lower()]))
    return df
