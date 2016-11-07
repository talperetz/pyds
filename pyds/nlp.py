"""
:Authors: Tal Peretz
:Date: 11/7/2016
:TL;DR: this module is responsible for all natural language processing aspects
"""

LANGUAGE_TO_ENCODING = {'he': 'iso-8859-8'}


def decode_cols(X, language, decode_columns=None):
    """
    given pandas DataFrame and the desired language of the text in columns
    returns the dataframe where columns are decoded according to desired language
    :param X: pandas DataFrame
    :param language: the desired language of the text in columns
    :param decode_columns: [list] columns to decode in X
    :return: X where columns are decoded according to desired language
    """
    if decode_columns:
        df = X[decode_columns].copy()
    else:
        df = X.copy()
    return df.apply(lambda col: df[col].str.decode(LANGUAGE_TO_ENCODING[language.lower()]))
