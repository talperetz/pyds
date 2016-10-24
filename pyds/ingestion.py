""" 
@author: Tal Peretz
@date: 10/14/2016
@TL;DR: this module gets relational data in several forms and returns a pandas DataFrame
"""

import pandas as pd

# supported file extensions
file_extension_to_read_attribute = {'csv': 'read_csv', 'excel': 'read_excel', 'hdf': 'read_hdf', 'sql': 'read_sql',
                                    'json': 'read_json', 'html': 'read_html', 'stata': 'read_stata', 'sas': 'read_sas',
                                    'pickle': 'read_pickle'}


def _get_file_extension(file_path):
    """
    given the path of a file
    returns it's extension as string
    :param file_path: the path of an input file
    :return: the extension of the file
    """
    if file_path is None:
        raise ValueError('file path is None')
    else:
        file_extension = file_path.split(".")[-1]
        if file_extension == "":
            raise ValueError('expected .[extension] file')
        elif file_extension not in file_extension_to_read_attribute.keys():
            raise ValueError('supported file types are \n %s' % file_extension_to_read_attribute.keys())
    return file_extension


def read(*args):
    """
    given a collection of file paths representing relational data
    returns a pandas pandas DataFrame of the data
    :param args: collection of file paths representing an input file
    :return: pandas pandas DataFrame
    """
    partial_dfs = []
    for index, file_path in enumerate(args):
        pd_read_function = getattr(pd, file_extension_to_read_attribute[_get_file_extension(file_path)])
        partial_dfs.append(pd_read_function(file_path))
    return pd.concat(partial_dfs)
