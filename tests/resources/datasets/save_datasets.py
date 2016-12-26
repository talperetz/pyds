""" 
:author: Tal Peretz
:date: 11/11/2016
:TL;DR: this module purpose is generating datasets for pyds tests
"""

import os

import pandas as pd
import sklearn.datasets

save_attribute_to_file_extension = {'to_excel': 'xls', 'to_html': 'html', 'to_json': 'json', 'to_pickle': 'pickle',
                                    'to_stata': 'stata', 'to_sql': 'sql', 'to_csv': 'csv', }
DATASETS_PATH = os.path.abspath("")

datasets = (
    sklearn.datasets.load_boston(),
    sklearn.datasets.fetch_california_housing())


def save_datasets(datasets_collection):
    for i, dataset in enumerate(datasets_collection):
        dataset_name = dataset['DESCR'].split('\n')[0]
        # build path variable, check if exists, if not create it
        path = DATASETS_PATH + '/' + dataset_name + '/'
        file_name = 'train.%s' % tuple(save_attribute_to_file_extension.values())[i]
        if not os.path.exists(path):
            os.makedirs(path)

            # build the dataframe in the form of data columns and target variable in one DataFrame
            df = pd.concat([pd.DataFrame(data=dataset['data'], columns=dataset['feature_names']),
                            pd.Series(data=dataset['target'], name='target')], axis=1)

            # save the resulting DataFrame in a format from save_attribute_to_file_extension
            getattr(df, tuple(save_attribute_to_file_extension.keys())[i])(path + file_name)


if __name__ == '__main__':
    save_datasets(datasets)
