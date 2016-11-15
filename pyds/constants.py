""" 
@author: Tal Peretz
@date: 10/30/2016
@TL;DR: this module holds all the pyds constants
"""

# ingestion

FILE_EXTENSION_TO_READ_ATTRIBUTE = {'csv': 'read_csv', 'xls': 'read_excel', 'xlsx': 'read_excel',
                                    'sql': 'read_sql', 'json': 'read_json', 'html': 'read_html', 'stata': 'read_stata',
                                    'sas': 'read_sas', 'pickle': 'read_pickle'}  # supported file extensions
NUMERIC_TYPES = ('int16', 'int32', 'int64', 'float16', 'float32', 'float64')
TEST_SPLIT_SIZE = 0.3

# cleaning
MISSING_VALUES_REPRESENTATION = 'NaN'  # indicates which types are considered as missing values in pandas DataFrame
DROP_ABOVE_NULL_THRESHOLD = 0.6  # percents [0.0 - 1.0]
ISOLATION_FOREST_N_ESTIMATORS = 100  # [int > 0] isolation forest is used for removing outliers in data
KNN_N_NEIGHBORS = 4  # [int > 0] knn is used to impute missing data

# exploration
CATEGORICAL_THRESHOLD = 10  # threshold use to differ categorical variables that are of numerical type
FIGURE_SIZE = 8  # matplotlib fig_size

# features engineering
NEG_INF_REPRESENTATION = -10

# ML
CLUSTERING_METRICS = ('completeness_score', 'homogeneity_score', 'mutual_info_score', 'adjusted_rand_score')
