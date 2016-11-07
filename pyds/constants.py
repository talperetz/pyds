""" 
@author: Tal Peretz
@date: 10/30/2016
@TL;DR: this module holds all the pyds constants
"""

TEST_SPLIT_SIZE = 0.3

# cleaning
DROP_ABOVE_NULL_THRESHOLD = 0.6  # percents [0.0 - 1.0]
ISOLATION_FOREST_N_ESTIMATORS = 100  # [int > 0] isolation forest is used for removing outliers in data
KNN_N_NEIGHBORS = 4  # [int > 0] knn is used to impute missing data

# exploration
CATEGORICAL_THRESHOLD = 10  # threshold use to differ categorical variables that are of numerical type
FIGURE_SIZE = 8  # matplotlib fig_size

# features engineering
NEG_INF_REPRESENTATION = -10
