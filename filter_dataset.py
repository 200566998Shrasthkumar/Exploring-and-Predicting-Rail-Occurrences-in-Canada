##########################################################################
# Author: Steve Kuruvilla, Shrasth Kumar                                 #
# Description: Data Transformation                                       #
##########################################################################

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows

def load_and_filter_dataset(file_path, null_threshold, exclude_suffix=None):
    dataset = pd.read_csv(file_path)
    if exclude_suffix:
        dataset = dataset.loc[:, ~dataset.columns.str.endswith(exclude_suffix)] # Exclude columns with suffix
    dataset = dataset.loc[:, dataset.isnull().sum() <= null_threshold] # Exclude columns with more than null_threshold null values
    dataset = dataset.dropna()
    return dataset

occurance_dataset = load_and_filter_dataset('datasets/OCCURRENCE_PUBLIC.csv', 5000, 'Fre')
injuries_dataset = load_and_filter_dataset('datasets/INJURIES_PUBLIC.csv', 10)
train_dataset = load_and_filter_dataset('datasets/TRAIN_PUBLIC.csv', 1000, 'Fre')
rolling_stock_dataset = load_and_filter_dataset('datasets/ROLLING_STOCK_PUBLIC.csv', 1000, 'Fre')
components_dataset = load_and_filter_dataset('datasets/COMPONENTS_PUBLIC.csv', 1000, 'Fre')

