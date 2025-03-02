##########################################################################
# Author: Steve Kuruvilla, Shrasth Kumar                                 #
# Description: Main Program                                              #
##########################################################################

import src.filter_dataset as filter_dataset
import pandas as pd
import src.model as train_and_evaluate_models

# Define file paths and parameters
datasets_info = [
    ('datasets\OCCURRENCE_PUBLIC.csv', 5000, 'Fre'),
    ('datasets\INJURIES_PUBLIC.csv', 10, None),
    ('datasets\TRAIN_PUBLIC.csv', 1000, 'Fre'),
    ('datasets\COMPONENTS_PUBLIC.csv', 1000, 'Fre')
]

# Load and save filtered datasets
filtered_datasets = {}
for path, threshold, filter_value in datasets_info:
    dataset_name = path.split('/')[-1].split('_')[0].lower()
    filtered_datasets[dataset_name] = filter_dataset.load_and_filter_dataset(path, threshold, filter_value)
    filtered_datasets[dataset_name].to_csv(f'datasets\filtered', index=False)

# Merging datasets
df = pd.merge(filtered_datasets['occurrence'], filtered_datasets['injuries'][['OccNo', 'TotalOffTrainFatalities']], on='OccNo', how='left')
df['TotalOffTrainFatalities'].fillna(0, inplace=True)

# Aggregate Train Details
train_agg = filtered_datasets['train'].groupby('OccNo').agg(NumTrains=('TrainSeq', 'nunique')).reset_index()
df = pd.merge(df, train_agg, on='OccNo', how='left')

# Feature Selection
features = ['OccurrenceTypeID', 'NumberTrainsInvolved', 'ActivityTypeID', 'SubdNameID',
            'TotalRSInvolved', 'DGCarsInvolvedIND', 'TotalOffTrainFatalities', 'NumTrains',
            'RegionOfOccurrence', 'OccYear']
target = 'TotalFatalInjuries'

# Call the function
train_and_evaluate_models(df, features, target)
