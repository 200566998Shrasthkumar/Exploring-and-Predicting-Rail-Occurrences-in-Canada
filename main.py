##########################################################################
# Author: Steve Kuruvilla, Shrasth Kumar                                 #
# Description: Main Program                                              #
##########################################################################

import src.filter_dataset as filter_dataset

occurance_dataset = filter_dataset.load_and_filter_dataset('.\\datasets\\COMPONENTS_PUBLIC.csv', 5000, 'Fre')
injuries_dataset = filter_dataset.load_and_filter_dataset('.\\datasets\\INJURIES_PUBLIC.csv', 10)
train_dataset = filter_dataset.load_and_filter_dataset('.\\datasets\\TRAIN_PUBLIC.csv', 1000, 'Fre')
components_dataset = filter_dataset.load_and_filter_dataset('.\\datasets\\COMPONENTS_PUBLIC.csv', 1000, 'Fre')
