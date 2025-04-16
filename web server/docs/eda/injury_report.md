---
sidebar_position: 1
---

# Injury Report

## Import Required Libraries

``` python
import pandas as pd
import src.filter_dataset as filter_dataset
```
## Filter Dataset

``` python
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None) 

def load_and_filter_dataset(file_path, null_threshold, exclude_suffix=None):
    dataset = pd.read_csv(file_path)
    if exclude_suffix:
        dataset = dataset.loc[:, ~dataset.columns.str.endswith(exclude_suffix)]
    dataset = dataset.loc[:, dataset.isnull().sum() <= null_threshold]
    dataset = dataset.dropna()
    return dataset

```

## Load the dataset

``` python
occurance_dataset = filter_dataset.load_and_filter_dataset('.\\datasets\\OCCURRENCE_PUBLIC.csv', 5000, 'Fre')
injuries_dataset = filter_dataset.load_and_filter_dataset('.\\datasets\\INJURIES_PUBLIC.csv', 10)
train_dataset = filter_dataset.load_and_filter_dataset('.\\datasets\\TRAIN_PUBLIC.csv', 1000, 'Fre')
components_dataset = filter_dataset.load_and_filter_dataset('.\\datasets\\COMPONENTS_PUBLIC.csv', 1000, 'Fre')

occurance_dataset.to_csv('.\\datasets\\filtered\\OCCURRENCE_PUBLIC.csv', index=False)
injuries_dataset.to_csv('.\\datasets\\filtered\\INJURIES_PUBLIC.csv', index=False)
train_dataset.to_csv('.\\datasets\\filtered\\TRAIN_PUBLIC.csv', index=False)
components_dataset.to_csv('.\\datasets\\filtered\\COMPONENTS_PUBLIC.csv', index=False)
```
## Define categories

``` python
categories = {
    "Employee Injuries": ["Offtrain_EmployeeFatal", "Offtrain_EmployeeSerious", "Offtrain_EmployeeMinor"],
    "RS Passenger Injuries": ["Offtrain_RS_PassengerFatal", "Offtrain_RS_PassengerSerious", "Offtrain_RS_PassengerMinor"],
    "Vehicle Operator Injuries": ["Offtrain_VehicleOperatorFatal", "Offtrain_VehicleOperatorSerious", "Offtrain_VehicleOperatorMinor"],
    "Vehicle Passenger Injuries": ["Offtrain_VehiclePassengerFatal", "Offtrain_VehiclePassengerSerious", "Offtrain_VehiclePassengerMinor"],
    "Pedestrian Injuries": ["Offtrain_PedestrianFatal", "Offtrain_PedestrianSerious", "Offtrain_PedestrianMinor"],
    "Trespasser Injuries": ["Offtrain_TrespasserFatal", "Offtrain_TrespasserSerious", "Offtrain_TrespasserMinor"],
}
```

## Group and aggregate

``` python
summary = {}
total_fatalities = 0
total_serious = 0
total_minor = 0

for category, cols in categories.items():
    fatalities = df[cols[0]].sum()
    serious_injuries = df[cols[1]].sum()
    minor_injuries = df[cols[2]].sum()
    total = fatalities + serious_injuries + minor_injuries
    
    # Update total counts
    total_fatalities += fatalities
    total_serious += serious_injuries
    total_minor += minor_injuries

    summary[category] = {
        "Fatalities": fatalities,
        "Serious Injuries": serious_injuries,
        "Minor Injuries": minor_injuries,
        "Total Cases": total,
    }
```
## Adding total row

``` python
summary["Total"] = {
    "Fatalities": total_fatalities,
    "Serious Injuries": total_serious,
    "Minor Injuries": total_minor,
    "Total Cases": total_fatalities + total_serious + total_minor,
}
```

## Convert to DataFrame

``` python
summary_df = pd.DataFrame.from_dict(summary, orient="index")
```
## Display DataFrame

``` python
summary_df.head(30)
```


    | Category                 | Fatalities | Serious Injuries | Minor Injuries | Total Cases |
    |--------------------------|------------|------------------|----------------|-------------|
    | Employee Injuries        | 122        | 130              | 1834           | 2086        |
    | RS Passenger Injuries    | 26         | 27               | 785            | 838         |
    | Vehicle Operator Injuries| 1101       | 842              | 3299           | 5242        |
    | Vehicle Passenger Injuries| 171       | 272              | 492            | 935         |
    | Pedestrian Injuries      | 305        | 135              | 113            | 553         |
    | Trespasser Injuries      | 2145       | 783              | 681            | 3609        |
    | **Total**                | **3870**   | **2189**         | **7204**       | **13263**   |

## Injury Statistics Respective of Incident ID

``` python
fatal_cols = [col for col in df.columns if "Fatal" in col]
serious_cols = [col for col in df.columns if "Serious" in col]
minor_cols = [col for col in df.columns if "Minor" in col]

df["Fatalities"] = df[fatal_cols].sum(axis=1)
df["Serious Injuries"] = df[serious_cols].sum(axis=1)
df["Minor Injuries"] = df[minor_cols].sum(axis=1)
df["Total Cases"] = df["Fatalities"] + df["Serious Injuries"] + df["Minor Injuries"]

result_df = df[["OccID", "OccNo", "Fatalities", "Serious Injuries", "Minor Injuries", "Total Cases"]] \
.sort_values(by="Total Cases", ascending=False)

result_df.head(10)
```

| ID     | OccID  | OccNo     | Fatalities | Serious Injuries | Minor Injuries | Total Cases |
|--------|--------|-----------|------------|------------------|----------------|-------------|
| 22604  | 31774  | R86Q0406  | 0          | 0                | 194            | 194         |
| 22627  | 31751  | R86C0490  | 46         | 0                | 142            | 188         |
| 13074  | 41702  | R91H0026  | 0          | 8                | 134            | 142         |
| 13075  | 41702  | R91H0026  | 0          | 8                | 134            | 142         |
| 14806  | 39945  | R90H0627  | 0          | 0                | 98             | 98          |
| 1915   | 118407 | R13D0054  | 94         | 0                | 0              | 94          |
| 18436  | 36158  | R88T2166  | 0          | 0                | 84             | 84          |
| 1888   | 118587 | R13T0192  | 12         | 20               | 40             | 72          |
| 16274  | 38429  | R89V1867  | 0          | 0                | 72             | 72          |
| 1889   | 118587 | R13T0192  | 12         | 20               | 40             | 72          |

