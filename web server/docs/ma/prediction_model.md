---
sidebar_position: 1
---

# Prediction Model

## Import Required Libraries

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from folium.plugins import MarkerCluster
from adjustText import adjust_text
```

## Function to Train and Test Model 

``` python
def evaluate_pipelines(X, y):

    title = "Model Performance"

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipelines = [
        ("Logistic Regression", LogisticRegression(random_state=42, solver='liblinear'), {'C': [0.01, 0.1, 1, 10]}), 
        ("Random Forest", RandomForestClassifier(random_state=42),  {'n_estimators': [50, 100], 'max_depth': [5, 10, 20]}),
        ("XGBoost", XGBClassifier(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}),
        ("KNN", KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9]}),
        ("Decision Tree", DecisionTreeClassifier(random_state=42), {'max_depth': [5, 10, 20]}),
    ]

    results = []
    best_metric = 0

    for pipeline_name, model, param_grid in pipelines:
        start_time = time.time()

        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        metric = accuracy_score(y_test, y_pred)
        end_time = time.time()
        duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))

        best_metric = max(best_metric, metric)

        results.append({
            "PIPELINE": pipeline_name,
            "DURATION": duration,
            "METRIC": metric,
            "BEST PARAMS":  "DEFAULT" if grid_search.best_params_ == {} else grid_search.best_params_
        })

    df = pd.DataFrame(results)
    table = tabulate(df, headers="keys", tablefmt="grid", floatfmt=".6f", showindex=False)
    title_line = title.center(len(table.splitlines()[0]))
    final_output = title_line + "\n" + table + "\n \n" + f"Best Metric: {best_metric}"
    return final_output

df = tuned_occurance_dataset

features = [
    'OccYear', 'OccTime', 'TimeZoneID', 'AccIncTypeID', 'ActivityTypeID',
    'NumberTrainsInvolved', 'SubdMileage',
    'SubdStartMileage', 'SubdEndMileage','TotalRSInvolved',
    'DGCarsInvolvedIND', 'NumTracksInvolved',
    'TotalFatalInjuries', 'TotalSeriousInjuries', 'TotalMinorInjuries',
    'TotalEmployeeInjuries', 'TotalPassengerInjuries', 'TotalOtherInjuries',
    'TotalMotoristInjuries', 'TotalMotorVehicleInjuries', 'TotalPedestrianInjuries',
    'TotalTrespasserInjuries'
]
target = 'OccurrenceTypeID'

table = evaluate_pipelines(df[features], df[target])
print(table)
```
## Model Performance Results

```
| PIPELINE            | DURATION   | METRIC   | BEST PARAMS                                 |
|---------------------|------------|----------|---------------------------------------------|
| Logistic Regression | 00:00:03   | 0.835230 | {"C": 0.1}                                  |
| Random Forest       | 00:00:17   | 0.999381 | {"max_depth": 20, "n_estimators": 50}       |
| XGBoost             | 00:00:01   | 0.999845 | {"learning_rate": 0.01, "n_estimators": 50} |
| KNN                 | 00:00:12   | 0.844986 | {"n_neighbors": 3}                          |
| Decision Tree       | 00:00:00   | 0.999845 | {"max_depth": 5}                            |

Best Metric: 0.999845141308556
```