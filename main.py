##########################################################################
# Author: Steve Kuruvilla, Shrasth Kumar                                 #
# Description: Main Program                                              #
##########################################################################

import src.filter_dataset as filter_dataset

occurance_dataset = filter_dataset.load_and_filter_dataset('.\\datasets\\OCCURRENCE_PUBLIC.csv', 5000, 'Fre')
injuries_dataset = filter_dataset.load_and_filter_dataset('.\\datasets\\INJURIES_PUBLIC.csv', 10)
train_dataset = filter_dataset.load_and_filter_dataset('.\\datasets\\TRAIN_PUBLIC.csv', 1000, 'Fre')
components_dataset = filter_dataset.load_and_filter_dataset('.\\datasets\\COMPONENTS_PUBLIC.csv', 1000, 'Fre')

occurance_dataset.to_csv('.\\datasets\\filtered\\OCCURRENCE_PUBLIC.csv', index=False)
injuries_dataset.to_csv('.\\datasets\\filtered\\INJURIES_PUBLIC.csv', index=False)
train_dataset.to_csv('.\\datasets\\filtered\\TRAIN_PUBLIC.csv', index=False)
components_dataset.to_csv('.\\datasets\\filtered\\COMPONENTS_PUBLIC.csv', index=False)


print(occurance_dataset.head(2))
print(injuries_dataset.head(2))
print(train_dataset.head(2))
print(components_dataset.head(2))

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

# Load your DataFrames (replace with your actual file paths)
df1 = pd.read_csv("Occurence_details.csv")  # Occurrence Details
df2 = pd.read_csv("Off_train_casualties.csv")  # Off-Train Injuries
df3 = pd.read_csv("Train_details.csv")  # Train Details

# 1. Data Integration
df = pd.merge(df1, df2[['OccNo', 'TotalOffTrainFatalities']], on='OccNo', how='left')
df['TotalOffTrainFatalities'].fillna(0, inplace=True)

# Aggregate Train Details (DataFrame 3)
train_agg = df3.groupby('OccNo').agg(
    NumTrains=('TrainSeq', 'nunique'),  # Number of unique trains involved
    # Add other aggregations if needed (e.g., types of trains)
).reset_index()
df = pd.merge(df, train_agg, on='OccNo', how='left')

# 2. Feature Selection (Expanded)
features = ['OccurrenceTypeID', 'NumberTrainsInvolved', 'ActivityTypeID', 'SubdNameID',
            'TotalRSInvolved', 'DGCarsInvolvedIND', 'TotalOffTrainFatalities', 'NumTrains']  # Added NumTrains
target = 'TotalFatalInjuries'

X = df[features]
y = df[target]

# 3. One-Hot Encoding
categorical_features = ['OccurrenceTypeID', 'ActivityTypeID', 'SubdNameID']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_features])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))
X = X.drop(categorical_features, axis=1)
X = pd.concat([X, X_encoded_df], axis=1)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators':,
    'max_depth': [None, 5, 10],
    # Add other hyperparameters to tune
}
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 6. Cross-Validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {-cv_scores.mean()}")  # Note the negative sign to get positive MSE

# 7. Model Evaluation
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)  # Added MAE
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")  # Print MAE
print(f"R-squared: {r2}")

# 8. Feature Importance
feature_importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))
