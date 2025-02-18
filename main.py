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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# Loading DataFrames
df1 = pd.read_csv("Occurence_details.csv")  # Occurrence Details
df2 = pd.read_csv("Off_train_casualties.csv")  # Off-Train Injuries
df3 = pd.read_csv("Train_details.csv")  # Train Details

# 1. Data Integration
df = pd.merge(df1, df2[['OccNo', 'TotalOffTrainFatalities']], on='OccNo', how='left')
df['TotalOffTrainFatalities'].fillna(0, inplace=True)

# Aggregate Train Details
train_agg = df3.groupby('OccNo').agg(
    NumTrains=('TrainSeq', 'nunique'),

).reset_index()
df = pd.merge(df, train_agg, on='OccNo', how='left')

# 2. Feature Selection 
features = ['OccurrenceTypeID', 'NumberTrainsInvolved', 'ActivityTypeID', 'SubdNameID',
            'TotalRSInvolved', 'DGCarsInvolvedIND', 'TotalOffTrainFatalities', 'NumTrains',
            'RegionOfOccurrence', 'OccYear']  # Added RegionOfOccurrence and OccYear
target = 'TotalFatalInjuries'

X = df[features]
y = df[target]

# 3. Preprocessing with Pipeline and ColumnTransformer
# Defining categorical and numerical features
categorical_features = ['OccurrenceTypeID', 'ActivityTypeID', 'SubdNameID', 'RegionOfOccurrence']
numerical_features = ['NumberTrainsInvolved', 'TotalRSInvolved', 'TotalOffTrainFatalities', 'NumTrains', 'OccYear']

# Create transformers for categorical and numerical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # Added StandardScaler for numerical features
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Hyperparameter Tuning using GridSearchCV with Pipeline
# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 5, 10],
    # Add other hyperparameters to tune
}
grid_search = GridSearchCV(estimator=pipeline,
                           param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 6. Cross-Validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {-cv_scores.mean()}")

# 7. Model Evaluation
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# 8. Feature Importance
# Get feature names from the preprocessor
feature_names = list(best_model.named_steps['preprocessor'].transformers_.named_steps['onehot'].get_feature_names_out(categorical_features)) + numerical_features

# Get feature importances
feature_importances = best_model.named_steps['model'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))

# 9. Plot Feature Importances
sorted_feature_importances = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_importances['Feature'], sorted_feature_importances['Importance'])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importances")
plt.show()
