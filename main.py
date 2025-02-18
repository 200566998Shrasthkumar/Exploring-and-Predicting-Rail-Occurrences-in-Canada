##########################################################################
# Author: Steve Kuruvilla, Shrasth Kumar                                 #
# Description: Unified ML Execution with Random Forest, Decision Tree,  #
#              and XGBoost for Incident Analysis                         #
##########################################################################

import filter_dataset as filter_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import matplotlib.pyplot as plt

# Loading filtered datasets
occurance_dataset = filter_dataset.load_and_filter_dataset('/dataset/datasets/filtered/OCCURRENCE_PUBLIC.csv', 5000, 'Fre')
injuries_dataset = filter_dataset.load_and_filter_dataset('/dataset/datasets/filtered/INJURIES_PUBLIC.csv', 10)
train_dataset = filter_dataset.load_and_filter_dataset('/dataset/datasets/filtered/TRAIN_PUBLIC.csv', 1000, 'Fre')
components_dataset = filter_dataset.load_and_filter_dataset('/dataset/datasets/filtered/COMPONENTS_PUBLIC.csv', 1000, 'Fre')

occurance_dataset.to_csv('.\\datasets\\filtered\\OCCURRENCE_PUBLIC.csv', index=False)
injuries_dataset.to_csv('.\\datasets\\filtered\\INJURIES_PUBLIC.csv', index=False)
train_dataset.to_csv('.\\datasets\\filtered\\TRAIN_PUBLIC.csv', index=False)
components_dataset.to_csv('.\\datasets\\filtered\\COMPONENTS_PUBLIC.csv', index=False)

# Merging datasets
df = pd.merge(occurance_dataset, injuries_dataset[['OccNo', 'TotalOffTrainFatalities']], on='OccNo', how='left')
df['TotalOffTrainFatalities'].fillna(0, inplace=True)

# Aggregate Train Details
train_agg = train_dataset.groupby('OccNo').agg(NumTrains=('TrainSeq', 'nunique')).reset_index()
df = pd.merge(df, train_agg, on='OccNo', how='left')

# Feature Selection
features = ['OccurrenceTypeID', 'NumberTrainsInvolved', 'ActivityTypeID', 'SubdNameID',
            'TotalRSInvolved', 'DGCarsInvolvedIND', 'TotalOffTrainFatalities', 'NumTrains',
            'RegionOfOccurrence', 'OccYear']
target = 'TotalFatalInjuries'

X = df[features]
y = df[target]

# Defining categorical and numerical features
categorical_features = ['OccurrenceTypeID', 'ActivityTypeID', 'SubdNameID', 'RegionOfOccurrence']
numerical_features = ['NumberTrainsInvolved', 'TotalRSInvolved', 'TotalOffTrainFatalities', 'NumTrains', 'OccYear']

# Create transformers
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
}

# Train and evaluate models
results = []

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MSE': mse,
        'MAE': mae,
        'R-squared': r2
    })
    
    print(f"{name} - MSE: {mse}, MAE: {mae}, R-squared: {r2}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plotting the results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

results_df.plot(kind='bar', x='Model', y='MSE', ax=axes[0], legend=False)
axes[0].set_title('Mean Squared Error')
axes[0].set_ylabel('MSE')

results_df.plot(kind='bar', x='Model', y='MAE', ax=axes[1], legend=False)
axes[1].set_title('Mean Absolute Error')
axes[1].set_ylabel('MAE')

results_df.plot(kind='bar', x='Model', y='R-squared', ax=axes[2], legend=False)
axes[2].set_title('R-squared')
axes[2].set_ylabel('R-squared')

plt.tight_layout()
plt.show()
    
