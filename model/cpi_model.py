# Define dependent variable
output = 'cost_per_impression'

# Setup
# Import standard libraries
import numpy as np
import pandas as pd

# Import dataset to a pandas dataframe
df = pd.read_csv(output + '.csv')

# Preprocess data
# Drop rows where budget is 0,
df = df[df.total_budget != 0]

# Drop columns with more than 25% missing data
df = df.drop(['num_events'], axis=1)
df = df.drop(['ticket_capacity'], axis=1)
df = df.drop(['average_ticket_price'], axis=1)
df = df.drop(['facebook_interest'], axis=1)
df = df.drop(['instagram_interest'], axis=1)
df = df.drop(['google_search_volume'], axis=1)
df = df.drop(['twitter_interest'], axis=1)

# Replace 0s with NaN where appropriate
columns = ['facebook_likes']
for column in columns:
  df[column].replace(0, np.nan, inplace=True)

# Put rare values into other bucket
threshold = 0.05
to_buckets = ['region', 'category', 'shop']
for column in to_buckets:
  results = df[column].count()
  groups = df.groupby([column])[column].count()
  for bucket in groups.index:
    if groups.loc[bucket] < results * threshold:
      df.loc[df[column] == bucket, column] = 'other'

# Change custom shop to other
df.loc[df['shop'] == 'custom', 'shop'] = 'other'

# Drop rows with NaN values
df.dropna(axis = 'index', inplace = True)

# Encode categorical data
df = pd.get_dummies(df, columns = ['region', 'locality', 'category', 'shop', 'tracking'],
                    prefix = ['region', 'locality', 'category', 'shop', 'tracking'],
                    drop_first = True)

# Build model
# Specify dependent variable vector y and independent variable matrix X
y = df.iloc[:, 0]
X = df.iloc[:, 1:]

# Build and fit regressor
# Random forest regression (no feature scaling needed)
from sklearn.ensemble import RandomForestRegressor
forest_regressor = RandomForestRegressor(
    n_estimators = 150,
    min_samples_split = 6,
    max_leaf_nodes = 7)
forest_regressor.fit(X.drop(['start_date', 'end_date'], axis=1), y)

# Save model
from sklearn.externals import joblib
joblib.dump(forest_regressor, 'cpi_model.pkl')

# Save training columns
columns = list(df.drop(['start_date', 'end_date'], axis=1).iloc[:, 1:].columns)
joblib.dump(columns, 'cpi_columns.pkl')

# Predict and print results
results = forest_regressor.predict(X.drop(['start_date', 'end_date'], axis=1)).round(4)
for result in results:
    print(result)
