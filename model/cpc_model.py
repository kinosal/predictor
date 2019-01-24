# Define dependent variable
output = 'cost_per_click'

# Setup
# Import libraries
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

# Put rare values into buckets
threshold = 0.05
to_buckets = ['region', 'category', 'shop']
for column in to_buckets:
  results = df[column].count()
  groups = df.groupby([column])[column].count()
  for shop in groups.index:
    if groups.loc[shop] < results * threshold:
      df.loc[df[column] == shop, column] = 'other'

# Change custom shop to other
df.loc[df['shop'] == 'custom', 'shop'] = 'other'

# Change pu and pv tracking to yes
df.loc[df['tracking'] == 'pu', 'tracking'] = 'yes'
df.loc[df['tracking'] == 'pv', 'tracking'] = 'yes'

# Drop rows with NaN values
df.dropna(axis = 'index', inplace = True)

# Encode categorical data
df = pd.get_dummies(df, columns = ['region', 'locality', 'category', 'shop', 'tracking'],
                    prefix = ['region', 'locality', 'category', 'shop', 'tracking'],
                    drop_first = False)
df = df.drop(['region_other', 'locality_multiple', 'category_other', 'shop_other', 'tracking_no'], axis=1)

# Build model
# Specify dependent variable vector y and independent variable matrix X
y = df.iloc[:, 0]
X = df.iloc[:, 1:]

# Build and fit regressor
# Random forest regression (no feature scaling needed)
from sklearn.ensemble import RandomForestRegressor
forest_regressor = RandomForestRegressor(
    n_estimators = 250,
    min_samples_split = 2,
    max_leaf_nodes = 8)
forest_regressor.fit(X.drop(['start_date', 'end_date'], axis=1), y)

# Save model
from sklearn.externals import joblib
joblib.dump(forest_regressor, 'cpc_model.pkl')

# Save training columns
columns = list(df.drop(['start_date', 'end_date'], axis=1).iloc[:, 1:].columns)
joblib.dump(columns, 'cpc_columns.pkl')

# Predict and print results
results = forest_regressor.predict(X.drop(['start_date', 'end_date'], axis=1)).round(4)
for result in results:
    print(result)
