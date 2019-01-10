# Import libraries
import numpy as np
import pandas as pd

# Import dataset to a pandas dataframe
df = pd.read_csv('phases_cpc.csv')

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
df = df.drop(['facebook_likes'], axis=1)

# Replace 0s with NaN where appropriate
# columns = ['facebook_likes']
# for column in columns:
#   df[column].replace(0, np.nan, inplace=True)

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

# Preprocess data
# Drop rows with NaN values
df.dropna(axis = 'index', inplace = True)

# Encode categorical data
df = pd.get_dummies(df, columns = ['region', 'locality', 'category', 'shop', 'tracking'],
                    prefix = ['region', 'locality', 'category', 'shop', 'tracking'],
                    drop_first = True)

# Specify dependent variable vector y and independent variable matrix X
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values

# Split dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Build and fit regressor
# Random forest regression (no feature scaling needed)
from sklearn.ensemble import RandomForestRegressor
forest_regressor = RandomForestRegressor(n_estimators = 250, random_state = 0)
forest_regressor.fit(X_train, y_train)

# Save model
from sklearn.externals import joblib
joblib.dump(forest_regressor, 'cpc_model.pkl')

# Save training columns
columns = list(df.iloc[:, 1:].columns)
joblib.dump(columns, 'cpc_columns.pkl')
