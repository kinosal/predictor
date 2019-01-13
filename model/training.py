# Setup
# Import standard libraries
import numpy as np
import pandas as pd

# Import dataset to a pandas dataframe
df = pd.read_csv('phases_cpi.csv')

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

# Build models
# Specify dependent variable vector y and independent variable matrix X
# Consider using .values for easier and more constistent modeling
y = df.iloc[:, 0]
X = df.iloc[:, 1:]

# Split dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['cost_per_impression'], axis=1), df['cost_per_impression'],
    test_size = 0.2, random_state = 0)

# Scale features
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(X.values)
X_train_scaled = sc_X.fit_transform(X_train.values)
X_test_scaled = sc_X.transform(X_test.values)
sc_y = StandardScaler()
y_scaled = sc_y.fit_transform(y.values.reshape(-1, 1)).flatten()
y_train_scaled = sc_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

# Build and fit regressors
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# Make scorer
from sklearn.metrics import make_scorer
def mean_relative(y_pred, y_true):
  return 1 - np.mean(np.abs((y_pred - y_true) / y_true))

mean_relative_score = make_scorer(mean_relative, greater_is_better = True)

# Linear regression (library includes feature scaling)
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()

# Decision tree regression (no feature scaling needed)
from sklearn.tree import DecisionTreeRegressor
tree_regressor = DecisionTreeRegressor()
tree_parameters = [{'min_samples_split': [2, 3, 4, 5, 6],
                    'max_leaf_nodes': [4, 5, 6, 7, 8]}]
tree_grid = GridSearchCV(estimator = tree_regressor,
                           param_grid = tree_parameters,
                           scoring = 'neg_mean_absolute_error',
                           cv = 5,
                           n_jobs = -1)
tree_grid_result = tree_grid.fit(X_train.drop(['start_date', 'end_date'], axis=1), y_train)
best_tree_score = tree_grid_result.best_score_
best_tree_parameters = tree_grid_result.best_params_
tree_regressor = DecisionTreeRegressor(
                   min_samples_split = best_tree_parameters['min_samples_split'],
                   max_leaf_nodes = best_tree_parameters['max_leaf_nodes'])

# Random forest regression (no feature scaling needed)
from sklearn.ensemble import RandomForestRegressor
forest_regressor = RandomForestRegressor()
forest_parameters = [{'n_estimators': [100, 150, 200, 250],
                      'min_samples_split': [2, 3, 4, 5, 6],
                      'max_leaf_nodes': [4, 5, 6, 7, 8]}]
forest_grid = GridSearchCV(estimator = forest_regressor,
                           param_grid = forest_parameters,
                           scoring = 'neg_mean_absolute_error',
                           cv = 5,
                           n_jobs = -1)
forest_grid_result = forest_grid.fit(X_train.drop(['start_date', 'end_date'], axis=1), y_train)
best_forest_score = forest_grid_result.best_score_
best_forest_parameters = forest_grid_result.best_params_
forest_regressor = RandomForestRegressor(
                   n_estimators = best_forest_parameters['n_estimators'],
                   min_samples_split = best_forest_parameters['min_samples_split'],
                   max_leaf_nodes = best_forest_parameters['max_leaf_nodes'])

# SVR (needs feature scaling)
from sklearn.svm import SVR
svr_regressor = SVR()
svr_parameters = [{'C': [0.01, 0.1, 1, 10, 20, 30], 'kernel': ['linear']},
              {'C': [0.01, 0.1, 1, 10, 20, 30], 'kernel': ['poly'], 'degree': [2, 3, 4, 5], 'gamma': [0.001, 0.01, 0.1, 1]},
              {'C': [0.01, 0.1, 1, 10, 20, 30], 'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1]}]
svr_grid = GridSearchCV(estimator = svr_regressor,
                        param_grid = svr_parameters,
                        scoring = 'neg_mean_absolute_error',
                        cv = 5,
                        n_jobs = -1)
svr_grid_result = svr_grid.fit(X_train_scaled, y_train_scaled)
best_svr_score = svr_grid_result.best_score_
best_svr_parameters = svr_grid_result.best_params_
svr_regressor = SVR(kernel = best_svr_parameters['kernel'],
                    C = best_svr_parameters['C'],
                    gamma = best_svr_parameters['gamma'])

# Evaluate models
# Cross-validation score
linear_score = np.mean(cross_val_score(estimator = linear_regressor, X = X_train, y = y_train, cv = 5, scoring = 'neg_mean_absolute_error'))
tree_score = np.mean(cross_val_score(estimator = tree_regressor, X = X_train.drop(['start_date', 'end_date'], axis=1), y = y_train, cv = 5, scoring = 'neg_mean_absolute_error'))
forest_score = np.mean(cross_val_score(estimator = forest_regressor, X = X_train.drop(['start_date', 'end_date'], axis=1), y = y_train, cv = 5, scoring = 'neg_mean_absolute_error'))
svr_score = np.mean(cross_val_score(estimator = svr_regressor, X = X_train_scaled, y = y_train_scaled, cv = 5, scoring = 'neg_mean_absolute_error'))

# Fit regressors on training set
linear_regressor.fit(X_train, y_train)
tree_regressor.fit(X_train.drop(['start_date', 'end_date'], axis=1), y_train)
forest_regressor.fit(X_train.drop(['start_date', 'end_date'], axis=1), y_train)
svr_regressor.fit(X_train_scaled, y_train_scaled)

# Predict test set results and calculate accuracy (1 - mean percentage error)
lin_accu = mean_relative(linear_regressor.predict(X_test), y_test)
tree_accu = mean_relative(tree_regressor.predict(X_test.drop(['start_date', 'end_date'], axis=1)), y_test)
forest_accu = mean_relative(forest_regressor.predict(X_test.drop(['start_date', 'end_date'], axis=1)), y_test)
svr_accu = mean_relative(sc_y.inverse_transform(svr_regressor.predict(X_test_scaled)), y_test)
