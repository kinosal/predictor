# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# Import dataset to a pandas dataframe
df = pd.read_csv('phases_cpi.csv')

# View summary
summary = df.describe()

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

# Put rare values and to bucket
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

# Describe again
summary = df.describe()

# Visualize distributions of numerical features
quan = list(df.loc[:,df.dtypes != 'object'].columns.values)
qual = list(df.loc[:,df.dtypes == 'object'].columns.values)
temp = pd.melt(df, value_vars = quan)
grid = sns.FacetGrid(temp, col = 'variable', col_wrap = 6, size = 3.0,
                     aspect = 0.8, sharex = False, sharey = False)
grid.map(sns.distplot, 'value')
plt.show()

# Visualize correlations between features
colormap = plt.cm.RdBu
plt.figure(figsize = (12,12))
sns.heatmap(df._get_numeric_data().astype(float).corr(), linewidths = 0.1,
            vmax = 1.0, square = True, cmap = colormap, linecolor = 'white',
            annot = True)
plt.show()

# Investigate correlation between dependent and independent variables
corr = df.corr(method = 'pearson').iloc[0]
corr.sort_values(ascending = True)

# Preprocess data
# Drop rows with NaN values
df.dropna(axis = 'index', inplace = True)

# Encode categorical data
df = pd.get_dummies(df, columns=['region', 'locality', 'category', 'shop', 'tracking'],
                    prefix=['region', 'locality', 'category', 'shop', 'tracking'],
                    drop_first=True)

# Specify dependent variable vector y and independent variable matrix X
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values

# Split dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Scale features
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(X)
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
sc_y = StandardScaler()
y_scaled = sc_y.fit_transform(y.reshape(-1, 1)).flatten()
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# Build and fit regressors
# Linear regression (library includes feature scaling)
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_lin = linear_regressor.predict(X_test)
lin_accu = np.average(np.absolute(np.divide(y_pred_lin, y_test) - 1))

# Implement backward elimination with OLS
import statsmodels.formula.api as smf
def backward_elimination_training(y, x, p):
  ars_old = 0
  elim = []
  while len(x[0]) > 0:
    regressor_OLS = smf.OLS(y, x).fit()
    ars = regressor_OLS.rsquared_adj
    if ars < ars_old - 0.001:
      print (summary_old)
      return [x_old, elim_old]
    else:
      maxp = max(regressor_OLS.pvalues)
      if maxp <= p:
        print (regressor_OLS.summary())
        return [x, elim]
      else:
        x_old = x
        ars_old = ars
        elim_old = elim[:]
        summary_old = regressor_OLS.summary()
        argmaxp = regressor_OLS.pvalues.argmax()
        elim.append(argmaxp)
        x = np.delete(x, argmaxp, 1)

# Method to run directly on the dataframe to keep column names for analysis
def backward_elimination_df(y, x, p):
  ars_old = 0
  elim = []
  while len(x.columns) > 0:
    regressor_OLS = smf.OLS(y, x).fit()
    ars = regressor_OLS.rsquared_adj
    if ars < ars_old - 0.001:
      print (summary_old)
      return [x_old, elim_old]
    else:
      maxp = max(regressor_OLS.pvalues)
      if maxp <= p:
        print (regressor_OLS.summary())
        return [x, elim]
      else:
        x_old = x
        ars_old = ars
        elim_old = elim[:]
        summary_old = regressor_OLS.summary()
        argmaxp = regressor_OLS.pvalues.argmax()
        elim.append(argmaxp)
        x = x_old.drop([argmaxp], axis=1)

# Add column of ones to df to account for linear regression intercept
df['intercept'] = 1
back_elim = backward_elimination_df(df['cost_per_impression'], df.drop(['cost_per_impression'], axis=1), 0.05)

# Add column of ones to X to account for linear regression intercept
X = np.append(arr = np.ones((X.shape[0], 1)).astype(int), values = X, axis = 1)

def adjust_test(x, elim):
  for i in range(len(elim)):
    x = np.delete(x, elim[i], 1)
  return x

back_elim = backward_elimination_training(y_train, X_train, 0.05)
X_opt = back_elim[0]
elim = back_elim[1]
linear_regressor_elim = LinearRegression()
linear_regressor_elim.fit(X_opt, y_train)
X_test_elim = adjust_test(X_test, elim)
y_pred_elim = linear_regressor_elim.predict(X_test_elim)
elim_accu = np.average(np.absolute(np.divide(y_pred_elim, y_test) - 1))

# Decision tree regression (no feature scaling needed)
from sklearn.tree import DecisionTreeRegressor
tree_regressor = DecisionTreeRegressor(random_state = 0)
tree_regressor.fit(X_train, y_train)
y_pred_tree = tree_regressor.predict(X_test)
tree_accu = np.average(np.absolute(np.divide(y_pred_tree, y_test) - 1))

# Random forest regression (no feature scaling needed)
from sklearn.ensemble import RandomForestRegressor
forest_regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
forest_regressor.fit(X_train, y_train)
y_pred_forest = forest_regressor.predict(X_test)
forest_accu = np.average(np.absolute(np.divide(y_pred_forest, y_test) - 1))

# SVR (needs feature scaling)
from sklearn.svm import SVR
svr_regressor = SVR()

from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.01, 0.1, 1, 10, 20, 30], 'kernel': ['linear']},
              {'C': [0.01, 0.1, 1, 10, 20, 30], 'kernel': ['poly'], 'degree': [2, 3, 4, 5], 'gamma': [0.001, 0.01, 0.1, 1]},
              {'C': [0.01, 0.1, 1, 10, 20, 30], 'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1]}]
grid_search = GridSearchCV(estimator = svr_regressor,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 5,
                           n_jobs = -1)
grid_search_result = grid_search.fit(X_train_scaled, y_train_scaled)
best_score = grid_search_result.best_score_
best_parameters = grid_search_result.best_params_

svr_regressor = SVR(kernel = best_parameters['kernel'],
                    C = best_parameters['C'], gamma = best_parameters['gamma'])
svr_regressor.fit(X_train_scaled, y_train_scaled)
y_pred_svr = sc_y.inverse_transform(svr_regressor.predict(X_test_scaled))
svr_accu = np.average(np.absolute(np.divide(y_pred_svr, y_test) - 1))
