# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataset to a pandas dataframe
df = pd.read_csv('purchases.csv')

# Develop new features
def add_mean(variable, periods):
  loc = df.columns.get_loc('1-' + variable)
  df[variable + '_mean_' + str(periods)] = df.iloc[:, loc:loc + periods].mean(axis = 1)

add_mean(variable = 'prepur', periods = 7)
add_mean(variable = 'prepur', periods = 28)
add_mean(variable = 'facebook-impressions', periods = 7)
add_mean(variable = 'facebook-impressions', periods = 28)
add_mean(variable = 'facebook-clicks', periods = 7)
add_mean(variable = 'facebook-clicks', periods = 28)
add_mean(variable = 'facebook-cost', periods = 7)
add_mean(variable = 'facebook-cost', periods = 28)
add_mean(variable = 'instagram-impressions', periods = 7)
add_mean(variable = 'instagram-impressions', periods = 28)
add_mean(variable = 'instagram-clicks', periods = 7)
add_mean(variable = 'instagram-clicks', periods = 28)
add_mean(variable = 'instagram-cost', periods = 7)
add_mean(variable = 'instagram-cost', periods = 28)
add_mean(variable = 'google_search-impressions', periods = 7)
add_mean(variable = 'google_search-impressions', periods = 28)
add_mean(variable = 'google_search-clicks', periods = 7)
add_mean(variable = 'google_search-clicks', periods = 28)
add_mean(variable = 'google_search-cost', periods = 7)
add_mean(variable = 'google_search-cost', periods = 28)

# Drop irrelevant features

drop_elements = []

for i in range(2, 29):
  drop_elements.append(str(i) + '-prepur')
  drop_elements.append(str(i) + '-facebook-impressions')
  drop_elements.append(str(i) + '-facebook-clicks')
  drop_elements.append(str(i) + '-facebook-cost')
  drop_elements.append(str(i) + '-instagram-impressions')
  drop_elements.append(str(i) + '-instagram-clicks')
  drop_elements.append(str(i) + '-instagram-cost')
  drop_elements.append(str(i) + '-google_search-impressions')
  drop_elements.append(str(i) + '-google_search-clicks')
  drop_elements.append(str(i) + '-google_search-cost')

for i in range(1, 29):
  drop_elements.append(str(i) + '-google_display-impressions')
  drop_elements.append(str(i) + '-google_display-clicks')
  drop_elements.append(str(i) + '-google_display-cost')
  drop_elements.append(str(i) + '-twitter-impressions')
  drop_elements.append(str(i) + '-twitter-clicks')
  drop_elements.append(str(i) + '-twitter-cost')

df = df.drop(drop_elements, axis = 1)

# Visualize distributions of numerical features
sns.distplot(df['purchases'])
sns.distplot(df['1-prepur'])
sns.distplot(df['prepur_mean_7'])
sns.distplot(df['1-facebook-impressions'])
sns.distplot(df['facebook-impressions_mean_7'])

quan = list(df.loc[:,df.dtypes != 'object'].columns.values)
qual = list(df.loc[:,df.dtypes == 'object'].columns.values)
temp = pd.melt(df, value_vars = quan)
grid = sns.FacetGrid(temp, col = "variable", col_wrap = 6, size = 3.0,
                     aspect = 0.8, sharex = False, sharey = False)
grid.map(sns.distplot, "value")
plt.show()

# Visualize correlations between features
colormap = plt.cm.RdBu
plt.figure(figsize=(28,24))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df._get_numeric_data().astype(float).corr(),linewidths=0.1,vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)

# Investigate correlation between dependent and independent variables
corr = df.corr(method = 'pearson').iloc[0]
corr.sort_values(ascending = False).head()
corr.sort_values(ascending = True).head()

# Preprocess data
# Encode categorical data
df = pd.get_dummies(df, columns=['weekday', 'month', 'week'], prefix=['weekday', 'month', 'week'])

# Specify dependent variable vector y and independent variable matrix X
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values

# Encode categorical data (alternative)
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 0] = labelencoder_X1.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
'''

# Split dataset into training and test set
# Not needed if cross-validation used for evaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Scale features
# Not needed for linear regression, decision trees (incl XGBoost) or random forest
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

# Polynomial regression (library includes feature scaling)
from sklearn.preprocessing import PolynomialFeatures
polynomial_transformer = PolynomialFeatures(degree = 2)
X_poly = polynomial_transformer.fit_transform(X)
X_train_poly = polynomial_transformer.fit_transform(X_train)
X_test_poly = polynomial_transformer.fit_transform(X_test)
linear_regressor_poly = LinearRegression()
linear_regressor_poly.fit(X_train_poly, y_train)

# SVR (needs feature scaling)
from sklearn.svm import SVR
svr_regressor = SVR(kernel = 'rbf')
svr_regressor.fit(X_train_scaled, y_train_scaled)

# Decision tree regression (no feature scaling needed)
from sklearn.tree import DecisionTreeRegressor
tree_regressor = DecisionTreeRegressor(random_state = 0)
tree_regressor.fit(X_train, y_train)

# Random forest regression (no feature scaling needed)
from sklearn.ensemble import RandomForestRegressor
forest_regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
forest_regressor.fit(X_train, y_train)

# ANN regression
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

def ann():
  ann = Sequential()
  ann.add(Dense(units = X_train.shape[1], kernel_initializer = 'normal', activation = 'relu', input_dim = X_train.shape[1]))
  ann.add(Dense(units = int(X_train.shape[1]/2), kernel_initializer = 'normal', activation = 'relu'))
  ann.add(Dense(units = 1, kernel_initializer = 'normal', activation = 'linear'))
  ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
  return ann

ann_regressor = KerasRegressor(build_fn = ann, batch_size = 5, epochs = 50)

# Embed feature scaling in pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
ann_estimators = []
ann_estimators.append(('standardize', StandardScaler()))
ann_estimators.append(('mlp', ann_regressor))
ann_pipeline = Pipeline(ann_estimators)

ann_pipeline.fit(X_train, y_train)

# XGBoost
from xgboost import XGBRegressor
xgb_regressor = XGBRegressor()
xgb_regressor.fit(X_train, y_train)

# Gradient Boosting
from sklearn import ensemble
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)

# KNN
from sklearn.neighbors import KNeighborsRegressor
knn_regressor = KNeighborsRegressor()

# Evaluate model
# Cross-validation score
from sklearn.metrics import make_scorer
def mean_relative(y_pred, y_test):
  return np.average(np.absolute(np.divide(y_pred, y_test) - 1))
mean_relative_error = make_scorer(mean_relative)

from sklearn.model_selection import KFold
kfold = KFold(n_splits = 5, random_state = 0)
scoring = 'neg_mean_squared_error'

from sklearn.model_selection import cross_val_score
scores_lin = cross_val_score(estimator = linear_regressor, X = X_train, y = y_train, cv = kfold, scoring = scoring)
scores_poly = cross_val_score(estimator = linear_regressor_poly, X = X_train_poly, y = y_train, cv = kfold, scoring = scoring)
scores_svr = cross_val_score(estimator = svr_regressor, X = X_train_scaled, y = y_train_scaled, cv = kfold, scoring = scoring)
scores_tree = cross_val_score(estimator = tree_regressor, X = X_train, y = y_train, cv = kfold, scoring = scoring)
scores_forest = cross_val_score(estimator = forest_regressor, X = X_train, y = y_train, cv = kfold, scoring = scoring)
scores_ann = cross_val_score(estimator = ann_pipeline, X = X_train, y = y_train, cv = kfold, scoring = scoring)
scores_xgb = cross_val_score(estimator = xgb_regressor, X = X_train, y = y_train, cv = kfold, scoring = scoring)
scores_knn = cross_val_score(estimator = knn_regressor, X = X_train, y = y_train, cv = kfold, scoring = scoring)

scores_avg_lin = scores_lin.mean()
scores_avg_poly = scores_poly.mean()
scores_avg_svr = scores_svr.mean()
scores_avg_tree = scores_tree.mean()
scores_avg_forest = scores_forest.mean()
scores_avg_ann = scores_ann.mean()

scores_std = scores.std()

# Predict test set results
y_pred_lin = linear_regressor.predict(X_test)
y_pred_poly = linear_regressor_poly.predict(X_test_poly)
y_pred_svr = sc_y.inverse_transform(svr_regressor.predict(X_test_scaled))
y_pred_tree = tree_regressor.predict(X_test)
y_pred_forest = forest_regressor.predict(X_test)
y_pred_ann = ann_pipeline.predict(X_test)
y_pred_xgb = xgb_regressor.predict(X_test)

# Apply grid search
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.01, 0.1, 1, 10, 20], 'kernel': ['linear']},
              {'C': [0.01, 0.1, 1, 10, 20], 'kernel': ['poly'], 'degree': [2, 3, 4, 5], 'gamma': [0.001, 0.01, 0.1, 1]},
              {'C': [0.01, 0.1, 1, 10, 20], 'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1]}]
grid_search = GridSearchCV(estimator = svr_regressor,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 5,
                           n_jobs = -1)
grid_search_result = grid_search.fit(X_train_scaled, y_train_scaled)
best_score = grid_search_result.best_score_
best_parameters = grid_search_result.best_params_

# Predict test set results after grid search
svr_regressor = SVR(kernel = best_parameters['kernel'], C = best_parameters['C'], gamma = best_parameters['gamma'])
svr_regressor.fit(X_train_scaled, y_train_scaled)
y_pred_svr = sc_y.inverse_transform(svr_regressor.predict(X_test_scaled))

# Overall accuracy
test_sum = sum(y_test)
lin_accu = sum(y_pred_lin) / test_sum
poly_accu = sum(y_pred_poly) / test_sum
svr_accu = sum(y_pred_svr) / test_sum
tree_accu = sum(y_pred_tree) / test_sum
forest_accu = sum(y_pred_forest) / test_sum

# Average daily accuracy
lin_accu_daily = np.average(np.absolute(np.divide(y_pred_lin, y_test) - 1))
poly_accu_daily = np.average(np.absolute(np.divide(y_pred_poly, y_test) - 1))
svr_accu_daily = np.average(np.absolute(np.divide(y_pred_svr, y_test) - 1))
tree_accu_daily = np.average(np.absolute(np.divide(y_pred_tree, y_test) - 1))
forest_accu_daily = np.average(np.absolute(np.divide(y_pred_forest, y_test) - 1))
ann_accu_daily = np.average(np.absolute(np.divide(y_pred_ann, y_test) - 1))

# Feature Selection and Importance
from sklearn.feature_selection import RFE
estimator = SVR(kernel = 'linear')
selector = RFE(estimator, 10, step = 1)
selector = selector.fit(X_train, y_train)
ranking = selector.ranking_

# Implement Backward Elimination with OLS (for linear regression)

# Add column of ones to X to account for linear regression intercept
X = np.append(arr = np.ones((181, 1)).astype(int), values = X, axis = 1)

import statsmodels.formula.api as sm
def backward_elimination(y, x, p):
  ars_old = 0
  elim = []
  while len(x[0]) > 0:
    regressor_OLS = sm.OLS(y, x).fit()
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

def adjust_test(x, elim):
  for i in range(len(elim)):
    x = np.delete(x, elim[i], 1)
  return x

back_elim = backward_elimination(y_train, X_train, 0.05)
X_opt = back_elim[0]
elim = back_elim[1]
linear_regressor_elim = LinearRegression()
linear_regressor_elim.fit(X_opt, y_train)
X_test_elim = adjust_test(X_test, elim)
y_pred_elim = linear_regressor_elim.predict(X_test_elim)
