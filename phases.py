# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataset to a pandas dataframe
df = pd.read_csv('phases_cpc.csv')

# Visualize distributions of numerical features
quan = list(df.loc[:,df.dtypes != 'object'].columns.values)
qual = list(df.loc[:,df.dtypes == 'object'].columns.values)
temp = pd.melt(df, value_vars = quan)
grid = sns.FacetGrid(temp, col = "variable", col_wrap = 6, size = 3.0,
                     aspect = 0.8, sharex = False, sharey = False)
grid.map(sns.distplot, "value")
plt.show()

# Visualize correlations between features
colormap = plt.cm.RdBu
plt.figure(figsize = (28,24))
sns.heatmap(df._get_numeric_data().astype(float).corr(), linewidths = 0.1,
            vmax = 1.0, square = True, cmap = colormap, linecolor = 'white',
            annot = True)

# Investigate correlation between dependent and independent variables
corr = df.corr(method = 'pearson').iloc[0]
corr.sort_values(ascending = False).head()
corr.sort_values(ascending = True).head()

# Preprocess data
# Encode categorical data
df = pd.get_dummies(df, columns=['start month', 'end month', 'country', 'region', 'content', 'shop'],
                    prefix=['start month', 'end month', 'weekday', 'month', 'week', 'shop'])

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

svr_regressor = SVR(kernel = best_parameters['kernel'],
                    C = best_parameters['C'], gamma = best_parameters['gamma'])
svr_regressor.fit(X_train_scaled, y_train_scaled)
y_pred_svr = sc_y.inverse_transform(svr_regressor.predict(X_test_scaled))
svr_accu = np.average(np.absolute(np.divide(y_pred_svr, y_test) - 1))
