# python -c 'import training; training.train("cost_per_impression", "pay_per_impression")'

# Import secrets
import config

# Import libraries
import numpy as np
import pandas as pd
import psycopg2 as pg
import pandas.io.sql as psql
import boto3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.externals import joblib

def train(output, filter = None):
    # Possible output values: 'cost_per_impression', 'cost_per_click', 'cost_per_purchase'
    # Possible filter values: 'pay_per_impression', 'pay_per_click'

    # Load phase data into dataframe
    connection = pg.connect(config.marketing_production)
    if filter == None:
        select = 'SELECT ' + output + ', ' + open('phases.sql', 'r').read()
    else:
        select = 'SELECT ' + output + ', ' + open('phases.sql', 'r').read() + 'AND ' + filter + ' = 1'
    df = pd.read_sql_query(select, connection)

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

    # Change pu and pv tracking to yes unless predicting cost per purchase,
    # else drop rows where tracking is no and then tracking column
    if output != 'cost_per_purchase':
        df.loc[df['tracking'] == 'pu', 'tracking'] = 'yes'
        df.loc[df['tracking'] == 'pv', 'tracking'] = 'yes'
    else:
        df = df[df.tracking != 'no']
        df = df.drop(['tracking'], axis=1)

    # Drop rows with NaN values
    df.dropna(axis = 'index', inplace = True)

    # Encode categorical data
    if output != 'cost_per_purchase':
        df = pd.get_dummies(df, columns = ['region', 'locality', 'category', 'shop', 'tracking'],
                            prefix = ['region', 'locality', 'category', 'shop', 'tracking'],
                            drop_first = False)
        df = df.drop(['region_other', 'locality_multiple', 'category_other',
                      'shop_other', 'tracking_no'], axis=1)
    else:
        df = pd.get_dummies(df, columns = ['region', 'locality', 'category', 'shop'],
                            prefix = ['region', 'locality', 'category', 'shop'],
                            drop_first = False)
        df = df.drop(['region_other', 'locality_multiple', 'category_other',
                      'shop_other'], axis=1)

    # Build models
    # Specify dependent variable vector y and independent variable matrix X
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    # Split dataset into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop([output], axis=1), df[output],
        test_size = 0.2, random_state = 0)

    # Scale features
    sc_X = StandardScaler()
    X_scaled = sc_X.fit_transform(X.values.astype(float))
    X_train_scaled = sc_X.fit_transform(X_train.values.astype(float))
    X_test_scaled = sc_X.transform(X_test.values.astype(float))
    sc_y = StandardScaler()
    y_scaled = sc_y.fit_transform(y.values.astype(float).reshape(-1, 1)).flatten()
    y_train_scaled = sc_y.fit_transform(y_train.values.astype(float).reshape(-1, 1)).flatten()

    # Build and fit regressors
    # Make scorer
    def mean_relative(y_pred, y_true):
      return 1 - np.mean(np.abs((y_pred - y_true) / y_true))

    mean_relative_score = make_scorer(mean_relative, greater_is_better = True)

    # Linear regression (library includes feature scaling)
    linear_regressor = LinearRegression()

    # Decision tree regression (no feature scaling needed)
    tree_regressor = DecisionTreeRegressor()
    tree_parameters = [{'min_samples_split': [4, 5, 6, 7, 8],
                        'max_leaf_nodes': [4, 5, 6, 7, 8]}]
    tree_grid = GridSearchCV(estimator = tree_regressor,
                               param_grid = tree_parameters,
                               scoring = mean_relative_score,
                               cv = 5,
                               n_jobs = -1,
                               iid = False)
    tree_grid_result = tree_grid.fit(X_train.drop(['start_date', 'end_date'], axis=1), y_train)
    best_tree_score = tree_grid_result.best_score_
    best_tree_parameters = tree_grid_result.best_params_
    tree_regressor = DecisionTreeRegressor(
                       min_samples_split = best_tree_parameters['min_samples_split'],
                       max_leaf_nodes = best_tree_parameters['max_leaf_nodes'])

    # Random forest regression (no feature scaling needed)
    forest_regressor = RandomForestRegressor()
    forest_parameters = [{'n_estimators': [100, 150, 200, 250],
                          'min_samples_split': [2, 3, 4, 5, 6],
                          'max_leaf_nodes': [4, 5, 6, 7, 8]}]
    forest_grid = GridSearchCV(estimator = forest_regressor,
                               param_grid = forest_parameters,
                               scoring = mean_relative_score,
                               cv = 5,
                               n_jobs = -1,
                               iid = False)
    forest_grid_result = forest_grid.fit(X_train.drop(['start_date', 'end_date'], axis=1), y_train)
    best_forest_score = forest_grid_result.best_score_
    best_forest_parameters = forest_grid_result.best_params_
    forest_regressor = RandomForestRegressor(
                       n_estimators = best_forest_parameters['n_estimators'],
                       min_samples_split = best_forest_parameters['min_samples_split'],
                       max_leaf_nodes = best_forest_parameters['max_leaf_nodes'])

    # SVR (needs feature scaling)
    def powerlist(start, times):
        array = []
        for i in range(0, times, 1):
            array.append(start * 2 ** i)
        return array
    svr_regressor = SVR()
    svr_parameters = [{'C': powerlist(0.01, 15), 'kernel': ['linear']},
                  {'C': powerlist(0.01, 15), 'kernel': ['poly'], 'degree': [2, 3, 4, 5], 'gamma': powerlist(0.0001, 15)},
                  {'C': powerlist(0.01, 15), 'kernel': ['rbf'], 'gamma': powerlist(0.0001, 15), 'epsilon': powerlist(0.0001, 15)}]
    svr_grid = GridSearchCV(estimator = svr_regressor,
                            param_grid = svr_parameters,
                            scoring = 'neg_mean_absolute_error',
                            cv = 5,
                            n_jobs = -1,
                            iid = False)
    svr_grid_result = svr_grid.fit(X_train_scaled, y_train_scaled)
    best_svr_score = svr_grid_result.best_score_
    best_svr_parameters = svr_grid_result.best_params_
    svr_regressor = SVR(kernel = best_svr_parameters['kernel'],
                        C = best_svr_parameters['C'],
                        gamma = best_svr_parameters['gamma'])

    # Evaluate models
    # Cross-validation score
    linear_score = np.mean(cross_val_score(estimator = linear_regressor, X = X_train, y = y_train, cv = 5, scoring = mean_relative_score))
    tree_score = np.mean(cross_val_score(estimator = tree_regressor, X = X_train.drop(['start_date', 'end_date'], axis=1), y = y_train, cv = 5, scoring = mean_relative_score))
    forest_score = np.mean(cross_val_score(estimator = forest_regressor, X = X_train.drop(['start_date', 'end_date'], axis=1), y = y_train, cv = 5, scoring = mean_relative_score))
    svr_score = np.mean(cross_val_score(estimator = svr_regressor, X = X_train_scaled, y = y_train_scaled, cv = 5, scoring = 'neg_mean_absolute_error'))

    # Fit regressors on training set
    linear_regressor.fit(X_train, y_train)
    tree_regressor.fit(X_train.drop(['start_date', 'end_date'], axis=1), y_train)
    forest_regressor.fit(X_train.drop(['start_date', 'end_date'], axis=1), y_train)
    svr_regressor.fit(X_train_scaled, y_train_scaled)

    # Predict test set results and calculate accuracy (1 - mean percentage error)
    linear_accu = mean_relative(linear_regressor.predict(X_test), y_test)
    tree_accu = mean_relative(tree_regressor.predict(X_test.drop(['start_date', 'end_date'], axis=1)), y_test)
    forest_accu = mean_relative(forest_regressor.predict(X_test.drop(['start_date', 'end_date'], axis=1)), y_test)
    svr_accu = mean_relative(sc_y.inverse_transform(svr_regressor.predict(X_test_scaled)), y_test)

    # Print model comparison
    print('linear_score: ' + str(linear_score))
    print('tree_score: ' + str(tree_score))
    print('forest_score: ' + str(forest_score))
    print('svr_score: ' + str(svr_score))
    print('linear_accu: ' + str(linear_accu))
    print('tree_accu: ' + str(tree_accu))
    print('forest_accu: ' + str(forest_accu))
    print('svr_accu: ' + str(svr_accu))
    print('best_tree_parameters: ' + str(best_tree_parameters))
    print('best_forest_parameters: ' + str(best_forest_parameters))
    print('best_svr_parameters: ' + str(best_svr_parameters))

    # Choose best model and fit with full dataset
    accuracies = {
        'linear_regressor': linear_accu,
        'tree_regressor': tree_accu,
        'forest_regressor': forest_accu,
        'svr_regressor': svr_accu
    }
    best_regressor = max(accuracies, key=accuracies.get)
    if best_regressor == 'tree_regressor' or best_regressor == 'forest_regressor':
        eval(best_regressor).fit(X.drop(['start_date', 'end_date'], axis=1), y)
    else:
        eval(best_regressor).fit(X, y)

    # Save model and columns to file
    joblib.dump(forest_regressor, output + '_model.pkl')
    columns = list(df.drop(['start_date', 'end_date'], axis=1).iloc[:, 1:].columns)
    joblib.dump(columns, output + '_columns.pkl')

    # Upload model and columns to S3
    s3 = boto3.client('s3')
    bucket_name = 'cpx-prediction'
    model_file = output + '_model.pkl'
    columns_file = output + '_columns.pkl'
    s3.upload_file(model_file, bucket_name, model_file)
    s3.upload_file(columns_file, bucket_name, columns_file)

    # Print actuals with predictions
    if best_regressor == 'tree_regressor' or best_regressor == 'forest_regressor':
        predictions = eval(best_regressor).predict(X.drop(['start_date', 'end_date'], axis=1)).round(4)
    else:
        predictions = eval(best_regressor).predict(X).round(4)

    print('actuals:')
    for actual in y:
        print(actual)

    print('predictions:')
    for prediction in predictions:
        print(prediction)
