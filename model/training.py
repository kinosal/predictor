"""Run train(output, constraint) for full pipeline to train, select and save
best model predicting phase performance, e.g.
python -c 'import training; training.train("cost_per_impression", "pay_per_impression")'
"""

# Import secrets
import config

# Import libraries
import numpy as np
import pandas as pd
import psycopg2 as pg
import boto3
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def mean_relative(y_pred, y_true):
    """Helper function to calculate mean relative deviation from two vectors
    = 1 - mean percentage error
    """
    return 1 - np.mean(np.abs((y_pred - y_true) / y_true))


def load(output, constraint):
    """Load phase data into dataframe"""

    connection = pg.connect(config.marketing_production)
    if constraint is None:
        select = 'SELECT ' + output + ', ' + open('phases.sql', 'r').read()
    else:
        select = 'SELECT ' + output + ', ' + open('phases.sql', 'r').read() + \
                 'AND ' + constraint + ' = 1'
    return pd.read_sql_query(select, connection)


def preprocess(data, output):
    """Preprocess data"""

    # Drop rows where budget is 0,
    data = data[data.budget != 0]

    # Drop columns with more than 25% missing data
    rows = data[output].count()
    for column in list(data.columns):
        if data[column].count() < rows * 0.75:
            data = data.drop([column], axis=1)

    # Replace 0s with NaN where appropriate
    columns = ['num_events', 'ticket_capacity',
               'average_ticket_price', 'facebook_likes']
    for column in list(set(columns).intersection(list(data.columns))):
        data[column].replace(0, np.nan, inplace=True)

    # Put rare values into other bucket
    threshold = 0.05
    to_buckets = ['region', 'category', 'shop']
    for column in list(set(to_buckets).intersection(list(data.columns))):
        results = data[column].count()
        groups = data.groupby([column])[column].count()
        for bucket in groups.index:
            if groups.loc[bucket] < results * threshold:
                data.loc[data[column] == bucket, column] = 'other'

    # Change custom shop to other
    data.loc[data['shop'] == 'custom', 'shop'] = 'other'

    # Change pu and pv tracking to yes unless predicting cost per purchase,
    # else drop rows where tracking is no and then tracking column
    if output != 'cost_per_purchase':
        data.loc[data['tracking'] == 'pu', 'tracking'] = 'yes'
        data.loc[data['tracking'] == 'pv', 'tracking'] = 'yes'
    else:
        data = data[data.tracking != 'no']
        data = data.drop(['tracking'], axis=1)

    # Drop rows with NaN values
    data.dropna(axis='index', inplace=True)

    # Encode categorical data
    if output != 'cost_per_purchase':
        data = pd.get_dummies(
            data,
            columns=['region', 'locality', 'category', 'shop', 'tracking'],
            prefix=['region', 'locality', 'category', 'shop', 'tracking'],
            drop_first=False)
        data = data.drop(['region_other', 'locality_multiple',
                          'category_other', 'shop_other', 'tracking_no'],
                         axis=1)
    else:
        data = pd.get_dummies(
            data,
            columns=['region', 'locality', 'category', 'shop'],
            prefix=['region', 'locality', 'category', 'shop'],
            drop_first=False)
        data = data.drop(['region_other', 'locality_multiple',
                          'category_other', 'shop_other'], axis=1)

    # Specify dependent variable vector y and independent variable matrix X
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]

    # Split dataset into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop([output], axis=1), data[output], test_size=0.2)

    # Scale features
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X.values.astype(float))
    X_train_scaled = X_scaler.transform(X_train.values.astype(float))
    X_test_scaled = X_scaler.transform(X_test.values.astype(float))
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(
        y_train.values.astype(float).reshape(-1, 1)).flatten()

    return [X, y, X_train, y_train, X_test, y_test,
            X_scaled, X_train_scaled, y_train_scaled, X_test_scaled, y_scaler]


def build(X_train, y_train, X_train_scaled, y_train_scaled):
    """Build and return models"""

    # Define helper function to create lists for search grids
    def powerlist(start, times):
        array = []
        for i in range(0, times, 1):
            array.append(start * 2 ** i)
        return array

    # Linear regression (library includes feature scaling)
    linear_regressor = LinearRegression()
    linear_score = np.mean(cross_val_score(
        estimator=linear_regressor, X=X_train, y=y_train,
        cv=5, scoring='r2'))

    # Decision tree regression (no feature scaling needed)
    tree_regressor = DecisionTreeRegressor()
    tree_parameters = [{'min_samples_split': list(range(2, 9, 1)),
                        'max_leaf_nodes': list(range(2, 9, 1)),
                        'criterion': ['mae']}]
    tree_grid = GridSearchCV(estimator=tree_regressor,
                             param_grid=tree_parameters,
                             scoring='r2',
                             cv=5,
                             n_jobs=-1,
                             iid=False)
    tree_grid_result = tree_grid.fit(
        X_train.drop(['start_date', 'end_date'], axis=1), y_train)
    best_tree_parameters = tree_grid_result.best_params_
    tree_score = tree_grid_result.best_score_
    tree_regressor = DecisionTreeRegressor(
        min_samples_split=best_tree_parameters['min_samples_split'],
        max_leaf_nodes=best_tree_parameters['max_leaf_nodes'],
        criterion='mae')

    # Random forest regression (no feature scaling needed)
    forest_regressor = RandomForestRegressor()
    forest_parameters = [{'n_estimators': powerlist(10, 5),
                          'min_samples_split': list(range(2, 9, 1)),
                          'max_leaf_nodes': list(range(2, 9, 1)),
                          'criterion': ['mae']}]
    forest_grid = GridSearchCV(estimator=forest_regressor,
                               param_grid=forest_parameters,
                               scoring='r2',
                               cv=5,
                               n_jobs=-1,
                               iid=False)
    forest_grid_result = forest_grid.fit(
        X_train.drop(['start_date', 'end_date'], axis=1), y_train)
    best_forest_parameters = forest_grid_result.best_params_
    forest_score = forest_grid_result.best_score_
    forest_regressor = RandomForestRegressor(
        n_estimators=best_forest_parameters['n_estimators'],
        min_samples_split=best_forest_parameters['min_samples_split'],
        max_leaf_nodes=best_forest_parameters['max_leaf_nodes'],
        criterion='mae')

    # SVR (needs feature scaling)
    svr_regressor = SVR()
    svr_parameters = [
        {'C': powerlist(0.01, 10), 'kernel': ['linear']},
        {'C': powerlist(0.01, 10), 'kernel': ['poly'], 'degree': [2, 3, 4, 5],
         'gamma': powerlist(0.0000001, 10)},
        {'C': powerlist(0.01, 10), 'kernel': ['rbf'],
         'gamma': powerlist(0.0000001, 10), 'epsilon': powerlist(0.0001, 10)}]
    svr_grid = GridSearchCV(estimator=svr_regressor,
                            param_grid=svr_parameters,
                            scoring='r2',
                            cv=5,
                            n_jobs=-1,
                            iid=False)
    svr_grid_result = svr_grid.fit(X_train_scaled, y_train_scaled)
    best_svr_parameters = svr_grid_result.best_params_
    svr_score = svr_grid_result.best_score_
    if best_svr_parameters['kernel'] == 'linear':
        svr_regressor = SVR(kernel=best_svr_parameters['kernel'],
                            C=best_svr_parameters['C'])
    elif best_svr_parameters['kernel'] == 'poly':
        svr_regressor = SVR(kernel=best_svr_parameters['kernel'],
                            C=best_svr_parameters['C'],
                            gamma=best_svr_parameters['gamma'])
    else:
        svr_regressor = SVR(kernel=best_svr_parameters['kernel'],
                            C=best_svr_parameters['C'],
                            gamma=best_svr_parameters['gamma'],
                            epsilon=best_svr_parameters['epsilon'])

    print('best_tree_parameters: ' + str(best_tree_parameters))
    print('best_forest_parameters: ' + str(best_forest_parameters))
    print('best_svr_parameters: ' + str(best_svr_parameters))
    print('linear_r2_score: ' + str(linear_score))
    print('tree_r2_score: ' + str(tree_score))
    print('forest_r2_score: ' + str(forest_score))
    print('svr_r2_score: ' + str(svr_score))

    return [linear_regressor, tree_regressor, forest_regressor, svr_regressor]


def evaluate(linear_regressor, tree_regressor, forest_regressor, svr_regressor,
             X_train, y_train, X_train_scaled, y_train_scaled,
             X_test, y_test, X_test_scaled, y_scaler):
    """Evaluate models and return best regressor"""

    # Fit regressors on training set
    linear_regressor.fit(X_train, y_train)
    tree_regressor.fit(
        X_train.drop(['start_date', 'end_date'], axis=1), y_train)
    forest_regressor.fit(
        X_train.drop(['start_date', 'end_date'], axis=1), y_train)
    svr_regressor.fit(X_train_scaled, y_train_scaled)

    # Predict training results and calculate accuracy
    linear_train_accu = mean_relative(linear_regressor.predict(
        X_train), y_train)
    tree_train_accu = mean_relative(tree_regressor.predict(
        X_train.drop(['start_date', 'end_date'], axis=1)), y_train)
    forest_train_accu = mean_relative(forest_regressor.predict(
        X_train.drop(['start_date', 'end_date'], axis=1)), y_train)
    svr_train_accu = mean_relative(y_scaler.inverse_transform(
        svr_regressor.predict(X_train_scaled)), y_train)

    # Predict test results and calculate accuracy
    linear_test_accu = mean_relative(linear_regressor.predict(X_test), y_test)
    tree_test_accu = mean_relative(tree_regressor.predict(
        X_test.drop(['start_date', 'end_date'], axis=1)), y_test)
    forest_test_accu = mean_relative(forest_regressor.predict(
        X_test.drop(['start_date', 'end_date'], axis=1)), y_test)
    svr_test_accu = mean_relative(y_scaler.inverse_transform(
        svr_regressor.predict(X_test_scaled)), y_test)

    accuracies = {'linear_regressor': linear_test_accu,
                  'tree_regressor': tree_test_accu,
                  'forest_regressor': forest_test_accu,
                  'svr_regressor': svr_test_accu}
    best_regressor = eval(max(accuracies, key=accuracies.get))

    print('linear_train_accu: ' + str(linear_train_accu))
    print('tree_train_accu: ' + str(tree_train_accu))
    print('forest_train_accu: ' + str(forest_train_accu))
    print('svr_train_accu: ' + str(svr_train_accu))
    print('linear_test_accu: ' + str(linear_test_accu))
    print('tree_test_accu: ' + str(tree_test_accu))
    print('forest_test_accu: ' + str(forest_test_accu))
    print('svr_test_accu: ' + str(svr_test_accu))

    return best_regressor


def save(regressor, X, output, constraint):
    """Save model and columns to file"""

    if str(regressor).split('(')[0] in (
            'DecisionTreeRegressor', 'RandomForestRegressor'):
        columns = list(
            X.drop(['start_date', 'end_date'], axis=1).columns)
    else:
        columns = list(X.columns)
    try:
        joblib.dump(
            regressor, output + '_' + constraint + '_model.pkl')
        joblib.dump(columns, output + '_' + constraint + '_columns.pkl')
    except TypeError:
        joblib.dump(regressor, output + '_model.pkl')
        joblib.dump(columns, output + '_columns.pkl')


def upload(output, constraint):
    """Upload model and columns to S3"""

    s3_connection = boto3.client('s3')
    bucket_name = 'cpx-prediction'
    model_file = output + '_' + constraint + '_model.pkl'
    columns_file = output + '_' + constraint + '_columns.pkl'
    s3_connection.upload_file(model_file, bucket_name, model_file)
    s3_connection.upload_file(columns_file, bucket_name, columns_file)


def print_results(regressor, X, X_scaled, y, y_scaler):
    """Print actuals and predictions"""

    if str(regressor).split('(')[0] in (
            'DecisionTreeRegressor', 'RandomForestRegressor'):
        predictions = regressor.predict(
            X.drop(['start_date', 'end_date'], axis=1)).round(4)
    elif str(regressor).split('(')[0] == 'SVR':
        predictions = y_scaler.inverse_transform(
            regressor.predict(X_scaled)).round(4)
    else:
        predictions = regressor.predict(X).round(4)

    print('actuals:')
    for actual in y:
        print(actual)

    print('predictions:')
    for prediction in predictions:
        print(prediction)


def train(output, constraint=None):
    """Complete training pipeline
    Possible output values:
    'cost_per_impression', 'cost_per_click', 'cost_per_purchase', 'click_rate'
    Possible constraint values: 'pay_per_impression', 'pay_per_click'
    """

    data = load(output, constraint)

    [X, y, X_train, y_train, X_test, y_test,
     X_scaled, X_train_scaled, y_train_scaled, X_test_scaled, y_scaler] \
        = preprocess(data, output)

    [linear_regressor, tree_regressor, forest_regressor, svr_regressor] \
        = build(X_train, y_train, X_train_scaled, y_train_scaled)

    best_regressor = evaluate(
        linear_regressor, tree_regressor, forest_regressor, svr_regressor,
        X_train, y_train, X_train_scaled, y_train_scaled, X_test, y_test,
        X_test_scaled, y_scaler)

    if str(best_regressor).split('(')[0] in (
            'DecisionTreeRegressor', 'RandomForestRegressor'):
        best_regressor.fit(X.drop(['start_date', 'end_date'], axis=1), y)
    else:
        best_regressor.fit(X, y)

    save(best_regressor, X, output, constraint)

    # upload(output, constraint)

    print_results(best_regressor, X, X_scaled, y, y_scaler)
