"""
Run train(output) for full pipeline to train, select and save
best model predicting campaign performance, e.g.
python -c 'import training; training.train(output="impressions")'
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
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def mean_relative(y_pred, y_true):
    """
    Helper function to calculate mean relative deviation from two vectors
    = 1 - mean percentage error
    """
    return 1 - np.mean(np.abs((y_pred - y_true) / y_true))


def load_data_from_postgres(output):
    """Load campaign data into dataframe from Postgres database"""
    connection = pg.connect(config.marketing_production)
    if output in ['impressions', 'clicks', 'purchases']:
        output = 'SUM(results.' + output + ') AS ' + output
    elif 'cost_per' in output:
        metric = output.replace('cost_per_', '') + 's'
        output = 'SUM(results.' + metric + ') AS ' + metric
    select = 'SELECT ' + output + ', ' + open('campaigns.sql', 'r').read()
    return pd.read_sql_query(select, connection)


def load_data_from_csv(filename):
    return pd.read_csv(filename)


def preprocess(data, output):
    """Preprocess data"""

    # Create 'cost_per_...' column and remove data where output is 0 or NaN
    if 'cost_per' in output:
        metric = output.replace('cost_per_', '') + 's'
        data = data[data[metric] > 0]
        data.insert(0, output, [row['cost'] / row[metric]
                                for index, row in data.iterrows()])
    else:
        data = data[data[output] > 0]

    # Drop columns with more than 50% missing data
    rows = data[output].count()
    for column in list(data.columns):
        if data[column].count() < rows * .5:
            data = data.drop([column], axis=1)

    # Drop rows with NaN values
    data.dropna(axis='index', inplace=True)

    # Put rare categorical values into other bucket
    categoricals = list(data.select_dtypes(include='object').columns)
    threshold = 0.1
    for column in categoricals:
        results = data[column].count()
        groups = data.groupby([column])[column].count()
        for bucket in groups.index:
            if groups.loc[bucket] < results * threshold:
                data.loc[data[column] == bucket, column] = 'other'

    # Encode categorical data
    for column in categoricals:
        if 'other' in data[column].unique():
            data = pd.get_dummies(data, columns=[column], prefix=[column],
                                  drop_first=False)
            data = data.drop([column + '_other'], axis=1)
        else:
            data = pd.get_dummies(data, columns=[column], prefix=[column],
                                  drop_first=True)

    # Specify dependent variable vector y and independent variable matrix X
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]

    # Split dataset into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop([output], axis=1), data[output], test_size=0.2)

    # Scale features
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X.values.astype(float))
    X_train_scaled = pd.DataFrame(
        data=X_scaler.transform(X_train.values.astype(float)),
        columns=X_train.columns)
    X_test_scaled = pd.DataFrame(
        data=X_scaler.transform(X_test.values.astype(float)),
        columns=X_test.columns)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(
        y.values.astype(float).reshape(-1, 1)).flatten()
    y_train_scaled = y_scaler.fit_transform(
        y_train.values.astype(float).reshape(-1, 1)).flatten()

    return [X, y, X_train, y_train, X_test, y_test, X_scaled, y_scaled,
            X_train_scaled, y_train_scaled, X_test_scaled, y_scaler]


def build(X_train, y_train, X_train_scaled, y_train_scaled,
          models=['linear', 'tree', 'forest', 'svr']):
    """Build and return models"""

    # Define helper function to create lists for search grids
    def powerlist(start, times):
        array = []
        for i in range(0, times, 1):
            array.append(start * 2 ** i)
        return array

    # Linear regression (library includes feature scaling)
    if 'linear' in models:
        linear_regressor = LinearRegression()
        linear_score = np.mean(cross_val_score(
            estimator=linear_regressor, X=X_train, y=y_train,
            cv=5, scoring=make_scorer(mean_relative)))
        print('Linear score: ' + str(linear_score))

    # Decision tree regression (no feature scaling needed)
    if 'tree' in models:
        tree_regressor = DecisionTreeRegressor()
        tree_parameters = [{'min_samples_split': list(range(2, 8, 1)),
                            'max_leaf_nodes': list(range(2, 8, 1)),
                            'criterion': ['mae', 'mse']}]
        tree_grid = GridSearchCV(estimator=tree_regressor,
                                 param_grid=tree_parameters,
                                 scoring=make_scorer(mean_relative),
                                 cv=5,
                                 n_jobs=-1,
                                 iid=False)
        tree_grid_result = tree_grid.fit(X_train, y_train)
        best_tree_parameters = tree_grid_result.best_params_
        tree_score = tree_grid_result.best_score_
        tree_regressor = DecisionTreeRegressor(
            min_samples_split=best_tree_parameters['min_samples_split'],
            max_leaf_nodes=best_tree_parameters['max_leaf_nodes'],
            criterion=best_tree_parameters['criterion'])
        print('Best tree params: ' + str(best_tree_parameters))
        print('Tree score: ' + str(tree_score))

    # Random forest regression (no feature scaling needed)
    if 'forest' in models:
        forest_regressor = RandomForestRegressor()
        forest_parameters = [{'n_estimators': powerlist(10, 4),
                              'min_samples_split': list(range(2, 8, 1)),
                              'max_leaf_nodes': list(range(2, 8, 1)),
                              'criterion': ['mae', 'mse']}]
        forest_grid = GridSearchCV(estimator=forest_regressor,
                                   param_grid=forest_parameters,
                                   scoring=make_scorer(mean_relative),
                                   cv=5,
                                   n_jobs=-1,
                                   iid=False)
        forest_grid_result = forest_grid.fit(X_train, y_train)
        best_forest_parameters = forest_grid_result.best_params_
        forest_score = forest_grid_result.best_score_
        forest_regressor = RandomForestRegressor(
            n_estimators=best_forest_parameters['n_estimators'],
            min_samples_split=best_forest_parameters['min_samples_split'],
            max_leaf_nodes=best_forest_parameters['max_leaf_nodes'],
            criterion=best_forest_parameters['criterion'])
        print('Best forest params: ' + str(best_forest_parameters))
        print('Forest score: ' + str(forest_score))

    # SVR (needs feature scaling)
    if 'svr' in models:
        svr_regressor = SVR()
        svr_parameters = [
            {'C': powerlist(0.01, 10), 'kernel': ['linear']},
            {'C': powerlist(0.01, 10), 'kernel': ['poly'],
             'degree': [2, 3, 4, 5], 'gamma': powerlist(0.0000001, 10)},
            {'C': powerlist(0.01, 10), 'kernel': ['rbf'],
             'gamma': powerlist(0.0000001, 10),
             'epsilon': powerlist(0.0001, 10)}]
        svr_grid = GridSearchCV(estimator=svr_regressor,
                                param_grid=svr_parameters,
                                scoring=make_scorer(mean_relative),
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
        print('Best SVR params: ' + str(best_svr_parameters))
        print('SVR score: ' + str(svr_score))

    regressors = []
    for model in models:
        regressors.append(eval(model + '_regressor'))

    return regressors


def evaluate(regressors,
             X_train, y_train, X_train_scaled, y_train_scaled,
             X_test, y_test, X_test_scaled, y_scaler):
    """Evaluate models and return best regressor"""

    # Fit regressors on training set
    for regressor in regressors:
        if 'svr' in str(regressor).lower():
            regressor.fit(X_train_scaled, y_train_scaled)
        else:
            regressor.fit(X_train, y_train)

    # Predict training results and calculate accuracy
    # (might slightly differ from training scores since evaluated on the
    # full training set without cross validation)
    training_accuracies = {}
    for regressor in regressors:
        if 'svr' in str(regressor).lower():
            training_accuracies[str(regressor)] = \
                mean_relative(y_scaler.inverse_transform(
                    regressor.predict(X_train_scaled)), y_train)
        else:
            training_accuracies[str(regressor)] = \
                mean_relative(regressor.predict(X_train), y_train)

    # Predict test results and calculate accuracy
    test_accuracies = {}
    for regressor in regressors:
        if 'svr' in str(regressor).lower():
            test_accuracies[str(regressor)] = \
                mean_relative(y_scaler.inverse_transform(
                    regressor.predict(X_test_scaled)), y_test)
        else:
            test_accuracies[str(regressor)] = \
                mean_relative(regressor.predict(X_test), y_test)

    best_regressor = eval(max(test_accuracies, key=test_accuracies.get))

    for regressor in regressors:
        print(str(regressor).split('(')[0] + '_train_accu: ' +
              str(training_accuracies[str(regressor)]))
        print(str(regressor).split('(')[0] + '_test_accu: ' +
              str(test_accuracies[str(regressor)]))

    return best_regressor


def save(regressor, X, output):
    """Save model and columns to file"""
    joblib.dump(regressor, output + '_model.pkl')
    joblib.dump(list(X.columns), output + '_columns.pkl')


def upload(output):
    """Upload model and columns to S3"""
    s3_connection = boto3.client('s3')
    bucket_name = 'cpx-prediction'
    model_file = output + '_model.pkl'
    columns_file = output + '_columns.pkl'
    s3_connection.upload_file(model_file, bucket_name, model_file)
    s3_connection.upload_file(columns_file, bucket_name, columns_file)


def print_results(regressor, X, X_scaled, y, y_scaler):
    """Print actuals and predictions"""

    if 'svr' in str(regressor).lower():
        predictions = y_scaler.inverse_transform(
            regressor.predict(X_scaled)).round(4)
    else:
        predictions = regressor.predict(X).round(4)

    print('\n### Results ###')
    print('\nActuals:')
    for actual in y:
        print(actual)

    print('\nPredictions:')
    for prediction in predictions:
        print(prediction)


def train(output, models=['linear', 'tree', 'forest', 'svr'], source='pg'):
    """Complete training pipeline"""

    if source == 'pg':
        data = load_data_from_postgres(output)
        print('Data loaded from Postgres.')
    elif source == 'csv':
        data = load_data_from_csv(output + '.csv')
        print('Data loaded from CSV.')
    else:
        print('Source not available.')
        return

    [X, y, X_train, y_train, X_test, y_test, X_scaled, y_scaled,
     X_train_scaled, y_train_scaled, X_test_scaled, y_scaler] \
        = preprocess(data, output)
    print('Data preprocessed.')

    regressors = build(
        X_train, y_train, X_train_scaled, y_train_scaled, models)

    best_regressor = evaluate(regressors, X_train, y_train, X_train_scaled,
                              y_train_scaled, X_test, y_test, X_test_scaled,
                              y_scaler)
    print('Regressors evaluated. Best regressor is:\n' + str(best_regressor))

    if 'svr' in str(best_regressor).lower():
        best_regressor.fit(X_scaled, y_scaled)
    else:
        best_regressor.fit(X, y)
    print('Regressor fit.')

    save(best_regressor, X, output)
    print('Regressor saved.')

    upload(output)
    print('Regressor uploaded.')

    print_results(best_regressor, X, X_scaled, y, y_scaler)
