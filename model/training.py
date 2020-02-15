"""
Run train(output) for full pipeline to train, select and save
best model predicting campaign performance, e.g.
python -c 'import training; training.train(output="impressions")'
"""

# Import libraries
import pandas as pd
import psycopg2 as pg
import boto3
import joblib

# Import secrets
import config

# Import modules
import helpers as hel
import preprocessing as pre
import regression as reg


def postgres_to_csv(environment='production'):
    connection = pg.connect(getattr(config, environment))
    select = open('campaigns.sql', 'r').read()
    data = pd.read_sql_query(select, connection)
    data.to_csv('campaigns.csv', index=False)


def trim(data, output):
    """Trim data to only include desired output metric"""
    if 'cost_per' in output:
        keep = output.replace('cost_per_', '') + 's'
    else:
        keep = output
    drop_candidates = ['id', 'impressions', 'clicks', 'purchases']
    drops = [drop for drop in drop_candidates if drop != keep]
    return data.drop(drops, axis=1)


def build(X_train, y_train, X_train_scaled, y_train_scaled, models):
    """Build and return regression models"""
    regression = \
        reg.Regression(X_train, y_train, X_train_scaled, y_train_scaled)
    regressors = []
    for model in models:
        regressors.append(getattr(regression, model)())
    return regressors


def evaluate(regressors,
             X_train, y_train, X_train_scaled, y_train_scaled,
             X_test, y_test, X_test_scaled, y_scaler):
    """
    Evaluate models by fitting on full training set and
    calculating training and test accuracies;
    return best regressor
    """

    training_accuracies = {}
    test_accuracies = {}
    for regressor in regressors:
        if 'SVR' in str(regressor):
            regressor.fit(X_train_scaled, y_train_scaled)
            training_accuracies[regressor] = hel.mean_relative_accuracy(
                y_scaler.inverse_transform(regressor.predict(
                    X_train_scaled)), y_train)
            test_accuracies[regressor] = hel.mean_relative_accuracy(
                y_scaler.inverse_transform(regressor.predict(
                    X_test_scaled)), y_test)
        else:
            regressor.fit(X_train, y_train)
            training_accuracies[regressor] = hel.mean_relative_accuracy(
                regressor.predict(X_train), y_train)
            test_accuracies[regressor] = hel.mean_relative_accuracy(
                regressor.predict(X_test), y_test)
        print(str(regressor).split('(')[0] + '_train_accu: ' +
              str(training_accuracies[regressor]))
        print(str(regressor).split('(')[0] + '_test_accu: ' +
              str(test_accuracies[regressor]))

    return max(test_accuracies, key=test_accuracies.get)


def save(regressor, X, output):
    """Save model and columns to file"""
    joblib.dump(regressor, './models/' + output + '_model.pkl')
    joblib.dump(list(X.columns), './models/' + output + '_columns.pkl')


def upload(output):
    """Upload model and columns to S3"""
    s3_connection = boto3.client('s3')
    bucket_name = 'cpx-prediction'
    model_file = output + '_model.pkl'
    model_local = './models/' + model_file
    columns_file = output + '_columns.pkl'
    columns_local = './models/' + columns_file
    s3_connection.upload_file(model_local, bucket_name, model_file)
    s3_connection.upload_file(columns_local, bucket_name, columns_file)


def print_results(regressor, X, X_scaled, y, y_scaler):
    """Print actuals and predictions"""

    if 'SVR' in str(regressor):
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


def train(output, update=False,
          models=['linear', 'tree', 'forest', 'svr']):
    """Complete training pipeline"""

    if update:
        postgres_to_csv()
        print('Data updated.')

    data = pd.read_csv('campaigns.csv')
    print('Data loaded.')

    data = trim(data, output)
    print('Data trimmed.')

    data = pre.data_pipeline(data, output)
    [X, y, X_train, y_train, X_test, y_test, X_scaled, y_scaled,
     X_train_scaled, y_train_scaled, X_test_scaled, y_scaler] \
        = pre.split_pipeline(data, output)
    print('Data preprocessed.')

    regressors = build(
        X_train, y_train, X_train_scaled, y_train_scaled, models)

    best_regressor = evaluate(regressors, X_train, y_train, X_train_scaled,
                              y_train_scaled, X_test, y_test, X_test_scaled,
                              y_scaler)
    print('Regressors evaluated. Best regressor is:\n' + str(best_regressor))

    if 'SVR' in str(best_regressor):
        best_regressor.fit(X_scaled, y_scaled)
    else:
        best_regressor.fit(X, y)
    print('Regressor fit.')

    print_results(best_regressor, X, X_scaled, y, y_scaler)

    save(best_regressor, X, output)
    print('Regressor saved.')

    upload(output)
    print('Regressor uploaded.')
