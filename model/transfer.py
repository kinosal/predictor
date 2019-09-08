"""
python -c 'import transfer; transfer.train_transfer("impressions")'
"""


import pandas as pd
import joblib
import training as tra
import preprocessing as pre


def get_predictions(output):
    # Load campaign data
    data = tra.load_data_from_postgres(output)

    # Preprocess data without train/test or y/X splitting
    data = pre.data_pipeline(data, output)

    # Load primary models
    direct_model = joblib.load(output + '_model.pkl')
    direct_columns = joblib.load(output + '_columns.pkl')
    cpx_model = joblib.load('cost_per_' + output[0:-1] + '_model.pkl')
    cpx_columns = joblib.load('cost_per_' + output[0:-1] + '_columns.pkl')

    # Calculate and save primary predictions
    predictions = pd.DataFrame(columns=[output, 'direct', 'cpx'])
    for index, row in data.iterrows():
        direct_row = pd.DataFrame([dict(row)]) \
                       .reindex(columns=direct_columns, fill_value=0)
        cpx_row = pd.DataFrame([dict(row)]) \
                    .reindex(columns=cpx_columns, fill_value=0)
        direct_prediction = int(direct_model.predict(direct_row)[0])
        cpx_prediction = int(row['cost'] / cpx_model.predict(cpx_row)[0])
        predictions.loc[index] = [row[output], direct_prediction,
                                  cpx_prediction]

    return predictions


def train_transfer(output, models=['linear', 'tree', 'forest', 'svr']):
    data = get_predictions(output)
    print('Primary predictions loaded.')

    [X, y, X_train, y_train, X_test, y_test, X_scaled, y_scaled,
     X_train_scaled, y_train_scaled, X_test_scaled, y_scaler] \
        = pre.split_pipeline(data, output)
    print('Data preprocessed.')

    regressors = tra.build(
        X_train, y_train, X_train_scaled, y_train_scaled, models)

    best_regressor = tra.evaluate(
        regressors, X_train, y_train, X_train_scaled, y_train_scaled,
        X_test, y_test, X_test_scaled, y_scaler)
    print('Regressors evaluated. Best regressor is:\n' + str(best_regressor))

    if 'SVR' in str(best_regressor):
        best_regressor.fit(X_scaled, y_scaled)
    else:
        best_regressor.fit(X, y)
    print('Regressor fit.')

    tra.print_results(best_regressor, X, X_scaled, y, y_scaler)

    tra.save(best_regressor, X, output + '_transfer')
    print('Regressor saved.')

    tra.upload(output + '_transfer')
    print('Regressor uploaded.')
