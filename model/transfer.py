"""
python -c 'import train_transfer; transfer.train_transfer("impressions")'
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import training


def get_predictions(output):
    # Load campaign data
    data = training.load_data(output)

    # Remove data where output is 0 or NaN
    data = data[data[output] > 0]

    # Drop columns with more than 25% missing data
    rows = data[output].count()
    for column in list(data.columns):
        if data[column].count() < rows * 0.75:
            data = data.drop([column], axis=1)

    # Drop rows with NaN values
    data.dropna(axis='index', inplace=True)

    # Load primary models
    direct_model = joblib.load(output + '_model.pkl')
    direct_columns = joblib.load(output + '_columns.pkl')
    cpx_model = joblib.load('cost_per_' + output[0:-1] + '_model.pkl')
    cpx_columns = joblib.load('cost_per_' + output[0:-1] + '_columns.pkl')

    # Calculate and save primary predictions
    categoricals = list(data.select_dtypes(include='object').columns)
    predictions = pd.DataFrame(columns=['output', 'direct', 'cpx'])
    for index, row in data.iterrows():
        for cat in categoricals:
            if row[cat]:
                row[cat + '_' + row[cat].lower()] = 1
                del row[cat]
        direct_row = pd.DataFrame([dict(row)]) \
                       .reindex(columns=direct_columns, fill_value=0)
        cpx_row = pd.DataFrame([dict(row)]) \
                    .reindex(columns=cpx_columns, fill_value=0)
        direct_prediction = int(direct_model.predict(direct_row)[0])
        cpx_prediction = int(row['cost'] / cpx_model.predict(cpx_row)[0])
        predictions.loc[index] = [row[output], direct_prediction,
                                  cpx_prediction]

    return predictions


def preprocess_transfer(data):
    # Specify dependent variable vector y and independent variable matrix X
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]

    # Split dataset into training and test set
    X_train, X_test, y_train, y_test = train_test_split(data.drop(
        ['output'], axis=1), data['output'], test_size=0.2)

    # Scale features
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X.values.astype(float))
    X_train_scaled = X_scaler.transform(X_train.values.astype(float))
    X_test_scaled = X_scaler.transform(X_test.values.astype(float))
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(
        y.values.astype(float).reshape(-1, 1)).flatten()
    y_train_scaled = y_scaler.fit_transform(
        y_train.values.astype(float).reshape(-1, 1)).flatten()

    return [X, y, X_train, y_train, X_test, y_test, X_scaled, y_scaled,
            X_train_scaled, y_train_scaled, X_test_scaled, y_scaler]


def train_transfer(output, models=['linear', 'tree', 'forest', 'svr']):
    data = get_predictions(output)

    [X, y, X_train, y_train, X_test, y_test, X_scaled, y_scaled,
     X_train_scaled, y_train_scaled, X_test_scaled, y_scaler] \
        = preprocess_transfer(data)

    regressors = training.build(
        X_train, y_train, X_train_scaled, y_train_scaled, models)

    best_regressor = training.evaluate(
        regressors, X_train, y_train, X_train_scaled, y_train_scaled,
        X_test, y_test, X_test_scaled, y_scaler)

    if 'svr' in str(best_regressor).lower():
        best_regressor.fit(X_scaled, y_scaled)
    else:
        best_regressor.fit(X, y)

    training.save(best_regressor, X, output + '_transfer')

    training.upload(output + '_transfer')

    training.print_results(best_regressor, X, X_scaled, y, y_scaler)
