"""Run calculate_importance(output, method) to return
feature importance dataframe, e.g.
python -c 'import importance; importance.calculate("impressions")'
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
import training


def create_dataframe(columns, values):
    """Helper function to create dataframe from column and value lists"""
    return pd.DataFrame({'column': columns, 'value': values}) \
             .sort_values('value', ascending=False) \
             .reset_index(drop=True)


def calculate(output, model):
    """Determine importance for features of used best model"""

    data = training.load_data(output)
    print('Data loaded.')

    # Add random column to data
    np.random.seed(seed=0)
    data['random'] = np.random.random(size=len(data))

    [X, y, X_train, y_train, X_test, y_test, X_scaled, y_scaled,
     X_train_scaled, y_train_scaled, X_test_scaled, y_scaler] \
        = training.preprocess(data, output)
    print('Data preprocessed.')

    regressor = training.build(
        X_train, y_train, X_train_scaled, y_train_scaled, [model])[0]

    model_clone = clone(regressor)

    # Set random_state for comparability
    model_clone.random_state = 0

    # Train and score the benchmark model
    if str(regressor).split('(')[0] == 'SVR':
        model_clone.fit(X_train_scaled, y_train_scaled)
        benchmark_score = training.mean_relative(
            y_scaler.inverse_transform(
                model_clone.predict(X_train_scaled)), y_train)
    else:
        model_clone.fit(X_train, y_train)
        benchmark_score = training.mean_relative(
            model_clone.predict(X_train), y_train)

    # Calculate and store feature importance benchmark deviation
    importances = []
    columns = X_train.columns
    i = 1
    for column in columns:
        model_clone = clone(regressor)
        model_clone.random_state = 0
        if str(regressor).split('(')[0] == 'SVR':
            model_clone.fit(X_train_scaled.drop(column, axis=1),
                            y_train_scaled)
            drop_col_score = training.mean_relative(model_clone.predict(
                X_train_scaled.drop(column, axis=1)), y_train_scaled)
        else:
            model_clone.fit(X_train.drop(column, axis=1), y_train)
            drop_col_score = training.mean_relative(
                model_clone.predict(X_train.drop(column, axis=1)), y_train)
        importances.append(benchmark_score - drop_col_score)
        i += 1

    importances_df = create_dataframe(
        columns=X_train.columns, values=importances)

    print('Importances:')
    for i in range(0, len(importances_df)):
        print(str(importances_df.iloc[i].column) + ': ' +
              str(importances_df.iloc[i].value))

    return importances_df
