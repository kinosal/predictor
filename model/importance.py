"""
Run calculate(output, model, source) to return feature importance dataframe,
e.g. python -c 'import importance; importance.calculate("impressions", "tree")'
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
import helpers as hel
import training as tra
import preprocessing as pre


def calculate(output, model, source='csv'):
    """Determine feature importance for spcified model"""

    if source == 'pg':
        data = tra.load_data_from_postgres(output)
        print('Data loaded from Postgres.')
    elif source == 'csv':
        data = tra.load_data_from_csv(output)
        print('Data loaded from CSV.')
    else:
        print('Source not available.')
        return

    # Add random column to data
    np.random.seed(seed=0)
    data['random'] = np.random.random(size=len(data))

    data = pre.data_pipeline(data, output)
    [_, _, X_train, y_train, _, _, _, _, X_train_scaled, y_train_scaled,
     _, y_scaler] = pre.split_pipeline(data, output)
    print('Data preprocessed.')

    regressor = \
        tra.build(X_train, y_train, X_train_scaled, y_train_scaled, [model])[0]

    model_clone = clone(regressor)

    # Set random_state for comparability
    model_clone.random_state = 0

    # Train and score the benchmark model
    if 'SVR' in str(regressor):
        model_clone.fit(X_train_scaled, y_train_scaled)
        benchmark_score = hel.mean_relative_accuracy(y_scaler.inverse_transform(
            model_clone.predict(X_train_scaled)), y_train)
    else:
        model_clone.fit(X_train, y_train)
        benchmark_score = \
            hel.mean_relative_accuracy(model_clone.predict(X_train), y_train)

    # Calculate and store feature importance benchmark deviation
    importances = []
    columns = X_train.columns
    i = 1
    for column in columns:
        model_clone = clone(regressor)
        model_clone.random_state = 0
        if 'SVR' in str(regressor):
            model_clone.fit(X_train_scaled.drop(column, axis=1),
                            y_train_scaled)
            drop_col_score = hel.mean_relative_accuracy(model_clone.predict(
                X_train_scaled.drop(column, axis=1)), y_train_scaled)
        else:
            model_clone.fit(X_train.drop(column, axis=1), y_train)
            drop_col_score = hel.mean_relative_accuracy(
                model_clone.predict(X_train.drop(column, axis=1)), y_train)
        importances.append(benchmark_score - drop_col_score)
        i += 1

    importances_df = \
        pd.DataFrame({'column': X_train.columns, 'value': importances}) \
          .sort_values('value', ascending=False).reset_index(drop=True)

    print('Importances:')
    for i in range(0, len(importances_df)):
        print(str(importances_df.iloc[i].column) + ': ' +
              str(importances_df.iloc[i].value))
