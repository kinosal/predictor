"""Run calculate_importance(output, constraint, method) to return
feature importance dataframe, e.g.
python -c 'import importance; importance.calculate("cost_per_impression", "pay_per_impression", "base")'
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
import training


def create_dataframe(columns, values):
    """Helper functions to create dataframe from column and value lists"""
    return pd.DataFrame({'column': columns, 'value': values}) \
             .sort_values('value', ascending=False) \
             .reset_index(drop=True)


def calculate(output, constraint=None, method='base'):
    """Determine importance for features of used best model"""
    data = training.load(output, constraint)

    # Add random column to data
    np.random.seed(seed=0)
    data['random'] = np.random.random(size=len(data))

    [X, y, X_train, y_train, X_test, y_test,
     X_scaled, X_train_scaled, y_train_scaled, X_test_scaled, y_scaler] \
        = training.preprocess(data, output)

    [linear_regressor, tree_regressor, forest_regressor, svr_regressor] \
        = training.build(X_train, y_train, X_train_scaled, y_train_scaled)

    best_regressor = training.evaluate(
        linear_regressor, tree_regressor, forest_regressor, svr_regressor,
        X_train, y_train, X_train_scaled, y_train_scaled, X_test, y_test,
        X_test_scaled, y_scaler)

    if method == 'base':
        if str(best_regressor).split('(')[0] == 'SVR':
            print('Base method not available for SVR')
            return
        elif str(best_regressor).split('(')[0] in (
                'DecisionTreeRegressor', 'RandomForestRegressor'):
            best_regressor.fit(X_train.drop(
                ['start_date', 'end_date'], axis=1), y_train)
            importances_df = create_dataframe(
                columns=X_train.drop(['start_date', 'end_date'], axis=1).columns,
                values=best_regressor.feature_importances_)
        else:
            best_regressor.fit(X_train, y_train)
            importances_df = create_dataframe(
                columns=X_train.columns,
                values=best_regressor.feature_importances_)

    elif method == 'clone':
        # Clone initially trained model
        model_clone = clone(best_regressor)

        # Set random_state for comparability
        model_clone.random_state = 0

        # Train and score the benchmark model
        if str(best_regressor).split('(')[0] == 'SVR':
            model_clone.fit(X_train_scaled, y_train_scaled)
            benchmark_score = training.mean_relative(
                y_scaler.inverse_transform(
                    svr_regressor.predict(X_train_scaled)), y_train)
        elif str(best_regressor).split('(')[0] in (
                'DecisionTreeRegressor', 'RandomForestRegressor'):
            model_clone.fit(
                X_train.drop(['start_date', 'end_date'], axis=1), y_train)
            benchmark_score = training.mean_relative(model_clone.predict(
                X_train.drop(['start_date', 'end_date'], axis=1)), y_train)
        else:
            model_clone.fit(X_train, y_train)
            benchmark_score = training.mean_relative(
                model_clone.predict(X_train), y_train)

        # Create list to store feature importances
        importances = []

        # Calculate and store feature importance benchmark deviation
        if str(best_regressor).split('(')[0] in (
                'DecisionTreeRegressor', 'RandomForestRegressor'):
            columns = X_train.drop(['start_date', 'end_date'], axis=1).columns
        else:
            columns = X_train.columns

        i = 0
        for column in columns:
            model_clone = clone(best_regressor)
            model_clone.random_state = 0
            if str(best_regressor).split('(')[0] == 'SVR':
                model_clone.fit(X_train_scaled[0:i, i+1:-1], y_train_scaled)
                drop_col_score = training.mean_relative(model_clone.predict(
                    X_train_scaled[0:i, i+1:-1]), y_train_scaled)
            elif str(best_regressor).split('(')[0] in (
                    'DecisionTreeRegressor', 'RandomForestRegressor'):
                model_clone.fit(X_train.drop(
                    ['start_date', 'end_date', column], axis=1), y_train)
                drop_col_score = training.mean_relative(
                    model_clone.predict(X_train.drop(
                        ['start_date', 'end_date', column], axis=1)), y_train)
            else:
                model_clone.fit(X_train.drop(column, axis=1), y_train)
                drop_col_score = training.mean_relative(
                    model_clone.predict(X_train.drop(column, axis=1)), y_train)
            importances.append(benchmark_score - drop_col_score)
            i += 1

        if str(best_regressor).split('(')[0] in (
                'DecisionTreeRegressor', 'RandomForestRegressor'):
            importances_df = create_dataframe(
                columns=X_train.drop(['start_date', 'end_date'], axis=1).columns,
                values=importances)
        else:
            importances_df = create_dataframe(
                columns=X_train.columns, values=importances)

    print('importances:')
    for i in range(0, len(importances_df)):
        print(str(importances_df.iloc[i].column) + ': ' +
              str(importances_df.iloc[i].value))

    return importances_df
