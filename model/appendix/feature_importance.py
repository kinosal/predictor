import numpy as np
import pandas as pd
from sklearn.base import clone
import training


def create_dataframe(columns, values):
    """Helper functions to create dataframe from column and value lists"""
    return pd.DataFrame({'feature': columns, 'feature_importance': values}) \
             .sort_values('feature_importance', ascending=False) \
             .reset_index(drop=True)


def calculate_importance(output, constraint=None, method='base'):
    """Determine importance for features of used best model"""
    data = training.load(output, constraint)

    # Add random column to data
    np.random.seed(seed=0)
    data['random'] = np.random.random(size=len(data))

    [X, y, X_train, y_train, X_test, y_test,
     X_train_scaled, y_train_scaled, X_test_scaled, y_scaler] \
        = training.preprocess(data, output)

    [linear_regressor, tree_regressor, forest_regressor, svr_regressor] \
        = training.build(X_train, y_train, X_train_scaled, y_train_scaled)

    best_regressor = training.evaluate(
        linear_regressor, tree_regressor, forest_regressor, svr_regressor,
        X_train, y_train, X_train_scaled, y_train_scaled, X_test, y_test,
        X_test_scaled, y_scaler)

    if method == 'base':
        best_regressor.fit(X, y)
        importance = create_dataframe(
            columns=X.columns, values=best_regressor.feature_importances_)
    elif method == 'clone':
        # Clone model to have the same specification as the one initially trained
        model_clone = clone(best_regressor)
        # Set random_state for comparability
        model_clone.random_state = 0
        # Train and score the benchmark model
        model_clone.fit(X, y)
        benchmark_score = model_clone.score(X, y)
        # Create list to store feature importances
        importances = []
        # Calculate and store feature importance benchmark deviation for all columns
        for column in X.columns:
            model_clone = clone(best_regressor)
            model_clone.random_state = 0
            model_clone.fit(X.drop(column, axis=1), y)
            drop_col_score = model_clone.score(X.drop(column, axis=1), y)
            importances.append(benchmark_score - drop_col_score)
        importance = create_dataframe(columns=X.columns, values=importances)
    else:
        importance = 'No method specified'

    return importance
