import pandas as pd
from sklearn.tree import export_graphviz
import helpers as hel

def save_dot(tree, columns):
    """Visualize decision tree results"""
    export_graphviz(tree, out_file='tree.dot', feature_names=columns)
# dot -Tpng tree.dot -o tree.png

def evaluate_params(model, params, X_train, y_train, X_test, y_test):
    """Evaluate model scores with different param values"""
    # model = 'DecisionTreeRegressor'
    # params = {'min_samples_leaf': list(range(2, 10, 1)),
    #           'criterion': ['mae', 'mse']}
    for param in params:
        results = pd.DataFrame(columns=[param, 'score'])
        for value in params[param]:
            regressor = eval(model + '(**{param: value})')
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            score = hel.mean_relative_accuracy(y_pred, y_test)
            results = results.append({param: value, 'score': score},
                                     ignore_index=True)
        print(results)
