# Visualize decision tree results
from sklearn.tree import export_graphviz
export_graphviz(regressor, out_file='tree.dot', feature_names=X_train.columns)
# dot -Tpng tree.dot -o tree.png

####

# Evaluate model scores with different param values
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import helpers

def evaluate_params(model):
    params = {'criterion': ['mae', 'mse'],
              'min_samples_leaf': list(range(2, 10, 1))}
    for param in params:
        results = pd.DataFrame(columns=[param, 'score'])
        for value in params[param]:
            regressor = DecisionTreeRegressor(**{param: value})
            regressor.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = helpers.mean_relative_accuracy(y_pred, y_test)
            results = results.append({param: value, 'score': score},
                                     ignore_index=True)
        print(results)
