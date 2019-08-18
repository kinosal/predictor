import numpy as np
import helpers
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class Regression:
    def __init__(self, X_train, y_train, X_train_scaled, y_train_scaled):
        self.X_train = X_train
        self.y_train = y_train
        self.X_train_scaled = X_train_scaled
        self.y_train_scaled = y_train_scaled
        self.scorer = make_scorer(helpers.mean_relative)

    def linear(self):
        """
        Contruct a linear regressor and calculate the training score using
        training data, 5-fold cross validation and a predefined scorer
        """

        linear_regressor = LinearRegression()
        linear_score = np.mean(cross_val_score(
            estimator=linear_regressor, X=self.X_train, y=self.y_train,
            cv=5, scoring=self.scorer))
        print('Linear score: ' + str(linear_score))
        return linear_regressor

    def tree(self):
        """
        Contruct a decision tree regressor and calculate the trainng score
        using training data and grid search with 5-fold cross validation
        to determine the best parameters
        """

        tree_regressor = DecisionTreeRegressor()
        tree_parameters = [{'min_samples_split': list(range(2, 8, 1)),
                            'max_leaf_nodes': list(range(2, 8, 1)),
                            'criterion': ['mae', 'mse']}]
        tree_grid = GridSearchCV(estimator=tree_regressor,
                                 param_grid=tree_parameters,
                                 scoring=self.scorer, cv=5, n_jobs=-1,
                                 iid=False)
        tree_grid_result = tree_grid.fit(self.X_train, self.y_train)
        best_tree_parameters = tree_grid_result.best_params_
        tree_score = tree_grid_result.best_score_
        print('Best tree params: ' + str(best_tree_parameters))
        print('Tree score: ' + str(tree_score))
        return DecisionTreeRegressor(
            min_samples_split=best_tree_parameters['min_samples_split'],
            max_leaf_nodes=best_tree_parameters['max_leaf_nodes'],
            criterion=best_tree_parameters['criterion'])

    def forest(self):
        """
        Contruct a random forest regressor and calculate the trainng score
        using training data and grid search with 5-fold cross validation
        to determine the best parameters
        """

        forest_regressor = RandomForestRegressor()
        forest_parameters = [{'n_estimators': helpers.powerlist(10, 2, 4),
                              'min_samples_split': list(range(2, 8, 1)),
                              'max_leaf_nodes': list(range(2, 8, 1)),
                              'criterion': ['mae', 'mse']}]
        forest_grid = GridSearchCV(estimator=forest_regressor,
                                   param_grid=forest_parameters,
                                   scoring=self.scorer, cv=5, n_jobs=-1,
                                   iid=False)
        forest_grid_result = forest_grid.fit(self.X_train, self.y_train)
        best_forest_parameters = forest_grid_result.best_params_
        forest_score = forest_grid_result.best_score_
        print('Best forest params: ' + str(best_forest_parameters))
        print('Forest score: ' + str(forest_score))
        return RandomForestRegressor(
            n_estimators=best_forest_parameters['n_estimators'],
            min_samples_split=best_forest_parameters['min_samples_split'],
            max_leaf_nodes=best_forest_parameters['max_leaf_nodes'],
            criterion=best_forest_parameters['criterion'])

    def svr(self):
        """
        Contruct a support vector regressor and calculate the trainng score
        using scaled training data and grid search with 5-fold cross validation
        to determine the best parameters
        """

        svr_regressor = SVR()
        svr_parameters = [
            {'C': helpers.powerlist(0.01, 2, 10), 'kernel': ['linear']},
            {'C': helpers.powerlist(0.01, 2, 10), 'kernel': ['poly'],
             'degree': [2, 3, 4, 5],
             'gamma': helpers.powerlist(0.0000001, 2, 10)},
            {'C': helpers.powerlist(0.01, 2, 10), 'kernel': ['rbf'],
             'gamma': helpers.powerlist(0.0000001, 2, 10),
             'epsilon': helpers.powerlist(0.0001, 2, 10)}]
        svr_grid = GridSearchCV(estimator=svr_regressor,
                                param_grid=svr_parameters,
                                scoring=self.scorer, cv=5, n_jobs=-1,
                                iid=False)
        svr_grid_result = \
            svr_grid.fit(self.X_train_scaled, self.y_train_scaled)
        best_svr_parameters = svr_grid_result.best_params_
        svr_score = svr_grid_result.best_score_
        print('Best SVR params: ' + str(best_svr_parameters))
        print('SVR score: ' + str(svr_score))
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
        return svr_regressor
