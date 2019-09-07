import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import statsmodels.api as sm
import helpers as hel


class Regression:
    """
    Regression instance contains functions to build linear, tree, forest and
    SVR models, expects scaled (for SVR) and unscaled (for others) trainig data
    """
    def __init__(self, X_train, y_train, X_train_scaled, y_train_scaled):
        self.X_train = X_train
        self.y_train = y_train
        self.X_train_scaled = X_train_scaled
        self.y_train_scaled = y_train_scaled
        self.scorer = make_scorer(hel.mean_relative_accuracy)

    def linear(self, verbose=0):
        """
        Contruct a linear regressor and calculate the training score using
        training data, 5-fold cross validation and a predefined scorer
        """

        # Output linear regression summary with coefficients and p-values
        # if desired
        if verbose != 0:
            model = sm.OLS(self.y_train, sm.add_constant(self.X_train)).fit()
            print(model.summary())

        linear_regressor = LinearRegression(fit_intercept=True, normalize=False,
                                            copy_X=True)
        linear_score = np.mean(cross_val_score(
            estimator=linear_regressor, X=self.X_train, y=self.y_train,
            cv=5, scoring=self.scorer))
        print('Linear score: ' + str(linear_score))
        return linear_regressor

    def tree(self):
        """
        Contruct a decision tree regressor and calculate the training score
        using training data and grid search with 5-fold cross validation
        to determine the best parameters
        """

        tree_parameters = [{'min_samples_leaf': list(range(2, 10, 1)),
                            'criterion': ['mae', 'mse'],
                            'random_state': [1]}]
        tree_grid = GridSearchCV(estimator=DecisionTreeRegressor(),
                                 param_grid=tree_parameters,
                                 scoring=self.scorer, cv=5, n_jobs=-1,
                                 iid=False)
        tree_grid_result = tree_grid.fit(self.X_train, self.y_train)
        best_tree_parameters = tree_grid_result.best_params_
        tree_score = tree_grid_result.best_score_
        print('Best tree params: ' + str(best_tree_parameters))
        print('Tree score: ' + str(tree_score))
        return DecisionTreeRegressor(
            min_samples_leaf=best_tree_parameters['min_samples_leaf'],
            criterion=best_tree_parameters['criterion'],
            random_state=1)

    def forest(self):
        """
        Contruct a random forest regressor and calculate the training score
        using training data and grid search with 5-fold cross validation
        to determine the best parameters
        """

        forest_parameters = [{'n_estimators': hel.powerlist(10, 2, 4),
                              'min_samples_leaf': list(range(2, 10, 1)),
                              'criterion': ['mae', 'mse'],
                              'random_state': [1], 'n_jobs': [-1]}]
        forest_grid = GridSearchCV(estimator=RandomForestRegressor(),
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
            min_samples_leaf=best_forest_parameters['min_samples_leaf'],
            criterion=best_forest_parameters['criterion'],
            random_state=1, n_jobs=-1)

    def svr(self):
        """
        Contruct a support vector regressor and calculate the training score
        using scaled training data and grid search with 5-fold cross validation
        to determine the best parameters
        """

        svr_parameters = [{'kernel': ['linear', 'rbf'],
                           'C': hel.powerlist(0.1, 2, 10),
                           'epsilon': hel.powerlist(0.01, 2, 10),
                           'gamma': ['scale']},
                          {'kernel': ['poly'],
                           'degree': list(range(2, 5, 1)),
                           'C': hel.powerlist(0.1, 2, 10),
                           'epsilon': hel.powerlist(0.01, 2, 10),
                           'gamma': ['scale']}]
        svr_grid = GridSearchCV(estimator=SVR(),
                                param_grid=svr_parameters,
                                scoring=self.scorer, cv=5, n_jobs=-1,
                                iid=False)
        svr_grid_result = svr_grid.fit(self.X_train_scaled, self.y_train_scaled)
        best_svr_parameters = svr_grid_result.best_params_
        svr_score = svr_grid_result.best_score_
        print('Best SVR params: ' + str(best_svr_parameters))
        print('SVR score: ' + str(svr_score))
        if best_svr_parameters['kernel'] == 'poly':
            svr_regressor = SVR(kernel=best_svr_parameters['kernel'],
                                degree=best_svr_parameters['degree'],
                                C=best_svr_parameters['C'],
                                epsilon=best_svr_parameters['epsilon'],
                                gamma='scale')
        else:
            svr_regressor = SVR(kernel=best_svr_parameters['kernel'],
                                C=best_svr_parameters['C'],
                                epsilon=best_svr_parameters['epsilon'],
                                gamma='scale')
        return svr_regressor
