import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Real
import statsmodels.api as sm
import helpers as hel


class Stopper:
    def __init__(self, search, max_stagnations=1):
        self.best_score = 0
        self.n_stagnations = 0
        self.max_stagnations = max_stagnations
        self.search = search

    def on_step(self, result):
        new_best_score = self.search.best_score_
        # print("New best score: %s" % new_best_score)
        if new_best_score <= self.best_score:
            self.n_stagnations += 1
            if self.n_stagnations > self.max_stagnations:
                return True
        self.best_score = new_best_score


class Regression:
    """
    Regression instance contains functions to build linear, tree, forest and
    SVR models, expects scaled (for SVR) and unscaled (for others) training data
    """
    def __init__(self, X_train, y_train, X_train_scaled, y_train_scaled):
        self.X_train = X_train
        self.y_train = y_train
        self.X_train_scaled = X_train_scaled
        self.y_train_scaled = y_train_scaled
        self.scorer = make_scorer(hel.mean_relative_accuracy)

    def linear(self, verbose=0):
        """
        Construct a linear regressor and calculate the training score using
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
        Construct a decision tree regressor and calculate the training score
        using training data and parameter search with 5-fold cross validation
        """

        # tree_parameters = [{'min_samples_leaf': list(range(2, 10, 1)),
        #                     'criterion': ['mae', 'mse'],
        #                     'random_state': [1]}]
        # tree_search = GridSearchCV(estimator=DecisionTreeRegressor(),
        #                            param_grid=tree_parameters,
        #                            scoring=self.scorer, cv=5, n_jobs=-1,
        #                            iid=False)
        tree_parameters = [{'min_samples_leaf': Integer(2, 10),
                            'criterion': ['mae', 'mse'],
                            'random_state': [1]}]
        tree_search = BayesSearchCV(
            estimator=DecisionTreeRegressor(), search_spaces=tree_parameters,
            scoring=self.scorer, cv=5, n_jobs=-1, n_iter=20
        )
        stopper = Stopper(tree_search)
        tree_search_result = tree_search.fit(
            self.X_train, self.y_train, callback=stopper.on_step)
        best_tree_parameters = dict(tree_search_result.best_params_)
        tree_score = tree_search_result.best_score_
        print('Best tree params: ' + str(best_tree_parameters))
        print('Tree score: ' + str(tree_score))
        return DecisionTreeRegressor(
            min_samples_leaf=best_tree_parameters['min_samples_leaf'],
            criterion=best_tree_parameters['criterion'],
            random_state=1)

    def forest(self):
        """
        Construct a random forest regressor and calculate the training score
        using training data and parameter search with 5-fold cross validation
        """

        # forest_parameters = [{'n_estimators': hel.powerlist(10, 2, 4),
        #                       'min_samples_leaf': list(range(2, 10, 1)),
        #                       'criterion': ['mae', 'mse'],
        #                       'random_state': [1], 'n_jobs': [-1]}]
        # forest_search = GridSearchCV(estimator=RandomForestRegressor(),
        #                              param_grid=forest_parameters,
        #                              scoring=self.scorer, cv=5, n_jobs=-1,
        #                              iid=False)
        forest_parameters = [{'n_estimators': Integer(10, 200),
                              'min_samples_leaf': Integer(2, 10),
                              'criterion': ['mae', 'mse'],
                              'random_state': [1], 'n_jobs': [-1]}]
        forest_search = BayesSearchCV(
            estimator=RandomForestRegressor(), search_spaces=forest_parameters,
            scoring=self.scorer, cv=5, n_jobs=-1, n_iter=20
        )
        stopper = Stopper(forest_search)
        forest_search_result = forest_search.fit(
            self.X_train, self.y_train, callback=stopper.on_step)
        best_forest_parameters = dict(forest_search_result.best_params_)
        forest_score = forest_search_result.best_score_
        print('Best forest params: ' + str(best_forest_parameters))
        print('Forest score: ' + str(forest_score))
        return RandomForestRegressor(
            n_estimators=best_forest_parameters['n_estimators'],
            min_samples_leaf=best_forest_parameters['min_samples_leaf'],
            criterion=best_forest_parameters['criterion'],
            random_state=1, n_jobs=-1)

    def svr(self):
        """
        Construct a support vector regressor and calculate the training score
        using scaled training data and parameter search with 5-fold cross validation
        """

        # svr_parameters = [{'kernel': ['linear', 'rbf'],
        #                    'C': hel.powerlist(0.1, 2, 10),
        #                    'epsilon': hel.powerlist(0.01, 2, 10),
        #                    'gamma': ['scale']},
        #                   {'kernel': ['poly'],
        #                    'degree': list(range(2, 5, 1)),
        #                    'C': hel.powerlist(0.1, 2, 10),
        #                    'epsilon': hel.powerlist(0.01, 2, 10),
        #                    'gamma': ['scale']}]
        # svr_search = GridSearchCV(estimator=SVR(),
        #                           param_grid=svr_parameters,
        #                           scoring=self.scorer, cv=5, n_jobs=-1,
        #                           iid=False)
        svr_parameters = [{'kernel': ['linear', 'rbf'],
                           'C': Real(0.1, 10),
                           'epsilon': Real(0.01, 1),
                           'gamma': ['scale']},
                          {'kernel': ['poly'],
                           'degree': Integer(2, 5),
                           'C': Real(0.1, 10),
                           'epsilon': Real(0.01, 1),
                           'gamma': ['scale']}]
        svr_search = BayesSearchCV(
            estimator=SVR(), search_spaces=svr_parameters,
            scoring=self.scorer, cv=5, n_jobs=-1, n_iter=20
        )
        stopper = Stopper(svr_search)
        svr_search_result = svr_search.fit(
            self.X_train_scaled, self.y_train_scaled, callback=stopper.on_step)
        best_svr_parameters = dict(svr_search_result.best_params_)
        svr_score = svr_search_result.best_score_
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

    def xgb(self):
        """
        Construct a gradient boosting regressor and calculate the training score
        using training data and parameter search with 5-fold cross validation

        ! XGBoost library does currently not install on AWS Lambda via Zappa !
        """

        estimator = XGBRegressor(booster='gbtree', objective='reg:squarederror')

        # Traditional Grid Search (slow)
        # xgb_parameters = [{
        #     'learning_rate': [x/100 for x in range(5, 10, 1)],
        #     'min_split_loss': [x/10 for x in range(1, 5, 1)],
        #     'max_depth': list(range(5, 10, 1)),
        #     'min_child_weight': list(range(1, 5, 1)),
        #     'colsample_bytree': [x/10 for x in range(5, 10, 1)],
        #     'random_state': [1]
        # }]
        #
        # xgb_search = GridSearchCV(
        #     estimator=estimator, param_grid=xgb_parameters,
        #     scoring=self.scorer, cv=5, n_jobs=-1, iid=False
        # )

        # Bayes Search (faster)
        xgb_parameters = {
            'learning_rate': Real(0.05, 0.5),
            'min_split_loss': Real(0.1, 0.5),
            'max_depth': Integer(5, 10),
            'min_child_weight': Integer(1, 5),
            'random_state': [1]
        }
        xgb_search = BayesSearchCV(
            estimator=estimator, search_spaces=xgb_parameters,
            scoring=self.scorer, cv=5, n_jobs=-1, n_iter=20
        )
        stopper = Stopper(xgb_search)
        xgb_search_result = xgb_search.fit(
            self.X_train, self.y_train, callback=stopper.on_step)
        best_xgb_parameters = dict(xgb_search_result.best_params_)
        xgb_score = xgb_search_result.best_score_

        print('Best XGB params: ' + str(best_xgb_parameters))
        print('XGB score: ' + str(xgb_score))

        return XGBRegressor(
            booster='gbtree', objective='reg:squarederror',
            learning_rate=best_xgb_parameters['learning_rate'],
            min_split_loss=best_xgb_parameters['min_split_loss'],
            max_depth=best_xgb_parameters['max_depth'],
            min_child_weight=best_xgb_parameters['min_child_weight'],
            random_state=1, n_jobs=-1)
