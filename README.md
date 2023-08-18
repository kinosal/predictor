# Marketing Performance Predictor

Source code for my blog post about [How to predict the success of your marketing campaign](https://medium.com/@nikolasschriefer/how-to-predict-the-success-of-your-marketing-campaign-579fbb153a97)

Contains the code to train a prediction model for regression results with linear, decision tree, random forest and support vector regressors and provides a simple Python (Flask) web app to predict ad impressions, clicks and purchases (conversions) for digital (social media and search) marketing campaigns [predictor.stagelink.com](https://predictor.stagelink.com)

## Structure

This project contains two parts, a regression model builder/trainer and a web app to predict results.

1) Model: Basic training pipeline in training.py, consuming first_glance.py (basic descriptive analysis), helpers.py (helper functions), preprocessing.py (data preprocessing) and regression.py (regression class with 4 different regressor methods).

2) App: Single page app defined in api.py. HTML views in templates, custom CSS in static/css.

## Contribution

Please submit any [issues](https://github.com/kinosal/predictor/issues) you have. If you have any ideas how to further improve the predictor please get in touch or feel free to fork this project and create a pull request with your proposed updates.

## Environment

- Python 3.9
- Flask (web framework)
- zappa (deployment to AWS lambda)
- numpy (Python computing package)
- pandas (Python data analytics library)
- statsmodels (Python statistics module)
- Matplotlib (Python plotting library)
- Seaborn (Python visualization library)
- Scikit-learn (Python machine learning library)
- Joblib (Python pipelining library)
- Psycopg (Python PostgreSQL adapter)
- Boto3 (AWS Python SDK)

## Prediction pipeline as in training.train()

1) Load data from CSV or (Postgres) database
2) Preprocess data
3) Build regressors (linear, tree, forest and SVR)
4) Evaluate regressors
5) Fit best regressor to data
6) Save best regressor
