import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_pipeline(data, output):
    """
    Preprocessing pipeline part 1: Transform full data frame
    Arguments: Pandas dataframe, output column (dependent variable)
    Returns: Modified dataframe
    """
    data = cost_per_metric(data, output) if 'cost_per' in output \
                                         else data[data[output] > 0]
    data = drop_columns(data, output, threshold=.5)
    data = data.dropna(axis='index')
    data = create_other_buckets(data, threshold=.1)
    data_cat = data
    data = one_hot_encode(data)
    return data, data_cat

def split_pipeline(data, output, encoded=True):
    """
    Preprocessing pipeline part 2: Split data into variables
    Arguments: Pandas dataframe, output column (dependent variable)
    Returns: List of scaled and unscaled dependent and independent variables
    """
    y, X = data.iloc[:, 0], data.iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop([output], axis=1), data[output], test_size=.2, random_state=1)
    if encoded:
        X_scaled, y_scaled, X_train_scaled, y_train_scaled, X_test_scaled, \
            y_scaler = scale(X, y, X_train, y_train, X_test)
        return [X, y, X_train, y_train, X_test, y_test, X_scaled, y_scaled,
                X_train_scaled, y_train_scaled, X_test_scaled, y_scaler]
    return [X, y, X_train, y_train, X_test, y_test]

def cost_per_metric(data, output):
    """Create 'cost_per_...' column and remove data where output is 0 or NaN"""
    metric = output.replace('cost_per_', '') + 's'
    data = data[data[metric] > 0]
    data.insert(0, output, [row['cost'] / row[metric]
                            for index, row in data.iterrows()])
    return data

def drop_columns(data, output, threshold=0.5):
    """Drop columns with more than threshold missing data"""
    rows = data[output].count()
    for column in list(data.columns):
        if data[column].count() < rows * threshold:
            data = data.drop([column], axis=1)
    return data

def create_other_buckets(data, threshold=0.1):
    """Put rare categorical values into other bucket"""
    categoricals = list(data.select_dtypes(include='object').columns)
    for column in categoricals:
        results = data[column].count()
        groups = data.groupby([column])[column].count()
        for bucket in groups.index:
            if groups.loc[bucket] < results * threshold:
                data.loc[data[column] == bucket, column] = 'other'
    return data

def one_hot_encode(data):
    """One-hot encode categorical data"""
    categoricals = list(data.select_dtypes(include='object').columns)
    for column in categoricals:
        if 'other' in data[column].unique():
            data = pd.get_dummies(data, columns=[column], prefix=[column],
                                  drop_first=False)
            data = data.drop([column + '_other'], axis=1)
        else:
            data = pd.get_dummies(data, columns=[column], prefix=[column],
                                  drop_first=True)
    return data

def scale(X, y, X_train, y_train, X_test):
    """Scale dependent and independent variables"""
    X_scaler, y_scaler = StandardScaler(), StandardScaler()

    X_scaled = X_scaler.fit_transform(X.values.astype(float))
    y_scaled = y_scaler.fit_transform(
        y.values.astype(float).reshape(-1, 1)).flatten()

    X_train_scaled = pd.DataFrame(data=X_scaler.transform(
        X_train.values.astype(float)), columns=X.columns)
    y_train_scaled = y_scaler.transform(
        y_train.values.astype(float).reshape(-1, 1)).flatten()

    X_test_scaled = pd.DataFrame(data=X_scaler.transform(
        X_test.values.astype(float)), columns=X.columns)

    return [X_scaled, y_scaled, X_train_scaled, y_train_scaled,
            X_test_scaled, y_scaler]
