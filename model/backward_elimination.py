# Implement backward elimination with OLS
import statsmodels.formula.api as smf
def backward_elimination_training(y, x, p):
  ars_old = 0
  elim = []
  while len(x[0]) > 0:
    regressor_OLS = smf.OLS(y, x).fit()
    ars = regressor_OLS.rsquared_adj
    if ars < ars_old - 0.001:
      print (summary_old)
      return [x_old, elim_old]
    else:
      maxp = max(regressor_OLS.pvalues)
      if maxp <= p:
        print (regressor_OLS.summary())
        return [x, elim]
      else:
        x_old = x
        ars_old = ars
        elim_old = elim[:]
        summary_old = regressor_OLS.summary()
        argmaxp = regressor_OLS.pvalues.argmax()
        elim.append(argmaxp)
        x = np.delete(x, argmaxp, 1)

# Method to run directly on the dataframe to keep column names for analysis
def backward_elimination_df(y, x, p):
  ars_old = 0
  elim = []
  while len(x.columns) > 0:
    regressor_OLS = smf.OLS(y, x).fit()
    ars = regressor_OLS.rsquared_adj
    if ars < ars_old - 0.001:
      print (summary_old)
      return [x_old, elim_old]
    else:
      maxp = max(regressor_OLS.pvalues)
      if maxp <= p:
        print (regressor_OLS.summary())
        return [x, elim]
      else:
        x_old = x
        ars_old = ars
        elim_old = elim[:]
        summary_old = regressor_OLS.summary()
        argmaxp = regressor_OLS.pvalues.argmax()
        elim.append(argmaxp)
        x = x_old.drop([argmaxp], axis=1)

# Add column of ones to df to account for linear regression intercept
df['intercept'] = 1
back_elim = backward_elimination_df(df['cost_per_impression'], df.drop(['cost_per_impression'], axis=1), 0.05)

# Add column of ones to X to account for linear regression intercept
X = np.append(arr = np.ones((X.shape[0], 1)).astype(int), values = X, axis = 1)

def adjust_test(x, elim):
  for i in range(len(elim)):
    x = np.delete(x, elim[i], 1)
  return x

back_elim = backward_elimination_training(y_train, X_train, 0.05)
X_opt = back_elim[0]
elim = back_elim[1]
linear_regressor_elim = LinearRegression()
linear_regressor_elim.fit(X_opt, y_train)
X_test_elim = adjust_test(X_test, elim)
y_pred_elim = linear_regressor_elim.predict(X_test_elim)
elim_accu = 1 - np.average(np.divide(np.absolute(y_pred_elim - y_test), y_test))
