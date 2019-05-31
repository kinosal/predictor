import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# View metrics summary
data.describe()

# Visualize distributions of numerical features
quan = list(data.loc[:,data.dtypes != 'object'].columns.values)
qual = list(data.loc[:,data.dtypes == 'object'].columns.values)
temp = pd.melt(data, value_vars = quan)
grid = sns.FacetGrid(temp, col = 'variable', col_wrap = 6, height = 3.0,
                     aspect = 0.8, sharex = False, sharey = False)
grid.map(sns.distplot, 'value')
plt.show()

# Visualize correlations between features
colormap = plt.cm.RdBu
plt.figure(figsize = (12,12))
sns.heatmap(data._get_numeric_data().astype(float).corr(), linewidths = 0.1,
            vmax = 1.0, square = True, cmap = colormap, linecolor = 'white',
            annot = True)
plt.show()

# Investigate correlation between dependent and independent variables
corr = data.corr(method = 'pearson').iloc[0]
corr.sort_values(ascending = True)

# Visualize tree results
import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(tree_regressor, out_file='tree.dot', feature_names=X_train.drop(['start_date', 'end_date'], axis=1).columns)
# dot -Tpng tree.dot -o tree.png
