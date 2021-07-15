from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint

# HalvingGridSearchCV
print('HalvingGridSearchCV')
X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(random_state=0)
param_grid = {"max_depth": [3, None],
              "min_samples_split": [5, 10]}
search = HalvingGridSearchCV(clf, param_grid, resource='n_estimators',
                             max_resources=10,
                             random_state=0).fit(X, y)
print(search.best_estimator_)
print(search.best_params_)
print()

# HalvingRandomSearchCV
print('HalvingRandomSearchCV')
X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(random_state=0)
np.random.seed(0)
param_distributions = {"max_depth": [3, None],
                       "min_samples_split": randint(2, 11)}
search = HalvingRandomSearchCV(clf, param_distributions,
                               resource='n_estimators',
                               max_resources=10,
                               random_state=0).fit(X, y)
print(search.best_estimator_)
print(search.best_params_)
print()
