import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import datasets 
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut

from sklearn.model_selection import TimeSeriesSplit


# load data
X, y = datasets.load_iris(return_X_y=True)
print(X.shape, y.shape)

# split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

# validation
score = clf.score(X_test, y_test)
print(f'validation score: {score}\n')

# cross validation score
scores = cross_val_score(clf, X, y, cv=5)
print(f'cross validation scores:\n{scores}\n')

# cross validation
scoring = ['precision_macro', 'recall_macro']
scores = cross_validate(clf, X, y, scoring=scoring)
print(f'cross validation:')
print(f'score keys:\n{sorted(scores.keys())}')
print(f"test_recall_macro scores:\n{scores['test_recall_macro'] }")
print()

# K fold
print("K Fold")
X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)
print(f'k < len of set')
print(f'K fold sample:')
print(f'train_set test_set')
for train, test in kf.split(X):
    print("%s %s" % (train, test))
kf = KFold(n_splits=4)
print(f'k == len of set, which equal to Leave One Out')
print(f'K fold sample:')
for train, test in kf.split(X):
    print("%s %s" % (train, test))
print()

# Repeat K Fold
print('Repeat K Fold')
print('make sure each item will in each iterator N times.')
print('this is used when item repeat N time in the data set.')
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
random_state = 12883823
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
print(f'Repeat K fold sample:')
for train, test in rkf.split(X):
    print("%s %s" % (train, test))
print()

# Leave One Out (LOO)
print('Leave One Out (LOO)')
print('each iterator take all samples expect one.')
X = [1, 2, 3, 4]
loo = LeaveOneOut()
print(f'LOO sample:')
for train, test in loo.split(X):
    print("%s %s" % (train, test))
print()

# Leave P Out (LPO)
print('Leave P Out (LPO)')
print('all the possible training/test sets by removing   samples from the complete set.')
X = np.ones(4)
lpo = LeavePOut(p=2)
print(f'LPO sample:')
for train, test in lpo.split(X):
    print("%s %s" % (train, test))

# Time Series Split
print()
print('Time Series Split ')
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=3)
print(tscv)
for train, test in tscv.split(X):
    print("%s %s" % (train, test))


"""
If the data ordering is not arbitrary (e.g. samples with the same class label are contiguous), 
shuffling it first may be essential to get a meaningful cross-validation result.
"""
# K fold (shuffle)
print()
print("K Fold (shuffle)")
X = ["a", "a", "b", "b", "c", "d"]
kf = KFold(n_splits=3, shuffle=True)
print(f'k < len of set')
print(f'K fold sample:')
print(f'train_set test_set')
for train, test in kf.split(X):
    print("%s %s" % (train, test))
