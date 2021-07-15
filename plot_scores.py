from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_validate

# load data
X, y = datasets.load_iris(return_X_y=True)
print(X.shape, y.shape)

# split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))

# cross validation
clf = svm.SVC(kernel='linear', C=1)
scoring = ['precision_macro', 'recall_macro']
scores = cross_validate(clf, X, y, scoring=scoring)
print(f'cross validation:')
print(f'score keys:\n{sorted(scores.keys())}')
print(f"test_recall_macro scores:\n{scores['test_recall_macro']}")
print()

score_name = f'test_{scoring[0]}'
plt.plot(scores[score_name])
plt.xlabel("cv")
plt.ylabel(score_name)
plt.legend([score_name])
plt.show()
