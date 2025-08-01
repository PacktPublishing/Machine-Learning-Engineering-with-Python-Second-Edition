# Inspired by https://maelfabien.github.io/machinelearning/HyperOpt/#the-data

# HyperOpt
from hyperopt import hp, tpe, STATUS_OK, Trials, fmin
# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import datasets

from functools import partial # to solve scoping problem when supplying more params' to objective function

import pprint

space = {
    'warm_start': hp.choice('warm_start', [True, False]),
    'fit_intercept': hp.choice('fit_intercept', [True, False]),
    'tol': hp.uniform('tol', 0.00001, 0.0001),
    'C': hp.uniform('C', 0.05, 2.5),
    'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear']),
    'max_iter': hp.choice('max_iter', range(10, 500))
}


def objective(params, n_folds, X, y):
    """Objective function for tuning logistic regression hyperparameters"""

    # Perform n_fold cross validation with hyperparameters
    clf = LogisticRegression(**params, random_state=42)
    scores = cross_val_score(clf, X, y, cv=n_folds, scoring='f1_macro')

    # Extract the best score
    max_score = max(scores)

    # Loss must be minimized
    loss = 1 - max_score

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


if __name__ == "__main__":
    n_folds = 5
    X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                        n_informative=2, n_redundant=2)

    train_samples = 100  # Samples used for training the models

    X_train = X[:train_samples]
    X_test = X[train_samples:]
    y_train = y[:train_samples]
    y_test = y[train_samples:]


    # Trials object to track progress
    trials = Trials()

    # Optimize
    best = fmin(
        fn=partial(objective, n_folds=n_folds, X=X_train, y=y_train),
        space=space,
        algo=tpe.suggest,
        max_evals=16,
        trials=trials
    )

    pprint.pprint(best)