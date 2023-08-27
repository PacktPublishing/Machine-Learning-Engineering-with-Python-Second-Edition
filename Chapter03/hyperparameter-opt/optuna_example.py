# Inspired by https://github.com/optuna/optuna/blob/master/examples/sklearn/sklearn_optuna_search_cv_simple.py

# https://www.freecodecamp.org/news/hyperparameter-optimization-techniques-machine-learning/

# Optuna
import optuna

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import datasets

from functools import partial # to solve scoping problem when supplying more params' to objective function



def objective(trial, n_folds, X, y):
    """Objective function for tuning logistic regression hyperparameters"""
    params = {
        'warm_start': trial.suggest_categorical('warm_start', [True, False]),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'tol': trial.suggest_uniform('tol', 0.00001, 0.0001),
        'C': trial.suggest_uniform('C', 0.05, 2.5),
        'solver': trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear']),
        'max_iter': trial.suggest_categorical('max_iter', range(10, 500))
    }
    # Perform n_fold cross validation with hyperparameters
    clf = LogisticRegression(**params, random_state=42)
    scores = cross_val_score(clf, X, y, cv=n_folds, scoring='f1_macro')

    # Extract the best score
    max_score = max(scores)

    # Loss must be minimized
    loss = 1 - max_score

    # Dictionary with information for evaluation
    return loss


if __name__ == "__main__":
    n_folds = 5
    X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                    n_informative=2, n_redundant=2)

    train_samples = 100  # Samples used for training the models

    X_train = X[:train_samples]
    X_test = X[train_samples:]
    y_train = y[:train_samples]
    y_test = y[train_samples:]

    study = optuna.create_study(direction='minimize')
    study.optimize(partial(objective, n_folds=n_folds, X=X_train, y=y_train), n_trials=16)

    print(study.best_trial.params)
    print(stu)
