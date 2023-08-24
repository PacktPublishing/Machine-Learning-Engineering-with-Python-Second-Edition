import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
# Imbalanced classes
from imblearn.over_sampling import SMOTE


def ingest_and_prep_data(
        bank_dataset: str = 'bank_data/bank.csv'
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv('bank_data/bank.csv', delimiter=';', decimal=',')
    
    # Assume there was some EDA and feature analysis to select below
    feature_cols = ['job', 'marital', 'education', 'contact', 'housing', 'loan', 'default', 'day']

    # Features and target
    X = df[feature_cols].copy()
    y = df['y'].apply(lambda x: 1 if x == 'yes' else 0).copy()

    # Train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42)

    # Feature engineering
    enc = OneHotEncoder(handle_unknown='ignore')
    X_train = enc.fit_transform(X_train)
    
    return X_train, X_test, y_train, y_test

def rebalance_classes(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sm = SMOTE()
    X_balanced, y_balanced = sm.fit_resample(X, y)
    return X_balanced, y_balanced

def get_hyperparam_grid() -> dict:
    # Hyperparameter optimisation
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]  # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    return random_grid
    

def get_randomised_rf_cv(random_grid: dict) -> sklearn.model_selection._search.RandomizedSearchCV:
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=random_grid,
        n_iter=100,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring='f1'
    )
    return rf_random
    
    
    
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = ingest_and_prep_data()
    
    X_balanced, y_balanced = rebalance_classes(X_train, y_train)
    
    rf_random = get_randomised_rf_cv(
        random_grid=get_hyperparam_grid()
        )
    
    rf_random.fit(X_balanced, y_balanced)
