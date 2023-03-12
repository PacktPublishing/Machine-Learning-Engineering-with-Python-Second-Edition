"""
Random Forest Classifier on the wine dataset.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
#import joblib

# Load the dataset
X, y = load_wine(return_X_y=True)
y = y == 2

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train the classifier
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)

# Dump file to joblib
# joblib.dump(rfc, 'rfc.joblib')
 