"""
Random Forest Classifier on the wine dataset.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import joblib

from sklearn.metrics import classification_report

# Load the dataset
X, y = load_wine(return_X_y=True)
y = y == 2

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train the classifier
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)

# Dump file to joblib
joblib.dump(rfc, 'rfc.joblib')

metrics = classification_report(y_true=y_test, y_pred=rfc.predict(X_test), output_dict=True)

print(metrics)
# {
#     'False': {
#         'precision': 0.96875,
#         'recall': 0.9393939393939394,
#         'f1-score': 0.9538461538461539,
#         'support': 33
#     },
#     'True': {
#         'precision': 0.8461538461538461,
#         'recall': 0.9166666666666666,
#         'f1-score': 0.8799999999999999,
#         'support': 12
#     },
#     'accuracy': 0.9333333333333333,
#     'macro avg': {
#         'precision': 0.9074519230769231,
#         'recall': 0.928030303030303,
#         'f1-score': 0.916923076923077,
#         'support': 45
#     },
#     'weighted avg': {
#         'precision': 0.9360576923076923,
#         'recall': 0.9333333333333333,
#         'f1-score': 0.9341538461538461,
#         'support': 45
#     }
# }
 