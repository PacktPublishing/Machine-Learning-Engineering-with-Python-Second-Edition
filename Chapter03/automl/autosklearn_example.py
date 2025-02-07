import numpy as np
import sklearn.datasets
import sklearn.metrics
import autosklearn.classification
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    per_run_time_limit=30
)

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

automl.fit(X_train, y_train, dataset_name='wine')

print(automl.show_models())
print(automl.sprint_statistics())
predictions = automl.predict(X_test)
print(sklearn.metrics.accuracy_score(y_test, predictions))