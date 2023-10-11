import numpy as np
import sklearn.datasets
import sklearn.metrics
import autosklearn.classification

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    per_run_time_limit=30
)

automl.fit(X_train, y_train, dataset_name='wine')

print(automl.show_models())
print(automl.sprint_statistics())
predictions = automl.predict(X_test)
sklearn.metrics.accuracy_score(y_test, predictions)
