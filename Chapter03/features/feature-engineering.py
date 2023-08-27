from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline

X, y = load_wine(return_X_y=True)

# Make a 70/30 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30,
                                                    test_size=0.30,
                                                    random_state=42)

# Fit ridge classifier to the data
no_scale_clf = make_pipeline(RidgeClassifier(tol=1e-2, solver="sag"))
no_scale_clf.fit(X_train, y_train)
y_pred_no_scale = no_scale_clf.predict(X_test)

# Fit a ridge classifier after performing standard scaling
std_scale_clf = make_pipeline(StandardScaler(), RidgeClassifier(tol=1e-2, solver="sag"))
std_scale_clf.fit(X_train, y_train)
y_pred_std_scale = std_scale_clf.predict(X_test)

# Prediction accuracies with and without scaling
print('\nAccuracy [no scaling]')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, y_pred_no_scale)))

print('\nClassification Report [no scaling]')
print(metrics.classification_report(y_test, y_pred_no_scale))

print('\nAccuracy [scaling]')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, y_pred_std_scale)))

print('\nClassification Report [scaling]')
print(metrics.classification_report(y_test, y_pred_std_scale))

