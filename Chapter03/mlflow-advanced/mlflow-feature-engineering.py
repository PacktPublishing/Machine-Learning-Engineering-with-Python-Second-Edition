from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline

# Note - Must set up https://www.mlflow.org/docs/latest/tracking.html#backend-stores
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from pprint import pprint

if __name__=="__main__":
    # assume you have already run 'start-mlflow-server.sh'
    mlflow.set_tracking_uri("http://localhost:5000")

    X, y = load_wine(return_X_y=True)

    # Make a 70/30 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.30,
                                                        random_state=42)

    with mlflow.start_run(run_name="YOUR_RUN_NAME") as run:
        params = {
            'tol': 1e-2,
            'solver': 'sag'
        }
        # Fit a ridge classifier after performing standard scaling
        std_scale_clf = make_pipeline(StandardScaler(), RidgeClassifier(**params))
        std_scale_clf.fit(X_train, y_train)
        y_pred_std_scale = std_scale_clf.predict(X_test)

        mlflow.log_metrics({
            'accuracy': metrics.accuracy_score(y_test, y_pred_std_scale),
            'precision': metrics.precision_score(y_test, y_pred_std_scale, average='macro'),
            'f1': metrics.f1_score(y_test, y_pred_std_scale, average='macro'),
            'recall': metrics.recall_score(y_test, y_pred_std_scale, average='macro')
        })

        mlflow.log_params(params)

        # Log the sklearn model and register as version 1
        mlflow.sklearn.log_model(
            sk_model=std_scale_clf,
            artifact_path="sklearn-model",
            registered_model_name="sk-learn-std-scale-clf"
        )

    # Fetch specific model and version ...
    model_name = "sk-learn-std-scale-clf"
    model_version = 1
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )
    model.predict(X_test)

    # Transition the model stage to 'Staging'
    client = MlflowClient()
    client.transition_model_version_stage(
        name="sk-learn-std-scale-clf",
        version=1,
        stage="Staging"
    )
    # Transition the model stage to 'Production'
    client = MlflowClient()
    client.transition_model_version_stage(
        name="sk-learn-std-scale-clf",
        version=1,
        stage="Production"
    )

    # Fetch model based on stage name ...
    stage = 'Production'
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{stage}"
    )
    # Search model versions of a given model name
    client = MlflowClient()
    for mv in client.search_model_versions("name='sk-learn-std-scale-clf'"):
        pprint(dict(mv), indent=4)

    # Transition the model stage to 'Archived'
    client = MlflowClient()
    client.transition_model_version_stage(
        name="sk-learn-std-scale-clf",
        version=1,
        stage="Archived"
    )



