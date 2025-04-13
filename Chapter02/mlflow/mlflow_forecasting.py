#BASED ON EXAMPLE FROM MLFLOW DOCS
# https://github.com/mlflow/mlflow/blob/master/examples/prophet/train.py
import pandas as pd
from prophet import Prophet

from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

import mlflow
import mlflow.pyfunc

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

class FbProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def load_context(self, context):
        from fbprophet import Prophet

        return

    def predict(self, context, model_input):
        future = self.model.make_future_dataframe(periods=model_input["periods"][0])
        return self.model.predict(future)


seasonality = {
    'yearly': True,
    'weekly': True,
    'daily': True
}

def train_predict(df_all_data, df_all_train_index, seasonality_params=seasonality):
    # grab split data
    df_train = df_all_data.copy().iloc[0:df_all_train_index]
    df_test = df_all_data.copy().iloc[df_all_train_index:]

    # Useful for multiple runs (only doing one run in this sample notebook)
    with mlflow.start_run():
        # create Prophet model
        model = Prophet(
            yearly_seasonality=seasonality_params['yearly'],
            weekly_seasonality=seasonality_params['weekly'],
            daily_seasonality=seasonality_params['daily']
        )
        # train and predict
        model.fit(df_train)

        # Evaluate Metrics
        #df_cv = cross_validation(model, initial="730 days", period="180 days", horizon="365 days")
        df_cv = cross_validation(model, initial="600 days", period="90 days", horizon="120 days")

        df_p = performance_metrics(df_cv)

        # Print out metrics
        print("  CV: \n%s" % df_cv.head())
        print("  Perf: \n%s" % df_p.head())

        # Log parameter, metrics, and model to MLflow
        mlflow.log_metric("rmse", df_p.loc[0, "rmse"])

        mlflow.pyfunc.log_model("model", python_model=FbProphetWrapper(model))
        print(
            "Logged model with URI: runs:/{run_id}/model".format(
                run_id=mlflow.active_run().info.run_id
            )
        )

    predicted = model.predict(df_test)
    return predicted, df_train, df_test


if __name__ == "__main__":
    # Read in Data
    df = pd.read_csv('C:/Users/DELL/PycharmProjects/Machine-Learning-Engineering-with-Python-Second-Edition/Chapter01/forecasting/train.csv')
    df.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)
    # Filter out store and item 1
    #print(df.columns)
    #print(df.head())

    #df_store1_item1 = df[(df['Store'] == 1) & (df['item'] == 1)].reset_index(drop=True)
    df_store1 = df[df['Store'] == 1].reset_index(drop=True)

    #print(f"{len(df_store1)} rows")
    #print(f"Start: {df_store1['ds'].min()}, End: {df_store1['ds'].max()}")
    #train_index = int(0.8 * df_store1_item1.shape[0])
    train_index = int(0.8 * df_store1.shape[0])
    """predicted, df_train, df_test = train_predict(
        df_all_data=df_store1_item1,
        df_all_train_index=train_index,
        seasonality_params=seasonality
    )"""

    predicted, df_train, df_test = train_predict(
        df_all_data=df_store1,
        df_all_train_index=train_index,
        seasonality_params=seasonality
    )
