from flask_restful import Resource, reqparse
from flask import jsonify
import numpy as np

post_parser = reqparse.RequestParser()
post_parser.add_argument(
    'store_number',
    type=int,
    required=True,  # need a store
    location="json",
    help='The numerical id of the store'
)

post_parser.add_argument(
    'forecast_start_date',
    type=str,
    location="json",
    help='start date for forecast in iso format YYYY-mm-DDTHH:MM:SS'
)


class ForecastHandler(Resource):
    def __init__(self, **kwargs):
        self.forecaster = kwargs['forecaster']

    def get(self):
        return {}

    def post(self):
        args = post_parser.parse_args()
        print(args)
        result = {"store_number": args["store_number"], "result": self.forecaster.forecast()}
        return jsonify(result)


class Forecaster(object):

    def __init__(self, model_config=None):
        # Do stuff using model config, for example configure MLFLow server addresses
        self.model = None  # placeholder for later

    def forecast(self, params={}, steps=10):
        if self.model is None:
            return np.random.random(steps).tolist()  # as a placeholder for actual forecast
