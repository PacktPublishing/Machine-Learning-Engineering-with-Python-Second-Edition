#  Main application file app.py. This web service is a very simple one for returning basic requests that can be built
# upon later.
from flask import Flask
from flask_restful import Api, Resource
from resources.forecast import ForecastHandler, Forecaster
import logging

app = Flask(__name__)
api = Api(app)

#forecaster = Forecaster()
#api.add_resource(ForecastHandler, '/forecast', resource_class_kwargs={'forecaster': forecaster})

if __name__ == '__main__':
    logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    logging.info('Main app sequence begun')
    app.run(debug=True, host='0.0.0.0', port=5000) # change debug=False in production
    logging.info('App finished')
