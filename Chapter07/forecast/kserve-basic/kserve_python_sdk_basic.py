'''
Based on https://www.kubeflow.org/docs/external-add-ons/kserve/first_isvc_kserve/

'''
from kubernetes import client 
from kserve import KServeClient
from kserve import constants
from kserve import utils
from kserve import V1beta1InferenceService
from kserve import V1beta1InferenceServiceSpec
from kserve import V1beta1PredictorSpec
from kserve import V1beta1SKLearnSpec


# MLFLOW


# KUBEFLOW
namespace = utils.get_default_target_namespace()

name='sklearn-iris'
kserve_version='v1beta1'
api_version = constants.KSERVE_GROUP + '/' + kserve_version

isvc = V1beta1InferenceService(api_version=api_version,
                               kind=constants.KSERVE_KIND,
                               metadata=client.V1ObjectMeta(
                                   name=name, namespace=namespace, annotations={'sidecar.istio.io/inject':'false'}),
                               spec=V1beta1InferenceServiceSpec(
                               predictor=V1beta1PredictorSpec(
                               sklearn=(V1beta1SKLearnSpec(
                                   storage_uri="gs://kfserving-samples/models/sklearn/iris"))))
)

KServe = KServeClient()
KServe.create(isvc)

KServe.get(name, namespace=namespace, watch=True, timeout_seconds=120)

'''

'''
import requests

isvc_resp = KServe.get(name, namespace=namespace)
isvc_url = isvc_resp['status']['address']['url']

print(isvc_url)

inference_input = {
  'instances': [
    [6.8,  2.8,  4.8,  1.4],
    [6.0,  3.4,  4.5,  1.6]
  ]
}

response = requests.post(isvc_url, json=inference_input)
print(response.text)