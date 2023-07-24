# Install the kserve quick start env
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.10/hack/quick_install.sh" | bash

# Create namespace for deployment
kubectl create namespace kserve-test

# Create an inference service
# NB: args: ["--enable_docs_url=True"] in model enables swaggerUI
kubectl apply -n kserve-test -f - <<EOF
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "sklearn-iris"
spec:
  predictor:
    model:
      args: ["--enable_docs_url=True"]
      modelFormat:
        name: sklearn
      storageUri: "gs://kfserving-examples/models/sklearn/1.0/model"
EOF

# Check inference service status
kubectl get inferenceservices sklearn-iris -n kserve-test