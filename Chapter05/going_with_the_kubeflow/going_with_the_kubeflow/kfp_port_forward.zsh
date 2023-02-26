# change `--n` if you deployed Kubeflow Pipelines into a different namespace
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80


# Step 2: the following code will create a kfp.Client() against your port-forwarded ml-pipeline-ui service:

# import kfp

# client = kfp.Client(host="http://localhost:3000")

# print(client.list_experiments())
