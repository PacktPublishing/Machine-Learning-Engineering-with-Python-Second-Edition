1. Run  kind create cluster --name mlewp-kserve --config cluster-config.yaml
2. Run kubectl get nodes to check nodes are there
3. Create a simple nginx deployment to test port forwarding

❯ kubectl create deployment nginx --image=nginx

deployment.apps/nginx created
❯ kubectl get deployments
kubectl get pods

NAME    READY   UP-TO-DATE   AVAILABLE   AGE
nginx   0/1     1            0           5s
NAME                    READY   STATUS              RESTARTS   AGE
nginx-76d6c9b8c-5dlp4   0/1     ContainerCreating   0          5s
❯ kubectl port-forward nginx-76d6c9b8c-5dlp4-n 8080:80
Error from server (NotFound): pods "nginx-76d6c9b8c-5dlp4-n" not found
❯ kubectl port-forward nginx-76d6c9b8c-5dlp4 8080:80
Forwarding from [::1]:8080 -> 80

4. Open a new terminal
5. Run a curl to check port-forwarding is working

❯ curl http://127.0.0.1:80

<html>
<head><title>404 Not Found</title></head>
<body>
<center><h1>404 Not Found</h1></center>
<hr><center>nginx</center>
</body>
</html>

# FastAPI service 

## Containerize
1. Define simple test fastapi service

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    input: str

class PredictionResponse(BaseModel):
    prediction: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Perform model inference here and return the prediction
    prediction = f"Prediction for input: {request.input}"
    return PredictionResponse(prediction=prediction)



2. Define simple Dockerfile

FROM tiangolo/uvicorn-gunicorn-fastapi:latest

COPY ./app /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

3. Build it
docker build -t custom-kserve-endpoint .

4. Log into docker hub 

docker login

5. Build and push to docker hub

docker build -t electricweegie/custom-kserve-endpoint:latest .
docker push electricweegie/custom-kserve-endpoint:latest

## Kubernetes deployment

1. Write the kubernetes manifest for the deployment (kserve-custom-endpoint.yaml):

apiVersion: "serving.kserve.io/v1alpha1"
kind: "Service"
metadata:
  name: "custom-kserve-endpoint"
spec:
  template:
    spec:
      containers:
        - image: "electricweegie/custom-kserve-endpoint:latest"
          name: "my-fastapi-app"
          ports:
            - containerPort: 80

2. Make sure you are using the correct cluster context

kubectl config use-context kind-mlewp-kserve-cluster

3. Deploy the service to the cluster


----
# Starting again - straight kubernetes deployment.

1. Running the docker container with the fast api app

docker run -d --platform linux/amd64 -p 80:8080 electricweegie/custom-kserve-endpoint

Had to specify platform to avoid an error and use port 80 locally as 8080 already has a binding (docker daemon?)

POST in postman to  http://0.0.0.0:80/predict with input

http://0.0.0.0:80/predict

gives 

{
    "prediction": "Prediction for input: hello world"
}

response.

THE KEY WAS TO EXPOSE THE SERVICE DUH!

❯ kubectl expose deployment fast-api-deployment --type=LoadBalancer --port=8080
service/fast-api-deployment exposed
❯ kubectl get services
NAME                  TYPE           CLUSTER-IP     EXTERNAL-IP   PORT(S)          AGE
fast-api-deployment   LoadBalancer   10.97.155.18   127.0.0.1     8080:31689/TCP   9s
fast-api-service      LoadBalancer   10.108.93.1    127.0.0.1     8000:30472/TCP   176m
kubernetes            ClusterIP      10.96.0.1      <none>        443/TCP          3h6m
❯ minikube service fast-|-----------|---------------------|-------------|---------------------------|
| NAMESPACE |        NAME         | TARGET PORT |            URL            |
|-----------|---------------------|-------------|---------------------------|
| default   | fast-api-deployment |        8080 | http://192.168.49.2:31689 |
|-----------|---------------------|-------------|---------------------------|
🏃  Starting tunnel for service fast-api-deployment.
|-----------|---------------------|-------------|------------------------|
| NAMESPACE |        NAME         | TARGET PORT |          URL           |
|-----------|---------------------|-------------|------------------------|
| default   | fast-api-deployment |             | http://127.0.0.1:52108 |
|-----------|---------------------|-------------|------------------------|
🎉  Opening service default/fast-api-deployment in default browser...
❗  Because you are using a Docker driver on darwin, the terminal needs to be open to run it.


postman post http://127.0.0.1:52108/predict

{
    "store_id": "4",
    "begin_date": "2023-03-01T00:00:00Z",
    "end_date": "2023-03-07T00:00:00Z",
    "input": "input"
}

return 

{
    "prediction": "Prediction for input: input"
}