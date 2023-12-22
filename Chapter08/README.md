# Chapter08 Supporting Notes

## Cluster Configuration

In the book, the two important cluster configuration steps are those for initializing the cluster and deploying the Fast API application.

All the ```.yaml``` files  mentioned are in ```Chapter08/cluster-configuration```.

### 1. Cluster initialization

```
kind create cluster
```
or
```
kind create cluster --config cluster-config-ch08.yaml 
```
or 
```
minikube start
```


### 2. Application deployment

```
kubectl apply –f direct-kube-deploy.yaml
``` 