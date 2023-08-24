# We do not have an external load balancer for ingress gateway so have to use
# the service node's port
export INGRESS_HOST=$(kubectl get po -l istio=ingressgateway -n istio-system -o jsonpath='{.items[0].status.hostIP}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].nodePort}')

#----------------------------------------------------------------------------
#If this does not work you should port-forward, will not work in production!
#----------------------------------------------------------------------------
# # 1. Run the below to forward to port 3001
# INGRESS_GATEWAY_SERVICE=$(kubectl get svc --namespace istio-system --selector="app=istio-ingressgateway" --output jsonpath='{.items[0].metadata.name}')
# kubectl port-forward --namespace istio-system svc/${INGRESS_GATEWAY_SERVICE} 3001:80
# #2. Open new terminal and execute
# export INGRESS_HOST=localhost
# export INGRESS_PORT=8080
#----------------------------------------------------------------------------
# Then you can run the curl command with the SERVICE_HOSTNAME below ...
#----------------------------------------------------------------------------


# Prep an inference file
cat <<EOF > "./iris-input.json"
{
  "instances": [
    [6.8,  2.8,  4.8,  1.4],
    [6.0,  3.4,  4.5,  1.6]
  ]
}
EOF

# We do not have DNS enabled so we will curl with the ingress gateway external IP
# using the HOST header
SERVICE_HOSTNAME=$(kubectl get inferenceservice sklearn-iris -n kserve-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)
curl -v -H "Host: ${SERVICE_HOSTNAME}" "http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/sklearn-iris:predict" -d @./iris-input.json
