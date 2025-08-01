#!/bin/bash

cd app/
# Building on macbook m2
docker buildx build --platform=linux/amd64 -t basic-ml-microservice:latest .
# Building on linux
#docker build -t basic-ml-microservice:latest .

#docker run --rm -it -p 8080:5000 basic-ml-microservice:latest
#curl http://localhost:8080/forecast
#curl -d store_number=1 -d forecast_start_date="2021-10-01T00:00:00"
#docker ps
