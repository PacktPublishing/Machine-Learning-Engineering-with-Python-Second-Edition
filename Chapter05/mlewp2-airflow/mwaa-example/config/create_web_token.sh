#!/bin/bash
HOST=d83b8c42-0367-4dfd-88c1-7137fd0f5096-vpce.c14.eu-west-1.airflow.amazonaws.com
YOUR_URL=https://$HOST/aws_mwaa/aws-console-sso?login=true#
WEB_TOKEN=$(aws mwaa create-web-login-token --region eu-west-1 --name mlewp2-airflow-dev-env --query WebToken --output text)
echo $YOUR_URL$WEB_TOKEN