# Copy and paste the following commands into your terminal for each stage.

# 1. Create your ECR repo.
aws ecr create-repository \
--repository-name basic-ml-microservice \
--image-scanning-configuration scanOnPush=true \
--region eu-west-1

# 2. 