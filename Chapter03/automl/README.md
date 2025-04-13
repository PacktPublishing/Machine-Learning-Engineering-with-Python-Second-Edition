# Autosklearn example
There are known issues around installing auto-sklearn on MacOS and Windows systems so I have set this up to run in a docker container.

To run this example just run the following (this assumes you have already run ```conda env create -f mlewp-chapter03.yml```):

```bash
docker build -t autosklearn .
docker run autosklearn
```

