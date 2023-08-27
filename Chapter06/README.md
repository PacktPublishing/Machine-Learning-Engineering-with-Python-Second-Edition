# Chapter06 Supporting Notes

## Environment configuration
For this chapter, yml file provided sets up your environment for Spark and Ray dev but you can also use Poetry to run the environment for Ray.

So, the following will be good for running the Spark examples
```
conda activate mlewp-chapter06.yml
```

But to run the Ray examples you can also run the examples with:

```
poetry run jupyter notebook
```

```
poetry run ray_air_basic.py
```

```
poetry run pytest
```
etc..