# Airflow 

Will use setup and some pointers from:

- https://airflow-tutorial.readthedocs.io
- https://airflow-tutorial.readthedocs.io/en/latest/first-airflow.html


Tutorial says directory should end up like this:

```bash
airflow                  # the root directory.
├── dags                 # root folder for all dags. files inside folders are not searched for dags.
│   ├── my_dag.py, # my dag (definitions of tasks/operators) including precedence.
│   └── ...
├── logs                 # logs for the various tasks that are run
│   └── my_dag           # DAG specific logs
│   │   ├── src1_s3      # folder for task-specific logs (log files are created by date of a run)
│   │   ├── src2_hdfs
│   │   ├── src3_s3
│   │   └── spark_task_etl
├── airflow.db           # SQLite database used by Airflow internally to track the status of each DAG.
├── airflow.cfg          # global configuration for Airflow (this can be overridden by config inside the file.)
└── ...
```
