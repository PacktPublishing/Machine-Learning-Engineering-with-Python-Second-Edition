# test_dag.py
# will try pushing with yml
# This code borrows heavily from https://airflow.apache.org/docs/apache-airflow/stable/tutorial.htmlfrom
from datetime import timedelta
import logging
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['andrewpmcmahon629@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'simple_demo',
    default_args=default_args,
    description='A simple DAG with a few Python tasks.',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(2),
    tags=['example'],
)

### PYTHON FUNCTIONS
def log_context(**kwargs):
    for key, value in kwargs.items():
        logging.info(f"Context key {key} = {value}")

def compute_product(a=None, b=None):
    logging.info(f"Inputs: a={a}, b={b}")
    if a == None or b == None:
        return None
    return a * b

t1 = PythonOperator(
    task_id="task1",
    python_callable=log_context,
    dag=dag
)

t2 = PythonOperator(
    task_id="task2",
    python_callable=compute_product,
    op_kwargs={'a': 3, 'b': 5},
    dag=dag
)

t1 >> t2