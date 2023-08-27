from datetime import timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'Andrew McMahon',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'email': ['example@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}


#instantiate DAG
dag = DAG(
    'classification_pipeline',
    default_args=default_args,
    description=’Basic pipeline for classifying the Wine Dataset',
    schedule_interval=timedelta(days=1), # run daily? check
)


get_data = BashOperator(
    task_id='get_data',
    bash_command='python3 /usr/local/airflow/scripts/get_data.py',
    dag=dag,
)

train_model= BashOperator(
    task_id='train_model',
    depends_on_past=False,
    bash_command='python3 /usr/local/airflow/scripts/train_model.py',
    retries=3,
    dag=dag,
)

# Persist to MLFlow
persist_model = BashOperator(
    task_id='persist_model',
    depends_on_past=False,
    bash_command=’python ……./persist_model.py,
    retries=3,
    dag=dag,
)

get_data >> train_model >> persist_model 
