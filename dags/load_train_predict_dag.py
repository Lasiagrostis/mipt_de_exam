from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

sys.path.append('/opt/airflow/etl')  # путь к скриптам

from load_data import load_breast_cancer_data
from train_model import train_logistic_regression
from predict import make_predictions

with DAG(
    dag_id="load_train_predict_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["MIPT"],
) as dag:

    load_data = PythonOperator(
        task_id="load_breast_cancer_data",
        python_callable=load_breast_cancer_data
    )

    train_model = PythonOperator(
        task_id="train_logistic_regression",
        python_callable=train_logistic_regression
    )
    
    predict_data = PythonOperator(
        task_id="make_predictions",
        python_callable=make_predictions
    )

load_data >> train_model >> predict_data