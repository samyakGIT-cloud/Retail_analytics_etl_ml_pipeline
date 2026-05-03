from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

PROJECT_PATH = r"C:\Users\SAMYAK\OneDrive\Desktop\Data Engineering\AMDOX INTERN\Neural-Retail"

with DAG(
    dag_id="neural_retail_pipeline",
    start_date=datetime(2026, 4, 1),
    schedule="@daily",
    catchup=False,
    tags=["retail", "etl", "ml"],
) as dag:

    run_etl = BashOperator(
        task_id="run_etl_pipeline",
        bash_command=f'cd "{PROJECT_PATH}" && py -3.12 etl_pipeline.py'
    )

    run_forecast = BashOperator(
        task_id="run_forecast_model",
        bash_command=f'cd "{PROJECT_PATH}" && py -3.12 forecast_model.py'
    )

    run_prophet = BashOperator(
        task_id="run_prophet_model",
        bash_command=f'cd "{PROJECT_PATH}" && py -3.12 prophet_forecast.py'
    )

    run_lstm = BashOperator(
        task_id="run_lstm_model",
        bash_command=f'cd "{PROJECT_PATH}" && py -3.12 LSTM_forecast.py'
    )

    run_etl >> run_forecast >> run_prophet >> run_lstm
