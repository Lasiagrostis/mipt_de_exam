import pandas as pd
import logging
from datetime import datetime
import os

if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = "/opt/airflow"

def load_breast_cancer_data():
    # Универсальная настройка корневой директории
    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    # project_root = '/opt/airflow'

    logs_dir = os.path.join(project_root, "logs", "data_quality")
    results_dir = os.path.join(project_root, "results", "data")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Настройка логгера
    logger = logging.getLogger("data_quality_logger")
    logger.setLevel(logging.INFO)

    today = datetime.today().strftime("%Y_%m_%d")
    log_file_path = os.path.join(logs_dir, f"data_log_{today}.log")

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file_path for h in logger.handlers):
        file_handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Названия колонок
    data_columns = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    df = pd.read_csv(url, header=None, names=data_columns)

    logger.info(f"Загружено строк: {len(df)}")
    logger.info("Количество пропущенных значений:\n" + str(df.isnull().sum().fillna(0)))

    save_path = os.path.join(results_dir, f"breast_cancer_data_{today}.csv")
    df.to_csv(save_path, index=False)
    logger.info(f"Данные сохранены в файл: {save_path}")
    
load_breast_cancer_data()