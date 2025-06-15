import pandas as pd
import logging
from datetime import datetime
import pickle
from sklearn.utils import resample
import os

if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = "/opt/airflow"

def make_predictions():
    # Получение корневой директории
    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    print(os.getcwd())

    models_dir = os.path.join(project_root, "results", "models")
    output_dir = os.path.join(project_root, "results", "predictions")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Начинаем загрузку данных для предсказаний")

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
    df = df.drop(columns=["diagnosis"])
    logging.info(f"Загружено строк: {len(df)}")

    # Увеличение объёма данных
    df_resampled = resample(df, replace=True, n_samples=3 * len(df), random_state=42)
    logging.info(f"После ресемплинга: {len(df_resampled)} строк")

    def load_latest_model():
        today = datetime.today().strftime("%Y_%m_%d")
        today_filename = f"model_{today}.pkl"
        today_path = os.path.join(models_dir, today_filename)

        if os.path.exists(today_path):
            logging.info(f"Загружаем модель за сегодня: {today_filename}")
            with open(today_path, "rb") as f:
                return pickle.load(f)

        all_files = [
            f for f in os.listdir(models_dir)
            if f.startswith("model_") and f.endswith(".pkl")
        ]
        if not all_files:
            raise FileNotFoundError(f"Нет сохранённых моделей в {models_dir}")

        def extract_date(f):
            date_str = f.replace("model_", "").replace(".pkl", "")
            return datetime.strptime(date_str, "%Y_%m_%d")

        latest_file = sorted(all_files, key=extract_date)[-1]
        logging.info(f"Модель за сегодня не найдена. Загружаем последнюю: {latest_file}")
        with open(os.path.join(models_dir, latest_file), "rb") as f:
            return pickle.load(f)

    model = load_latest_model()

    # Предсказание
    X = df_resampled.drop(columns=["id"])
    predictions = model.predict(X)
    df_resampled["prediction"] = predictions
    df_resampled["prediction"] = df_resampled["prediction"].map({1: "M", 0: "B"})
    logging.info("Предсказания завершены")

    # Сохранение
    today = datetime.today().strftime("%Y_%m_%d")
    output_path = os.path.join(output_dir, f"predictions_{today}.csv")
    df_resampled.to_csv(output_path, index=False)
    logging.info(f"Предсказания сохранены в файл: {output_path}")

make_predictions()