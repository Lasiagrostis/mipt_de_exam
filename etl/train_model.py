import pandas as pd
import os
import pickle
import logging
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = "/opt/airflow"
    
def load_latest_data():
    # Получаем путь до директории проекта
    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    results_dir = os.path.join(project_root, "results", "data")

    today = datetime.today().strftime("%Y_%m_%d")
    today_filename = f"breast_cancer_data_{today}.csv"
    today_path = os.path.join(results_dir, today_filename)

    if os.path.exists(today_path):
        logging.info(f"Загружаем файл за сегодня: {today_filename}")
        return pd.read_csv(today_path)

    all_files = [
        f for f in os.listdir(results_dir)
        if f.startswith("breast_cancer_data_") and f.endswith(".csv")
    ]
    if not all_files:
        raise FileNotFoundError(f"Нет ни одного сохранённого датасета в {results_dir}")

    def extract_date(f):
        date_str = f.replace("breast_cancer_data_", "").replace(".csv", "")
        return datetime.strptime(date_str, "%Y_%m_%d")

    latest_file = sorted(all_files, key=extract_date)[-1]
    logging.info(f"Сегодняшний файл не найден. Загружаем последний: {latest_file}")
    return pd.read_csv(os.path.join(results_dir, latest_file))


def train_logistic_regression():
    # Получаем путь до директории проекта
    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    model_dir = os.path.join(project_root, "results", "models")
    os.makedirs(model_dir, exist_ok=True)

    df = load_latest_data()
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop(columns=["id", "diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    logging.info(f"\n{report}")

    today = datetime.today().strftime("%Y_%m_%d")
    model_path = os.path.join(model_dir, f"model_{today}.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Модель сохранена в: {model_path}")

    # Сохраняем метрики
    metrics_path = os.path.join(model_dir, f"model_{today}_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Train date: {datetime.today().strftime('%Y-%m-%d')}\n")
        f.write("=== Logistic Model Results ===\n")
        f.write(report)
        f.write(f"\nAccuracy: {accuracy:.4f}\n")

    logging.info(f"Метрики модели сохранены в: {metrics_path}")

train_logistic_regression()