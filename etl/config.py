import os

def get_project_root():
    """Определяет корневую директорию проекта в зависимости от среды выполнения."""
    # Приоритетная переменная — задаётся вручную или через Airflow env
    if "PROJECT_ROOT" in os.environ:
        return os.environ["PROJECT_ROOT"]

    # Проверка на переменные окружения Airflow
    if "AIRFLOW_HOME" in os.environ:
        return os.environ["AIRFLOW_HOME"]

    # Автоопределение Docker-контейнера
    try:
        with open('/proc/1/cgroup', 'rt') as f:
            if any('docker' in line for line in f):
                return "/opt/airflow"
    except Exception:
        pass

    # По умолчанию — локальная директория
    return os.getcwd()
