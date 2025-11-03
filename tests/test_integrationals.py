import os
import subprocess
from pathlib import Path

DATA_DIR = Path("data/processed")

def run_script(script_path, *args):
    """Вспомогательная функция для запуска скрипта как subprocess."""
    result = subprocess.run(
        ["python", script_path, *map(str, args)],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"Script {script_path} failed"


def test_prepare_data():
    """Проверяем, что prepare_data.py успешно создаёт prepared.csv"""
    output_path = DATA_DIR / "prepared.csv"
    run_script("scripts/data/prepare_data.py")
    assert output_path.exists(), "prepared.csv не был создан"


def test_validate_data():
    """Проверяем, что validate_data.py отрабатывает без ошибок"""
    input_path = DATA_DIR / "prepared.csv"
    run_script("scripts/data/validate_data.py", input_path)


def test_feature_engineering():
    """Проверяем, что feature_engineering.py создаёт featured.csv"""
    input_path = DATA_DIR / "prepared.csv"
    output_path = DATA_DIR / "featured.csv"
    run_script("scripts/data/feature_engineering.py", input_path, output_path)
    assert output_path.exists(), "featured.csv не был создан"


def test_train_model():
    """Проверяем, что train_model.py отрабатывает и создаёт модель"""
    input_path = DATA_DIR / "featured.csv"
    run_script("scripts/models/train_model.py", input_path)
    # Проверим, что в mlruns появилась запись
    mlruns_path = Path("mlruns")
    assert mlruns_path.exists(), "MLflow не создал директорию mlruns"