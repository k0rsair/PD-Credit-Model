import pytest
import pandas as pd
import pandera.errors as pa_errors
from scripts.data.validate_data import validate_credit_data


def test_validate_credit_data_passes(sample_df):
    """Проверка, что корректные данные проходят валидацию."""
    validated = validate_credit_data(sample_df)
    assert isinstance(validated, pd.DataFrame)


def test_validate_credit_data_fails_on_invalid_age(sample_df):
    """Проверка, что некорректный возраст вызывает ошибку."""
    sample_df.loc[0, "AGE"] = 150
    with pytest.raises(pa_errors.SchemaErrors):
        validate_credit_data(sample_df)


@pytest.fixture
def sample_df():
    """Минимальный пример корректных данных."""
    data = {
        "ID": [1, 2],
        "LIMIT_BAL": [20000.0, 30000.0],
        "SEX": [1, 2],
        "EDUCATION": [2, 1],
        "MARRIAGE": [1, 2],
        "AGE": [25, 40],
        "PAY_0": [0, -1],
        "PAY_2": [2, 2],
        "PAY_3": [1, 1],
        "PAY_4": [0, 0],
        "PAY_5": [-1, -1],
        "PAY_6": [-1, -1],
        "BILL_AMT1":[3913.1,2000.0],
        "BILL_AMT2":[3102.3,3000.0],
        "BILL_AMT3":[689.1,6000.3],
        "BILL_AMT4":[0.0,0.0],
        "BILL_AMT5":[0.0,0.0],
        "BILL_AMT6":[0.0,0.0],
        "PAY_MEAN": [0.0, -0.2],
        "PAY_MAX": [0, 0],
        "PAY_MIN": [0, -1],
        "AGE_BINNED": [0, 2],
        "PAY_AMT1": [0.0, 0.0],
        "PAY_AMT2" : [689.0,6000.0],
        "PAY_AMT3" : [0.0,0.0],
        "PAY_AMT4" : [0.0,0.0],
        "PAY_AMT5" : [0.0,0.0],
        "PAY_AMT6" : [0.0,0.0],
        "default.payment.next.month": [0, 1],
    }
    return pd.DataFrame(data)