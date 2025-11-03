import sys
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema
from pandera.errors import SchemaError


def get_credit_schema() -> DataFrameSchema:
    """Возвращает схему проверки данных"""
    return pa.DataFrameSchema(
        columns={

            #  Основные характеристики клиента
            "ID": Column(int, Check.gt(0), unique=True, nullable=False),

            "LIMIT_BAL": Column(float, Check.ge(0), nullable=False),

            "SEX": Column(int, Check.isin([1, 2]), nullable=False),

            "EDUCATION": Column(int, Check.isin([1, 2, 3, 4, 5, 6]), nullable=False),

            "MARRIAGE": Column(int, Check.isin([1, 2, 3]), nullable=False),

            "AGE": Column(int, [Check.ge(18), Check.le(100)], nullable=False),

            #  Статусы платежей
            **{
                f"PAY_{i}": Column(
                    int,
                    Check.isin([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                    nullable=False
                )
                for i in [0, 2, 3, 4, 5, 6]
            },

            #  Счета и платежи
            **{
                f"BILL_AMT{i}": Column(float, nullable=False)
                for i in range(1, 7)
            },
            **{
                f"PAY_AMT{i}": Column(float, Check.ge(0), nullable=False)
                for i in range(1, 7)
            },

            #  Целевая переменная
            "default.payment.next.month": Column(int, Check.isin([0, 1]), nullable=False),
        },

        #  Межколоночные проверки
        checks=[
            # Проверка логики последовательности PAY_*
            # Check(check_pay_consistency, error="Нарушена логика последовательности PAY_*"),
            # Возраст и статус брака не противоречат — минимальный возраст для 'married' ≥ 18
            Check(
                lambda df: (
                        (df["MARRIAGE"] != 1) | (df["AGE"] >= 18)
                ).all(),
                error="Married клиент не может быть младше 18 лет."
            ),

            # Лимит должен быть больше нуля и не экстремально мал (например, > 10 000 NT$)
            Check(
                lambda df: df["LIMIT_BAL"].median() > 10000,
                error="Слишком маленький медианный LIMIT_BAL — возможно, ошибка загрузки."
            ),
        ],
    )


def validate_credit_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Проверяет DataFrame по схеме Pandera.
    Возвращает валидированные данные или выбрасывает исключение SchemaErrors.
    """
    schema = get_credit_schema()
    validated = schema.validate(data, lazy=True)
    return validated

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_data.py <path_to_csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    try:
        df = pd.read_csv(csv_path)
        validate_credit_data(df)
        print("✅ Validation passed")
    except SchemaError as e:
        print("❌ Validation failed")
        print(str(e))