import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATOS_CSV = os.path.join(PROJECT_DIR, '01_Datos_Sucios', 'loan_data.csv')


def test_datos_existen():
    assert os.path.exists(DATOS_CSV), "El archivo de datos no existe"


def test_cargar_datos():
    df = pd.read_csv(DATOS_CSV)
    assert len(df) > 0, "El dataframe esta vacio"
    assert 'loan_status' in df.columns, "Falta la columna objetivo"


def test_columnas_esperadas():
    df = pd.read_csv(DATOS_CSV)
    columnas = ['person_age', 'person_gender', 'person_education',
                'person_income', 'person_emp_exp', 'person_home_ownership',
                'loan_amnt', 'loan_intent', 'loan_int_rate',
                'loan_percent_income', 'cb_person_cred_hist_length',
                'credit_score', 'previous_loan_defaults_on_file', 'loan_status']
    for col in columnas:
        assert col in df.columns, f"Falta la columna {col}"


def test_validacion_rangos():
    df = pd.read_csv(DATOS_CSV)
    assert df['person_age'].min() > 0, "Hay edades negativas"
    assert df['loan_int_rate'].min() > 0, "Hay tasas de interes negativas"
    assert set(df['loan_status'].unique()).issubset({0, 1}), "loan_status debe ser 0 o 1"


def test_target_binario():
    df = pd.read_csv(DATOS_CSV)
    valores = df['loan_status'].unique()
    assert len(valores) == 2, "loan_status debe tener exactamente 2 valores"
