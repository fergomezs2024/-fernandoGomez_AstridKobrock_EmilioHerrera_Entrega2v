import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import sqlite3
import os
import logging
import time
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATOS_DIR = os.path.join(PROJECT_DIR, '01_Datos_Sucios')
RESULTADOS_DIR = os.path.join(PROJECT_DIR, '03_EDA_Resultados')

os.makedirs(RESULTADOS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RESULTADOS_DIR, 'log_ejecucion.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()


# ============================================================
# ETAPA 1: INGESTA DE DATOS
# ============================================================
log.info("=" * 60)
log.info("ETAPA 1: INGESTA DE DATOS")
log.info("=" * 60)

inicio_ingesta = time.time()

ruta_datos = os.path.join(DATOS_DIR, 'loan_data.csv')
df = pd.read_csv(ruta_datos)

tiempo_ingesta = time.time() - inicio_ingesta

log.info(f"Datos cargados desde: {ruta_datos}")
log.info(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
log.info(f"Tiempo de ingesta: {tiempo_ingesta:.2f} segundos")
log.info(f"Primeras 5 filas:")
log.info(f"\n{df.head()}")
log.info(f"Tipos de datos:")
log.info(f"\n{df.dtypes}")


# ============================================================
# ETAPA 2: LIMPIEZA Y TRANSFORMACION DE DATOS
# ============================================================
log.info("=" * 60)
log.info("ETAPA 2: LIMPIEZA Y TRANSFORMACION")
log.info("=" * 60)

inicio_limpieza = time.time()

log.info("Valores nulos por columna:")
log.info(f"\n{df.isnull().sum()}")

filas_antes = len(df)

df.dropna(inplace=True)
log.info(f"Filas eliminadas por nulos: {filas_antes - len(df)}")

filas_antes_outliers = len(df)
df = df[df['person_age'] >= 18]
df = df[df['person_age'] <= 100]
df = df[df['person_income'] >= 0]
df = df[df['person_emp_exp'] >= 0]
log.info(f"Filas eliminadas por outliers: {filas_antes_outliers - len(df)}")
log.info(f"Filas restantes despues de limpieza: {len(df)}")

log.info(f"loan_int_rate - Min: {df['loan_int_rate'].min()}, Max: {df['loan_int_rate'].max()}, Promedio: {df['loan_int_rate'].mean():.2f}")
log.info("Los valores de loan_int_rate ya estan en porcentaje, no se necesita conversion.")

genero_original = df['person_gender'].copy()

columnas_categoricas = ['person_gender', 'person_education', 'person_home_ownership',
                        'loan_intent', 'previous_loan_defaults_on_file']

le = LabelEncoder()
for col in columnas_categoricas:
    log.info(f"Codificando {col}: {list(df[col].unique())}")
    df[col] = le.fit_transform(df[col])

tiempo_limpieza = time.time() - inicio_limpieza
log.info(f"Tiempo de limpieza: {tiempo_limpieza:.2f} segundos")


# ============================================================
# ETAPA 3: VALIDACION ESTRUCTURAL Y SEMANTICA
# ============================================================
log.info("=" * 60)
log.info("ETAPA 3: VALIDACION ESTRUCTURAL Y SEMANTICA")
log.info("=" * 60)

inicio_validacion = time.time()
errores_validacion = 0

log.info("--- Validacion Estructural ---")

columnas_esperadas = ['person_age', 'person_gender', 'person_education', 'person_income',
                      'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
                      'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                      'credit_score', 'previous_loan_defaults_on_file', 'loan_status']

for col in columnas_esperadas:
    if col in df.columns:
        log.info(f"  [OK] Columna '{col}' presente")
    else:
        log.error(f"  [ERROR] Columna '{col}' faltante")
        errores_validacion += 1

nulos_restantes = df.isnull().sum().sum()
if nulos_restantes == 0:
    log.info(f"  [OK] No hay valores nulos en el dataset")
else:
    log.error(f"  [ERROR] Quedan {nulos_restantes} valores nulos")
    errores_validacion += 1

log.info(f"  [OK] Cantidad de filas: {len(df)}")
log.info(f"  [OK] Cantidad de columnas: {len(df.columns)}")

log.info("--- Validacion Semantica ---")

if df['person_age'].min() >= 18 and df['person_age'].max() <= 100:
    log.info(f"  [OK] Edades en rango valido (18-100): min={df['person_age'].min()}, max={df['person_age'].max()}")
else:
    log.error(f"  [ERROR] Edades fuera de rango")
    errores_validacion += 1

if df['person_income'].min() >= 0:
    log.info(f"  [OK] Ingresos no negativos: min={df['person_income'].min()}")
else:
    log.error(f"  [ERROR] Existen ingresos negativos")
    errores_validacion += 1

if df['credit_score'].min() >= 300 and df['credit_score'].max() <= 850:
    log.info(f"  [OK] Credit score en rango valido (300-850): min={df['credit_score'].min()}, max={df['credit_score'].max()}")
else:
    log.warning(f"  [AVISO] Credit score fuera del rango tipico (300-850): min={df['credit_score'].min()}, max={df['credit_score'].max()}")

if df['loan_int_rate'].min() > 0 and df['loan_int_rate'].max() <= 100:
    log.info(f"  [OK] Tasa de interes en rango valido (0-100%): min={df['loan_int_rate'].min()}, max={df['loan_int_rate'].max()}")
else:
    log.error(f"  [ERROR] Tasa de interes fuera de rango")
    errores_validacion += 1

if set(df['loan_status'].unique()) == {0, 1}:
    log.info(f"  [OK] Variable objetivo loan_status es binaria (0 y 1)")
else:
    log.error(f"  [ERROR] loan_status tiene valores inesperados: {df['loan_status'].unique()}")
    errores_validacion += 1

completitud = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
log.info(f"  [OK] Completitud del dataset: {completitud:.1f}%")

tiempo_validacion = time.time() - inicio_validacion
log.info(f"Errores de validacion encontrados: {errores_validacion}")
log.info(f"Tiempo de validacion: {tiempo_validacion:.2f} segundos")

if errores_validacion > 0:
    log.error("VALIDACION FALLIDA - Revisar errores antes de continuar")
else:
    log.info("VALIDACION EXITOSA - Datos listos para carga")


# ============================================================
# ETAPA 4: CARGA DE DATOS
# ============================================================
log.info("=" * 60)
log.info("ETAPA 4: CARGA DE DATOS")
log.info("=" * 60)

inicio_carga = time.time()

df.to_csv(os.path.join(RESULTADOS_DIR, 'datos_limpios.csv'), index=False)
log.info(f"Datos limpios guardados en: 03_EDA_Resultados/datos_limpios.csv")

ruta_db = os.path.join(RESULTADOS_DIR, 'prestamos.db')
conexion = sqlite3.connect(ruta_db)
df.to_sql('prestamos', conexion, if_exists='replace', index=False)

cursor = conexion.cursor()
cursor.execute("SELECT COUNT(*) FROM prestamos")
filas_cargadas = cursor.fetchone()[0]
log.info(f"Datos cargados en SQLite: 03_EDA_Resultados/prestamos.db")
log.info(f"Tabla 'prestamos' creada con {filas_cargadas} filas")

conexion.close()

tiempo_carga = time.time() - inicio_carga
log.info(f"Tiempo de carga: {tiempo_carga:.2f} segundos")


# ============================================================
# ANALISIS EXPLORATORIO (EDA)
# ============================================================
log.info("=" * 60)
log.info("ANALISIS EXPLORATORIO (EDA)")
log.info("=" * 60)

log.info("Estadisticas descriptivas:")
log.info(f"\n{df.describe()}")

log.info("Distribucion de loan_status:")
log.info(f"\n{df['loan_status'].value_counts()}")

plt.figure(figsize=(6, 4))
df['loan_status'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Distribucion de Incumplimiento')
plt.xlabel('Loan Status (0=Pagado, 1=Incumplimiento)')
plt.ylabel('Cantidad')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(RESULTADOS_DIR, 'distribucion_target.png'))
plt.close()
log.info("Grafica guardada: 03_EDA_Resultados/distribucion_target.png")

plt.figure(figsize=(6, 4))
plt.hist(df['person_age'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribucion de Edad de Solicitantes')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig(os.path.join(RESULTADOS_DIR, 'distribucion_edad.png'))
plt.close()
log.info("Grafica guardada: 03_EDA_Resultados/distribucion_edad.png")

plt.figure(figsize=(6, 4))
df.boxplot(column='loan_int_rate', by='loan_status')
plt.title('Tasa de Interes por Estado del Prestamo')
plt.suptitle('')
plt.xlabel('Loan Status (0=Pagado, 1=Incumplimiento)')
plt.ylabel('Tasa de Interes (%)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTADOS_DIR, 'tasa_interes_por_status.png'))
plt.close()
log.info("Grafica guardada: 03_EDA_Resultados/tasa_interes_por_status.png")


# ============================================================
# MODELO DE CLASIFICACION
# ============================================================
log.info("=" * 60)
log.info("MODELO DE CLASIFICACION")
log.info("=" * 60)

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log.info(f"Datos de entrenamiento: {X_train.shape[0]} filas")
log.info(f"Datos de prueba: {X_test.shape[0]} filas")

log.info("-" * 40)
log.info("MODELO 1: Arbol de Decision")

arbol = DecisionTreeClassifier(max_depth=5, random_state=42)
arbol.fit(X_train, y_train)
pred_arbol = arbol.predict(X_test)

acc_arbol = accuracy_score(y_test, pred_arbol)
log.info(f"Accuracy: {acc_arbol:.4f}")
log.info(f"Matriz de confusion:\n{confusion_matrix(y_test, pred_arbol)}")
log.info(f"Reporte:\n{classification_report(y_test, pred_arbol)}")

plt.figure(figsize=(20, 10))
plot_tree(arbol, feature_names=list(X.columns), class_names=['Pagado', 'Incumplido'],
          filled=True, rounded=True, fontsize=8, max_depth=3)
plt.title('Arbol de Decision - Como decide el modelo')
plt.tight_layout()
plt.savefig(os.path.join(RESULTADOS_DIR, 'arbol_decision.png'), dpi=150)
plt.close()
log.info("Grafica guardada: 03_EDA_Resultados/arbol_decision.png")

log.info("-" * 40)
log.info("MODELO 2: Regresion Logistica")

logistica = LogisticRegression(max_iter=1000, random_state=42)
logistica.fit(X_train, y_train)
pred_logistica = logistica.predict(X_test)

acc_logistica = accuracy_score(y_test, pred_logistica)
log.info(f"Accuracy: {acc_logistica:.4f}")
log.info(f"Matriz de confusion:\n{confusion_matrix(y_test, pred_logistica)}")
log.info(f"Reporte:\n{classification_report(y_test, pred_logistica)}")

if acc_arbol > acc_logistica:
    mejor_modelo = "Arbol de Decision"
    mejor_acc = acc_arbol
    predicciones = pred_arbol
else:
    mejor_modelo = "Regresion Logistica"
    mejor_acc = acc_logistica
    predicciones = pred_logistica

log.info(f"Mejor modelo: {mejor_modelo} con accuracy {mejor_acc:.4f}")


# ============================================================
# ANALISIS DE EQUIDAD
# ============================================================
log.info("=" * 60)
log.info("ANALISIS DE EQUIDAD POR GENERO")
log.info("=" * 60)

genero_test = genero_original.loc[X_test.index]

accs_genero = {}
for genero in ['male', 'female']:
    mascara = genero_test == genero
    if mascara.sum() > 0:
        acc = accuracy_score(y_test[mascara], predicciones[mascara])
        accs_genero[genero] = acc
        log.info(f"Accuracy para {genero}: {acc:.4f} ({mascara.sum()} muestras)")

plt.figure(figsize=(6, 4))
plt.bar(accs_genero.keys(), accs_genero.values(), color=['steelblue', 'salmon'])
plt.title('Accuracy por Genero')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(RESULTADOS_DIR, 'equidad_genero.png'))
plt.close()
log.info("Grafica guardada: 03_EDA_Resultados/equidad_genero.png")

diferencia = abs(list(accs_genero.values())[0] - list(accs_genero.values())[1])
log.info(f"Diferencia de accuracy entre generos: {diferencia:.4f}")
if diferencia < 0.05:
    log.info("La diferencia es menor al 5%, el modelo es razonablemente equitativo.")
else:
    log.warning("La diferencia es mayor al 5%, se deberia revisar posible sesgo.")


# ============================================================
# REPORTE FINAL Y KPIs DEL PIPELINE
# ============================================================
log.info("=" * 60)
log.info("REPORTE FINAL Y KPIs DEL PIPELINE")
log.info("=" * 60)

tiempo_total = tiempo_ingesta + tiempo_limpieza + tiempo_validacion + tiempo_carga
filas_procesadas = len(df)
filas_descartadas = 45000 - len(df)
tasa_completitud = completitud

log.info(f"--- KPIs de Monitoreo del Pipeline ---")
log.info(f"Tiempo total del pipeline: {tiempo_total:.2f} segundos")
log.info(f"  - Ingesta: {tiempo_ingesta:.2f}s")
log.info(f"  - Limpieza: {tiempo_limpieza:.2f}s")
log.info(f"  - Validacion: {tiempo_validacion:.2f}s")
log.info(f"  - Carga: {tiempo_carga:.2f}s")
log.info(f"Filas procesadas: {filas_procesadas}")
log.info(f"Filas descartadas: {filas_descartadas}")
log.info(f"Completitud: {tasa_completitud:.1f}%")
log.info(f"Errores de validacion: {errores_validacion}")

with open(os.path.join(RESULTADOS_DIR, 'reporte.txt'), 'w', encoding='utf-8') as f:
    f.write("REPORTE - Prediccion de Incumplimiento de Prestamos\n")
    f.write("=" * 60 + "\n\n")

    f.write("PIPELINE DataOps\n")
    f.write("-" * 40 + "\n")
    f.write(f"Etapa 1 - Ingesta: {tiempo_ingesta:.2f}s\n")
    f.write(f"Etapa 2 - Limpieza: {tiempo_limpieza:.2f}s\n")
    f.write(f"Etapa 3 - Validacion: {tiempo_validacion:.2f}s\n")
    f.write(f"Etapa 4 - Carga: {tiempo_carga:.2f}s\n")
    f.write(f"Tiempo total: {tiempo_total:.2f}s\n\n")

    f.write("KPIs DEL PIPELINE\n")
    f.write("-" * 40 + "\n")
    f.write(f"Filas originales: 45000\n")
    f.write(f"Filas despues de limpieza: {filas_procesadas}\n")
    f.write(f"Filas descartadas: {filas_descartadas}\n")
    f.write(f"Completitud: {tasa_completitud:.1f}%\n")
    f.write(f"Errores de validacion: {errores_validacion}\n\n")

    f.write("RESULTADOS DE LOS MODELOS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Arbol de Decision - Accuracy: {acc_arbol:.4f}\n")
    f.write(f"Regresion Logistica - Accuracy: {acc_logistica:.4f}\n")
    f.write(f"Mejor modelo: {mejor_modelo} ({mejor_acc:.4f})\n\n")

    f.write("ANALISIS DE EQUIDAD\n")
    f.write("-" * 40 + "\n")
    for genero, acc in accs_genero.items():
        f.write(f"Accuracy {genero}: {acc:.4f}\n")
    f.write(f"Diferencia: {diferencia:.4f}\n")

log.info("Reporte guardado en: 03_EDA_Resultados/reporte.txt")
log.info("")
log.info("=" * 60)
log.info("PIPELINE COMPLETADO EXITOSAMENTE")
log.info("Resultados en la carpeta '03_EDA_Resultados/'")
log.info("=" * 60)
