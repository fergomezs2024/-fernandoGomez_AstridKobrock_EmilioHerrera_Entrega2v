"""
============================================================
GENERADOR DE METRICAS PARA EL DASHBOARD WEB (DESPLIEGUE AUTOMATICO)
Prediccion de Incumplimiento de Prestamos - EV Parcial 3
============================================================
Este script es el "puente de datos" entre el pipeline de IA y la
pagina web publicada en GitHub Pages.

  1. Lee el dataset real (loan_data.csv).
  2. Reutiliza la limpieza y el entrenamiento del Parcial 3
     (Arbol de Decision, Regresion Logistica y Random Forest).
  3. Calcula todas las metricas (accuracy, precision, recall, F1,
     AUC, Gini, matriz de confusion, curva ROC, importancia).
  4. Exporta TODO a 'site/metrics.json'.

El dashboard (site/index.html) lee ese JSON y se actualiza solo.
GitHub Actions ejecuta este script en la nube en cada push, de modo
que la web siempre refleja el ultimo entrenamiento (conexion automatica).

Ejecutar localmente:  python 06_Despliegue_Web/generar_metrics.py
"""

import os
import json
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score, roc_curve)

warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Rutas robustas (funcionan en local y en GitHub Actions)
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)               # raiz del repo
SITE_DIR = os.path.join(SCRIPT_DIR, 'site')
os.makedirs(SITE_DIR, exist_ok=True)

# Busca el CSV en las ubicaciones posibles del proyecto
CANDIDATOS_CSV = [
    os.path.join(PROJECT_DIR, '01_Datos_Sucios', 'loan_data.csv'),
    os.path.join(PROJECT_DIR, '08_Entrega_Parcial_3', 'Entrega_Final', 'Datos', 'loan_data.csv'),
    os.path.join(SCRIPT_DIR, 'loan_data.csv'),
]
CSV_PATH = next((p for p in CANDIDATOS_CSV if os.path.exists(p)), None)
if CSV_PATH is None:
    raise FileNotFoundError(
        "No se encontro loan_data.csv. Ubicaciones probadas:\n  - "
        + "\n  - ".join(CANDIDATOS_CSV))

print(f"[1/5] Leyendo dataset: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
filas_originales = len(df)

# ------------------------------------------------------------
# Limpieza (identica al pipeline del Parcial 3)
# ------------------------------------------------------------
df.dropna(inplace=True)
df = df[(df['person_age'] >= 18) & (df['person_age'] <= 100)]
df = df[df['person_income'] >= 0]
df = df[df['person_emp_exp'] >= 0]
filas_limpias = len(df)
tasa_incumplimiento = float(df['loan_status'].mean())
print(f"[2/5] Limpieza: {filas_originales} -> {filas_limpias} filas "
      f"| tasa incumplimiento {tasa_incumplimiento:.3f}")

# ------------------------------------------------------------
# Codificacion de categoricas
# ------------------------------------------------------------
categoricas = ['person_gender', 'person_education', 'person_home_ownership',
               'loan_intent', 'previous_loan_defaults_on_file']
le = LabelEncoder()
for col in categoricas:
    df[col] = le.fit_transform(df[col])

X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
print(f"[3/5] Particion estratificada: train={len(X_train)} | test={len(X_test)}")

# ------------------------------------------------------------
# Entrenamiento de los 3 modelos
# ------------------------------------------------------------
modelos = {
    'Arbol de Decision': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Regresion Logistica': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1),
}


def downsample_roc(fpr, tpr, n=80):
    """Reduce la curva ROC a ~n puntos para un JSON liviano."""
    if len(fpr) <= n:
        return [[round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr, tpr)]
    idx = np.linspace(0, len(fpr) - 1, n).astype(int)
    return [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx]


resultados = {}
modelos_json = []
roc_json = {}

print("[4/5] Entrenando modelos...")
for nombre, modelo in modelos.items():
    t0 = time.time()
    modelo.fit(X_train, y_train)
    t_entreno = time.time() - t0

    t0 = time.time()
    pred = modelo.predict(X_test)
    t_infer = time.time() - t0
    proba = modelo.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)
    fila = {
        'nombre': nombre,
        'accuracy': round(float(accuracy_score(y_test, pred)), 4),
        'precision': round(float(precision_score(y_test, pred)), 4),
        'recall': round(float(recall_score(y_test, pred)), 4),
        'f1': round(float(f1_score(y_test, pred)), 4),
        'auc': round(float(auc), 4),
        'gini': round(float(2 * auc - 1), 4),
        't_entreno': round(float(t_entreno), 3),
        't_infer': round(float(t_infer), 4),
    }
    modelos_json.append(fila)
    resultados[nombre] = dict(modelo=modelo, pred=pred, proba=proba, auc=auc)

    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_json[nombre] = downsample_roc(fpr, tpr)
    print(f"      {nombre:22s} AUC={auc:.4f}  Recall={fila['recall']:.4f}")

# Mejor modelo por AUC
mejor_nombre = max(resultados, key=lambda k: resultados[k]['auc'])
mejor = resultados[mejor_nombre]

cm = confusion_matrix(y_test, mejor['pred'])
tn, fp, fn, tp = [int(v) for v in cm.ravel()]

importancias = []
if hasattr(mejor['modelo'], 'feature_importances_'):
    serie = pd.Series(mejor['modelo'].feature_importances_, index=X.columns)
    serie = serie.sort_values(ascending=False).head(8)
    importancias = [{'variable': k, 'valor': round(float(v), 4)}
                    for k, v in serie.items()]

# ------------------------------------------------------------
# Exportar metrics.json
# ------------------------------------------------------------
salida = {
    'generado_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
    'dataset': {
        'filas_originales': int(filas_originales),
        'filas_limpias': int(filas_limpias),
        'tasa_incumplimiento': round(tasa_incumplimiento, 4),
        'train': int(len(X_train)),
        'test': int(len(X_test)),
    },
    'modelos': modelos_json,
    'mejor': mejor_nombre,
    'matriz_confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
    'importancias': importancias,
    'roc': roc_json,
}

destino = os.path.join(SITE_DIR, 'metrics.json')
with open(destino, 'w', encoding='utf-8') as f:
    json.dump(salida, f, ensure_ascii=False, indent=2)

print(f"[5/5] Metricas exportadas -> {destino}")
print(f"      Mejor modelo: {mejor_nombre} "
      f"(AUC={mejor['auc']:.4f}, Gini={2*mejor['auc']-1:.4f})")
print("Listo. El dashboard web leera este archivo automaticamente.")
