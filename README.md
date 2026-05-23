# Prediccion de Incumplimiento de Prestamos

Pipeline DataOps que procesa 45,000 solicitudes de prestamos, limpia los datos, entrena modelos de clasificacion y predice si un solicitante va a pagar o no.

Proyecto academico para la materia **Gestion de Datos para IA (ITY1101)**.

---

## Que hace este proyecto

1. **Lee** un archivo CSV con 45,000 solicitudes de prestamos
2. **Limpia** los datos: elimina edades imposibles, valores nulos y outliers
3. **Valida** que los datos cumplan reglas de negocio (estructura y semantica)
4. **Carga** los datos limpios en un CSV procesado y una base de datos SQLite
5. **Analiza** los datos con graficos y estadisticas
6. **Entrena** dos modelos de IA (Arbol de Decision y Regresion Logistica)
7. **Evalua** equidad por genero para verificar que el modelo no discrimina
8. **Genera** un reporte con KPIs de monitoreo del pipeline

Todo se ejecuta con **un solo comando** y tarda menos de 5 segundos.

---

## Estructura del proyecto

```
GestionDeDatos/
├── 01_Datos_Sucios/
│   └── loan_data.csv              # Dataset original (45,000 filas)
├── 02_Ingesta_y_Pipeline/
│   ├── analisis.py                # Script principal del pipeline
│   ├── test_analisis.py           # Tests automatizados
│   └── requirements.txt           # Dependencias de Python
├── 03_EDA_Resultados/             # Se genera al ejecutar el pipeline
│   ├── datos_limpios.csv          # Datos procesados
│   ├── prestamos.db               # Base de datos SQLite
│   ├── reporte.txt                # Reporte con resultados y KPIs
│   ├── log_ejecucion.log          # Registro detallado de ejecucion
│   ├── distribucion_target.png    # Grafico: pagados vs incumplidos
│   ├── distribucion_edad.png      # Grafico: edades de solicitantes
│   ├── tasa_interes_por_status.png # Grafico: tasa de interes por estado
│   ├── arbol_decision.png         # Grafico: visualizacion del modelo
│   ├── equidad_genero.png         # Grafico: accuracy por genero
│   └── correlacion.png            # Matriz de correlacion
├── 04_Pagina_Web/
│   └── dashboard.html             # Dashboard visual interactivo
├── 05_Manual/
│   └── MANUAL_PRESENTACION.txt    # Guia para presentacion y estudio
└── README.md
```

---

## Requisitos previos

### Python 3.10 o superior

Descargar desde [python.org/downloads](https://www.python.org/downloads/).

> **Importante:** durante la instalacion, marcar la casilla **"Add Python to PATH"**.

Para verificar que esta instalado, abrir una terminal y escribir:

```
python --version
```

Debe mostrar algo como `Python 3.12.x` o superior.

### Git (opcional, solo para clonar el repositorio)

Descargar desde [git-scm.com/downloads](https://git-scm.com/downloads). Instalar con opciones por defecto.

---

## Instalacion paso a paso

### 1. Descargar el proyecto

**Opcion A — Con Git:**
```bash
git clone https://github.com/TU_USUARIO/GestionDeDatos.git
cd GestionDeDatos
```

**Opcion B — Sin Git:**

Descargar el ZIP desde GitHub (boton verde "Code" > "Download ZIP"), descomprimir y abrir la carpeta en una terminal.

### 2. Crear un entorno virtual (recomendado)

Esto aisla las librerias del proyecto para que no interfieran con otras instalaciones de Python.

**Windows (PowerShell o CMD):**
```
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Si funciono, veras `(venv)` al inicio de la linea en la terminal.

### 3. Instalar dependencias

```
pip install -r 02_Ingesta_y_Pipeline/requirements.txt
```

Esto instala:

| Libreria | Para que se usa |
|---|---|
| pandas | Leer, limpiar y manipular datos (como un Excel potente) |
| scikit-learn | Entrenar modelos de inteligencia artificial |
| matplotlib | Generar graficos |
| seaborn | Graficos estadisticos avanzados |
| pytest | Ejecutar pruebas automatizadas |

---

## Como ejecutar

### Ejecutar el pipeline completo

```
python 02_Ingesta_y_Pipeline/analisis.py
```

El script procesa todo de principio a fin y muestra el progreso en la terminal. Al terminar, los resultados quedan en la carpeta `03_EDA_Resultados/`.

### Ejecutar los tests

```
pytest 02_Ingesta_y_Pipeline/test_analisis.py -v
```

Deben salir 5 tests en verde (PASSED). Verifican que los datos se cargaron bien, que no hay nulos, que las edades estan en rango, etc.

### Ver el dashboard

Abrir el archivo `04_Pagina_Web/dashboard.html` en cualquier navegador (Chrome, Firefox, Edge). Es una pagina local que no necesita servidor ni internet.

---

## Que hace cada etapa del pipeline

### Etapa 1 — Ingesta
Lee el archivo `01_Datos_Sucios/loan_data.csv` con pandas. Carga 45,000 filas y 14 columnas en memoria.

### Etapa 2 — Limpieza y Transformacion
- Elimina filas con valores nulos
- Elimina outliers: edades menores a 18 o mayores a 100, ingresos negativos, experiencia laboral negativa
- Convierte texto a numeros (ej: "female" → 0, "male" → 1) para que los modelos de IA puedan procesarlos
- Resultado: 44,993 filas limpias (7 descartadas)

### Etapa 3 — Validacion
Verifica que los datos cumplan reglas:
- **Estructural:** estan las 14 columnas, no hay nulos, cantidad correcta de filas
- **Semantica:** edades en rango 18-100, ingresos positivos, credit score valido, variable objetivo binaria (0/1)

### Etapa 4 — Carga
Guarda los datos limpios en dos formatos:
- `datos_limpios.csv` — archivo plano, se puede abrir en Excel
- `prestamos.db` — base de datos SQLite con tabla `prestamos`

### Analisis Exploratorio (EDA)
Genera graficos de distribucion de edad, incumplimiento y tasa de interes. Calcula estadisticas descriptivas.

### Modelos de Clasificacion
Entrena dos modelos y compara resultados:

| Modelo | Accuracy |
|---|---|
| Arbol de Decision | 91.28% |
| Regresion Logistica | 88.54% |

El Arbol de Decision gana y ademas es interpretable: se puede visualizar como toma cada decision.

### Analisis de Equidad
Mide accuracy por genero para verificar que el modelo no discrimina:
- Hombres: 91.54%
- Mujeres: 90.95%
- Diferencia: 0.59% (menor al umbral del 5%)

---

## Resultados generados

Despues de ejecutar el pipeline, la carpeta `03_EDA_Resultados/` contiene:

| Archivo | Descripcion |
|---|---|
| `datos_limpios.csv` | Dataset procesado y limpio (44,993 filas) |
| `prestamos.db` | Base de datos SQLite con los datos cargados |
| `reporte.txt` | Resumen con metricas de los modelos y KPIs |
| `log_ejecucion.log` | Registro paso a paso con timestamps |
| `distribucion_target.png` | Pagados (78%) vs incumplidos (22%) |
| `distribucion_edad.png` | Distribucion de edad de solicitantes |
| `tasa_interes_por_status.png` | Tasa de interes segun estado del prestamo |
| `arbol_decision.png` | Visualizacion del arbol de decision |
| `equidad_genero.png` | Comparacion de accuracy por genero |
| `correlacion.png` | Matriz de correlacion entre variables |

---

## Columnas del dataset

| Columna | Descripcion |
|---|---|
| `person_age` | Edad del solicitante |
| `person_gender` | Genero (male/female) |
| `person_education` | Nivel educativo |
| `person_income` | Ingreso anual en dolares |
| `person_emp_exp` | Anos de experiencia laboral |
| `person_home_ownership` | Tipo de vivienda (RENT, OWN, MORTGAGE, OTHER) |
| `loan_amnt` | Monto del prestamo solicitado |
| `loan_intent` | Proposito del prestamo |
| `loan_int_rate` | Tasa de interes (%) |
| `loan_percent_income` | Prestamo como porcentaje del ingreso |
| `cb_person_cred_hist_length` | Antiguedad del historial crediticio (anos) |
| `credit_score` | Puntaje crediticio (300-850) |
| `previous_loan_defaults_on_file` | Incumplimientos previos (Yes/No) |
| `loan_status` | **Variable objetivo** — 0: pago, 1: incumplio |

---

## Tecnologias utilizadas

- **Python 3** — Lenguaje de programacion
- **pandas** — Manipulacion de datos
- **scikit-learn** — Modelos de machine learning
- **matplotlib / seaborn** — Visualizacion
- **SQLite** — Base de datos relacional
- **pytest** — Testing automatizado
- **HTML/CSS/JS** — Dashboard visual

---

## Solucion de problemas

**"python no se reconoce como comando"**
Python no esta en el PATH. Reinstalar y marcar "Add Python to PATH".

**Error al instalar dependencias:**
```
python -m pip install --upgrade pip
python -m pip install -r 02_Ingesta_y_Pipeline/requirements.txt
```

**Los graficos no se generan:**
El script usa el backend `Agg` de matplotlib (no necesita pantalla). Si hay error, verificar que matplotlib esta instalado: `pip show matplotlib`.

**El dashboard no muestra graficos:**
Ejecutar primero el pipeline para que se generen los archivos `.png` en `03_EDA_Resultados/`.
