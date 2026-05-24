FROM python:3.12-slim

WORKDIR /app

COPY 02_Ingesta_y_Pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 01_Datos_Sucios/ 01_Datos_Sucios/
COPY 02_Ingesta_y_Pipeline/ 02_Ingesta_y_Pipeline/
COPY 03_EDA_Resultados/ 03_EDA_Resultados/
COPY 04_Pagina_Web/ 04_Pagina_Web/

RUN mkdir -p 03_EDA_Resultados

CMD ["python", "02_Ingesta_y_Pipeline/analisis.py"]
