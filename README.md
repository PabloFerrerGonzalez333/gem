# PFG-TestDataScience-1 [![ci](https://github.com/PabloFerrerGonzalez333/PFG-TestDataScience-1/actions/workflows/ci.yml/badge.svg)](https://github.com/PabloFerrerGonzalez333/PFG-TestDataScience-1/actions/workflows/ci.yml)

Pipeline de **ciencia de datos end-to-end** para el dataset de Telco (descarga → preparación de features → entrenamiento → evaluación → inferencia → documentación → API).

> **Tecnologías:** Python · Poetry · Uvicorn/FastAPI (serving) · Sphinx (docs)

---

## 📦 Requisitos

- **Python 3.11** (recomendado)
- **Poetry** para la gestión del entorno y dependencias

---

## 🚀 Instalación

Clona el repo y resuelve dependencias:

```bash
pip install poetry
```

```bash
poetry install
```

> Esto creará un entorno virtual aislado y descargará las dependencias definidas en `pyproject.toml`.

---

## 🧭 Estructura del proyecto

```
.
├── README.md                # Descripción del proyecto
├── pyproject.toml           # Configuración de dependencias (Poetry)
├── notebooks/               # Jupyter notebooks (EDA, features, modelos, tracking)
│   ├── 01_eda.ipynb         # Análisis exploratorio de datos (Exploratory Data Analysis)
│   ├── 02_features.ipynb    # Creación y transformación de features.
│   ├── 03_models.ipynb      # Entrenamiento y validación de modelos.
│   └── 04_mlflow.ipynb      # Registro y seguimiento de experimentos con MLflow.
├── reports/                 # Resultados renderizados en HTML y figuras
├── docs/                    # Documentación generada con Sphinx
├── src/                     # Código fuente principal
│   ├── data/
│   │   └── make_dataset.py        # Descarga y limpieza de datos
│   ├── features/
│   │   ├── build_features.py      # Construcción y preparación de variables
│   │   └── feature_selection.py   # Métodos de selección de *features*
│   ├── models/
│   │   ├── models.py              # Definición de modelos y grids de hiperparámetros
│   │   ├── train.py               # Entrenamiento, validación anidada y guardado de modelos
│   │   └── predict.py             # Predicciones por lotes con modelos entrenados
│   └── serving/
│       └── app.py                 # API de predicción (FastAPI)
```

### Descripción de módulos principales

- **`src/data/make_dataset.py`**: descarga, limpia y transforma los datos originales para dejarlos listos para modelado.  
- **`src/features/build_features.py`**: construcción de *features*, codificación, escalado y partición de datos.  
- **`src/features/feature_selection.py`**: utilidades de selección de variables (varianza, correlación, colinealidad, RFECV).  
- **`src/models/models.py`**: define los clasificadores disponibles y sus grids de hiperparámetros.  
- **`src/models/train.py`**: entrena los modelos con validación cruzada anidada, selecciona umbral óptimo y guarda artefactos.  
- **`src/models/predict.py`**: permite generar predicciones en lote a partir de un modelo guardado.  
- **`src/serving/app.py`**: API con FastAPI para exponer los modelos en producción.  

---

## ⚡️ Guía rápida (TL;DR)

1. **Instala dependencias**

```bash
poetry install
```

2. **Genera la documentación**

- **Windows:**
  ```bash
  .\docs\make.bat html
  ```
- **Linux/macOS:**
  ```bash
  make -C docs html
  ```

6. **Sirve la API**

```bash
poetry run uvicorn src.serving.app:app --host 127.0.0.1 --port 8000
```

- Documentación: `GET http://localhost:8000/documentation`

3. **Descarga datos 'Telco Churn'**

```bash
poetry run python src/data/make_dataset.py --out data/raw/telco.csv
```

4. **Preparación de los datos**

```bash
poetry run python src/features/build_features.py --in data/preprocessed/telco_preprocessed.xlsx --out data/processed --kind cc
```

5. **Entrena, evalúa y guarda modelo**

```bash
poetry run python src/models/train.py --data data/processed --models models
```

> El script tarda entorno a una hora, guarda el modelo entrenado y un artefacto con las métricas.


7. **Ejemplo de predicción**

```bash
poetry run uvicorn src.serving.app:app --host 127.0.0.1 --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data @sample.json
```

## 📊 MLflow

```bash
mlflow ui
```

> Posteriormente, ejecutar el notebook 04_mlflow.ipynb.
