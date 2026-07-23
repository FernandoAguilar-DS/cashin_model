# Modelo Activación 30d + Tipo de Primera Transacción — v1.1.0

Artefactos: **`artifacts_MLOps_v1.1.0/`**
Notebook: `jose_arredondo_HistGB_Horizont_30D_First_Tx_Type_1.1.5_MLOps.ipynb`
Job equivalente: `batch_inference.py`

Esta versión incluye dos cambios independientes:

1. **Remapeo de los eventos transaccionales** que definen las clases del modelo multiclase (afecta el significado del modelo).
2. **Reescritura de la sección `# Simple execution`** para inferencia batch reproducible (afecta cómo se ejecuta, no el modelo).

Más el **reentrenamiento** del notebook con los artefactos publicados en `artifacts_MLOps_v1.1.0/`.

---

## 1. Remapeo de eventos transaccionales

### 1.1 Antes (v1.1.4)

```python
CASH_IN_TX = {
    "CASH_IN_AT_OXXO","CASH_IN_AT_OXXO_QR","CASH_OUT_WITH_CARD_AT_OXXO",
    "CASH_OUT_AT_OXXO","CASH_OUT_AT_MERCHANT","CARD_PURCHASE","CARD_ATM_WITHDRAWAL"
}
SPEI_TX = {"SPEI_CASH_IN","TRANSFER_TO_CARD","TRANSFER_TO_CLABE" "P2P_TRANSFER_TARGET",
           "P2P_TRANSFER_TARGET_CLABE","P2P_TRANSFER_TARGET","P2P_TRANSFER_TARGET_CARD"}

P2P_TX = {"P2P_TRANSFER_TARGET_CLABE","P2P_TRANSFER_TARGET","P2P_TRANSFER_TARGET_CARD",
          "P2P_TRANSFER_SOURCE_CARD","P2P_TRANSFER_SOURCE_CLABE","P2P_TRANSFER_SOURCE",
          "IN_APP_PURCHASE_TAE","IN_APP_PURCHASE_BILLPAYMENT","QR_MERCHANT_PAYMENT",
          "GIFT_CARD_PURCHASE","INTERNATIONAL_REMITTANCE_CASH_IN"}
```

### 1.2 Ahora (v1.1.0)

```python
CASH_IN_TX = {"CASH_IN_AT_OXXO","CASH_IN_AT_OXXO_QR"}
SPEI_TX    = {"SPEI_CASH_IN"}
P2P_TX     = {"P2P_TRANSFER_TARGET","P2P_TRANSFER_TARGET_CLABE",
              "P2P_TRANSFER_TARGET","P2P_TRANSFER_TARGET_CARD"}
```

El criterio pasó de una agrupación amplia por canal (que mezclaba entradas, salidas y compras) a una definición estricta de **método de fondeo de la primera transacción**: efectivo en OXXO, transferencia SPEI recibida, o transferencia P2P recibida.

`TX_TYPE_NAMES` **no cambia**: `0 = Cash_In`, `1 = SPEI`, `2 = P2P`.

### 1.3 Defectos del mapeo anterior que este cambio corrige

Los tres se verificaron ejecutando el código anterior:

**a) Coma faltante → token fantasma.** En `SPEI_TX`, `"TRANSFER_TO_CLABE" "P2P_TRANSFER_TARGET"` (sin coma) activaba la concatenación implícita de literales de Python y producía la cadena única `"TRANSFER_TO_CLABEP2P_TRANSFER_TARGET"`, que no corresponde a ningún evento real. Consecuencia: **`TRANSFER_TO_CLABE` nunca estuvo mapeado** y el set tenía 6 elementos en vez de los 7 aparentes.

**b) Solape entre clases con precedencia silenciosa.** Tres eventos aparecían en `SPEI_TX` **y** en `P2P_TX`:

`P2P_TRANSFER_TARGET`, `P2P_TRANSFER_TARGET_CLABE`, `P2P_TRANSFER_TARGET_CARD`

Como `TX_TYPE_MAP` se construye con `{**cash_in, **spei, **p2p}`, ganaba el último diccionario: quedaban etiquetados como **P2P (clase 2)**, no como SPEI. La clase SPEI efectiva del modelo anterior era, en la práctica, solo `SPEI_CASH_IN` + `TRANSFER_TO_CARD`. En el mapeo nuevo **no hay solapes**.

**c) Clase 0 semánticamente mezclada.** `CASH_IN_TX` incluía salidas de efectivo y compras (`CASH_OUT_AT_OXXO`, `CASH_OUT_AT_MERCHANT`, `CARD_PURCHASE`, `CARD_ATM_WITHDRAWAL`, `CASH_OUT_WITH_CARD_AT_OXXO`), que no son entradas de dinero. Bajo la etiqueta "Cash_In" se estaba modelando en realidad "actividad en canal físico" — consistente con `TX_TRINARY_NAME = {0:"FISICAS", 1:"DIGITALES", 2:"P2P"}`, pero no con el nombre de la clase ni con el uso de negocio.

### 1.4 Diferencia exacta entre ambos mapeos

| | Antes | Ahora |
|---|---|---|
| Eventos mapeados | 21 | 6 |
| Solapes entre clases | 3 (SPEI ∩ P2P) | 0 |
| Tokens inválidos | 1 | 0 |

**Ningún evento cambia de clase.** Los 6 que sobreviven conservan su etiqueta:

| Evento | Clase |
|---|---|
| `CASH_IN_AT_OXXO` | 0 — Cash_In |
| `CASH_IN_AT_OXXO_QR` | 0 — Cash_In |
| `SPEI_CASH_IN` | 1 — SPEI |
| `P2P_TRANSFER_TARGET` | 2 — P2P |
| `P2P_TRANSFER_TARGET_CLABE` | 2 — P2P |
| `P2P_TRANSFER_TARGET_CARD` | 2 — P2P |

**15 eventos salen del mapeo** (pasan a `NaN` y por lo tanto quedan fuera del entrenamiento y de la evaluación del modelo multiclase):

| Evento | Clase anterior |
|---|---|
| `CASH_OUT_AT_OXXO` | 0 |
| `CASH_OUT_AT_MERCHANT` | 0 |
| `CASH_OUT_WITH_CARD_AT_OXXO` | 0 |
| `CARD_PURCHASE` | 0 |
| `CARD_ATM_WITHDRAWAL` | 0 |
| `TRANSFER_TO_CARD` | 1 |
| `TRANSFER_TO_CLABEP2P_TRANSFER_TARGET` (token inválido) | 1 |
| `P2P_TRANSFER_SOURCE` | 2 |
| `P2P_TRANSFER_SOURCE_CARD` | 2 |
| `P2P_TRANSFER_SOURCE_CLABE` | 2 |
| `IN_APP_PURCHASE_TAE` | 2 |
| `IN_APP_PURCHASE_BILLPAYMENT` | 2 |
| `QR_MERCHANT_PAYMENT` | 2 |
| `GIFT_CARD_PURCHASE` | 2 |
| `INTERNATIONAL_REMITTANCE_CASH_IN` | 2 |

### 1.5 Consecuencias operativas

- **Población de entrenamiento menor.** En la celda de entrenamiento, `y_tx_coded = y_tx_raw.map(TX_TYPE_MAP)` y `mask_train_tx = y_tx_coded.notna()`: los usuarios activados cuya primera transacción sea alguno de los 15 eventos removidos ya no entran al modelo multiclase. El tamaño real de la caída depende de la mezcla de la ventana de datos — **está por medir** (ver §5).
- **La recomendación de canal es condicional.** En scoring el modelo asigna una de las 3 clases a **todos** los usuarios, incluidos aquellos cuya primera transacción real sería un evento no mapeado. La lectura correcta es: *"dado que la primera transacción sea uno de estos tres métodos de fondeo, cuál es el más probable"*, no *"qué hará el usuario"*.
- **Los scores no son comparables entre versiones.** `score_cash_in`, `score_spei` y `score_p2p` conservan el nombre pero cambian de significado. No compares distribuciones contra `preds_w30_tx.csv` de v1.1.4 ni interpretes un PSI alto contra esa referencia como *drift* de datos: es un cambio de definición. La línea base de monitoreo debe reiniciarse con los artefactos de v1.1.0.
- **El modelo de activación W30 no se ve afectado por este cambio.** Su etiqueta es `label_activated_30d`, independiente de `TX_TYPE_MAP`. Si sus métricas cambiaron, es por el reentrenamiento (ventana de datos), no por el remapeo.

---

## 2. Sección `# Simple execution` → ejecución MLOps

La sección se conservó: `read_data()`, `load_artifacts()`, `_sanitize_tx_class_name()`, `_get_binary_prob()` y `score_batch()` mantienen su contrato. `score_batch()` solo suma tres parámetros opcionales que, en `None`, reproducen el comportamiento original.

**El bloqueo principal.** La sección no podía ejecutarse con el kernel reiniciado. `joblib`/`pickle` no serializa el código de las clases, solo la referencia `módulo.NombreClase`; los `.pkl` se generaron dentro del notebook, donde `FeatureBuilder`, `Activation30Model` y `TxTypeModel` viven en `__main__`. En un kernel limpio:

```
AttributeError: Can't get attribute 'FeatureBuilder' on <module '__main__'>
```

Además `FeatureBuilder.transform()` lee **globales** del módulo (`CFG`, `GENDER_MAP`, `LEAKY_ALWAYS`, `assert_no_regex_leak`, …), no solo `self`.

**Solución.** La celda de bootstrap extrae del notebook las secciones `# Config`, `# FeatureBuilder`, `# Modelo W30` y `# Model type of transaction`, las materializa como `serving_lib.py` dentro de la carpeta de artefactos, reconstruye `CFG` desde `config.json` y registra las clases en `__main__` antes de deserializar. A partir de la primera ejecución, la carpeta de artefactos es autosuficiente y el notebook deja de ser necesario.

| # | Hallazgo | Cambio |
|---|---|---|
| 1 | `score_batch()` usa `np` pero la celda de imports no importaba numpy | `import numpy as np` |
| 2 | Los `.pkl` no cargan en kernel limpio | bootstrap + `serving_lib.py` |
| 3 | `transform()` indexa `df[CFG.label_col]`; en inferencia esa columna no existe → `KeyError` | `prepare_raw_for_inference()` inyecta etiqueta dummy (no se usa para puntuar) |
| 4 | `X` se pasaba al modelo sin alinear a `feature_names_` | `align_features()` reindexa al orden exacto de entrenamiento |
| 5 | El nombrado `p_tx_*` recorría `classes_`, pero `TxTypeModel.predict_proba` **siempre** devuelve columnas en orden `[0,1,2]` | mapping por posición desde `model_tx_type_summary.json` |
| 6 | Salida con vocabulario distinto al de entrenamiento | esquema canónico + columnas originales |
| 7 | Sin trazabilidad del run | `run_metadata.json` por corrida |
| 8 | Todo dependía de BigQuery | `read_batch()` acepta `csv` / `parquet` / `pickle` / `dataframe` |

---

## 3. Contenido de `artifacts_MLOps_v1.1.0/`

| Archivo | Descripción |
|---|---|
| `config.json` | Configuración de entrenamiento (tabla origen, columnas, seeds) |
| `train_val_test_info.json` | Tamaños, fechas y tasas de activación por split |
| `feature_builder.pkl` | Objeto `FeatureBuilder` (feature engineering + `RobustScaler`) |
| `feature_builder_meta.json` | Columnas numéricas, OHE de estados, tipo de scaler |
| `model_activation_30d.pkl` | Modelo W30 (folds + calibradores isotónicos) |
| `model_activation_30d_summary.json` | Métricas OOF/holdout + `feature_names` |
| `model_tx_type.pkl` | Modelo multiclase Cash_In / SPEI / P2P |
| `model_tx_type_summary.json` | Hiperparámetros y mapping de clases |
| `preds_w30_tx.csv` | Predicciones de entrenamiento (referencia de monitoreo) |
| `serving_lib.py` | Contrato de código para deserializar. Se genera solo la primera vez; **súbelo al repo junto con los `.pkl`** |

---

## 4. Cómo ejecutar

Apunta `ARTIFACTS_DIR` a la carpeta nueva:

```python
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "artifacts_MLOps_v1.1.0"))
```

**Notebook:** reinicia el kernel, ve directo a `# Simple execution` y ejecuta sus celdas. No hace falta correr nada anterior.

**Job:**

```bash
export ARTIFACTS_DIR=artifacts_MLOps_v1.1.0
python batch_inference.py                                  # tabla completa
python batch_inference.py --limit 1000                     # prueba rápida
python batch_inference.py --source parquet --path batch.parquet
```

Salidas: `outputs/run_id=<RUN_ID>/preds_w30_tx_batch.csv` y `run_metadata.json`.

---

## 5. Métricas del reentrenamiento — por completar

No tengo acceso a `artifacts_MLOps_v1.1.0/`, así que **estas cifras no están verificadas**. Complétalas desde `model_activation_30d_summary.json` y `train_val_test_info.json` antes de publicar el README.

| Concepto | v1.1.4 | v1.1.0 |
|---|---|---|
| Ventana de datos (min → max `signup_date`) | por completar | por completar |
| Filas train / holdout | por completar | por completar |
| W30 — AP OOF / holdout | 0.7697 / — | por completar |
| W30 — AUC OOF / holdout | 0.6602 / — | por completar |
| W30 — Brier OOF / holdout | 0.2800 / — | por completar |
| TxType — usuarios activados con clase válida | por completar | por completar |
| TxType — distribución de clases | por completar | por completar |
| TxType — accuracy holdout | por completar | por completar |

> Las cifras de la columna v1.1.4 provienen de los outputs guardados en el notebook 1.1.4 (bloque "MÉTRICAS OOF (Train)"); confírmalas contra el summary antes de citarlas fuera del equipo.

---

## 6. Notas de riesgo

- **Versionado inconsistente.** El notebook 1.1.4 escribe `ARTIFACT_VERSION = "1.1.4"` dentro del README que genera, el README previo del folder decía `"1.2.0"` y ahora la carpeta es `v1.1.0`. Unifica los tres (nombre de carpeta, valor en `save_training_artifacts()` y README) antes de que alguien tenga que reconstruir qué modelo generó un score.
- **Versiones de librerías.** Deserializar objetos de scikit-learn entre versiones distintas puede fallar o cambiar resultados sin aviso. Fija en el entorno de scoring las mismas versiones de `scikit-learn`, `numpy`, `pandas` y `joblib` usadas en el entrenamiento.
- **`serving_lib.py` se genera por extracción del notebook.** Funciona, pero es frágil ante ediciones de las celdas de clases. La solución de fondo es invertir la dependencia: un módulo compartido que importen tanto el entrenamiento como el serving, y picklear desde ahí.
- **`write_predictions_to_bq()` no está probada.** Valida permisos y la versión de `google-cloud-bigquery` antes de usarla en producción.

---

## 7. Changelog

**v1.1.0**
- Remapeo de eventos transaccionales: de 21 a 6 eventos, criterio de método de fondeo. Se eliminan el token inválido por coma faltante y el solape SPEI/P2P.
- Sección `# Simple execution` reescrita para inferencia batch ejecutable solo con artefactos.
- Nuevo artefacto `serving_lib.py`; nuevas salidas `run_metadata.json` y CSV por `run_id`.
- Reentrenamiento completo del notebook.

**v1.1.4 (anterior)**
- Modelo W30 (HistGB + calibración isotónica por folds) y modelo multiclase de tipo de primera transacción.
