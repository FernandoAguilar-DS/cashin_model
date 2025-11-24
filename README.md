# Model 1.2.0 – Activación 30d + Tipo de Transacción

    Este folder contiene los artefactos de entrenamiento del modelo de Activación 30d
    y el modelo multiclase de tipo de primera transacción.

    Archivos clave:
    - config.json                         → Configuración de entrenamiento (tabla origen, columnas, seeds, etc.)
    - train_val_test_info.json            → Tamaños, fechas y tasas de activación por split.
    - feature_builder.pkl                 → Objeto FeatureBuilder (feature engineering + scaler).
    - feature_builder_meta.json           → Listas de columnas numéricas y OHE, tipo de scaler.
    - model_activation_30d.pkl            → Modelo W30 entrenado (folds + calibradores).
    - model_activation_30d_summary.json   → Métricas OOF/Holdout + features usadas.
    - model_tx_type.pkl                   → Modelo multiclase Cash_In / SPEI / P2P.
    - model_tx_type_summary.json          → Hiperparámetros y mapping de clases.
    - preds_w30_tx.csv                    → Predicciones por usuario (scores + recomendación de canal).
    
    ARTIFACT_VERSION = "1.2.0"
    ARTIFACTS_DIR = "artifacts"
