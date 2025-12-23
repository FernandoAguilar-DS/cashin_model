-- ====== PARAMS ======
DECLARE ACT_WINDOW_DAYS INT64 DEFAULT 30;
DECLARE TARGET_TX_COUNT INT64 DEFAULT 5;

DECLARE COHORT_START DATE DEFAULT DATE '2025-01-01';
DECLARE COHORT_END DATE DEFAULT DATE_SUB(CURRENT_DATE("America/Mexico_City"), INTERVAL 30 DAY);

DECLARE TX_SCAN_END  DATE DEFAULT DATE_ADD(CURRENT_DATE("America/Mexico_City"), INTERVAL 1 DAY);

DECLARE ACT_TYPES ARRAY<STRING> DEFAULT [
  'CARD_ATM_WITHDRAWAL','CARD_PURCHASE','CASH_IN_AT_OXXO','CASH_IN_VIA_CARD',
  'CASH_IN_VIA_CREDIT_CARD','CASH_OUT_AT_OXXO','CASH_OUT_REMITTANCE',
  'CASH_OUT_WITH_CARD_AT_OXXO','IN_APP_PURCHASE_BILLPAYMENT','IN_APP_PURCHASE_TAE',
  'P2P_TRANSFER_SOURCE','P2P_TRANSFER_SOURCE_CARD','P2P_TRANSFER_SOURCE_CLABE',
  'P2P_TRANSFER_TARGET','P2P_TRANSFER_TARGET_CARD','P2P_TRANSFER_TARGET_CLABE',
  'SPEI_CASH_IN','TRANSFER_TO_CARD','TRANSFER_TO_CLABE',
  'CASH_OUT_AT_MERCHANT','INTERNATIONAL_REMITTANCE_CASH_IN','GIFT_CARD_PURCHASE',
  'PUBLIC_TRANSPORT_CHARGE','QR_MERCHANT_PAYMENT','CASH_IN_AT_OXXO_QR'
];

-- ====== CTES ======
WITH
-- CTE de Premia (de Q2)
premia_map AS (
  SELECT userid AS user_id_sbo, accountid AS premia_accountid
  FROM `daf-dp-trusted-prod.coa_mastertables.tbl_users`
),

-- USERS + CONFIRMACIONES (Modificado con campos de Q2)
users AS (
  SELECT
    a.userIdentifier AS user_id,
    a.createdAccountDate AS signup_ts,                               -- TIMESTAMP
    DATE(a.createdAccountDate, "America/Mexico_City") AS signup_date,
    a.userTypeIdentifier,
    a.channelUserIdentifier,
    a.accountLevel,
    a.stateName,
    a.Card_linked_date,
    a.IsActive,
    -- Normaliza confirmaciones (soporta DATE/TIMESTAMP)
    TIMESTAMP(CAST(a.phone_confirmation_date AS DATETIME), "America/Mexico_City")  AS phone_conf_ts_raw,
    TIMESTAMP(CAST(a.email_confirmation_date AS DATETIME), "America/Mexico_City")  AS email_conf_ts_raw,

    -- Nuevas columnas de Q2
    g.genderType AS gender,
    u.userTypeDetail AS user_type,
    c.channelDetail,
    DATE(a.birthDate, "America/Mexico_City") AS birth_date,
    a.birthState

  FROM `spin-dp-trusted-prod.spin_account.tbl_dim_user` a
  -- Nuevos JOINS de Q2
  LEFT JOIN `spin-dp-trusted-prod.spin_catalogs.tbl_dim_gender` g
    ON a.genderIdentifier = g.genderIdentifier
  LEFT JOIN `spin-dp-trusted-prod.spin_catalogs.tbl_dim_user_channel` c
    ON a.channelUserIdentifier = c.channelUserIdentifier
  LEFT JOIN `spin-dp-trusted-prod.spin_catalogs.tbl_dim_user_type` u
    ON a.userTypeIdentifier = u.userTypeIdentifier
  WHERE DATE(a.createdAccountDate, "America/Mexico_City") >= COHORT_START
    AND DATE(a.createdAccountDate, "America/Mexico_City") <  COHORT_END
),
conf AS (
  SELECT
    user_id, signup_date, signup_ts, userTypeIdentifier, channelUserIdentifier,
    accountLevel, stateName, Card_linked_date, IsActive,
    -- Nuevas columnas de Q2 (passthrough)
    gender, user_type, channelDetail, birth_date, birthState,

    -- Lógica de Q1 (más robusta)
    IF(phone_conf_ts_raw < signup_ts, NULL, phone_conf_ts_raw) AS phone_conf_ts,
    IF(email_conf_ts_raw < signup_ts, NULL, email_conf_ts_raw) AS email_conf_ts,
    CAST(phone_conf_ts_raw IS NOT NULL AS INT64) AS phn_confir,
    CAST(email_conf_ts_raw IS NOT NULL AS INT64) AS email_confir,
    CAST(phone_conf_ts_raw IS NOT NULL AND phone_conf_ts_raw < TIMESTAMP_ADD(signup_ts, INTERVAL 7 DAY) AS INT64) AS phn_confir_d7,
    CAST(email_conf_ts_raw IS NOT NULL AND email_conf_ts_raw < TIMESTAMP_ADD(signup_ts, INTERVAL 7 DAY) AS INT64) AS email_confir_d7
  FROM users
),

-- ====== TRANSACCIONES FILTRADAS ======
tx_raw AS (
  SELECT
    t.userIdentifier AS user_id,
    DATE(t.transactionDate, "America/Mexico_City") AS tx_date,   -- particionado
    tt.transactionType AS tx_type,
    t.transactionIdentifier AS tx_id,
    SAFE_DIVIDE(t.transactionAmount, 100) AS tx_amount
  FROM `spin-dp-trusted-prod.spin_transaction.tbl_fact_transaction` t
  JOIN conf u ON u.user_id = t.userIdentifier
  LEFT JOIN `spin-dp-trusted-prod.spin_catalogs.tbl_dim_transaction_type` tt
    ON t.transactionTypeIdentifier = tt.transactionTypeIdentifier
  WHERE DATE(t.transactionDate, "America/Mexico_City") >= COHORT_START
    AND DATE(t.transactionDate, "America/Mexico_City") <  TX_SCAN_END
    AND tt.transactionType IN UNNEST(ACT_TYPES)
    AND COALESCE(t.isReversedFlag, FALSE) = FALSE
),

-- Dedup por transacción
tx_dedup AS (
  SELECT
    tx_id,
    ANY_VALUE(user_id)  AS user_id,
    ANY_VALUE(tx_date)  AS tx_date,
    ANY_VALUE(tx_type)  AS tx_type,
    ANY_VALUE(tx_amount) AS tx_amount
  FROM tx_raw
  GROUP BY tx_id
),

-- Agregado por día/tipo
tx_daily AS (
  SELECT
    user_id, tx_date, tx_type,
    COUNT(*)            AS tx_count,
    SUM(tx_amount)      AS tx_amount
  FROM tx_dedup
  GROUP BY 1,2,3
),

-- Primer y última tx EVER (Modificado con lifespan_days de Q2)
tx_span AS (
  SELECT
    user_id,
    MIN(tx_date) AS first_tx_date,
    MAX(tx_date) AS latest_tx_date,
    -- Nueva columna de Q2
    DATE_DIFF(MAX(tx_date), MIN(tx_date), DAY) + 1 AS lifespan_days
  FROM tx_dedup
  GROUP BY user_id
),

-- Detalle de la primera tx (tipo/importe en ese día)
first_tx_details AS (
  SELECT
    s.user_id,
    s.first_tx_date,
    ANY_VALUE(d.tx_type) AS first_tx_type,
    SUM(d.tx_amount)     AS first_tx_amount
  FROM tx_span s
  JOIN tx_daily d
    ON d.user_id = s.user_id AND d.tx_date = s.first_tx_date
  GROUP BY s.user_id, s.first_tx_date
),

-- Activación dentro de [signup, signup+30)
activation_30d AS (
  SELECT
    u.user_id,
    MIN(d.tx_date) AS first_tx_in_30d
  FROM conf u
  LEFT JOIN tx_daily d
    ON d.user_id = u.user_id
   AND d.tx_date >= u.signup_date
   AND d.tx_date <  DATE_ADD(u.signup_date, INTERVAL ACT_WINDOW_DAYS DAY)
  GROUP BY u.user_id
),

-- Suma de transacciones en [signup, signup+30)
tx_in_signup_win AS (
  SELECT
    u.user_id,
    SUM(d.tx_count)  AS tx_30d_count,
    SUM(d.tx_amount) AS tx_30d_amount
  FROM conf u
  LEFT JOIN tx_daily d
    ON d.user_id = u.user_id
   AND d.tx_date >= u.signup_date
   AND d.tx_date <  DATE_ADD(u.signup_date, INTERVAL ACT_WINDOW_DAYS DAY)
  GROUP BY u.user_id
),

-- Suma de transacciones en [first_tx, first_tx+30)
tx_from_activation AS (
  SELECT
    u.user_id,
    SUM(d.tx_count) AS tx_30d_from_activation
  FROM conf u
  JOIN tx_span s   ON s.user_id = u.user_id
  LEFT JOIN tx_daily d
    ON d.user_id = u.user_id
   AND d.tx_date >= s.first_tx_date
   AND d.tx_date <  DATE_ADD(s.first_tx_date, INTERVAL ACT_WINDOW_DAYS DAY)
  GROUP BY u.user_id
)

SELECT
  c.user_id,
  c.signup_date,
  
  -- Esta es la columna que incluye la fecha y hora del registro
  c.signup_ts,
  
  -- Columnas demográficas (de c, expandidas por Q2)
  c.userTypeIdentifier, c.channelUserIdentifier, c.accountLevel, c.stateName,
  c.gender, c.user_type, c.channelDetail, c.birth_date, c.birthState, -- <-- NUEVAS
  c.Card_linked_date, c.IsActive,

  -- Confirmaciones limpias + flags D7 (de c)
  c.phn_confir, c.email_confir,
  c.phone_conf_ts, c.email_conf_ts,
  c.phn_confir_d7, c.email_confir_d7,
  CAST(c.phn_confir_d7 = 1 AND c.email_confir_d7 = 1 AS INT64) AS both_confir_d7,

  -- Premia (de pm) 
  pm.premia_accountid,
  IF(pm.premia_accountid IS NOT NULL, 1, 0) AS has_premia,

  -- Activación / labels (de s, a, win)
  s.first_tx_date        AS activation_date_ever,
  a.first_tx_in_30d      AS activation_date_30d,
  IF(a.first_tx_in_30d IS NULL, 0, 1) AS label_activated_30d,

  win.tx_30d_count,
  win.tx_30d_amount,
  IF(COALESCE(win.tx_30d_count,0) >= TARGET_TX_COUNT, 1, 0) AS label_5tx_30d,

  -- Detalles primera tx (de fa)
  fa.first_tx_type,
  fa.first_tx_amount,
  CASE
    WHEN fa.first_tx_type LIKE '%P2P%' THEN 'P2P'
    WHEN fa.first_tx_type LIKE '%QR%' THEN 'QR'
    WHEN fa.first_tx_type LIKE '%SPEI%' OR fa.first_tx_type LIKE '%TRANSFER%' THEN 'SPEI/Transfer'
    WHEN fa.first_tx_type LIKE '%CASH_IN_AT_OXXO%' THEN 'CashIn_OXXO'
    WHEN fa.first_tx_type LIKE '%CARD%' THEN 'Card'
    ELSE 'Other'
  END AS activation_channel,

  -- Métricas de actividad (de s, txfa, y calculadas)
  s.latest_tx_date,
  s.lifespan_days, 
  DATE_DIFF(CURRENT_DATE("America/Mexico_City"), s.latest_tx_date, DAY) AS days_since_last,
  txfa.tx_30d_from_activation,
  DATE_DIFF(s.first_tx_date, c.signup_date, DAY) AS days_to_first_activation 

FROM conf c
LEFT JOIN tx_span              s    ON s.user_id = c.user_id
LEFT JOIN activation_30d       a    ON a.user_id = c.user_id
LEFT JOIN tx_in_signup_win     win  ON win.user_id = c.user_id
LEFT JOIN first_tx_details     fa   ON fa.user_id = c.user_id
LEFT JOIN tx_from_activation   txfa ON txfa.user_id = c.user_id
LEFT JOIN premia_map           pm   ON c.user_id = pm.user_id_sbo 
ORDER BY c.user_id;