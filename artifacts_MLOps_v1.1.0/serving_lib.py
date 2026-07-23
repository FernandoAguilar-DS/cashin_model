# ---------------------------------------------------------------
# serving_lib.py - AUTOGENERADO desde el notebook de entrenamiento.
# NO EDITAR A MANO. Contiene el contrato de codigo necesario para
# deserializar los artefactos (Config, FeatureBuilder,
# Activation30Model, TxTypeModel) y las globales que usan sus metodos.
# ---------------------------------------------------------------
from __future__ import annotations

import json
import re
import typing as t
import unicodedata
from dataclasses import dataclass, fields as _dc_fields
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import holidays
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.utils import Bunch
from sklearn.utils.class_weight import compute_sample_weight

RNG_SEED = 42

# ============ CODIGO EXTRAIDO DEL NOTEBOOK (verbatim) ============
# --- [# Config] celda 4 ---
@dataclass
class Config:
    project_id: str = "spin-aip-singularity-comp-sb"
    table_fqn: str = "spin-aip-singularity-comp-sb.model_activation.dataste_model_activation_timewindow_30D_V-1-5-0"
    label_col: str = "label_activated_30d"
    signup_ts_col: str = "signup_ts"
    signup_date_col: str = "signup_date"
    tz_local: str = "America/Mexico_City"
    embargo_days: int = 3      
    holdout_days: int = 14 
    n_splits: int = 5
    train_sample_frac: float = 1.0
    random_state: int = RNG_SEED
    lift_fracs: t.Tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)

CFG = Config()

# Leakage / IDs a ignorar en features
LEAKY_ALWAYS = {
    # labels y derivados
    "y_w0","y_w1","y_w7","y_w30","y_cum30","label_activated_30d","label_5tx_30d",
    # info post-activación / post-window
    "activation_date_ever","activation_date_30d","days_to_first_activation",
    "tx_30d_count","tx_30d_amount","tx_30d_from_activation", "first_tx_type","first_tx_amount","latest_tx_date",
    # confirmaciones si no son estrictamente previas al cutoff de cada horizonte
#    "phn_confir","email_confir",
    "phn_confir_d7","email_confir_d7","both_confir_d7",'Card_linked_date',"activation_*","*_30d_*","*latest_tx*",
    #ID
    "user_id", "userid","channelUserIdentifier", "premia_accountid", "accountid", "member_id", "spin_user_id", "id"
}

LEAK_BAN = {
    "y_w0","y_w1","y_w7","y_w30","y_cum30","label_activated_30d","label_5tx_30d",
    "activation_date_ever","activation_date_30d","days_to_first_activation",
    "latest_tx_date","tx_30d_count","tx_30d_amount","tx_30d_from_activation",
    "first_tx_type","first_tx_amount","activation_channel"
}

# Regex anti-leak (además del set LEAKY_ALWAYS existente)
LEAK_PATTERNS = [
    r"(^|_)activation(_|$)", r"(^|_)first_tx(_|$)", r"(^|_)latest_tx(_|$)",
    r"(^|_)tx_30d(_|$)", r"(^|_)days_to_first(_|$)", r"(^|_)from_activation(_|$)"
]
def assert_no_regex_leak(df_like: pd.DataFrame):
    bad = []
    for c in df_like.columns:
        for pat in LEAK_PATTERNS:
            if re.search(pat, c, flags=re.IGNORECASE):
                bad.append(c); break
    assert len(bad) == 0, f"LEAKAGE by regex: quita columnas {sorted(set(bad))}"
def assert_no_labelish_cols(df_like):
    inter = [c for c in df_like.columns if c in LEAK_BAN]
    assert len(inter) == 0, f"LEAKAGE: quita columnas {inter}"


EXTRA_DROP_TS = {"phone_conf_ts", "email_conf_ts"}

# Normalización de estados (stateName -> siglas)
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Z ]", "", s.upper())
    return re.sub(r"\s+", " ", s).strip()

STATE_TO_ABBR = {
    "AGUASCALIENTES":"AG","BAJA CALIFORNIA":"BC","BAJA CALIFORNIA SUR":"BS","CAMPECHE":"CM",
    "CHIAPAS":"CS","CHIHUAHUA":"CH","CIUDAD DE MEXICO":"DF","COAHUILA":"CO","COLIMA":"CL",
    "DURANGO":"DG","GUANAJUATO":"GT","GUERRERO":"GR","HIDALGO":"HG","JALISCO":"JA","MEXICO":"EM",
    "MICHOACAN":"MI","MORELOS":"MO","NAYARIT":"NA","NUEVO LEON":"NL","OAXACA":"OA","PUEBLA":"PU",
    "QUERETARO":"QT","QUINTANA ROO":"QR","SAN LUIS POTOSI":"SL","SINALOA":"SI","SONORA":"SO",
    "TABASCO":"TB","TAMAULIPAS":"TM","TLAXCALA":"TL","VERACRUZ":"VE","YUCATAN":"YU","ZACATECAS":"ZA"
}
STATE_SYNONYMS = {"CDMX":"CIUDAD DE MEXICO","ESTADO DE MEXICO":"MEXICO","EDOMEX":"MEXICO"}

# birthState canon + buckets regionales
CANON = {
    "SR":"SO","SO":"SO","VZ":"VE","VE":"VE","YN":"YU","YU":"YU","JC":"JA","JA":"JA","MC":"MI","MI":"MI",
    "TS":"TM","TM":"TM","TC":"TB","TB":"TB","CC":"CL","CL":"CL","DF":"DF","EM":"EM","NL":"NL","BC":"BC",
    "BS":"BS","SI":"SI","NA":"NA","DG":"DG","ZA":"ZA","AG":"AG","SL":"SL","HG":"HG","MO":"MO","TL":"TL",
    "PU":"PU","QT":"QT","GT":"GT","OA":"OA","CM":"CM","CS":"CS","CO":"CO","GR":"GR","QR":"QR","CH":"CH",
    "MS":"MI","MN":"MI","SP":"SL","NE":"NL","OC":"OA","PL":"PU","NT":"NA","ZS":"ZA","AS":"AG","UN":"OT", None:"OT"
}
REGION_BUCKET = {
    "BC":1,"SO":1,"CH":1,"CO":1,"NL":1,"TM":1,                # Norte
    "BS":2,"SI":2,"NA":2,"DG":2,"ZA":2,                       # Norte-Occidente
    "JA":3,"AG":3,"CL":3,"MI":3,"SL":3,                       # Centro-Norte
    "DF":4,"EM":4,"HG":4,"MO":4,"TL":4,"PU":4,"QT":4,"GT":4,  # Centro-País
    "CS":5,"TB":5,"CM":5,"YU":5,"QR":5,"OA":5,"GR":5,"VE":5,  # Sur-Sureste
    "OT":0
}

# Mapeos categóricos
GENDER_MAP = {"female":1, "male":0}
USER_TYPE_MAP = {"HYBRID":0, "DIGITAL":1, "ANALOG":2}
CHANNEL_DETAIL_MAP = {"ORGANIC":0,"COLLABORATOR":1,"POS":2,"SPIN_PREMIA":3,"DIGITAL_ORGANIC":4,"DIGITAL":5}

#CASH_IN_TX = {"CASH_IN_AT_OXXO","CASH_IN_AT_OXXO_QR"}

#SPEI_TX = {"SPEI_CASH_IN"}

#P2P_TX = {"P2P_TRANSFER_TARGET","P2P_TRANSFER_TARGET_CLABE","P2P_TRANSFER_TARGET","P2P_TRANSFER_TARGET_CARD"}

CASH_IN_TX = {
    "CASH_IN_AT_OXXO","CASH_IN_AT_OXXO_QR","CASH_OUT_WITH_CARD_AT_OXXO", "CASH_OUT_AT_OXXO","CASH_OUT_AT_MERCHANT","CARD_PURCHASE","CARD_ATM_WITHDRAWAL"
}
SPEI_TX = {"SPEI_CASH_IN","TRANSFER_TO_CARD","TRANSFER_TO_CLABE" "P2P_TRANSFER_TARGET","P2P_TRANSFER_TARGET_CLABE","P2P_TRANSFER_TARGET","P2P_TRANSFER_TARGET_CARD"}
   
P2P_TX ={"P2P_TRANSFER_TARGET_CLABE","P2P_TRANSFER_TARGET","P2P_TRANSFER_TARGET_CARD","P2P_TRANSFER_SOURCE_CARD","P2P_TRANSFER_SOURCE_CLABE",
    "P2P_TRANSFER_SOURCE","IN_APP_PURCHASE_TAE","IN_APP_PURCHASE_BILLPAYMENT","QR_MERCHANT_PAYMENT",
    "GIFT_CARD_PURCHASE","INTERNATIONAL_REMITTANCE_CASH_IN"}
    
TX_TYPE_MAP = {
    **{k: 0 for k in CASH_IN_TX},  # 0 = Cash_In
    **{k: 1 for k in SPEI_TX},     # 1 = SPEI
    **{k: 2 for k in P2P_TX}       # 2 = P2P
}
TX_TYPE_NAMES = {
    0: "Cash_In",
    1: "SPEI",
    2: "P2P"
}
TX_TRINARY_NAME = {0:"FISICAS", 1:"DIGITALES", 2:"P2P"}


# --- [# FeatureBuilder] celda 6 ---
class FeatureBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ohe_states_: t.List[str] = []
        # NUEVO: Pipeline de normalización
        self.scaler_: RobustScaler | None = None
        self.numeric_cols_: t.List[str] = []
        self._fitted_scaler: bool = False

    @staticmethod
    def _state_to_abbr(s: t.Any) -> str:
        if pd.isna(s): return "OT"
        s = str(s)
        if s.upper() in REGION_BUCKET: return s.upper()
        s2 = _norm(s)
        s2 = STATE_SYNONYMS.get(s2, s2)
        return STATE_TO_ABBR.get(s2, "OT")

    @staticmethod
    def _canon_birthstate(s: t.Any) -> str:
        if pd.isna(s): return "OT"
        s = str(s).upper()
        return CANON.get(s, s if s in REGION_BUCKET else "OT")

    def _mk_time_feats(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = pd.to_datetime(df[self.cfg.signup_ts_col], utc=True).dt.tz_convert(ZoneInfo(self.cfg.tz_local))
        df = df.copy()
        df["signup_dow"] = ts.dt.weekday.astype("int16")
        df["signup_week"] = ts.dt.isocalendar().week.astype("int16")
        df["signup_month"] = ts.dt.month.astype("int16")
        hr = ts.dt.hour
        df["signup_daypart"] = np.select([(hr>=5)&(hr<=11),(hr>=12)&(hr<=17)],[0,1],default=2).astype("int8")
        years = list({d.year for d in pd.to_datetime(df[self.cfg.signup_date_col]).dt.date})
        mx_hol = holidays.MX(years=years)
        dates = pd.to_datetime(df[self.cfg.signup_date_col]).dt.date
        df["is_holiday_mx"] = dates.map(lambda d: 1 if d in mx_hol else 0).astype("int8")
        day = ts.dt.day
        eom = (ts + pd.offsets.MonthEnd(0)).dt.day
        df["near_payday_any"] = ((np.abs(day-1)<=3)|(np.abs(day-15)<=3)|(np.abs(day-eom)<=3)).astype("int8")
        df["near_payday_1st"] = (np.abs(day-1)<=3).astype("int8")
        df["near_payday_15"]  = (np.abs(day-15)<=3).astype("int8")
        df["near_payday_eom"] = (np.abs(day-eom)<=3).astype("int8")
        return df

    def fit(self, df: pd.DataFrame):
        st = df["stateName"].map(self._state_to_abbr)
        self.ohe_states_ = sorted(st.dropna().unique().tolist())
        if "OT" not in self.ohe_states_: self.ohe_states_.append("OT")
        return self

    def transform(self, df: pd.DataFrame) -> Bunch:
        df = df.copy()
        s_ts = pd.to_datetime(df[CFG.signup_ts_col], utc=True)

        # Categóricas core
        df["gender_bin"] = df["gender"].map(GENDER_MAP).astype("float32")
        df["user_type_tri"] = df["user_type"].map(USER_TYPE_MAP).astype("float32")
        df["channel_detail_code"] = df["channelDetail"].map(CHANNEL_DETAIL_MAP).astype("float32")

        # birth bucket + edad
        bcanon = df["birthState"].map(self._canon_birthstate)
        df["birth_bucket"] = bcanon.map(REGION_BUCKET).astype("float32")
        bdate = pd.to_datetime(df["birth_date"], errors="coerce", utc=True)
        df["age_years"] = ((s_ts - bdate).dt.days/365.25).astype("float32")

        # Time features
        df = self._mk_time_feats(df)

        # state OHE
        st = df["stateName"].map(self._state_to_abbr)
        for ab in self.ohe_states_:
            df[f"state_{ab}"] = (st==ab).astype("int8")

        # Confirmaciones / flags: Int nulos a 0
        if "phone_conf_ts" in df.columns:
            phn_ts = pd.to_datetime(df["phone_conf_ts"], errors="coerce", utc=True)
            df["phn_confir"] = (phn_ts < s_ts).fillna(False).astype("int8")
        else:
            df["phn_confir"] = 0

        if "email_conf_ts" in df.columns:
            email_ts = pd.to_datetime(df["email_conf_ts"], errors="coerce", utc=True)
            df["email_confir"] = (email_ts < s_ts).fillna(False).astype("int8")
        else:
            df["email_confir"] = 0

        # Card_linked_date -> deltas sin fuga
        if "Card_linked_date" in df.columns:
            card_dt = pd.to_datetime(df["Card_linked_date"], errors="coerce", utc=True)
            # Cambiar <= por < (estrictamente antes)
            before = card_dt < s_ts  
            lag_days = (s_ts - card_dt).dt.days.astype("float32")
            df["card_linked_before_signup"] = before.fillna(False).astype("int8")
            # Solo crear lag_days si ocurrió ANTES (no <=)
            df["card_linked_lag_days"] = np.where(before, lag_days, np.nan).astype("float32")
            df = df.drop(columns=["Card_linked_date"])

        LEAKY_FEATURES = {
            'lifespan_days',        #  LEAKAGE CONFIRMADO (corr=0.535)
            'days_since_last',      #  Sospechoso (corr=0.329)
        }

            
        # Armar X
        drop_cols = set(LEAKY_ALWAYS) | {
            "stateName","gender","user_type","channelDetail","birthState","birth_date",
            CFG.signup_date_col, CFG.signup_ts_col
        } | set(EXTRA_DROP_TS) | LEAKY_FEATURES  # ← MODIFICAR ESTA LÍNEA

        drop_cols = [c for c in drop_cols if c in df.columns]
        X = df.drop(columns=drop_cols, errors="ignore")

        # Limpiar tipos
        # 1) Si queda algún dtype extension de BigQuery -> fuera
        bad_ext = [c for c in X.columns if "db_dtypes" in str(X[c].dtype).lower() or "dbdate" in str(X[c].dtype).lower()]
        X = X.drop(columns=bad_ext, errors="ignore")

        # 2) Casts seguros
        for c in X.columns:
            if pd.api.types.is_integer_dtype(X[c]) or str(X[c].dtype).startswith("Int"):
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype("float32")
            elif pd.api.types.is_float_dtype(X[c]):
                X[c] = X[c].astype("float32")
            elif pd.api.types.is_bool_dtype(X[c]):
                X[c] = X[c].astype("int8")
            elif pd.api.types.is_datetime64_any_dtype(X[c]):
                X = X.drop(columns=[c])

        # 3) Objetos -> fuera
        obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            X = X.drop(columns=obj_cols)

        # NUEVO: Normalización con RobustScaler
        if not self._fitted_scaler:
            # Primera vez: identificar columnas numéricas y fit scaler
            numeric_cols = [c for c in X.columns
                            if not c.startswith('state_')  # Excluir one-hot de estados
                            and X[c].dtype == 'float32'     # Solo float32
                            and X[c].nunique() > 10]        # Excluir binarias
            
            self.numeric_cols_ = numeric_cols
            
            if len(numeric_cols) > 0:
                self.scaler_ = RobustScaler()
                self.scaler_.fit(X[numeric_cols])
                self._fitted_scaler = True
                print(f'✓ RobustScaler fitted con {len(numeric_cols)} features numéricas')
            else:
                print('⚠️  No se encontraron features numéricas para normalizar')
                self._fitted_scaler = True
        
        # Aplicar normalización si existe
        if self.scaler_ is not None and len(self.numeric_cols_) > 0:
            existing_cols = [c for c in self.numeric_cols_ if c in X.columns]
            if len(existing_cols) > 0:
                X[existing_cols] = self.scaler_.transform(X[existing_cols])

        # y binaria
        y = pd.to_numeric(df[CFG.label_col], errors="coerce").fillna(0).astype(int).values
        
        assert_no_regex_leak(X)
        LEAK_BAN = LEAKY_ALWAYS | {"first_tx_type", "first_tx_amount"}
        assert_no_labelish_cols(X)
        
        meta = pd.DataFrame({
            "user_id": df.get("user_id", pd.Series(index=df.index, dtype="object")),
            "signup_date": pd.to_datetime(df[CFG.signup_date_col], errors="coerce"),
            "gender": df["gender"].astype(str),
            "channelDetail": df["channelDetail"].astype(str),
            "state_abbr": st.astype(str)
        })
        return Bunch(X=X, y=y, meta=meta)


# --- [# Modelo W30] celda 8 ---
def lift_at_k(y_true, y_score, frac: float) -> float:
    n = len(y_true); k = max(1, int(n*frac))
    idx = np.argpartition(-y_score, k-1)[:k]
    top_pos = y_true[idx].sum()
    base_rate = y_true.mean()
    expected_pos = k*base_rate
    return float(top_pos/expected_pos) if expected_pos>0 else np.nan

class Activation30Model:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_folds_: t.List[HistGradientBoostingClassifier] = []
        self.iso_folds_: t.List[IsotonicRegression] = []
        self.metrics_ = {}
        self.oof_pred_ = None
        self.feature_names_: t.List[str] = []

    def _make_model(self):
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
            l2_regularization=1.0, max_bins=255, random_state=self.cfg.random_state
        )
    def _apply_embargo(self, train_idx: np.ndarray, val_idx: np.ndarray, 
                       time_order: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
        """Purga registros del train/val que violen el embargo temporal"""
        if self.cfg.embargo_days == 0:
            return train_idx, val_idx

        # Convertir a pandas Timestamp para operaciones seguras
        train_dates = pd.to_datetime(time_order[train_idx])
        val_dates = pd.to_datetime(time_order[val_idx])

        # Última fecha de train
        last_train_date = train_dates.max()

        # Calcular embargo cutoff
        embargo_cutoff = last_train_date + pd.Timedelta(days=self.cfg.embargo_days)

        # Filtrar validación
        val_valid_mask = val_dates >= embargo_cutoff
        val_filtered = val_idx[val_valid_mask]

        # VALIDACIÓN CRÍTICA: Asegurar mínimo de muestras
        min_samples = 100  # mínimo razonable para HistGradientBoosting
        if len(val_filtered) < min_samples:
            print(f" WARNING: Embargo dejó {len(val_filtered)} muestras (< {min_samples}). "
                  f"Usando fold completo sin embargo.")
            return train_idx, val_idx

        removed = len(val_idx) - len(val_filtered)
        print(f"✓ Embargo aplicado: {removed} muestras removidas de validación "
              f"({removed/len(val_idx)*100:.1f}%)")

        return train_idx, val_filtered

    def fit_cv(self, X: pd.DataFrame, y: np.ndarray, time_order: np.ndarray):
        # ordenar temporal
        order = np.argsort(time_order)
        X = X.iloc[order].reset_index(drop=True)
        y = y[order]
        time_order_sorted = pd.to_datetime(time_order[order])

        tss = TimeSeriesSplit(n_splits=self.cfg.n_splits)
        oof = np.zeros(len(X), dtype=float)
        self.model_folds_.clear()
        self.iso_folds_.clear()
        self.feature_names_ = X.columns.tolist()

        for f, (tr, va) in enumerate(tss.split(X)):
            # embargo con fechas ya ordenadas
            tr, va = self._apply_embargo(tr, va, time_order_sorted)

            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y[tr], y[va]

            sw = compute_sample_weight(class_weight="balanced", y=ytr).astype("float32")

            m = self._make_model()
            m.fit(Xtr, ytr, sample_weight=sw)

            p_raw = m.predict_proba(Xva)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip").fit(p_raw, yva)
            p_cal = iso.transform(p_raw)
            oof[va] = p_cal

            self.model_folds_.append(m)
            self.iso_folds_.append(iso)
            print(f"[Fold {f+1}] AP={average_precision_score(yva,p_cal):.4f} | "
                  f"AUC={roc_auc_score(yva,p_cal):.4f} | "
                  f"Brier={brier_score_loss(yva,p_cal):.4f}")

        self.oof_pred_ = oof
        self.metrics_ = {
            "OOF_AP": average_precision_score(y, oof),
            "OOF_AUC": roc_auc_score(y, oof),
            "OOF_Brier": brier_score_loss(y, oof),
            **{f"OOF_Lift@{int(fr*100)}%": lift_at_k(y, oof, fr) for fr in self.cfg.lift_fracs}
        }
        return self


    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(X), dtype=float)
        for m, iso in zip(self.model_folds_, self.iso_folds_):
            preds += iso.transform(m.predict_proba(X)[:,1])
        return preds / max(1, len(self.model_folds_))


# --- [# Model type of transaction] celda 10 ---
class TxTypeModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_: HistGradientBoostingClassifier | None = None
        self.classes_: np.ndarray | None = None

    def _make_model(self):
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
            l2_regularization=1.0, max_bins=255, random_state=self.cfg.random_state
        )

    def fit(self, X: pd.DataFrame, y_tx: np.ndarray):
        # Pesos por clase inversos a la frecuencia
        classes, counts = np.unique(y_tx, return_counts=True)
        inv_freq = {c: (counts.sum()/ (len(classes)*cnt)) for c, cnt in zip(classes, counts)}
        sw = np.array([inv_freq[v] for v in y_tx], dtype="float32")

        m = self._make_model()
        m.fit(X, y_tx, sample_weight=sw)
        self.model_ = m
        self.classes_ = classes
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        proba = self.model_.predict_proba(X)
        # asegurar columnas en orden [0,1,2]
        out = np.zeros((len(X), 3), dtype=float)
        for i, c in enumerate(self.classes_):
            out[:, int(c)] = proba[:, i]
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ================= UTILIDADES DE SERVING =========================
def cfg_from_config_json(config_path):
    """Reconstruye Config desde artifacts/config.json (o defaults)."""
    config_path = Path(config_path)
    if not config_path.exists():
        return Config()
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    valid = {f.name for f in _dc_fields(Config)}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    if isinstance(kwargs.get("lift_fracs"), (list, tuple)):
        kwargs["lift_fracs"] = tuple(float(x) for x in kwargs["lift_fracs"])
    for key in ("embargo_days", "holdout_days", "n_splits", "random_state"):
        if kwargs.get(key) is not None:
            kwargs[key] = int(kwargs[key])
    if kwargs.get("train_sample_frac") is not None:
        kwargs["train_sample_frac"] = float(kwargs["train_sample_frac"])
    return Config(**kwargs)


def set_active_cfg(cfg):
    """FeatureBuilder.transform() lee el CFG GLOBAL de este modulo."""
    global CFG
    CFG = cfg
    return CFG


def register_in_main(names=("Config", "FeatureBuilder",
                            "Activation30Model", "TxTypeModel")):
    """Publica las clases en __main__ para que pickle resuelva por nombre."""
    import sys
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    this_mod = sys.modules[__name__]
    for name in names:
        if hasattr(this_mod, name):
            setattr(main_mod, name, getattr(this_mod, name))
