#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os

# Set before NumPy/scikit-learn imports for the most reproducible CPU execution.
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json, re, unicodedata, random
import os, json, re, unicodedata
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, log_loss
)


RAMAN_CSV = "../raman_added2.csv"
CLINICAL_CSV = "../CLINICAL.csv"
OUT_DIR = "./publication_clinical_reproduction_out_CV5"

DISEASE_LABELS = ["Liver", "Bile", "Pancreas"]

RAMAN_SAMPLE_COL = "Sample"
RAMAN_LABEL_COL  = "Group"

CLIN_SAMPLE_COL  = "Sample"
CLIN_LABEL_COL   = "Diagnosis"   
CLIN_PATIENT_COL = "PatientID"   


LOCKED_INPUT_DIR = "./publication_clinical_inputs_CV5"
STAGE2_SPLITS_CSV = os.path.join(LOCKED_INPUT_DIR, "selected_rep_SPLITS.csv")
ARCHIVED_FOLDS_CSV = os.path.join(LOCKED_INPUT_DIR, "selected_rep_PHASE2_HYPERPARAMS.csv")


SELECTED_REP = 23


REQUIRE_LOCKED_REP = True

# Outer CV
N_SPLITS_OUTER = 3
N_REPEATS = 30
SEED0 = 0
MAX_OUTER_TRIES = 10000


PHASE2_N_JOBS = -1

# Meta-folds inside outer-train to build OOF Raman features
N_META_SPLITS = 3

# Views (Raman partitions)
VIEWS = [("all", "all"), ("fp", "fingerprint"), ("ch", "ch")]

FIXED_MULTI_BY_VIEW = {
    "all": {"n_pc": 100, "C": 10.0,  "gamma": "scale"},
    "fp":  {"n_pc": 100, "C": 100.0, "gamma": "scale"},
    "ch":  {"n_pc": 100, "C": 10.0,  "gamma": "scale"},
}

# Fusion weights 
FUSION_W = {"all": 0.50, "fp": 0.20, "ch": 0.30}


ADD_RAMAN_LOGIT_FEATURES = False
ADD_RAMAN_PAIRWISE_LOGODDS = False
ADD_RAMAN_META_FEATURES = False
PROB_EPS = 1e-6


PHASE2_Cs = np.logspace(-3, 3, 13).tolist()  # [0.001 ... 1000]
PHASE2_L1RATIOS = [0.0, 0.05, 0.1, 0.25]
PHASE2_INNER_CV = 3
PHASE2_MAX_ITER = 30000
PHASE2_SCORING = "neg_log_loss"


LOCK_PHASE2_HYPERPARAMS_FROM_ARCHIVE = True


FALLBACK_TUNE_N_SPLITS = 30
FALLBACK_TUNE_TEST_FRAC = 0.25
FALLBACK_TUNE_SEED_OFFSET = 7777

FALLBACK_DEFAULT_C = 1.0
FALLBACK_DEFAULT_L1RATIO = 0.1

# Outputs toggles (existing)
SAVE_SPLITS = True
SAVE_PATIENT_PREDS = True
SAVE_COEFS = True
RUN_ABLATION = True  # Set False for fastest main-result reproduction

# Insight settings
TOPK_EXPLAIN = 12
BOOTSTRAP_CI = True
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 123

CSV_ENCODING = "utf-8-sig"

ENABLE_STAGEGROUP = False
STAGEGROUP_RULE = "I_II_vs_III_IV"


LT_MULTIPLIER = 0.5
GT_MULTIPLIER = 1.0

MISSING_TOKENS = {
    "", "na", "n/a", "nan", "none", "null", "-", "--",
    "nd", "n.d.", "not detected", "not_detected", "missing"
}


NUMERIC_CLIN_COLS = [
    "Age",
    "AFP(ng/ml)",
    "CA19-9(U/ml)",
    "CEA(ng/ml)",
    "ALT(U/L)",
    "AST(U/L)",
    "Total Bilirubin(umol/L)",
]


INSIGHT_EXTRA_NUMERIC_COLS = [
    "Total Bilirubin(μmol/L)",  # tolerate alt header for plotting/missingness
]


CATEGORICAL_CLIN_COLS = [
    "Sex",
    "Stage",
    # "Smoking History",
    # "Alcohol Consumption",
    "Diabetes Mellitus",
    "Viral Hepatitis Status",
]

# Subgroup cols to export/merge if present (analysis only)
SUBGROUP_COLS_TO_EXPORT = [
    "Sex", "Stage",
    "Smoking History", "Alcohol Consumption",
    "Diabetes Mellitus", "Viral Hepatitis Status",
]

LIVER_LABS = ["ALT(U/L)", "AST(U/L)", "Total Bilirubin(umol/L)", "Total Bilirubin(μmol/L)"]
TUMOR_MARKERS = ["AFP(ng/ml)", "CA19-9(U/ml)", "CEA(ng/ml)"]
RISK_FACTORS = ["Smoking History", "Alcohol Consumption", "Diabetes Mellitus", "Viral Hepatitis Status"]
SEX_COL = "Sex"
STAGE_COL = "Stage"

MIN_CAT_FREQ = 2


FORBIDDEN_MODEL_FEATURE_TOKENS = ()

def _canonical_feature_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())

def assert_no_forbidden_model_features(feature_cols: List[str], where: str):
    bad = [
        str(c) for c in feature_cols
        if any(tok in _canonical_feature_name(c) for tok in FORBIDDEN_MODEL_FEATURE_TOKENS)
    ]
    if bad:
        raise RuntimeError(
            f"Forbidden cell-count feature(s) detected in {where}: {bad}. "
            "Cell-count variables must not be used for model fitting."
        )

def set_global_reproducibility(seed: int):
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))


def read_csv_robust(path: str, dtype=None) -> pd.DataFrame:
    encs = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
    last_err = None
    for enc in encs:
        try:
            return pd.read_csv(path, dtype=dtype, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    try:
        return pd.read_csv(path, dtype=dtype)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV {path}. Last errors: {last_err} / {e}")

def to_csv_safe(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)


def _make_ohe(min_freq: int = MIN_CAT_FREQ):
    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_freq,
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _parse_json_list(x: str) -> List[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(t) for t in v]
        if isinstance(v, (set, tuple)):
            return [str(t) for t in list(v)]
        return [str(v)]
    except Exception:
        if s.startswith("[") and s.endswith("]"):
            s2 = s.strip("[]").strip()
            if not s2:
                return []
            return [t.strip().strip('"').strip("'") for t in s2.split(",") if t.strip()]
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        return [s]

def load_stage2_splits_csv(path: str) -> pd.DataFrame:
    df = read_csv_robust(path, dtype=str)
    need = ["rep", "fold", "outer_train_samples_json", "outer_test_samples_json"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Stage-2 splits CSV missing column '{c}'. Found columns: {df.columns.tolist()}")
    df = df.copy()
    df["rep"] = df["rep"].astype(int)
    df["fold"] = df["fold"].astype(int)
    df["outer_train_list"] = df["outer_train_samples_json"].apply(_parse_json_list)
    df["outer_test_list"]  = df["outer_test_samples_json"].apply(_parse_json_list)
    df = df.sort_values(["rep", "fold"]).reset_index(drop=True)
    return df


_RE_NUM_DOT0 = re.compile(r'^\s*(\d+)\.0\s*$')

def _norm_id_basic(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    m = _RE_NUM_DOT0.match(s)
    if m:
        return m.group(1)
    return s

def _norm_id_drop_leading_zeros_if_numeric(s: str) -> str:
    if not s:
        return s
    if s.isdigit():
        try:
            return str(int(s))
        except Exception:
            return s.lstrip("0") or "0"
    return s

def build_sample_keys(raman_samples: np.ndarray, clin_samples: np.ndarray):
    r_basic = np.array([_norm_id_basic(x) for x in raman_samples], dtype=str)
    c_basic = np.array([_norm_id_basic(x) for x in clin_samples], dtype=str)
    if len(np.intersect1d(r_basic, c_basic)) > 0:
        return r_basic, c_basic, "basic"

    r2 = np.array([_norm_id_drop_leading_zeros_if_numeric(_norm_id_basic(x)) for x in raman_samples], dtype=str)
    c2 = np.array([_norm_id_drop_leading_zeros_if_numeric(_norm_id_basic(x)) for x in clin_samples], dtype=str)
    if len(np.intersect1d(r2, c2)) > 0:
        return r2, c2, "drop_leading_zeros"

    return r_basic, c_basic, "basic"

_ROMAN_UNICODE_MAP = {"Ⅰ":"I", "Ⅱ":"II", "Ⅲ":"III", "Ⅳ":"IV", "Ⅴ":"V", "Ⅵ":"VI", "Ⅶ":"VII", "Ⅷ":"VIII", "Ⅸ":"IX", "Ⅹ":"X"}
_ARABIC_TO_ROMAN = {1:"I",2:"II",3:"III",4:"IV",5:"V",6:"VI",7:"VII",8:"VIII",9:"IX",10:"X"}

def _clean_text_nfkc(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("\uFFFD", "").replace("�", "")
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_stage_value_strict(x) -> str:
    s = _clean_text_nfkc(x)
    if not s or s.lower() in MISSING_TOKENS:
        return "Unknown"
    if "�" in s:
        return "Unknown"
    for k, v in _ROMAN_UNICODE_MAP.items():
        s = s.replace(k, v)
    s = s.strip()
    if re.fullmatch(r"(?i)stage", s.strip()):
        return "Unknown"
    s = re.sub(r"(?i)^\s*stage\s*", "", s).strip()
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return "Unknown"
    m = re.match(r"^((?:I{1,3}|IV|V|VI{0,3}|IX|X)|[0-9]{1,2})\s*([A-C]?)\b", s, flags=re.IGNORECASE)
    if not m:
        return "Unknown"
    core = m.group(1).upper()
    suf  = m.group(2).upper() if m.group(2) else ""
    if core.isdigit():
        n = int(core)
        if n in _ARABIC_TO_ROMAN:
            core = _ARABIC_TO_ROMAN[n]
        else:
            return "Unknown"
    return f"Stage {core}{suf}"

def stagegroup_from_stage(stage_norm: str) -> str:
    s = normalize_stage_value_strict(stage_norm)
    if s == "Unknown":
        return "Unknown"
    m = re.search(r"\bStage\s*(I{1,3}|IV)\b", s, flags=re.IGNORECASE)
    if not m:
        return "Unknown"
    core = m.group(1).upper()
    if STAGEGROUP_RULE == "I_II_vs_III_IV":
        if core in ("I", "II"):
            return "Early"
        if core in ("III", "IV"):
            return "Late"
    return "Unknown"


def get_spec_cols_and_wav(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    cols, wav = [], []
    for c in df.columns:
        try:
            wav.append(float(c))
            cols.append(c)
        except Exception:
            pass
    order = np.argsort(wav)
    cols = [cols[i] for i in order]
    wav = np.array([wav[i] for i in order], dtype=float)
    return cols, wav

def select_region(cols: List[str], wav: np.ndarray, region: str) -> List[str]:
    if region == "all":
        m = np.ones_like(wav, dtype=bool)
    elif region == "fingerprint":
        m = (wav >= 600) & (wav <= 1800)
    elif region == "ch":
        m = (wav >= 2800) & (wav <= 3100)
    else:
        raise ValueError(region)
    out = [c for c, keep in zip(cols, m) if keep]
    if len(out) == 0:
        raise ValueError(f"No spectral columns selected for region={region}. Check wavelength headers.")
    return out

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, PROB_EPS, 1 - PROB_EPS)
    return np.log(p / (1 - p))

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def agg_probs_to_sample_logit(sample_ids: np.ndarray, probs: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    probs = np.clip(probs, PROB_EPS, 1 - PROB_EPS)
    z = logit(probs)
    dfz = pd.DataFrame(z, columns=[f"z_{c}" for c in class_names])
    dfz["SampleKey"] = sample_ids
    z_mean = dfz.groupby("SampleKey").mean()
    P = sigmoid(z_mean.values)
    P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    return pd.DataFrame(P, index=z_mean.index, columns=[f"p_{c}" for c in class_names])

def class_counts_str(y_str: np.ndarray, classes: List[str]) -> str:
    d = {c: int(np.sum(y_str == c)) for c in classes}
    return json.dumps(d, ensure_ascii=False)

def _entropy(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, 1.0)
    return -(P * np.log(P)).sum(axis=1)

def _margin(P: np.ndarray) -> np.ndarray:
    s = np.sort(P, axis=1)
    return (s[:, -1] - s[:, -2])

def _set_json(a: set) -> str:
    return json.dumps(sorted(list(a)), ensure_ascii=False)


def augment_raman_features_df(
    Xr: pd.DataFrame,
    classes: List[str],
    prefix_in: str = "raman_p_",
    prefix_out_logit: str = "raman_logit_",
    prefix_out_ratio: str = "raman_logodds_",
) -> pd.DataFrame:
    Xr2 = Xr.copy()
    prob_cols = [f"{prefix_in}{c}" for c in classes]
    missing = [c for c in prob_cols if c not in Xr2.columns]
    if missing:
        raise RuntimeError(f"augment_raman_features_df missing prob cols: {missing}")

    P = Xr2[prob_cols].to_numpy(dtype=float)
    P = np.clip(P, PROB_EPS, 1.0 - PROB_EPS)
    P = P / (P.sum(axis=1, keepdims=True) + 1e-12)

    if ADD_RAMAN_LOGIT_FEATURES:
        Z = logit(P)
        for j, c in enumerate(classes):
            Xr2[f"{prefix_out_logit}{c}"] = Z[:, j]

    if ADD_RAMAN_PAIRWISE_LOGODDS:
        for a in classes:
            for b in classes:
                if a == b:
                    continue
                pa = np.clip(Xr2[f"{prefix_in}{a}"].to_numpy(dtype=float), PROB_EPS, 1.0)
                pb = np.clip(Xr2[f"{prefix_in}{b}"].to_numpy(dtype=float), PROB_EPS, 1.0)
                Xr2[f"{prefix_out_ratio}{a}_over_{b}"] = np.log(pa / pb)

    if ADD_RAMAN_META_FEATURES:
        Xr2["raman_entropy"] = _entropy(P)
        Xr2["raman_margin"]  = _margin(P)
        Xr2["raman_maxprob"] = P.max(axis=1)

    return Xr2


def train_svm_pca_fixed(Xtr: np.ndarray, ytr: np.ndarray, best: Dict, seed: int):
    if len(np.unique(ytr)) < 2:
        raise ValueError("Stage-1 training set has <2 classes at cell level.")
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)

    n_feat = int(Xtr_s.shape[1])
    n_samp = int(Xtr_s.shape[0])
    max_pc = max(1, min(n_feat, n_samp - 1))
    n_pc = int(min(int(best["n_pc"]), max_pc))

    pca = PCA(n_components=n_pc, random_state=seed)
    Ztr = pca.fit_transform(Xtr_s)

    svm = SVC(
        kernel="rbf",
        C=float(best["C"]),
        gamma=str(best["gamma"]),
        probability=True,
        class_weight="balanced",
        cache_size=2000,
        random_state=seed,
    )
    svm.fit(Ztr, ytr)
    return sc, pca, svm

def predict_proba_full(sc: StandardScaler, pca: PCA, svm: SVC, X: np.ndarray, n_classes: int) -> np.ndarray:
    P_small = svm.predict_proba(pca.transform(sc.transform(X)))
    cols_present = svm.classes_.astype(int)
    P = np.zeros((P_small.shape[0], n_classes), dtype=float)
    P[:, cols_present] = P_small
    return P

def train_view_model_fixed(
    df_raman: pd.DataFrame,
    spec_cols: List[str],
    classes: List[str],
    train_sample_keys: set,
    seed: int,
    fixed_multi: Dict,
    raman_label_col: str,
):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    tr_idx = np.where(df_raman["SampleKey"].isin(train_sample_keys).values)[0]
    if len(tr_idx) == 0:
        raise RuntimeError("No Raman cells found for meta-train patients. Check SampleKey mapping.")

    Xtr = df_raman.iloc[tr_idx][spec_cols].to_numpy(np.float32)
    y_str = df_raman.iloc[tr_idx][raman_label_col].astype(str).values
    ytr = np.array([class_to_idx[s] for s in y_str], dtype=int)

    sc, pca, svm = train_svm_pca_fixed(Xtr, ytr, fixed_multi, seed=seed)
    return dict(sc=sc, pca=pca, svm=svm, spec_cols=spec_cols, classes=classes, n_classes=n_classes)

def predict_patient_probs_view(df_raman: pd.DataFrame, model: Dict, sample_keys_set: set) -> pd.DataFrame:
    spec_cols = model["spec_cols"]
    classes = model["classes"]
    n_classes = model["n_classes"]

    idx = np.where(df_raman["SampleKey"].isin(sample_keys_set).values)[0]
    if len(idx) == 0:
        return pd.DataFrame()

    X = df_raman.iloc[idx][spec_cols].to_numpy(np.float32)
    samp = df_raman.iloc[idx]["SampleKey"].astype(str).values
    p_cell = predict_proba_full(model["sc"], model["pca"], model["svm"], X, n_classes=n_classes)
    return agg_probs_to_sample_logit(samp, p_cell, classes)

def fuse_patient_probs_fixed(P_all: pd.DataFrame, P_fp: pd.DataFrame, P_ch: pd.DataFrame, classes: List[str]) -> pd.DataFrame:
    idx = P_all.index
    P_fp = P_fp.reindex(idx)
    P_ch = P_ch.reindex(idx)
    if P_fp.isna().any().any():
        P_fp = P_fp.fillna(P_all)
    if P_ch.isna().any().any():
        P_ch = P_ch.fillna(P_all)

    A = P_all.to_numpy()
    F = P_fp.to_numpy()
    C = P_ch.to_numpy()

    P = FUSION_W["all"] * A + FUSION_W["fp"] * F + FUSION_W["ch"] * C
    P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    return pd.DataFrame(P, index=idx, columns=[f"p_{c}" for c in classes])


def make_constrained_outer_splits(samples: np.ndarray, yS: np.ndarray, seed0: int, n_classes: int):
    required = list(range(n_classes))
    for tries in range(1, MAX_OUTER_TRIES + 1):
        seed_try = seed0 + 10_000 * tries
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=seed_try)
        folds = list(sgkf.split(samples, yS, groups=samples))
        ok = True
        for tr, te in folds:
            present_te = set(np.unique(yS[te]).tolist())
            present_tr = set(np.unique(yS[tr]).tolist())
            if (not all(r in present_te for r in required)) or (not all(r in present_tr for r in required)):
                ok = False
                break
        if ok:
            for fold_id, (tr, te) in enumerate(folds, 1):
                yield fold_id, tr, te, seed_try, tries
            return
    raise RuntimeError(
        f"Could not find an outer split meeting constraints after {MAX_OUTER_TRIES} tries. "
        f"N_SPLITS_OUTER={N_SPLITS_OUTER}, counts={np.bincount(yS)}. "
        f"Try reducing N_SPLITS_OUTER."
    )


def metrics_multiclass(y_true: np.ndarray, P: np.ndarray, n_classes: int):
    yhat = P.argmax(axis=1)
    acc = float(accuracy_score(y_true, yhat))
    bal = float(balanced_accuracy_score(y_true, yhat)) if len(np.unique(y_true)) >= 2 else np.nan
    f1m = float(f1_score(y_true, yhat, average="macro", labels=list(range(n_classes)), zero_division=0))
    auc = np.nan
    try:
        if len(np.unique(y_true)) == n_classes:
            auc = float(roc_auc_score(y_true, P, multi_class="ovr"))
    except Exception:
        auc = np.nan
    return acc, bal, f1m, auc

def confusion_flat(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    return ",".join(map(str, cm.flatten().tolist()))


_RE_INEQ = re.compile(r"^\s*(<=|>=|<|>)\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$")
_RE_NUM  = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_RE_RANGE = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*[-–]\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$")

def parse_numeric_robust(x, lt_mult: float = LT_MULTIPLIER, gt_mult: float = GT_MULTIPLIER) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return np.nan
    s = s.replace(",", "").strip()

    m = _RE_INEQ.match(s)
    if m:
        op = m.group(1)
        try:
            v = float(m.group(2))
        except Exception:
            return np.nan
        if op in ("<", "<="):
            return float(lt_mult) * v
        else:
            return float(gt_mult) * v

    m2 = _RE_RANGE.match(s)
    if m2:
        try:
            a = float(m2.group(1))
            b = float(m2.group(2))
            return 0.5 * (a + b)
        except Exception:
            return np.nan

    if _RE_NUM.match(s):
        try:
            return float(s)
        except Exception:
            return np.nan

    s2 = re.sub(r"[^0-9eE\.\+\-]", "", s)
    if _RE_NUM.match(s2):
        try:
            return float(s2)
        except Exception:
            return np.nan
    return np.nan

def preprocess_clinical_locked(df_clin: pd.DataFrame) -> pd.DataFrame:
    df = df_clin.copy()
    df.columns = df.columns.astype(str).str.strip()
    if CLIN_SAMPLE_COL not in df.columns:
        raise ValueError(f"Clinical CSV missing '{CLIN_SAMPLE_COL}'")
    df[CLIN_SAMPLE_COL] = df[CLIN_SAMPLE_COL].astype(str).str.strip()

    # Parse model numeric cols
    for col in NUMERIC_CLIN_COLS:
        if col in df.columns:
            df[col] = df[col].apply(parse_numeric_robust).astype(float)

    # Parse insight-only numeric cols 
    for col in INSIGHT_EXTRA_NUMERIC_COLS:
        if col in df.columns and col not in NUMERIC_CLIN_COLS:
            df[col] = df[col].apply(parse_numeric_robust).astype(float)

    # Parse categorical cols used by model
    for col in CATEGORICAL_CLIN_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_clean_text_nfkc)
            df[col] = df[col].replace({"nan": "", "None": "", "": ""}).fillna("")
            if col == "Stage":
                df[col] = df[col].apply(normalize_stage_value_strict)
            else:
                s = df[col].astype(str).str.strip()
                s = s.replace(list(MISSING_TOKENS), "")
                s = s.replace({"": "Unknown"}).fillna("Unknown")
                df[col] = s

    # Also normalize subgroup cols if present 
    for col in SUBGROUP_COLS_TO_EXPORT:
        if col in df.columns and col not in CATEGORICAL_CLIN_COLS:
            s = df[col].apply(_clean_text_nfkc).astype(str).str.strip()
            s = s.replace(list(MISSING_TOKENS), "")
            s = s.replace({"": "Unknown"}).fillna("Unknown")
            if col == "Stage":
                s = s.apply(normalize_stage_value_strict)
            df[col] = s

    return df

def reduce_clinical_to_one_row_per_sample(df_clin: pd.DataFrame) -> pd.DataFrame:
    df = df_clin.copy()
    df.columns = df.columns.astype(str).str.strip()
    df[CLIN_SAMPLE_COL] = df[CLIN_SAMPLE_COL].astype(str).str.strip()
    if df[CLIN_SAMPLE_COL].is_unique:
        return df

    def first_non_unknown(series):
        s = series.dropna().astype(str)
        if len(s) == 0:
            return np.nan
        s2 = s[s != "Unknown"]
        if len(s2) > 0:
            return s2.iloc[0]
        return s.iloc[0]

    agg = {}
    for c in df.columns:
        if c == CLIN_SAMPLE_COL:
            continue
        # numeric columns (model + insight extras)
        if c in (NUMERIC_CLIN_COLS + INSIGHT_EXTRA_NUMERIC_COLS):
            agg[c] = "median"
        else:
            agg[c] = first_non_unknown

    return df.groupby(CLIN_SAMPLE_COL, as_index=False).agg(agg)

def get_clinical_feature_cols(df_c: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    cols_num = [c for c in NUMERIC_CLIN_COLS if c in df_c.columns]
    cols_cat = [c for c in CATEGORICAL_CLIN_COLS if c in df_c.columns]
    feature_cols = cols_num + cols_cat
    if len(feature_cols) == 0:
        raise ValueError("No clinical feature columns found. Check clinical CSV headers / lists.")
    assert_no_forbidden_model_features(feature_cols, "clinical feature allowlist")
    return feature_cols, cols_num, cols_cat

def build_phase2_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", _make_ohe(min_freq=MIN_CAT_FREQ)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_oof_raman_features(
    df_raman: pd.DataFrame,
    sample_keys_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    classes: List[str],
    all_cols: List[str],
    wav: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    binc = np.bincount(y_outer_train, minlength=len(classes))
    minc = int(binc.min())
    meta_splits = int(min(N_META_SPLITS, minc))
    if meta_splits < 2:
        raise RuntimeError(f"Too few patients per class in outer-train for OOF meta-CV. counts={binc.tolist()}")

    sgkf = StratifiedGroupKFold(n_splits=meta_splits, shuffle=True, random_state=seed)
    feats = []

    for k, (tr, va) in enumerate(sgkf.split(sample_keys_outer_train, y_outer_train, groups=sample_keys_outer_train), 1):
        meta_train = set(sample_keys_outer_train[tr].tolist())
        meta_val   = set(sample_keys_outer_train[va].tolist())

        models = {}
        for vname, region in VIEWS:
            cols = select_region(all_cols, wav, region)
            fixed = FIXED_MULTI_BY_VIEW[vname]
            models[vname] = train_view_model_fixed(
                df_raman=df_raman,
                spec_cols=cols,
                classes=classes,
                train_sample_keys=meta_train,
                seed=seed + 1000 * k + (11 if vname=="all" else (23 if vname=="fp" else 37)),
                fixed_multi=fixed,
                raman_label_col=RAMAN_LABEL_COL,
            )

        P_all = predict_patient_probs_view(df_raman, models["all"], meta_val)
        P_fp  = predict_patient_probs_view(df_raman, models["fp"],  meta_val)
        P_ch  = predict_patient_probs_view(df_raman, models["ch"],  meta_val)
        if P_all.empty:
            raise RuntimeError("Meta-val produced empty patient probs; check SampleKey construction.")

        P_fused = fuse_patient_probs_fixed(P_all, P_fp, P_ch, classes=classes)
        P_fused = P_fused.rename(columns={f"p_{c}": f"raman_p_{c}" for c in classes})
        P_fused["meta_fold"] = k
        feats.append(P_fused.reset_index())

    oof = pd.concat(feats, ignore_index=True)
    if oof["SampleKey"].duplicated().any():
        dup = oof[oof["SampleKey"].duplicated()]["SampleKey"].tolist()[:10]
        raise RuntimeError(f"OOF Raman features duplicated per sample; example dups: {dup}")
    return oof.set_index("SampleKey")

def build_raman_features_for_test(
    df_raman: pd.DataFrame,
    train_keys: set,
    test_keys: set,
    classes: List[str],
    all_cols: List[str],
    wav: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    models = {}
    for vname, region in VIEWS:
        cols = select_region(all_cols, wav, region)
        fixed = FIXED_MULTI_BY_VIEW[vname]
        models[vname] = train_view_model_fixed(
            df_raman=df_raman,
            spec_cols=cols,
            classes=classes,
            train_sample_keys=train_keys,
            seed=seed + (11 if vname=="all" else (23 if vname=="fp" else 37)),
            fixed_multi=fixed,
            raman_label_col=RAMAN_LABEL_COL,
        )

    P_all = predict_patient_probs_view(df_raman, models["all"], test_keys)
    P_fp  = predict_patient_probs_view(df_raman, models["fp"],  test_keys)
    P_ch  = predict_patient_probs_view(df_raman, models["ch"],  test_keys)
    if P_all.empty:
        raise RuntimeError("Outer-test produced empty patient probs; check SampleKey construction.")

    P_fused = fuse_patient_probs_fixed(P_all, P_fp, P_ch, classes=classes)
    P_fused = P_fused.rename(columns={f"p_{c}": f"raman_p_{c}" for c in classes})
    P_fused.index.name = "SampleKey"
    return P_fused


def _fit_phase2_with_inner_cv(pipe: Pipeline, Xtr: pd.DataFrame, ytr: np.ndarray) -> Pipeline:
    pipe.fit(Xtr, ytr)
    return pipe

def _tune_phase2_outertrain_fallback(
    pre: ColumnTransformer,
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    seed: int,
) -> Tuple[float, float, str]:
    ytr = np.asarray(ytr, dtype=int)
    n = len(ytr)
    n_classes = len(np.unique(ytr))
    if n < 8 or n_classes < 2:
        return float(FALLBACK_DEFAULT_C), float(FALLBACK_DEFAULT_L1RATIO), "fixed_default_too_small"

    test_size = float(FALLBACK_TUNE_TEST_FRAC)
    min_test = max(1, int(np.ceil(n * test_size)))
    if min_test < n_classes:
        test_size = min(0.5, float(n_classes) / float(n) + 0.05)

    splitter = StratifiedShuffleSplit(
        n_splits=FALLBACK_TUNE_N_SPLITS,
        test_size=test_size,
        random_state=int(seed) + FALLBACK_TUNE_SEED_OFFSET
    )

    best = (np.inf, float(FALLBACK_DEFAULT_C), float(FALLBACK_DEFAULT_L1RATIO))
    any_ok = False

    for C in PHASE2_Cs:
        for l1r in PHASE2_L1RATIOS:
            losses = []
            ok = True
            for tr_idx, va_idx in splitter.split(np.zeros(n), ytr):
                y_va = ytr[va_idx]
                if len(np.unique(y_va)) < 2:
                    ok = False
                    break

                X_tr = Xtr.iloc[tr_idx]
                X_va = Xtr.iloc[va_idx]
                y_tr2 = ytr[tr_idx]

                clf = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=float(l1r),
                    C=float(C),
                    multi_class="multinomial",
                    class_weight="balanced",
                    max_iter=PHASE2_MAX_ITER,
                    n_jobs=PHASE2_N_JOBS,
                    random_state=int(seed),
                )
                pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
                try:
                    pipe.fit(X_tr, y_tr2)
                    P = pipe.predict_proba(X_va)
                    losses.append(log_loss(y_va, P, labels=sorted(np.unique(ytr).tolist())))
                except Exception:
                    ok = False
                    break

            if ok and len(losses) > 0:
                any_ok = True
                m = float(np.mean(losses))
                if m < best[0]:
                    best = (m, float(C), float(l1r))

    if not any_ok:
        return float(FALLBACK_DEFAULT_C), float(FALLBACK_DEFAULT_L1RATIO), "fixed_default_no_valid_splits"

    return best[1], best[2], "outertrain_shuffle_tuned"

def fit_phase2_pipeline_robust(
    pre: ColumnTransformer,
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    inner_cv: StratifiedKFold,
    seed: int,
    archived_hp: Dict = None,
) -> Tuple[Pipeline, Dict]:
    if archived_hp is not None:
        C_arch = float(archived_hp["C"])
        l1_arch = float(archived_hp["l1_ratio"])
        method_arch = str(archived_hp.get("method", "inner_cv"))
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=l1_arch,
            C=C_arch,
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=PHASE2_MAX_ITER,
            n_jobs=PHASE2_N_JOBS,
            random_state=int(seed),
        )
        pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        return pipe, dict(
            method=method_arch,
            C=C_arch,
            l1_ratio=l1_arch,
            fit_mode="archived_hyperparameter_refit",
        )

    try:
        clf_cv = LogisticRegressionCV(
            Cs=PHASE2_Cs,
            cv=inner_cv,
            penalty="elasticnet",
            solver="saga",
            l1_ratios=PHASE2_L1RATIOS,
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=PHASE2_MAX_ITER,
            n_jobs=PHASE2_N_JOBS,
            random_state=int(seed),
            scoring=PHASE2_SCORING,
            refit=True,
        )
        pipe = Pipeline(steps=[("pre", pre), ("clf", clf_cv)])
        pipe = _fit_phase2_with_inner_cv(pipe, Xtr, ytr)

        C_used = float(np.mean(pipe.named_steps["clf"].C_)) if hasattr(pipe.named_steps["clf"], "C_") else np.nan
        l1r = getattr(pipe.named_steps["clf"], "l1_ratio_", None)
        l1_used = float(np.mean(l1r)) if l1r is not None else np.nan
        return pipe, dict(method="inner_cv", C=C_used, l1_ratio=l1_used, fit_mode="inner_cv_refit")
    except Exception as e:
        C_fb, l1_fb, tag = _tune_phase2_outertrain_fallback(pre, Xtr, ytr, seed=seed)

        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=float(l1_fb),
            C=float(C_fb),
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=PHASE2_MAX_ITER,
            n_jobs=PHASE2_N_JOBS,
            random_state=int(seed),
        )
        pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        return pipe, dict(method=tag, C=float(C_fb), l1_ratio=float(l1_fb), fit_mode="fallback_refit", inner_cv_error=str(e)[:300])


def extract_coefs_long(pipe: Pipeline, classes: List[str]) -> pd.DataFrame:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    feat_names = pre.get_feature_names_out()
    coef = clf.coef_
    rows = []
    for ci, cname in enumerate(classes):
        for fi, fn in enumerate(feat_names):
            rows.append({"class": cname, "feature": str(fn), "coef": float(coef[ci, fi])})
    return pd.DataFrame(rows)


def _topk_contrib(feature_names: np.ndarray, z_row: np.ndarray, coef_row: np.ndarray, topk: int) -> List[Tuple[str, float]]:
    contrib = z_row * coef_row
    idx = np.argsort(-np.abs(contrib))[:topk]
    return [(str(feature_names[i]), float(contrib[i])) for i in idx]


def _split_num_cat(feature_cols: List[str], clin_num_cols: List[str], clin_cat_cols: List[str]) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in feature_cols if c in clin_num_cols]
    cat_cols = [c for c in feature_cols if c in clin_cat_cols]
    return num_cols, cat_cols

def build_ablation_specs(clin_feature_cols: List[str], raman_feat_cols: List[str]) -> Dict[str, Dict]:
    drops_labs = [c for c in LIVER_LABS if c in clin_feature_cols]
    drops_markers = [c for c in TUMOR_MARKERS if c in clin_feature_cols]
    drops_risk = [c for c in RISK_FACTORS if c in clin_feature_cols]
    drops_sex = [SEX_COL] if SEX_COL in clin_feature_cols else []
    return {
        "clinical_only": dict(use_clinical=True,  use_raman=False, drop_cols=[]),
        "raman_only":    dict(use_clinical=False, use_raman=True,  drop_cols=[]),
        "full":          dict(use_clinical=True,  use_raman=True,  drop_cols=[]),
        "full_minus_liver_labs":    dict(use_clinical=True, use_raman=True, drop_cols=drops_labs),
        "full_minus_tumor_markers": dict(use_clinical=True, use_raman=True, drop_cols=drops_markers),
        "full_minus_risk_factors":  dict(use_clinical=True, use_raman=True, drop_cols=drops_risk),
        "full_minus_sex":           dict(use_clinical=True, use_raman=True, drop_cols=drops_sex),
    }


def bootstrap_ci_metric(y_true: np.ndarray, P: np.ndarray, metric_fn, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 5:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(y_true[idx], P[idx]))
    lo = float(np.nanpercentile(vals, 2.5))
    hi = float(np.nanpercentile(vals, 97.5))
    return (lo, hi)

def subgroup_metrics_table(df_preds: pd.DataFrame, classes: List[str], subgroup_col: str):
    n_classes = len(classes)
    out = []
    for val, sub in df_preds.groupby(subgroup_col, dropna=False):
        y_true = sub["y_true"].to_numpy(int)
        P = sub[[f"p_{c}" for c in classes]].to_numpy(float)
        yhat = P.argmax(axis=1)

        present = sorted(np.unique(y_true).tolist())
        n_present = int(len(present))

        acc = float(accuracy_score(y_true, yhat)) if len(y_true) else np.nan
        bal = float(balanced_accuracy_score(y_true, yhat)) if n_present >= 2 else np.nan

        f1_all = float(f1_score(y_true, yhat, average="macro", labels=list(range(n_classes)), zero_division=0)) if len(y_true) else np.nan
        f1_pres = float(f1_score(y_true, yhat, average="macro", labels=present, zero_division=0)) if n_present >= 1 else np.nan

        ll = float(log_loss(y_true, P, labels=list(range(n_classes)))) if len(y_true) else np.nan

        acc_ci = (np.nan, np.nan)
        bal_ci = (np.nan, np.nan)
        f1a_ci = (np.nan, np.nan)
        f1p_ci = (np.nan, np.nan)
        ll_ci  = (np.nan, np.nan)

        if BOOTSTRAP_CI and len(y_true) >= 10:
            acc_ci = bootstrap_ci_metric(y_true, P, lambda yt, PP: accuracy_score(yt, PP.argmax(axis=1)), BOOTSTRAP_N, BOOTSTRAP_SEED)
            if n_present >= 2:
                bal_ci = bootstrap_ci_metric(y_true, P, lambda yt, PP: balanced_accuracy_score(yt, PP.argmax(axis=1)), BOOTSTRAP_N, BOOTSTRAP_SEED)
            f1a_ci = bootstrap_ci_metric(y_true, P, lambda yt, PP: f1_score(yt, PP.argmax(axis=1), average="macro", labels=list(range(n_classes)), zero_division=0), BOOTSTRAP_N, BOOTSTRAP_SEED)
            f1p_ci = bootstrap_ci_metric(y_true, P, lambda yt, PP: f1_score(yt, PP.argmax(axis=1), average="macro", labels=sorted(np.unique(yt).tolist()), zero_division=0), BOOTSTRAP_N, BOOTSTRAP_SEED)
            ll_ci  = bootstrap_ci_metric(y_true, P, lambda yt, PP: log_loss(yt, PP, labels=list(range(n_classes))), BOOTSTRAP_N, BOOTSTRAP_SEED)

        counts = np.bincount(y_true, minlength=n_classes).tolist()

        out.append(dict(
            subgroup=subgroup_col,
            value=str(val),
            n=int(len(sub)),
            n_classes_present=n_present,
            present_classes_json=json.dumps([classes[i] for i in present], ensure_ascii=False),
            class_counts_json=json.dumps({classes[i]: int(counts[i]) for i in range(n_classes)}, ensure_ascii=False),

            acc=acc, acc_ci_lo=acc_ci[0], acc_ci_hi=acc_ci[1],
            bal_acc=bal, bal_ci_lo=bal_ci[0], bal_ci_hi=bal_ci[1],

            macro_f1_all=f1_all, f1_all_ci_lo=f1a_ci[0], f1_all_ci_hi=f1a_ci[1],
            macro_f1_present=f1_pres, f1_pres_ci_lo=f1p_ci[0], f1_pres_ci_hi=f1p_ci[1],

            logloss=ll, ll_ci_lo=ll_ci[0], ll_ci_hi=ll_ci[1],
        ))
    return pd.DataFrame(out)

def subgroup_confusions_table(df_preds: pd.DataFrame, classes: List[str], subgroup_col: str, min_n: int = 6):
    n_classes = len(classes)
    out = []
    for val, sub in df_preds.groupby(subgroup_col, dropna=False):
        if len(sub) < min_n:
            continue
        y_true = sub["y_true"].to_numpy(int)
        P = sub[[f"p_{c}" for c in classes]].to_numpy(float)
        yhat = P.argmax(axis=1)
        out.append(dict(
            subgroup=subgroup_col,
            value=str(val),
            n=int(len(sub)),
            cm_flat=confusion_flat(y_true, yhat, n_classes),
        ))
    return pd.DataFrame(out)

def _summary_stats_by_group(df_long: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    def q(x, p):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return np.nan
        return float(np.quantile(x, p))
    g = df_long.groupby(group_cols, dropna=False)
    out = g["value"].agg(
        n="count",
        n_nonnull=lambda s: int(np.sum(pd.notna(s))),
        n_missing=lambda s: int(np.sum(pd.isna(s))),
        mean=lambda s: float(np.nanmean(s)) if np.isfinite(s.astype(float)).any() else np.nan,
        std=lambda s: float(np.nanstd(s.astype(float), ddof=1)) if np.isfinite(s.astype(float)).sum() >= 2 else np.nan,
        median=lambda s: float(np.nanmedian(s.astype(float))) if np.isfinite(s.astype(float)).any() else np.nan,
        q1=lambda s: q(s, 0.25),
        q3=lambda s: q(s, 0.75),
    ).reset_index()
    out["iqr"] = out["q3"] - out["q1"]
    return out



def load_archived_phase2_hyperparameters(path: str, selected_rep: int) -> Dict[Tuple[int, int], Dict]:
    if not isinstance(path, str) or not path.strip() or not os.path.exists(path):
        raise FileNotFoundError(
            "LOCK_PHASE2_HYPERPARAMS_FROM_ARCHIVE=True requires ARCHIVED_FOLDS_CSV "
            "to point to the original selected multimodal_disease3_folds.csv."
        )
    df = read_csv_robust(path)
    need = {"rep", "fold", "phase2_C", "phase2_l1_ratio"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"ARCHIVED_FOLDS_CSV is missing columns: {sorted(missing)}")

    df = df.copy()
    df["rep"] = pd.to_numeric(df["rep"], errors="coerce").astype("Int64")
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce").astype("Int64")
    df = df[df["rep"] == int(selected_rep)].copy()
    if len(df) < 1:
        raise ValueError(f"No archived fold rows found for SELECTED_REP={selected_rep}")
    if df["fold"].duplicated().any():
        dup = df[df["fold"].duplicated()]["fold"].tolist()
        raise ValueError(f"Duplicate archived fold hyperparameter rows for rep={selected_rep}: {dup}")

    hp = {}
    for _, r in df.iterrows():
        C = pd.to_numeric(pd.Series([r["phase2_C"]]), errors="coerce").iloc[0]
        l1 = pd.to_numeric(pd.Series([r["phase2_l1_ratio"]]), errors="coerce").iloc[0]
        if pd.isna(C) or pd.isna(l1):
            raise ValueError(f"Bad archived phase2 hyperparameters at rep={r['rep']}, fold={r['fold']}")
        method = str(r["phase2_hp_method"]) if "phase2_hp_method" in df.columns else "inner_cv"
        hp[(int(r["rep"]), int(r["fold"]))] = {"C": float(C), "l1_ratio": float(l1), "method": method}
    return hp


def run_multimodal_disease3():
    set_global_reproducibility(SEED0)
    os.makedirs(OUT_DIR, exist_ok=True)
    classes = list(DISEASE_LABELS)
    n_classes = len(classes)

    
    out_csv = os.path.join(OUT_DIR, "multimodal_disease3_folds.csv")
    sum_csv = os.path.join(OUT_DIR, "multimodal_disease3_SUMMARY.csv")
    splits_csv = os.path.join(OUT_DIR, "multimodal_disease3_SPLITS.csv")
    preds_csv = os.path.join(OUT_DIR, "multimodal_disease3_PATIENT_PREDS.csv")
    coefs_long_csv = os.path.join(OUT_DIR, "multimodal_disease3_COEFS_LONG.csv")
    coef_agg_csv = os.path.join(OUT_DIR, "multimodal_disease3_COEF_AGG.csv")
    ablation_csv = os.path.join(OUT_DIR, "multimodal_disease3_ABLATION.csv")

   
    ci_ab_folds_csv   = os.path.join(OUT_DIR, "clinical_insight_ablation_folds.csv")
    ci_ab_sum_csv     = os.path.join(OUT_DIR, "clinical_insight_ablation_summary.csv")
    mis_csv           = os.path.join(OUT_DIR, "misclassified_full.csv")
    mis_exp_csv       = os.path.join(OUT_DIR, "misclassified_explanations_full.csv")
    subgroup_met_csv  = os.path.join(OUT_DIR, "subgroup_metrics_full.csv")
    subgroup_cm_csv   = os.path.join(OUT_DIR, "subgroup_confusions_full.csv")
    stage_diff_csv    = os.path.join(OUT_DIR, "stage_difficulty_full.csv")
    case_csv          = os.path.join(OUT_DIR, "case_studies_full_vs_raman.csv")
    violin_long_csv   = os.path.join(OUT_DIR, "violin_box_ready_outertest_long.csv")
    violin_sum_csv    = os.path.join(OUT_DIR, "violin_box_ready_outertest_summary.csv")
    miss_csv          = os.path.join(OUT_DIR, "clinical_missingness_outertest.csv")
    contrib_csv       = os.path.join(OUT_DIR, "patient_contributions_full_long.csv")

    # clean core outputs 
    for p in [out_csv, sum_csv, splits_csv, preds_csv, coefs_long_csv, coef_agg_csv, ablation_csv]:
        if os.path.exists(p):
            os.remove(p)
    # clean insight outputs
    for p in [ci_ab_folds_csv, ci_ab_sum_csv, mis_csv, mis_exp_csv, subgroup_met_csv, subgroup_cm_csv,
              stage_diff_csv, case_csv, violin_long_csv, violin_sum_csv, miss_csv, contrib_csv]:
        if os.path.exists(p):
            os.remove(p)

    
    df_r = read_csv_robust(RAMAN_CSV, dtype=None)
    df_r.columns = df_r.columns.astype(str).str.strip()
    df_r[RAMAN_SAMPLE_COL] = df_r[RAMAN_SAMPLE_COL].astype(str).str.strip()
    df_r[RAMAN_LABEL_COL]  = df_r[RAMAN_LABEL_COL].astype(str).str.strip()

    df_r = df_r[df_r[RAMAN_LABEL_COL].isin(classes)].copy()
    if df_r.empty:
        raise ValueError("After filtering Raman to disease-only labels, df is empty.")

    all_cols, wav = get_spec_cols_and_wav(df_r)

    sample_group_raw = df_r.groupby(RAMAN_SAMPLE_COL)[RAMAN_LABEL_COL].first()
    raman_samples_raw = sample_group_raw.index.values.astype(str)

   
    df_c = read_csv_robust(CLINICAL_CSV, dtype=str)
    df_c = preprocess_clinical_locked(df_c)
    df_c = reduce_clinical_to_one_row_per_sample(df_c)
    df_c[CLIN_SAMPLE_COL] = df_c[CLIN_SAMPLE_COL].astype(str).str.strip()
    clin_samples_raw = df_c[CLIN_SAMPLE_COL].values.astype(str)

    
    r_key, c_key, mode_used = build_sample_keys(raman_samples_raw, clin_samples_raw)
    raman_raw_to_key = dict(zip(raman_samples_raw.tolist(), r_key.tolist()))

    df_r["SampleKey"] = df_r[RAMAN_SAMPLE_COL].map(raman_raw_to_key).astype(str)
    df_c["SampleKey"] = c_key

    df_r = df_r[df_r["SampleKey"].astype(str) != ""].copy()
    df_c = df_c[df_c["SampleKey"].astype(str) != ""].copy()

    lab_nunique = df_r.groupby("SampleKey")[RAMAN_LABEL_COL].nunique()
    bad = lab_nunique[lab_nunique > 1]
    if len(bad) > 0:
        ex = bad.index[:10].tolist()
        raise ValueError(f"Raman label inconsistency for SampleKey. Examples: {ex}")

    samplekey_group = df_r.groupby("SampleKey")[RAMAN_LABEL_COL].first()
    samples_all = samplekey_group.index.values.astype(str)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    common = np.intersect1d(samples_all, df_c["SampleKey"].values.astype(str))
    if len(common) < 6:
        raise ValueError(f"Too few common patients between Raman and clinical: {len(common)}")

    samples = common
    yS = np.array([class_to_idx[samplekey_group.loc[s]] for s in samples], dtype=int)

    df_c_idx = df_c.set_index("SampleKey").loc[samples].reset_index()
    clinical_feature_cols, clin_num_cols, clin_cat_cols = get_clinical_feature_cols(df_c_idx)
    assert_no_forbidden_model_features(clinical_feature_cols, "Phase-2 clinical features")

    
    stage2_df = None
    use_stage2_alignment = False
    if isinstance(STAGE2_SPLITS_CSV, str) and STAGE2_SPLITS_CSV.strip() and os.path.exists(STAGE2_SPLITS_CSV):
        stage2_df = load_stage2_splits_csv(STAGE2_SPLITS_CSV)
        stage2_df = stage2_df[stage2_df["rep"] == int(SELECTED_REP)].copy()
        if len(stage2_df) < 1:
            raise RuntimeError(f"No saved split rows found for rep={SELECTED_REP}.")
        if stage2_df["fold"].duplicated().any():
            raise RuntimeError(f"Duplicate fold rows found for rep={SELECTED_REP}.")
        use_stage2_alignment = True
        all_known = set(samples.tolist())
        for _, rr in stage2_df.iterrows():
            if not set(rr["outer_train_list"]).issubset(all_known) or not set(rr["outer_test_list"]).issubset(all_known):
                raise RuntimeError("Stage-2 splits CSV has SampleKeys not in current aligned dataset.")

    rows = []
    coef_rows_all = []
    ablation_rows_simple = []

    # --- clinical insight accumulators (outer-test only) ---
    pred_rows_full_for_insight = []     # will be merged w clinical for subgroup/violin/mis
    mis_exp_rows = []                  # misclassified explanations
    contrib_rows = []                  # per-patient top-k contributions (long)
    case_rows = []                     # corrected by full vs raman_only
    ci_ablation_rows = []              # fold-wise ablation (same as ablation_rows_simple, but kept separate)

    total = int(len(stage2_df)) if use_stage2_alignment else (N_REPEATS * N_SPLITS_OUTER)
    pbar = tqdm(total=total, desc="Multimodal disease3 (repeat×fold)", dynamic_ncols=True)

    def _iter_splits():
        if use_stage2_alignment:
            for _, srow in stage2_df.iterrows():
                rep_num = int(srow["rep"])
                fold_id = int(srow["fold"])
                base_seed = (SEED0 + (rep_num - 1))
                outer_seed_used = int(srow["outer_seed_used"]) if "outer_seed_used" in srow and str(srow["outer_seed_used"]).strip() else np.nan
                outer_tries = int(srow["outer_tries"]) if "outer_tries" in srow and str(srow["outer_tries"]).strip() else np.nan
                outer_train_set = set(srow["outer_train_list"])
                outer_test_set  = set(srow["outer_test_list"])
                
                outer_train_keys = np.asarray(srow["outer_train_list"], dtype=str)
                outer_test_keys  = np.asarray(srow["outer_test_list"], dtype=str)
                yield rep_num, fold_id, base_seed, outer_seed_used, outer_tries, outer_train_keys, outer_test_keys
        else:
            for rep in range(N_REPEATS):
                base_seed = SEED0 + rep
                outer_gen = make_constrained_outer_splits(samples, yS, seed0=base_seed, n_classes=n_classes)
                for fold_id, tr_idx, te_idx, outer_seed_used, outer_tries in outer_gen:
                    yield (rep + 1), fold_id, base_seed, outer_seed_used, outer_tries, samples[tr_idx], samples[te_idx]

    if REQUIRE_LOCKED_REP and not use_stage2_alignment:
        raise RuntimeError(
            "Publication script requires STAGE2_SPLITS_CSV and a valid SELECTED_REP. "
            "It will not regenerate or search repeats."
        )

    archived_phase2_hp = {}
    if LOCK_PHASE2_HYPERPARAMS_FROM_ARCHIVE:
        archived_phase2_hp = load_archived_phase2_hyperparameters(ARCHIVED_FOLDS_CSV, SELECTED_REP)
        missing_hp = []
        for _, rr in stage2_df.iterrows():
            key = (int(rr["rep"]), int(rr["fold"]))
            if key not in archived_phase2_hp:
                missing_hp.append(key)
        if missing_hp:
            raise RuntimeError(f"Missing archived Phase-2 hyperparameters for folds: {missing_hp}")

  
    df_c_keyed = df_c_idx.set_index("SampleKey")

    for rep_num, fold_id, base_seed, outer_seed_used, outer_tries, outer_train_keys, outer_test_keys in _iter_splits():
        outer_train = set(outer_train_keys.tolist())
        outer_test  = set(outer_test_keys.tolist())

        # OOF Raman on outer-train
        y_outer_train = np.array([class_to_idx[samplekey_group.loc[s]] for s in outer_train_keys], dtype=int)
        oof_raman = build_oof_raman_features(
            df_raman=df_r,
            sample_keys_outer_train=outer_train_keys,
            y_outer_train=y_outer_train,
            classes=classes,
            all_cols=all_cols,
            wav=wav,
            seed=base_seed + 100 + fold_id,
        )

        # Raman features for outer-test
        te_raman = build_raman_features_for_test(
            df_raman=df_r,
            train_keys=outer_train,
            test_keys=outer_test,
            classes=classes,
            all_cols=all_cols,
            wav=wav,
            seed=base_seed + 500 + fold_id,
        )

        # Clinical train/test
        Xc_tr = df_c_keyed.loc[outer_train_keys, clinical_feature_cols].copy()
        Xc_te = df_c_keyed.loc[outer_test_keys, clinical_feature_cols].copy()
        y_tr = np.array([class_to_idx[samplekey_group.loc[s]] for s in Xc_tr.index.values], dtype=int)
        y_te = np.array([class_to_idx[samplekey_group.loc[s]] for s in Xc_te.index.values], dtype=int)

        # Merge Raman probs
        Xr_tr = oof_raman.loc[Xc_tr.index].copy()
        Xr_te = te_raman.loc[Xc_te.index].copy()

        # add optional Raman features (B & C) to BOTH train/test
        Xr_tr_aug = augment_raman_features_df(Xr_tr, classes=classes)
        Xr_te_aug = augment_raman_features_df(Xr_te, classes=classes)

        X_tr_full = pd.concat([Xc_tr, Xr_tr_aug], axis=1)
        X_te_full = pd.concat([Xc_te, Xr_te_aug], axis=1)

        # Phase-2 feature columns
        raman_base_cols = [c for c in Xr_tr.columns if c.startswith("raman_p_")]
        raman_extra_cols = [c for c in Xr_tr_aug.columns if c not in raman_base_cols]
        raman_extra_cols = [c for c in raman_extra_cols if c != "meta_fold"]
        raman_feat_cols = raman_base_cols + raman_extra_cols
        phase2_cols_full = clinical_feature_cols + raman_feat_cols
        assert_no_forbidden_model_features(raman_feat_cols, "Phase-2 Raman features")
        assert_no_forbidden_model_features(phase2_cols_full, "full Phase-2 model")

        # Inner CV feasibility
        minclass = int(np.min(np.bincount(y_tr, minlength=n_classes)))
        inner_splits = int(min(PHASE2_INNER_CV, minclass))
        if inner_splits < 2:
            inner_splits = 2 
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=int(base_seed) + 999)

        # FULL model robust fit
        num_cols_full = [c for c in clin_num_cols if c in phase2_cols_full] + raman_feat_cols
        cat_cols_full = [c for c in clin_cat_cols if c in phase2_cols_full]
        pre_full = build_phase2_preprocessor(num_cols=num_cols_full, cat_cols=cat_cols_full)

        archived_hp = archived_phase2_hp.get((int(rep_num), int(fold_id))) if LOCK_PHASE2_HYPERPARAMS_FROM_ARCHIVE else None
        pipe_full, hp_info = fit_phase2_pipeline_robust(
            pre=pre_full,
            Xtr=X_tr_full[phase2_cols_full],
            ytr=y_tr,
            inner_cv=inner_cv,
            seed=int(base_seed) + 999 + 100*fold_id,
            archived_hp=archived_hp,
        )

        P_te_full = pipe_full.predict_proba(X_te_full[phase2_cols_full])

        acc, bal, f1m, auc = metrics_multiclass(y_te, P_te_full, n_classes=n_classes)
        ll = float(log_loss(y_te, P_te_full, labels=list(range(n_classes))))
        yhat_full = P_te_full.argmax(axis=1).astype(int)
        cm = confusion_matrix(y_te, yhat_full, labels=np.arange(n_classes)).flatten().tolist()

        row = dict(
            rep=int(rep_num),
            fold=int(fold_id),
            samplekey_mode=mode_used,
            base_seed=int(base_seed),
            outer_seed_used=int(outer_seed_used) if outer_seed_used == outer_seed_used else np.nan,
            outer_tries=int(outer_tries) if outer_tries == outer_tries else np.nan,
            n_train_patients=int(len(outer_train)),
            n_test_patients=int(len(outer_test)),
            train_counts_json=class_counts_str(samplekey_group.loc[list(outer_train)].values.astype(str), classes),
            test_counts_json=class_counts_str(samplekey_group.loc[list(outer_test)].values.astype(str), classes),
            w_all=float(FUSION_W["all"]),
            w_fp=float(FUSION_W["fp"]),
            w_ch=float(FUSION_W["ch"]),
            add_raman_logit=int(bool(ADD_RAMAN_LOGIT_FEATURES)),
            add_raman_pairwise=int(bool(ADD_RAMAN_PAIRWISE_LOGODDS)),
            add_raman_meta=int(bool(ADD_RAMAN_META_FEATURES)),
            acc=float(acc),
            bal_acc=float(bal),
            macro_f1=float(f1m),
            auc_ovr=float(auc) if auc == auc else np.nan,
            logloss=float(ll),
            cm_flat=",".join(map(str, cm)),
            phase2_hp_method=str(hp_info.get("method", "")),
            phase2_fit_mode=str(hp_info.get("fit_mode", "")),
            phase2_C=float(hp_info.get("C", np.nan)) if hp_info.get("C", np.nan) == hp_info.get("C", np.nan) else np.nan,
            phase2_l1_ratio=float(hp_info.get("l1_ratio", np.nan)) if hp_info.get("l1_ratio", np.nan) == hp_info.get("l1_ratio", np.nan) else np.nan,
            n_phase2_features=int(len(phase2_cols_full)),
        )
        rows.append(row)
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

        # splits
        if SAVE_SPLITS:
            srow = dict(
                rep=int(rep_num),
                fold=int(fold_id),
                base_seed=int(base_seed),
                outer_seed_used=int(outer_seed_used) if outer_seed_used == outer_seed_used else np.nan,
                outer_tries=int(outer_tries) if outer_tries == outer_tries else np.nan,
                outer_train_n=int(len(outer_train)),
                outer_test_n=int(len(outer_test)),
                outer_train_samples_json=_set_json(outer_train),
                outer_test_samples_json=_set_json(outer_test),
            )
            pd.DataFrame([srow]).to_csv(splits_csv, mode="a", header=not os.path.exists(splits_csv), index=False)

        # patient preds  + accumulate for insights
        if SAVE_PATIENT_PREDS:
            dfp = pd.DataFrame({
                "SampleKey": X_te_full.index.values.astype(str),
                "rep": int(rep_num),
                "fold": int(fold_id),
                "y_true": y_te.astype(int),
                "y_true_name": [classes[int(i)] for i in y_te],
            })
            for j, cname in enumerate(classes):
                dfp[f"p_{cname}"] = P_te_full[:, j]
            dfp["y_pred"] = yhat_full
            dfp["y_pred_name"] = [classes[int(i)] for i in dfp["y_pred"].values]
            dfp["correct"] = (dfp["y_pred"].values == dfp["y_true"].values).astype(int)
            dfp["margin"] = _margin(P_te_full)
            dfp["entropy"] = _entropy(P_te_full)

            for cname in classes:
                rc = f"raman_p_{cname}"
                dfp[rc] = Xr_te.loc[dfp["SampleKey"].values, rc].values if rc in Xr_te.columns else np.nan

            if ADD_RAMAN_META_FEATURES:
                for c in ["raman_entropy", "raman_margin", "raman_maxprob"]:
                    if c in Xr_te_aug.columns:
                        dfp[c] = Xr_te_aug.loc[dfp["SampleKey"].values, c].values

            dfp.to_csv(preds_csv, mode="a", header=not os.path.exists(preds_csv), index=False)

            # accumulate for later clinical merge + insight outputs
            pred_rows_full_for_insight.append(dfp.copy())

        # coefs 
        if SAVE_COEFS:
            dcoef = extract_coefs_long(pipe_full, classes=classes)
            dcoef.insert(0, "rep", int(rep_num))
            dcoef.insert(1, "fold", int(fold_id))
            dcoef.to_csv(coefs_long_csv, mode="a", header=not os.path.exists(coefs_long_csv), index=False)
            coef_rows_all.append(dcoef)

        
        try:
            pre = pipe_full.named_steps["pre"]
            clf = pipe_full.named_steps["clf"]
            feat_names = pre.get_feature_names_out()
            Z_te = pre.transform(X_te_full[phase2_cols_full])
            Z_te = np.asarray(Z_te, dtype=float)
            coefs = clf.coef_.copy()
           
        except Exception:
            Z_te = None
            feat_names = None
            coefs = None

        if Z_te is not None and feat_names is not None and coefs is not None:
            keys_test = X_te_full.index.values.astype(str)
            for i, sk in enumerate(keys_test):
                true_k = int(y_te[i])
                pred_k = int(yhat_full[i])
                correct = int(true_k == pred_k)

                # contribution long rows: predicted + true class
                for which, kcls in [("pred", pred_k), ("true", true_k)]:
                    contrib = Z_te[i] * coefs[kcls]
                    idx = np.argsort(-np.abs(contrib))[:TOPK_EXPLAIN]
                    for j in idx:
                        contrib_rows.append(dict(
                            rep=int(rep_num),
                            fold=int(fold_id),
                            SampleKey=str(sk),
                            which=which,
                            class_name=classes[kcls],
                            feature=str(feat_names[j]),
                            contribution=float(contrib[j]),
                            z_value=float(Z_te[i, j]),
                            coef=float(coefs[kcls, j]),
                            correct=int(correct),
                            y_true_name=classes[true_k],
                            y_pred_name=classes[pred_k],
                        ))

                # misclassified explanation json
                if correct == 0:
                    top_pred = _topk_contrib(feat_names, Z_te[i], coefs[pred_k], TOPK_EXPLAIN)
                    top_true = _topk_contrib(feat_names, Z_te[i], coefs[true_k], TOPK_EXPLAIN)
                    mis_exp_rows.append(dict(
                        rep=int(rep_num),
                        fold=int(fold_id),
                        SampleKey=str(sk),
                        y_true_name=classes[true_k],
                        y_pred_name=classes[pred_k],
                        top_pred_features=json.dumps(top_pred, ensure_ascii=False),
                        top_true_features=json.dumps(top_true, ensure_ascii=False),
                    ))

        # ablation  + case studies via raman_only compare
        P_raman_only = None
        yhat_raman_only = None

        if RUN_ABLATION:
            specs = build_ablation_specs(clinical_feature_cols, raman_feat_cols)
            for ab_name, sp in specs.items():
                use_clin = bool(sp["use_clinical"])
                use_ram  = bool(sp["use_raman"])
                drop_cols = list(sp.get("drop_cols", []))

                feat_cols = []
                if use_clin:
                    feat_cols += [c for c in clinical_feature_cols if c not in set(drop_cols)]
                if use_ram:
                    feat_cols += raman_feat_cols
                if len(feat_cols) == 0:
                    continue
                assert_no_forbidden_model_features(
                    feat_cols, f"Phase-2 ablation '{ab_name}'"
                )

                clin_used = [c for c in feat_cols if c in clinical_feature_cols]
                num_used, cat_used = _split_num_cat(clin_used, clin_num_cols, clin_cat_cols)
                if use_ram:
                    num_used = num_used + [c for c in feat_cols if c in raman_feat_cols]  # all numeric

                pre = build_phase2_preprocessor(num_cols=num_used, cat_cols=cat_used)
                pipe_ab, hp_ab = fit_phase2_pipeline_robust(
                    pre=pre,
                    Xtr=X_tr_full[feat_cols],
                    ytr=y_tr,
                    inner_cv=inner_cv,
                    seed=int(base_seed) + 222 + 100*fold_id,
                )

                P_te = pipe_ab.predict_proba(X_te_full[feat_cols])
                accx, balx, f1x, aucx = metrics_multiclass(y_te, P_te, n_classes=n_classes)
                llx = float(log_loss(y_te, P_te, labels=list(range(n_classes))))

                ab_row = dict(
                    rep=int(rep_num), fold=int(fold_id), model=ab_name,
                    add_raman_logit=int(bool(ADD_RAMAN_LOGIT_FEATURES)),
                    add_raman_pairwise=int(bool(ADD_RAMAN_PAIRWISE_LOGODDS)),
                    add_raman_meta=int(bool(ADD_RAMAN_META_FEATURES)),
                    acc=float(accx), bal_acc=float(balx), macro_f1=float(f1x),
                    auc_ovr=float(aucx) if aucx == aucx else np.nan,
                    logloss=float(llx),
                    n_test_patients=int(len(outer_test)),
                    hp_method=str(hp_ab.get("method","")),
                    C=float(hp_ab.get("C", np.nan)),
                    l1_ratio=float(hp_ab.get("l1_ratio", np.nan)),
                )
                ablation_rows_simple.append(ab_row)
                ci_ablation_rows.append(ab_row)

                if ab_name == "raman_only":
                    P_raman_only = P_te
                    yhat_raman_only = P_te.argmax(axis=1).astype(int)

        # case studies: corrected by full vs raman_only (outer-test only)
        if P_raman_only is not None and yhat_raman_only is not None:
            keys_test = X_te_full.index.values.astype(str)
            for i, sk in enumerate(keys_test):
                corr_full = int(y_te[i] == yhat_full[i])
                corr_r = int(y_te[i] == yhat_raman_only[i])
                if corr_full != corr_r:
                    case_rows.append(dict(
                        rep=int(rep_num),
                        fold=int(fold_id),
                        SampleKey=str(sk),
                        y_true=classes[int(y_te[i])],
                        pred_full=classes[int(yhat_full[i])],
                        pred_raman=classes[int(yhat_raman_only[i])],
                        full_correct=int(corr_full),
                        raman_correct=int(corr_r),
                        full_p=json.dumps({classes[j]: float(P_te_full[i, j]) for j in range(n_classes)}, ensure_ascii=False),
                        raman_p=json.dumps({classes[j]: float(P_raman_only[i, j]) for j in range(n_classes)}, ensure_ascii=False),
                    ))

        tqdm.write(
            f"[rep {rep_num} fold {fold_id}] "
            f"acc={acc:.3f} bal={bal:.3f} f1={f1m:.3f} ll={ll:.3f} "
            f"hp={row['phase2_hp_method']} C={row['phase2_C']:.4g} l1={row['phase2_l1_ratio']:.2f} "
            f"| B(logit)={row['add_raman_logit']} B(ratio)={row['add_raman_pairwise']} C(meta)={row['add_raman_meta']} "
            f"feat={row['n_phase2_features']}"
        )
        pbar.update(1)

    pbar.close()

    res = pd.DataFrame(rows)

    # coef agg (existing)
    if SAVE_COEFS and len(coef_rows_all) > 0:
        allc = pd.concat(coef_rows_all, ignore_index=True)
        allc["nonzero"] = (np.abs(allc["coef"].values) > 1e-8).astype(int)
        agg = allc.groupby(["class", "feature"], as_index=False).agg(
            coef_mean=("coef", "mean"),
            coef_std=("coef", "std"),
            nonzero_freq=("nonzero", "mean"),
            n=("coef", "count"),
        )
        agg.to_csv(coef_agg_csv, index=False, encoding=CSV_ENCODING)

    # ablation save (existing)
    if RUN_ABLATION and len(ablation_rows_simple) > 0:
        pd.DataFrame(ablation_rows_simple).to_csv(ablation_csv, index=False, encoding=CSV_ENCODING)

    # summary (existing)
    summary = dict(
        samplekey_mode=mode_used,
        n_rows=int(len(res)),
        add_raman_logit=int(bool(ADD_RAMAN_LOGIT_FEATURES)),
        add_raman_pairwise=int(bool(ADD_RAMAN_PAIRWISE_LOGODDS)),
        add_raman_meta=int(bool(ADD_RAMAN_META_FEATURES)),
        acc_mean=float(res["acc"].mean()) if len(res) else float("nan"),
        acc_std=float(res["acc"].std()) if len(res) else float("nan"),
        bal_mean=float(res["bal_acc"].mean()) if len(res) else float("nan"),
        bal_std=float(res["bal_acc"].std()) if len(res) else float("nan"),
        f1_mean=float(res["macro_f1"].mean()) if len(res) else float("nan"),
        f1_std=float(res["macro_f1"].std()) if len(res) else float("nan"),
        logloss_mean=float(res["logloss"].mean()) if len(res) else float("nan"),
        auc_mean=float(res["auc_ovr"].dropna().mean()) if len(res) and res["auc_ovr"].notna().any() else float("nan"),
        hp_methods_counts=res["phase2_hp_method"].value_counts(dropna=False).to_dict() if len(res) else {},
        phase2_fit_modes_counts=res["phase2_fit_mode"].value_counts(dropna=False).to_dict() if len(res) and "phase2_fit_mode" in res.columns else {},
        selected_rep=int(SELECTED_REP),
        locked_rep=bool(REQUIRE_LOCKED_REP),
        stage2_splits_csv=str(STAGE2_SPLITS_CSV),
        archived_folds_csv=str(ARCHIVED_FOLDS_CSV),
        lock_phase2_hyperparams_from_archive=bool(LOCK_PHASE2_HYPERPARAMS_FROM_ARCHIVE),
        fallback_default=dict(C=FALLBACK_DEFAULT_C, l1_ratio=FALLBACK_DEFAULT_L1RATIO),
        fallback_tuner=dict(n_splits=FALLBACK_TUNE_N_SPLITS, test_frac=FALLBACK_TUNE_TEST_FRAC),
        enable_stagegroup=bool(ENABLE_STAGEGROUP),
        stagegroup_rule=str(STAGEGROUP_RULE),
    )
    pd.DataFrame([summary]).to_csv(sum_csv, index=False, encoding=CSV_ENCODING)

  
    # 1) contributions + mis_explanations + case studies
    if len(contrib_rows) > 0:
        to_csv_safe(pd.DataFrame(contrib_rows), contrib_csv)
    if len(mis_exp_rows) > 0:
        to_csv_safe(pd.DataFrame(mis_exp_rows), mis_exp_csv)
    if len(case_rows) > 0:
        to_csv_safe(pd.DataFrame(case_rows), case_csv)

    # 2) clinical insight ablation folds + summary (separate files)
    if len(ci_ablation_rows) > 0:
        df_ci_ab = pd.DataFrame(ci_ablation_rows)
        to_csv_safe(df_ci_ab, ci_ab_folds_csv)

        summ_rows = []
        for m, sub in df_ci_ab.groupby("model"):
            summ_rows.append(dict(
                model=m,
                n_rows=int(len(sub)),
                acc_mean=float(sub["acc"].mean()),
                acc_std=float(sub["acc"].std()),
                bal_mean=float(sub["bal_acc"].mean()),
                bal_std=float(sub["bal_acc"].std()),
                f1_mean=float(sub["macro_f1"].mean()),
                f1_std=float(sub["macro_f1"].std()),
                logloss_mean=float(sub["logloss"].mean()),
                logloss_std=float(sub["logloss"].std()),
            ))
        to_csv_safe(pd.DataFrame(summ_rows).sort_values("model"), ci_ab_sum_csv)

    # 3) Build merged outer-test table with clinical columns for subgroup/violin/misclassified
    if len(pred_rows_full_for_insight) > 0:
        df_preds = pd.concat(pred_rows_full_for_insight, ignore_index=True)

       
        df_clin_all = df_c.copy()
        # ensure normalized stage + stagegroup
        if STAGE_COL in df_clin_all.columns:
            df_clin_all[STAGE_COL] = df_clin_all[STAGE_COL].apply(normalize_stage_value_strict)
        df_clin_all["StageNorm"] = df_clin_all[STAGE_COL].apply(normalize_stage_value_strict) if STAGE_COL in df_clin_all.columns else "Unknown"
        df_clin_all["StageGroup"] = df_clin_all["StageNorm"].apply(stagegroup_from_stage) if ENABLE_STAGEGROUP else "Disabled"

        # keep clinical columns for merge (everything except raw sample col maybe)
        clin_keep = ["SampleKey"] + list({*NUMERIC_CLIN_COLS, *INSIGHT_EXTRA_NUMERIC_COLS, *SUBGROUP_COLS_TO_EXPORT, "StageNorm", "StageGroup"})
        clin_keep = [c for c in clin_keep if c in df_clin_all.columns]
        df_clin_keep = df_clin_all[clin_keep].drop_duplicates("SampleKey")

        df_full = df_preds.merge(df_clin_keep, on="SampleKey", how="left")

        # fill subgroup columns with Unknown if missing
        for c in SUBGROUP_COLS_TO_EXPORT:
            if c not in df_full.columns:
                df_full[c] = "Unknown"
            df_full[c] = df_full[c].fillna("Unknown").astype(str)
            df_full.loc[df_full[c].str.strip().eq(""), c] = "Unknown"

        if "StageNorm" not in df_full.columns:
            df_full["StageNorm"] = "Unknown"
        if "StageGroup" not in df_full.columns:
            df_full["StageGroup"] = "Disabled"

        # misclassified_full.csv
        df_mis = df_full[df_full["correct"].astype(int) == 0].copy()
        to_csv_safe(df_mis, mis_csv)

        # subgroup metrics/confusions (outer-test only)
        subgroup_cols = [
            "Viral Hepatitis Status",
            "Smoking History",
            "Alcohol Consumption",
            "Diabetes Mellitus",
            SEX_COL,
            "StageNorm",
        ]
        if ENABLE_STAGEGROUP:
            subgroup_cols.append("StageGroup")

        all_sub = []
        all_cm = []
        for col in subgroup_cols:
            if col not in df_full.columns:
                df_full[col] = "Unknown"
            all_sub.append(subgroup_metrics_table(df_full, classes, col))
            all_cm.append(subgroup_confusions_table(df_full, classes, col, min_n=6))

        if len(all_sub) > 0:
            to_csv_safe(pd.concat(all_sub, ignore_index=True), subgroup_met_csv)
        if len(all_cm) > 0:
            to_csv_safe(pd.concat(all_cm, ignore_index=True), subgroup_cm_csv)

        # stage difficulty (outer-test only)
        def stage_diff_table(group_col: str) -> pd.DataFrame:
            tmp = df_full.copy()
            return (
                tmp.groupby(group_col, dropna=False)
                .agg(
                    n=("SampleKey", "count"),
                    error_rate=("correct", lambda s: 1.0 - float(np.mean(s.astype(int)))),
                    entropy_mean=("entropy", "mean"),
                    entropy_std=("entropy", "std"),
                    margin_mean=("margin", "mean"),
                    margin_std=("margin", "std"),
                )
                .reset_index()
                .rename(columns={group_col: "group"})
            )

        st1 = stage_diff_table("StageNorm")
        if ENABLE_STAGEGROUP:
            st2 = stage_diff_table("StageGroup")
            st = pd.concat([st1.assign(group_type="StageNorm"), st2.assign(group_type="StageGroup")], ignore_index=True)
        else:
            st = st1.assign(group_type="StageNorm")
        to_csv_safe(st, stage_diff_csv)

        # violin/box plot-ready (outer-test only)
        numeric_cols_present = [c for c in (NUMERIC_CLIN_COLS + INSIGHT_EXTRA_NUMERIC_COLS) if c in df_full.columns]
        raman_prob_cols = [f"raman_p_{c}" for c in classes if f"raman_p_{c}" in df_full.columns]
        prob_cols = [f"p_{c}" for c in classes if f"p_{c}" in df_full.columns]
        extra_cols = ["entropy", "margin"]
        value_cols = numeric_cols_present + raman_prob_cols + prob_cols + extra_cols

        id_cols = [
            "rep", "fold", "SampleKey",
            "y_true_name", "y_pred_name", "correct",
            SEX_COL, "StageNorm", "StageGroup",
            "Viral Hepatitis Status", "Smoking History", "Alcohol Consumption", "Diabetes Mellitus",
        ]
        id_cols = [c for c in id_cols if c in df_full.columns]

        df_long = df_full[id_cols + value_cols].melt(
            id_vars=id_cols,
            value_vars=value_cols,
            var_name="feature",
            value_name="value"
        )
        to_csv_safe(df_long, violin_long_csv)

        df_sumplot = _summary_stats_by_group(df_long, group_cols=["y_true_name", "feature"])
        to_csv_safe(df_sumplot, violin_sum_csv)

        # missingness by class for numeric clinical features
        miss_rows = []
        for feat in numeric_cols_present:
            for cls in classes:
                sub = df_full[df_full["y_true_name"] == cls]
                n = int(len(sub))
                if n == 0:
                    continue
                nmiss = int(pd.isna(sub[feat]).sum())
                miss_rows.append(dict(
                    y_true_name=cls,
                    feature=feat,
                    n=n,
                    n_missing=nmiss,
                    missing_rate=float(nmiss / max(1, n)),
                ))
        to_csv_safe(pd.DataFrame(miss_rows), miss_csv)

   
    print("Saved:", out_csv)
    print("Saved:", sum_csv)
    if SAVE_SPLITS: print("Saved:", splits_csv)
    if SAVE_PATIENT_PREDS: print("Saved:", preds_csv)
    if SAVE_COEFS:
        print("Saved:", coefs_long_csv)
        print("Saved:", coef_agg_csv)
    if RUN_ABLATION:
        print("Saved:", ablation_csv)

    # insight outputs (may be empty depending on data/options)
    for p in [ci_ab_folds_csv, ci_ab_sum_csv, mis_csv, mis_exp_csv, subgroup_met_csv, subgroup_cm_csv,
              stage_diff_csv, case_csv, violin_long_csv, violin_sum_csv, miss_csv, contrib_csv]:
        if os.path.exists(p):
            print("Saved:", p)

    print("SUMMARY:", json.dumps(summary, indent=2, ensure_ascii=False))
    return res

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    run_multimodal_disease3()

if __name__ == "__main__":
    main()
