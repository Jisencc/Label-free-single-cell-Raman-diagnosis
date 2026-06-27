#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, confusion_matrix, recall_score

import joblib

RAMAN_CSV = "../raman_added2.csv"
CLINICAL_CSV = "../CLINICAL.csv"
OUT_DIR = "./publication_MIL_disease3_reproduction_out_CV4"

LOCKED_INPUT_DIR = "./publication_MIL_clinical_disease3_inputs_CV4"
LOCKED_RUNS_CSV = os.path.join(LOCKED_INPUT_DIR, "selected_rep_MIL_DISEASE3_LOCKED_RUNS.csv")
CHECKPOINT_DIR = os.path.join(LOCKED_INPUT_DIR, "checkpoints")
EXPECTED_SELECTED_REP = 5  # fill with the selected rep used to create the locked input folder

DISEASE_LABELS = ["Liver", "Bile", "Pancreas"]
RAMAN_SAMPLE_COL = "Sample"
RAMAN_LABEL_COL = "Group"
CLIN_SAMPLE_COL = "Sample"

VIEWS = [("all", "all"), ("fp", "fingerprint"), ("ch", "ch")]
USE_VIEW_FUSION = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_ENCODING = "utf-8-sig"


SAVE_ATTN_EXPORT = True
ATTN_EXPORT_TOPK = 200
ATTN_CELL_ID_COL = "Cell"
ATTN_DIRNAME = "attn"
COPY_LOCKED_AUDIT_ARTIFACTS_TO_OUT = True  # copy selected tuned/checkpoint artifacts to OUT_DIR for a complete selected-output folder


LT_MULTIPLIER = 0.5
GT_MULTIPLIER = 1.0
MISSING_TOKENS = {
    "", "na", "n/a", "nan", "none", "null", "-", "--",
    "nd", "n.d.", "not detected", "not_detected", "missing"
}
NUMERIC_CLIN_COLS = [
    "Age", "AFP(ng/ml)", "CA19-9(U/ml)", "CEA(ng/ml)",
    "ALT(U/L)", "AST(U/L)", "Total Bilirubin(umol/L)", "Total Bilirubin(μmol/L)",
]
CATEGORICAL_CLIN_COLS = [
    "Sex", "Stage", "Smoking History", "Alcohol Consumption",
    "Diabetes Mellitus", "Viral Hepatitis Status",
]


def configure_torch_runtime(deterministic: bool = False, allow_tf32: bool = False):
    try:
        torch.use_deterministic_algorithms(bool(deterministic))
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    except Exception:
        pass

configure_torch_runtime(deterministic=False, allow_tf32=False)

@dataclass
class MILParams:
    embed_dim: int = 128
    attn_dim: int = 128
    clin_embed_dim: int = 64
    dropout: float = 0.35
    lr: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 120
    max_instances_train: Optional[int] = 128

class AttnMILMultiModal(nn.Module):
    def __init__(self, in_dim: int, clin_in_dim: int, n_classes: int,
                 embed_dim: int, attn_dim: int, clin_embed_dim: int, dropout: float):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.attn = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1))
        self.clin = nn.Sequential(
            nn.Linear(clin_in_dim, clin_embed_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(clin_embed_dim, clin_embed_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim + clin_embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, x_pad: torch.Tensor, mask: torch.Tensor, c: torch.Tensor):
        H = self.embed(x_pad)
        a = self.attn(H).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf"))
        w = torch.softmax(a, dim=1)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        M = torch.sum(H * w.unsqueeze(-1), dim=1)
        C = self.clin(c)
        return self.head(torch.cat([M, C], dim=1))

    def forward_with_attn(self, x_pad: torch.Tensor, mask: torch.Tensor, c: torch.Tensor):
        H = self.embed(x_pad)
        a = self.attn(H).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf"))
        w = torch.softmax(a, dim=1)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        M = torch.sum(H * w.unsqueeze(-1), dim=1)
        C = self.clin(c)
        return self.head(torch.cat([M, C], dim=1)), w


def read_csv_robust(path: str, dtype=None) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(path, dtype=dtype, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=dtype)

def to_csv_safe(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)

_RE_NUM_DOT0 = re.compile(r"^\s*(\d+)\.0\s*$")
def _norm_id_basic(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    m = _RE_NUM_DOT0.match(s)
    return m.group(1) if m else s

def _norm_id_drop_leading_zeros_if_numeric(s: str) -> str:
    if s and s.isdigit():
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

def get_spec_cols_and_wav(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    cols, wav = [], []
    for c in df.columns:
        try:
            wav.append(float(c)); cols.append(c)
        except Exception:
            pass
    order = np.argsort(wav)
    return [cols[i] for i in order], np.array([wav[i] for i in order], dtype=float)

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
    if not out:
        raise ValueError(f"No spectral columns selected for region={region}")
    return out

def apply_temperature_multiclass(P: np.ndarray, T: float) -> np.ndarray:
    P = np.clip(P, 1e-12, 1.0)
    logits = np.log(P) / float(T)
    logits -= logits.max(axis=1, keepdims=True)
    expv = np.exp(logits)
    return expv / (expv.sum(axis=1, keepdims=True) + 1e-12)

def entropy(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, 1.0)
    return -np.sum(P * np.log(P), axis=1)

def margin_true_vs_best_other(y_true: np.ndarray, P: np.ndarray) -> np.ndarray:
    out = np.zeros(len(y_true), dtype=float)
    for i in range(len(y_true)):
        out[i] = float(P[i, int(y_true[i])] - np.max(np.delete(P[i], int(y_true[i]))))
    return out

def class_counts_str(y_str: np.ndarray, classes: List[str]) -> str:
    return json.dumps({c: int(np.sum(y_str == c)) for c in classes}, ensure_ascii=False)

_RE_INEQ = re.compile(r"^\s*(<=|>=|<|>)\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$")
_RE_NUM = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
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
        v = float(m.group(2))
        return float(lt_mult) * v if m.group(1) in ("<", "<=") else float(gt_mult) * v
    m2 = _RE_RANGE.match(s)
    if m2:
        return 0.5 * (float(m2.group(1)) + float(m2.group(2)))
    if _RE_NUM.match(s):
        return float(s)
    s2 = re.sub(r"[^0-9eE\.\+\-]", "", s)
    return float(s2) if _RE_NUM.match(s2) else np.nan

def preprocess_clinical_locked(df_clin: pd.DataFrame) -> pd.DataFrame:
    df = df_clin.copy()
    df.columns = df.columns.astype(str).str.strip()
    if CLIN_SAMPLE_COL not in df.columns:
        raise ValueError(f"Clinical CSV missing {CLIN_SAMPLE_COL!r}")
    df[CLIN_SAMPLE_COL] = df[CLIN_SAMPLE_COL].astype(str).str.strip()
    for col in NUMERIC_CLIN_COLS:
        if col in df.columns:
            df[col] = df[col].apply(parse_numeric_robust).astype(float)
    for col in CATEGORICAL_CLIN_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"}).fillna("Unknown")
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
        return s2.iloc[0] if len(s2) else s.iloc[0]
    agg = {c: ("median" if c in NUMERIC_CLIN_COLS else first_non_unknown) for c in df.columns if c != CLIN_SAMPLE_COL}
    return df.groupby(CLIN_SAMPLE_COL, as_index=False).agg(agg)

def build_bags_from_df(df: pd.DataFrame, spec_cols: List[str]) -> Dict[str, np.ndarray]:
    bags = {}
    for sid, g in df.groupby("SampleKey", sort=False):
        X = g[spec_cols].to_numpy(np.float32)
        if X.ndim == 2 and X.shape[0] > 0:
            bags[str(sid)] = X
    return bags

class MILBagsClinDataset(Dataset):
    def __init__(self, sample_ids, bags, y_by_sample, clin_X_by_sample, mean, std):
        self.sample_ids = list(sample_ids)
        self.bags = bags
        self.y_by_sample = y_by_sample
        self.clin_X_by_sample = clin_X_by_sample
        self.mean = mean
        self.std = std
    def __len__(self):
        return len(self.sample_ids)
    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        X = (self.bags[sid] - self.mean) / (self.std + 1e-8)
        c = self.clin_X_by_sample[sid]
        y = int(self.y_by_sample[sid])
        return torch.from_numpy(X).float(), torch.from_numpy(c).float(), torch.tensor(y, dtype=torch.long), sid

def collate_pad(batch):
    xs, cs, ys, sids = zip(*batch)
    lens = [x.shape[0] for x in xs]
    D = xs[0].shape[1]
    T = max(lens)
    B = len(xs)
    x_pad = torch.zeros((B, T, D), dtype=torch.float32)
    mask = torch.zeros((B, T), dtype=torch.bool)
    c = torch.stack(cs, dim=0)
    y = torch.stack(ys, dim=0)
    for i, x in enumerate(xs):
        t = x.shape[0]
        x_pad[i, :t] = x
        mask[i, :t] = True
    return x_pad, mask, c, y, list(sids)

def build_clin_matrix_dict(df_clin_keyed: pd.DataFrame, pre, feat_cols: List[str], sample_list: List[str]) -> Dict[str, np.ndarray]:
    X = pre.transform(df_clin_keyed.loc[sample_list, feat_cols])
    X = np.asarray(X, dtype=np.float32)
    return {sid: X[i] for i, sid in enumerate(sample_list)}

@torch.no_grad()
def predict_probs_multimodal_mil(model, bags, y_by_sample, clin_X_by_sample, sample_list, mean, std, batch_size, device):
    ds = MILBagsClinDataset(sample_list, bags, y_by_sample, clin_X_by_sample, mean, std)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_pad)
    model.eval()
    all_ids, all_p = [], []
    for x_pad, mask, c, y, sids in dl:
        logits = model(x_pad.to(device), mask.to(device), c.to(device))
        P = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)
        all_ids.extend(sids)
        all_p.append(P)
    return all_ids, np.concatenate(all_p, axis=0) if all_p else np.zeros((0, 0), dtype=np.float64)

@torch.no_grad()
def extract_attn_multimodal_mil(model, bags, y_by_sample, clin_X_by_sample, sample_list, mean, std, batch_size, device):
    ds = MILBagsClinDataset(sample_list, bags, y_by_sample, clin_X_by_sample, mean, std)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_pad)
    model.eval()
    out = {}
    for x_pad, mask, c, y, sids in dl:
        _, w = model.forward_with_attn(x_pad.to(device), mask.to(device), c.to(device))
        w = w.detach().cpu().numpy().astype(np.float64)
        mask_np = mask.detach().cpu().numpy().astype(bool)
        for i, sid in enumerate(sids):
            out[str(sid)] = w[i][mask_np[i]]
    return out

def build_cell_ids_by_samplekey(df_r: pd.DataFrame) -> Dict[str, List[str]]:
    out = {}
    use_col = ATTN_CELL_ID_COL in df_r.columns
    for sid, g in df_r.groupby("SampleKey", sort=False):
        out[str(sid)] = g[ATTN_CELL_ID_COL].astype(str).tolist() if use_col else [str(i) for i in range(len(g))]
    return out

def attn_entropy_1d(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 1e-12, 1.0)
    w = w / (w.sum() + 1e-12)
    return float(-(w * np.log(w)).sum())

def export_attention_topk_and_summary(
    *, rep: int, fold: int, view_name: str, sample_list: List[str],
    attn_by_sample: Dict[str, np.ndarray], cell_ids_by_sample: Dict[str, List[str]],
    y_true_by_sample: Dict[str, int], y_pred_by_sample: Dict[str, int],
    prob_by_sample: Dict[str, np.ndarray], classes: List[str],
    out_cells_csv: str, out_summary_csv: str, topk: int,
):
    rows_cells, rows_sum = [], []
    for sid in sample_list:
        w = attn_by_sample.get(sid)
        ids = cell_ids_by_sample.get(sid)
        if w is None or ids is None:
            continue
        w = np.asarray(w, dtype=float)
        n = int(w.shape[0])
        if n == 0:
            continue
        if len(ids) != n:
            m = min(len(ids), n)
            w = w[:m]
            ids = ids[:m]
            n = m
        k = int(max(1, min(int(topk), n)))
        idx = np.argsort(-w)[:k]
        P = prob_by_sample[sid]
        for rnk, j in enumerate(idx, 1):
            prow = {
                "rep": int(rep), "fold": int(fold), "view": str(view_name),
                "SampleKey": str(sid), "cell_id": str(ids[int(j)]),
                "attn_weight": float(w[int(j)]), "attn_rank": int(rnk), "n_cells": int(n),
                "y_true": int(y_true_by_sample[sid]), "y_true_name": classes[int(y_true_by_sample[sid])],
                "y_pred": int(y_pred_by_sample[sid]), "y_pred_name": classes[int(y_pred_by_sample[sid])],
                "correct": int(int(y_true_by_sample[sid]) == int(y_pred_by_sample[sid])),
            }
            for ci, cname in enumerate(classes):
                prow[f"p_{cname}"] = float(P[ci])
            rows_cells.append(prow)
        w_sorted = np.sort(w)[::-1]
        ent = attn_entropy_1d(w)
        top_ids = [str(ids[int(j)]) for j in idx[:min(10, len(idx))]]
        srow = {
            "rep": int(rep), "fold": int(fold), "view": str(view_name), "SampleKey": str(sid),
            "n_cells": int(n),
            "y_true": int(y_true_by_sample[sid]), "y_true_name": classes[int(y_true_by_sample[sid])],
            "y_pred": int(y_pred_by_sample[sid]), "y_pred_name": classes[int(y_pred_by_sample[sid])],
            "correct": int(int(y_true_by_sample[sid]) == int(y_pred_by_sample[sid])),
            "attn_top1": float(w_sorted[0]),
            f"attn_top{int(k)}_mass": float(w_sorted[:k].sum()),
            "attn_entropy": float(ent),
            "attn_eff_cells_expH": float(np.exp(ent)),
            "top_cell_ids_10": json.dumps(top_ids, ensure_ascii=False),
        }
        for ci, cname in enumerate(classes):
            srow[f"p_{cname}"] = float(P[ci])
        rows_sum.append(srow)

    base_cols_cells = [
        "rep","fold","view","SampleKey","cell_id","attn_weight","attn_rank","n_cells",
        "y_true","y_true_name","y_pred","y_pred_name","correct",
        *[f"p_{c}" for c in classes],
    ]
    base_cols_sum = [
        "rep","fold","view","SampleKey","n_cells",
        "y_true","y_true_name","y_pred","y_pred_name","correct",
        "attn_top1", f"attn_top{int(topk)}_mass", "attn_entropy", "attn_eff_cells_expH", "top_cell_ids_10",
        *[f"p_{c}" for c in classes],
    ]
    to_csv_safe(pd.DataFrame(rows_cells) if rows_cells else pd.DataFrame(columns=base_cols_cells), out_cells_csv)
    to_csv_safe(pd.DataFrame(rows_sum) if rows_sum else pd.DataFrame(columns=base_cols_sum), out_summary_csv)

def copy_locked_audit_artifacts_to_out():
    if not COPY_LOCKED_AUDIT_ARTIFACTS_TO_OUT:
        return
    import shutil
    for dirname in ("tuned", "checkpoints"):
        src_dir = os.path.join(LOCKED_INPUT_DIR, dirname)
        dst_dir = os.path.join(OUT_DIR, dirname)
        if os.path.isdir(src_dir):
            os.makedirs(dst_dir, exist_ok=True)
            for name in sorted(os.listdir(src_dir)):
                src_file = os.path.join(src_dir, name)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, os.path.join(dst_dir, name))

def balanced_acc_from_recall(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> float:
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return float(np.mean(rec)) if rec.size else float("nan")

def metrics_safe_multiclass(y_true: np.ndarray, P: np.ndarray, n_classes: int):
    yhat = P.argmax(axis=1)
    acc = float(accuracy_score(y_true, yhat))
    bal = balanced_acc_from_recall(y_true, yhat, labels=list(range(n_classes)))
    f1m = float(f1_score(y_true, yhat, average="macro", labels=list(range(n_classes)), zero_division=0))
    auc = np.nan
    try:
        if len(np.unique(y_true)) == n_classes:
            auc = float(roc_auc_score(y_true, P, multi_class="ovr"))
    except Exception:
        auc = np.nan
    return acc, bal, f1m, auc

def load_torch_payload(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def load_view_model_from_checkpoint(row: pd.Series, view: str, view_bags, df_c_keyed, sample_list, device: str):
    ckpt_name = str(row.get(f"{view}_ckpt_file", ""))
    if not ckpt_name:
        ckpt_name = f"rep{int(row['rep']):02d}_fold{int(row['fold']):02d}_view{view}_E{int(row['E_star']):03d}.pt"
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    payload = load_torch_payload(ckpt_path, device=device)
    pre_name = payload.get("preprocessor_path", ckpt_name.split(f"_view{view}_")[0] + f"_view{view}_pre.joblib")
    pre_path = os.path.join(CHECKPOINT_DIR, pre_name)
    if not os.path.exists(pre_path):
        raise FileNotFoundError(f"Missing clinical preprocessor: {pre_path}")
    pre = joblib.load(pre_path)
    feat_cols = list(payload["feat_cols"])
    clin_X = build_clin_matrix_dict(df_c_keyed, pre, feat_cols, sample_list)
    clin_dim = int(next(iter(clin_X.values())).shape[0])
    params = MILParams(**payload.get("params", {}))
    in_dim = next(iter(view_bags.values())).shape[1]
    model = AttnMILMultiModal(
        in_dim=in_dim, clin_in_dim=clin_dim, n_classes=len(DISEASE_LABELS),
        embed_dim=params.embed_dim, attn_dim=params.attn_dim,
        clin_embed_dim=params.clin_embed_dim, dropout=params.dropout,
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    return dict(model=model, mean=payload["mean"], std=payload["std"], pre=pre, feat_cols=feat_cols, clin_X=clin_X, params=params)

def regenerate_outer_split(samples: np.ndarray, yS: np.ndarray, outer_seed_used: int, fold_id: int, n_splits: int):
    sgkf = StratifiedGroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(outer_seed_used))
    folds = list(sgkf.split(samples, yS, groups=samples))
    tr, te = folds[int(fold_id) - 1]
    return set(samples[tr].tolist()), set(samples[te].tolist())

def prepare_data():
    classes = list(DISEASE_LABELS)
    df_r = read_csv_robust(RAMAN_CSV)
    df_r.columns = df_r.columns.astype(str).str.strip()
    df_r[RAMAN_SAMPLE_COL] = df_r[RAMAN_SAMPLE_COL].astype(str).str.strip()
    df_r[RAMAN_LABEL_COL] = df_r[RAMAN_LABEL_COL].astype(str).str.strip()
    df_r = df_r[df_r[RAMAN_LABEL_COL].isin(classes)].copy()
    if df_r.empty:
        raise ValueError("Raman disease-only subset is empty.")

    sample_group_raw = df_r.groupby(RAMAN_SAMPLE_COL)[RAMAN_LABEL_COL].first()
    raman_samples_raw = sample_group_raw.index.values.astype(str)

    df_c = read_csv_robust(CLINICAL_CSV, dtype=str)
    df_c = preprocess_clinical_locked(df_c)
    df_c = reduce_clinical_to_one_row_per_sample(df_c)
    df_c[CLIN_SAMPLE_COL] = df_c[CLIN_SAMPLE_COL].astype(str).str.strip()
    clin_samples_raw = df_c[CLIN_SAMPLE_COL].values.astype(str)

    r_key, c_key, mode_used = build_sample_keys(raman_samples_raw, clin_samples_raw)
    df_r["SampleKey"] = df_r[RAMAN_SAMPLE_COL].map(dict(zip(raman_samples_raw.tolist(), r_key.tolist()))).astype(str)
    df_c["SampleKey"] = c_key
    df_r = df_r[df_r["SampleKey"].astype(str) != ""].copy()
    df_c = df_c[df_c["SampleKey"].astype(str) != ""].copy()

    samplekey_group = df_r.groupby("SampleKey")[RAMAN_LABEL_COL].first()
    samples_all = samplekey_group.index.values.astype(str)
    common = np.intersect1d(samples_all, df_c["SampleKey"].values.astype(str))
    if len(common) < 6:
        raise ValueError(f"Too few common patients between Raman and clinical: {len(common)}")

    le = LabelEncoder().fit(classes)
    yS = le.transform(samplekey_group.loc[common].values.astype(str))
    samples = common.astype(str)
    y_by_sample = {sid: int(le.transform([samplekey_group.loc[sid]])[0]) for sid in samples}

    all_cols, wav = get_spec_cols_and_wav(df_r)
    view_bags = {}
    for vname, region in VIEWS:
        cols = select_region(all_cols, wav, region)
        view_bags[vname] = build_bags_from_df(df_r, cols)

    df_c_keyed = df_c.set_index("SampleKey").copy()
    missing_c = [s for s in samples if s not in df_c_keyed.index]
    if missing_c:
        raise ValueError(f"Some Raman patients missing clinical rows. Example: {missing_c[:10]}")

    return classes, le, df_r, df_c_keyed, samplekey_group, samples, yS, y_by_sample, view_bags, mode_used

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    copy_locked_audit_artifacts_to_out()
    locked = read_csv_robust(LOCKED_RUNS_CSV)
    locked["rep"] = pd.to_numeric(locked["rep"], errors="coerce").astype(int)
    locked["fold"] = pd.to_numeric(locked["fold"], errors="coerce").astype(int)
    locked = locked[locked["rep"] == int(EXPECTED_SELECTED_REP)].copy()
    if locked.empty:
        raise ValueError(f"No locked rows found for EXPECTED_SELECTED_REP={EXPECTED_SELECTED_REP}")
    locked = locked.sort_values("fold").reset_index(drop=True)
    n_splits = int(locked["fold"].max())

    classes, le, df_r, df_c_keyed, samplekey_group, samples, yS, y_by_sample, view_bags, mode_used = prepare_data()
    n_classes = len(classes)
    assert n_classes == 3

    print("\n[Publication MIL replay] Disease-only patient counts:")
    for c in classes:
        print(f"  {c}: {int(np.sum(samplekey_group.loc[samples].values.astype(str) == c))}")

    rows = []
    patient_rows = []
    attn_dir = os.path.join(OUT_DIR, ATTN_DIRNAME)
    if SAVE_ATTN_EXPORT:
        os.makedirs(attn_dir, exist_ok=True)
        cell_ids_by_sample = build_cell_ids_by_samplekey(df_r)

    for _, row_locked in locked.iterrows():
        rep = int(row_locked["rep"])
        fold_id = int(row_locked["fold"])
        outer_seed_used = int(row_locked["outer_seed_used"])
        outer_train, outer_test = regenerate_outer_split(samples, yS, outer_seed_used, fold_id, n_splits=n_splits)
        idx_te = sorted(list(outer_test))
        y_te = np.array([y_by_sample[s] for s in idx_te], dtype=int)
        y_te_str = np.array([le.inverse_transform([y_by_sample[s]])[0] for s in idx_te], dtype=object)

        final_models = {}
        Pte = {}
        attn_by_view = {}
        for vname, _ in VIEWS:
            fm = load_view_model_from_checkpoint(row_locked, vname, view_bags[vname], df_c_keyed, idx_te, DEVICE)
            final_models[vname] = fm
            _, P = predict_probs_multimodal_mil(
                model=fm["model"], bags=view_bags[vname], y_by_sample=y_by_sample,
                clin_X_by_sample=fm["clin_X"], sample_list=idx_te,
                mean=fm["mean"], std=fm["std"],
                batch_size=max(2, int(fm["params"].batch_size)), device=DEVICE,
            )
            Pte[vname] = P
            if SAVE_ATTN_EXPORT:
                attn_by_view[vname] = extract_attn_multimodal_mil(
                    model=fm["model"], bags=view_bags[vname], y_by_sample=y_by_sample,
                    clin_X_by_sample=fm["clin_X"], sample_list=idx_te,
                    mean=fm["mean"], std=fm["std"],
                    batch_size=max(2, int(fm["params"].batch_size)), device=DEVICE,
                )

        w_all = float(row_locked["w_all"])
        w_fp = float(row_locked["w_fp"])
        w_ch = float(row_locked["w_ch"])
        T = float(row_locked["T"])
        use_view_fusion = bool(int(row_locked.get("use_view_fusion", 1)))
        if use_view_fusion:
            P_raw = w_all * Pte["all"] + w_fp * Pte["fp"] + w_ch * Pte["ch"]
        else:
            P_raw = Pte["all"]
        P_raw = P_raw / (P_raw.sum(axis=1, keepdims=True) + 1e-12)
        P_T = apply_temperature_multiclass(P_raw, T)

        acc, bal, f1m, auc = metrics_safe_multiclass(y_te, P_T, n_classes)
        ll = float(log_loss(y_te, P_T, labels=list(range(n_classes))))
        yhat = P_T.argmax(axis=1)
        cm = confusion_matrix(y_te, yhat, labels=list(range(n_classes))).flatten().tolist()

        if SAVE_ATTN_EXPORT:
            y_true_map = {sid: int(y_by_sample[sid]) for sid in idx_te}
            y_pred_map = {sid: int(yhat[i]) for i, sid in enumerate(idx_te)}
            prob_map = {sid: P_T[i].astype(float) for i, sid in enumerate(idx_te)}
            for vname, _ in VIEWS:
                out_cells = os.path.join(attn_dir, f"rep{rep:02d}_fold{fold_id:02d}_view{vname}_outertest_cells_topK.csv")
                out_sum = os.path.join(attn_dir, f"rep{rep:02d}_fold{fold_id:02d}_view{vname}_outertest_patient_summary.csv")
                export_attention_topk_and_summary(
                    rep=rep, fold=fold_id, view_name=vname, sample_list=idx_te,
                    attn_by_sample=attn_by_view.get(vname, {}), cell_ids_by_sample=cell_ids_by_sample,
                    y_true_by_sample=y_true_map, y_pred_by_sample=y_pred_map, prob_by_sample=prob_map,
                    classes=list(le.classes_), out_cells_csv=out_cells, out_summary_csv=out_sum,
                    topk=int(ATTN_EXPORT_TOPK),
                )
            if use_view_fusion:
                fused_attn = {}
                for sid in idx_te:
                    wa = attn_by_view.get("all", {}).get(sid)
                    wf = attn_by_view.get("fp", {}).get(sid)
                    wc = attn_by_view.get("ch", {}).get(sid)
                    if wa is None or wf is None or wc is None:
                        continue
                    m = min(len(wa), len(wf), len(wc))
                    if m > 0:
                        fused_attn[sid] = (w_all * wa[:m] + w_fp * wf[:m] + w_ch * wc[:m]).astype(np.float64)
                out_cells = os.path.join(attn_dir, f"rep{rep:02d}_fold{fold_id:02d}_viewfused_outertest_cells_topK.csv")
                out_sum = os.path.join(attn_dir, f"rep{rep:02d}_fold{fold_id:02d}_viewfused_outertest_patient_summary.csv")
                export_attention_topk_and_summary(
                    rep=rep, fold=fold_id, view_name="fused", sample_list=idx_te,
                    attn_by_sample=fused_attn, cell_ids_by_sample=cell_ids_by_sample,
                    y_true_by_sample=y_true_map, y_pred_by_sample=y_pred_map, prob_by_sample=prob_map,
                    classes=list(le.classes_), out_cells_csv=out_cells, out_summary_csv=out_sum,
                    topk=int(ATTN_EXPORT_TOPK),
                )

        ent = entropy(P_T)
        mar = margin_true_vs_best_other(y_te, P_T)
        clin_sub = df_c_keyed.loc[idx_te].copy()
        for col in CATEGORICAL_CLIN_COLS:
            if col not in clin_sub.columns:
                clin_sub[col] = "Unknown"

        for i, sk in enumerate(idx_te):
            prow = dict(
                rep=rep, fold=fold_id, SampleKey=str(sk), samplekey_mode=str(mode_used),
                outer_seed_used=int(outer_seed_used), outer_tries=int(row_locked.get("outer_tries", -1)),
                use_view_fusion=int(use_view_fusion), w_all=w_all, w_fp=w_fp, w_ch=w_ch, T=T,
                E_star=int(row_locked.get("E_star", -1)),
                y_true=int(y_te[i]), y_true_name=str(le.inverse_transform([int(y_te[i])])[0]),
                y_pred=int(yhat[i]), y_pred_name=str(le.inverse_transform([int(yhat[i])])[0]),
                correct=int(int(y_te[i]) == int(yhat[i])),
                entropy=float(ent[i]), margin=float(mar[i]),
            )
            for j, c in enumerate(le.classes_):
                prow[f"p_{c}"] = float(P_T[i, j])
                prow[f"p_raw_{c}"] = float(P_raw[i, j])
            for col in CATEGORICAL_CLIN_COLS:
                prow[col] = str(clin_sub.iloc[i][col]) if col in clin_sub.columns else "Unknown"
            patient_rows.append(prow)

        outrow = dict(
            rep=rep, fold=fold_id, outer_seed_used=int(outer_seed_used), outer_tries=int(row_locked.get("outer_tries", -1)),
            samplekey_mode=str(mode_used), n_train_patients=int(len(outer_train)), n_test_patients=int(len(idx_te)),
            outer_test_counts_json=class_counts_str(y_te_str, list(le.classes_)),
            use_view_fusion=int(use_view_fusion), w_all=w_all, w_fp=w_fp, w_ch=w_ch, T=T,
            E_star=int(row_locked.get("E_star", -1)), acc=float(acc), bal_acc=float(bal), macro_f1=float(f1m),
            auc_ovr=float(auc) if auc == auc else np.nan, logloss=float(ll), cm_flat=",".join(map(str, cm)),
        )
        for vname, _ in VIEWS:
            outrow[f"{vname}_ckpt_file"] = str(row_locked.get(f"{vname}_ckpt_file", ""))
            outrow[f"{vname}_final_epochs"] = int(row_locked.get(f"{vname}_final_epochs", row_locked.get("E_star", -1)))
        rows.append(outrow)
        print(f"[rep {rep} fold {fold_id}] acc={acc:.3f} bal={bal:.3f} f1={f1m:.3f} ll={ll:.3f} T={T:.2f} w=({w_all:.1f},{w_fp:.1f},{w_ch:.1f})")

    res = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "multimodal_MIL_disease3_phase2_checkpoint_replay.csv")
    sum_csv = os.path.join(OUT_DIR, "multimodal_MIL_disease3_phase2_checkpoint_replay_SUMMARY.csv")
    to_csv_safe(res, out_csv)
    summary = dict(
        task="multimodal_MIL_phase2_disease3_checkpoint_replay",
        n_rows=int(len(res)),
        acc_mean=float(res["acc"].mean()), acc_std=float(res["acc"].std()),
        bal_mean=float(res["bal_acc"].mean()), bal_std=float(res["bal_acc"].std()),
        f1_mean=float(res["macro_f1"].mean()), f1_std=float(res["macro_f1"].std()),
        logloss_mean=float(res["logloss"].mean()),
        auc_mean=float(res["auc_ovr"].dropna().mean()) if res["auc_ovr"].notna().any() else float("nan"),
        selected_rep=int(EXPECTED_SELECTED_REP),
        use_view_fusion=int(USE_VIEW_FUSION),
        curves_exported=0,
        attention_exported=int(SAVE_ATTN_EXPORT),
        locked_audit_artifacts_copied=int(COPY_LOCKED_AUDIT_ARTIFACTS_TO_OUT),
    )
    to_csv_safe(pd.DataFrame([summary]), sum_csv)

    pred_csv = os.path.join(OUT_DIR, "patient_preds_full.csv")
    mis_csv = os.path.join(OUT_DIR, "misclassified_full.csv")
    dfp = pd.DataFrame(patient_rows)
    to_csv_safe(dfp, pred_csv)
    to_csv_safe(dfp[dfp["correct"] == 0].copy(), mis_csv)

    print("\nSaved:", out_csv)
    print("Saved:", sum_csv)
    print("Saved:", pred_csv)
    print("Saved:", mis_csv)
    if SAVE_ATTN_EXPORT:
        print("Saved:", os.path.join(OUT_DIR, ATTN_DIRNAME))
    if COPY_LOCKED_AUDIT_ARTIFACTS_TO_OUT:
        print("Copied selected locked tuned/checkpoint artifacts into OUT_DIR if present.")
    print("No curve folder or curve files are generated by this public replay script.")
    print("SUMMARY:", json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
