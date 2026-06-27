#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, confusion_matrix, recall_score


RAMAN_CSV = "raman_added2.csv"
OUT_DIR = "./publication_MIL_disease3_reproduction_out_CV5"

LOCKED_INPUT_DIR = "./publication_stage1_disease3_MIL_inputs_CV5"
LOCKED_RUNS_CSV = os.path.join(LOCKED_INPUT_DIR, "selected_rep_MIL_DISEASE3_RAMANONLY_LOCKED_RUNS.csv")
EXPECTED_SELECTED_REP = 3 

DISEASE_LABELS = ["Liver", "Bile", "Pancreas"]

ENCODED_CLASS_ORDER = ["Bile", "Liver", "Pancreas"]

VIEWS = [("all", "all"), ("fp", "fingerprint"), ("ch", "ch")]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_ENCODING = "utf-8-sig"


SAVE_ATTENTION_EXPORTS = True
ATTN_EXPORT_TOPK = 200
ATTN_EXPORT_VIEWS = "active"  # "active" or "all_only"
ATTN_CELL_ID_COL = "Cell"
COPY_LOCKED_AUDIT_ARTIFACTS_TO_OUT = True 


def configure_torch_runtime(deterministic: bool = False):
    try:
        torch.use_deterministic_algorithms(bool(deterministic))
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    except Exception:
        pass

configure_torch_runtime(deterministic=False)

@dataclass
class MILParams:
    embed_dim: int = 128
    attn_dim: int = 128
    dropout: float = 0.35
    lr: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_instances_train: Optional[int] = 128

class AttnMILMulti(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, embed_dim: int, attn_dim: int, dropout: float):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.attn = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1))
        self.cls = nn.Linear(embed_dim, n_classes)

    def forward(self, x_pad: torch.Tensor, mask: torch.Tensor):
        H = self.embed(x_pad)
        a = self.attn(H).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf"))
        w = torch.softmax(a, dim=1)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        M = torch.sum(H * w.unsqueeze(-1), dim=1)
        return self.cls(M)

    def forward_with_attn(self, x_pad: torch.Tensor, mask: torch.Tensor):
        H = self.embed(x_pad)
        a = self.attn(H).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf"))
        w = torch.softmax(a, dim=1)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        M = torch.sum(H * w.unsqueeze(-1), dim=1)
        return self.cls(M), w


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


def parse_json_list(x) -> List[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(t) for t in v]
        return [str(v)]
    except Exception:
        if s.startswith("[") and s.endswith("]"):
            s = s.strip("[]")
        return [t.strip().strip('"').strip("'") for t in s.split(",") if t.strip()]


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


def _entropy(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, 1.0)
    return -(P * np.log(P)).sum(axis=1)


def _margin_top1_top2(P: np.ndarray) -> np.ndarray:
    s = np.sort(P, axis=1)
    return s[:, -1] - s[:, -2]


def class_counts_str(y_str: np.ndarray, classes: List[str]) -> str:
    return json.dumps({c: int(np.sum(y_str == c)) for c in classes}, ensure_ascii=False)


def balanced_acc_from_recall(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> float:
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return float(np.mean(rec)) if rec.size else float("nan")


def metrics_safe_multiclass(y_true: np.ndarray, P: np.ndarray, n_classes: int):
    labels = list(range(n_classes))
    yhat = P.argmax(axis=1)
    acc = float(accuracy_score(y_true, yhat))
    bal = balanced_acc_from_recall(y_true, yhat, labels=labels)
    f1m = float(f1_score(y_true, yhat, average="macro", labels=labels, zero_division=0))
    auc = np.nan
    try:
        if len(np.unique(y_true)) == n_classes:
            auc = float(roc_auc_score(y_true, P, multi_class="ovr"))
    except Exception:
        auc = np.nan
    ll = float(log_loss(y_true, P, labels=list(range(n_classes))))
    return acc, bal, f1m, auc, ll


def build_bags_from_df(df: pd.DataFrame, spec_cols: List[str]) -> Dict[str, np.ndarray]:
    bags = {}
    for sid, g in df.groupby("Sample"):
        X = g[spec_cols].to_numpy(np.float32)
        if X.ndim == 2 and X.shape[0] > 0:
            bags[str(sid)] = X
    return bags


def build_cell_ids_from_df(df: pd.DataFrame, cell_id_col: str = "Cell") -> Dict[str, List[str]]:
    out = {}
    for sid, g in df.groupby("Sample"):
        if cell_id_col in g.columns:
            ids = g[cell_id_col].astype(str).tolist()
        else:
            ids = [str(i) for i in range(len(g))]
        out[str(sid)] = ids
    return out

class MILBagsDataset(Dataset):
    def __init__(self, sample_ids: List[str], bags: Dict[str, np.ndarray], y_by_sample: Dict[str, int], mean: np.ndarray, std: np.ndarray):
        self.sample_ids = list(sample_ids)
        self.bags = bags
        self.y_by_sample = y_by_sample
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sid = self.sample_ids[idx]
        X = (self.bags[sid] - self.mean) / (self.std + 1e-8)
        y = int(self.y_by_sample[sid])
        return torch.from_numpy(X).float(), torch.tensor(y, dtype=torch.long), sid


def collate_pad(batch):
    xs, ys, sids = zip(*batch)
    lens = [x.shape[0] for x in xs]
    D = xs[0].shape[1]
    T = max(lens)
    B = len(xs)
    x_pad = torch.zeros((B, T, D), dtype=torch.float32)
    mask = torch.zeros((B, T), dtype=torch.bool)
    y = torch.stack(ys, dim=0)
    for i, x in enumerate(xs):
        t = x.shape[0]
        x_pad[i, :t] = x
        mask[i, :t] = True
    return x_pad, mask, y, list(sids)

@torch.no_grad()
def predict_probs_mil_multiclass(model: nn.Module, bags: Dict[str, np.ndarray], y_by_sample: Dict[str, int],
                                 sample_list: List[str], mean: np.ndarray, std: np.ndarray,
                                 batch_size: int, device: str) -> Tuple[List[str], np.ndarray]:
    ds = MILBagsDataset(sample_list, bags, y_by_sample, mean, std)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_pad)
    model.eval()
    ids, ps = [], []
    for x_pad, mask, y, sids in dl:
        logits = model(x_pad.to(device), mask.to(device))
        P = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)
        ids.extend(sids); ps.append(P)
    P = np.concatenate(ps, axis=0) if ps else np.zeros((0, 0), dtype=np.float64)
    return ids, P


def _safe_json_list(x) -> str:
    return json.dumps(list(x), ensure_ascii=False)


def _safe_json_float_list(x) -> str:
    return json.dumps([float(v) for v in x], ensure_ascii=False)

@torch.no_grad()
def export_attention_topk_outer_test(
    out_cells_csv: str,
    out_sum_csv: str,
    rep: int,
    fold: int,
    classes: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    P_raw: np.ndarray,
    P_cal: np.ndarray,
    sample_ids: List[str],
    views_to_export: List[str],
    view_bags: Dict[str, Dict[str, np.ndarray]],
    view_cell_ids: Dict[str, Dict[str, List[str]]],
    outer_models: Dict[str, dict],
    topk: int,
):
    rows_cells, rows_sum = [], []
    for i, sid in enumerate(sample_ids):
        yt = int(y_true[i]); yp = int(y_pred[i])
        for vname in views_to_export:
            if vname not in outer_models or sid not in view_bags.get(vname, {}):
                continue
            model = outer_models[vname]["model"]
            mean = outer_models[vname]["mean"]
            std = outer_models[vname]["std"]
            X_raw = view_bags[vname][sid]
            ids = view_cell_ids[vname].get(sid, [str(k) for k in range(X_raw.shape[0])])
            X = (X_raw.astype(np.float32) - mean.astype(np.float32)) / (std.astype(np.float32) + 1e-8)
            Tn, D = int(X.shape[0]), int(X.shape[1])
            x_pad = torch.zeros((1, Tn, D), dtype=torch.float32, device=DEVICE)
            mask = torch.ones((1, Tn), dtype=torch.bool, device=DEVICE)
            x_pad[0, :Tn] = torch.from_numpy(X).to(DEVICE)
            _, w = model.forward_with_attn(x_pad, mask)
            attn_w = w.detach().cpu().numpy().astype(np.float64)[0]
            if len(ids) != Tn:
                ids = [str(k) for k in range(Tn)]
            kk = int(min(max(1, topk), Tn))
            idx = np.argsort(-attn_w)[:kk]
            top_ids = [ids[j] for j in idx.tolist()]
            top_w = [float(attn_w[j]) for j in idx.tolist()]
            rows_sum.append({
                "SampleKey": sid, "rep": int(rep), "fold": int(fold), "view": str(vname),
                "y_true": yt, "y_true_name": str(classes[yt]),
                "y_pred": yp, "y_pred_name": str(classes[yp]),
                "correct": int(yt == yp), "topk": kk,
                "top_cell_ids_json": _safe_json_list(top_ids),
                "top_cell_weights_json": _safe_json_float_list(top_w),
                "max_attn": float(np.max(attn_w)),
                "sum_topk_attn": float(np.sum(np.array(top_w, dtype=np.float64))),
                "entropy": float(_entropy(P_cal[[i], :])[0]),
                "margin": float(_margin_top1_top2(P_cal[[i], :])[0]),
            })
            for rank, j in enumerate(idx.tolist(), 1):
                rows_cells.append({
                    "SampleKey": sid, "rep": int(rep), "fold": int(fold), "view": str(vname),
                    "y_true": yt, "y_true_name": str(classes[yt]),
                    "y_pred": yp, "y_pred_name": str(classes[yp]),
                    "correct": int(yt == yp), "rank": int(rank),
                    "cell_idx_in_bag": int(j), "cell_id": str(ids[j]), "attn_w": float(attn_w[j]),
                })
    if rows_cells:
        pd.DataFrame(rows_cells).to_csv(out_cells_csv, mode="a", header=not os.path.exists(out_cells_csv), index=False, encoding=CSV_ENCODING)
    if rows_sum:
        pd.DataFrame(rows_sum).to_csv(out_sum_csv, mode="a", header=not os.path.exists(out_sum_csv), index=False, encoding=CSV_ENCODING)


def torch_load_public(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint(row: pd.Series, view: str, n_classes: int) -> Dict:
    rel = str(row.get(f"{view}_checkpoint_rel_file", "")).strip()
    if not rel:
        rep = int(row["rep"]); fold = int(row["fold"])
        rel = str(Path("models_outer") / f"rep{rep:02d}_fold{fold:02d}_view_{view}_outer_Estar.pt")
    path = Path(LOCKED_INPUT_DIR) / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint for view={view}: {path}")
    payload = torch_load_public(path)
    params_dict = dict(payload.get("params", {}))
    params = MILParams(**{k: v for k, v in params_dict.items() if k in MILParams.__dataclass_fields__}) if params_dict else MILParams()
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    in_dim = int(len(mean))
    model = AttnMILMulti(in_dim=in_dim, n_classes=n_classes,
                         embed_dim=int(params.embed_dim), attn_dim=int(params.attn_dim), dropout=float(params.dropout)).to(DEVICE)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return dict(model=model, mean=mean, std=std, params=params, payload=payload, path=path)


def copy_locked_audit_artifacts():
    if not COPY_LOCKED_AUDIT_ARTIFACTS_TO_OUT:
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in [
        "selected_rep_MIL_DISEASE3_RAMANONLY_LOCKED_RUNS.csv",
        "selected_rep_EXPECTED_METRICS.csv",
        "selected_rep_EXPECTED_SPLITS.csv",
        "selected_rep_EXPECTED_PATIENT_PREDS.csv",
        "selected_rep_EXPECTED_MISCLASSIFIED.csv",
        "selected_rep_EXPECTED_INNER_VAL_PREDS.csv",
        "selected_rep_EXPECTED_ATTN_CELLS.csv",
        "selected_rep_EXPECTED_ATTN_PATIENT_SUMMARY.csv",
    ]:
        src = Path(LOCKED_INPUT_DIR) / name
        if src.exists():
            shutil.copy2(src, Path(OUT_DIR) / name)
    for dname in ["models_outer", "tuned_params"]:
        src = Path(LOCKED_INPUT_DIR) / dname
        if src.exists():
            dst = Path(OUT_DIR) / dname
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def validate_locked(locked: pd.DataFrame):
    reps = sorted(pd.to_numeric(locked["rep"], errors="coerce").dropna().astype(int).unique().tolist())
    if reps != [int(EXPECTED_SELECTED_REP)]:
        raise ValueError(f"Expected selected rep {EXPECTED_SELECTED_REP}, but locked CSV contains reps {reps}")
    expected_order = sorted(DISEASE_LABELS)
    if list(ENCODED_CLASS_ORDER) != expected_order:
        raise ValueError(f"ENCODED_CLASS_ORDER should match sorted(DISEASE_LABELS)={expected_order}; got {ENCODED_CLASS_ORDER}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    locked = read_csv_robust(LOCKED_RUNS_CSV)
    required = ["rep", "fold", "outer_test_samples_json", "use_view_fusion", "w_all", "w_fp", "w_ch", "T"]
    missing = [c for c in required if c not in locked.columns]
    if missing:
        raise ValueError(f"Locked CSV missing columns: {missing}. Found: {locked.columns.tolist()}")
    validate_locked(locked)
    copy_locked_audit_artifacts()

    classes = list(ENCODED_CLASS_ORDER)
    n_classes = len(classes)
    label_to_int = {c: i for i, c in enumerate(classes)}

    df = read_csv_robust(RAMAN_CSV)
    df.columns = df.columns.astype(str).str.strip()
    for c in ["Batch", "Group", "Sample", "CellType", "Cell"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    if "Sample" not in df.columns or "Group" not in df.columns:
        raise ValueError("Raman CSV must contain at least Sample and Group columns.")

    # Disease-only subset. Keep model label order as LabelEncoder's sorted class order.
    df_dis = df[df["Group"].isin(set(DISEASE_LABELS))].copy()
    sample_group = df_dis.groupby("Sample")["Group"].first()
    all_samples = sample_group.index.astype(str).values
    y_by_sample = {str(s): int(label_to_int[str(sample_group.loc[s])]) for s in all_samples}

    all_cols, wav = get_spec_cols_and_wav(df_dis)
    view_bags, view_cell_ids = {}, {}
    for vname, region in VIEWS:
        cols = select_region(all_cols, wav, region)
        view_bags[vname] = build_bags_from_df(df_dis, cols)
        view_cell_ids[vname] = build_cell_ids_from_df(df_dis, cell_id_col=ATTN_CELL_ID_COL)

    out_csv = os.path.join(OUT_DIR, "stage1_MIL_disease3_results.csv")
    sum_csv = os.path.join(OUT_DIR, "stage1_MIL_disease3_SUMMARY.csv")
    preds_csv = os.path.join(OUT_DIR, "stage1_PATIENT_PREDS_disease3.csv")
    mis_csv = os.path.join(OUT_DIR, "stage1_MISCLASSIFIED_disease3.csv")
    splits_csv = os.path.join(OUT_DIR, "stage1_SPLITS_disease3.csv")
    attn_cells_csv = os.path.join(OUT_DIR, "stage1_ATTN_CELLS_disease3.csv")
    attn_sum_csv = os.path.join(OUT_DIR, "stage1_ATTN_PATIENT_SUMMARY_disease3.csv")
    for p in [out_csv, sum_csv, preds_csv, mis_csv, splits_csv, attn_cells_csv, attn_sum_csv]:
        if os.path.exists(p):
            os.remove(p)

    rows = []
    locked = locked.copy()
    locked["fold"] = pd.to_numeric(locked["fold"], errors="coerce").astype(int)
    locked = locked.sort_values("fold")

    # Recreate selected split table from locked rows.
    split_rows = []

    for _, row in locked.iterrows():
        rep = int(row["rep"]); fold = int(row["fold"])
        idx_te = parse_json_list(row.get("outer_test_samples_json", ""))
        if not idx_te:
            raise ValueError(f"Missing outer_test_samples_json for rep={rep}, fold={fold}")
        missing_samples = [s for s in idx_te if s not in y_by_sample]
        if missing_samples:
            raise ValueError(f"Locked outer-test samples not found in Raman CSV. Examples: {missing_samples[:10]}")

        y_te = np.array([y_by_sample[s] for s in idx_te], dtype=int)
        y_te_name = np.array([classes[int(y_by_sample[s])] for s in idx_te], dtype=object)
        use_view_fusion = bool(int(float(row.get("use_view_fusion", 1))))
        active_views = [v for v, _ in VIEWS] if use_view_fusion else ["all"]
        models = {v: load_checkpoint(row, v, n_classes=n_classes) for v in active_views}

        Pte = {}
        for v in active_views:
            _, P = predict_probs_mil_multiclass(
                model=models[v]["model"], bags=view_bags[v], y_by_sample=y_by_sample,
                sample_list=idx_te, mean=models[v]["mean"], std=models[v]["std"],
                batch_size=max(2, int(models[v]["params"].batch_size)), device=DEVICE,
            )
            Pte[v] = P

        w_all = float(row.get("w_all", 1.0)); w_fp = float(row.get("w_fp", 0.0)); w_ch = float(row.get("w_ch", 0.0))
        if use_view_fusion:
            P_raw = w_all * Pte["all"] + w_fp * Pte["fp"] + w_ch * Pte["ch"]
        else:
            P_raw = Pte["all"]
        P_raw = P_raw / (P_raw.sum(axis=1, keepdims=True) + 1e-12)
        T = float(row.get("T", 1.0))
        P_cal = apply_temperature_multiclass(P_raw, T)
        yhat = P_cal.argmax(axis=1)
        acc, bal, f1m, auc, ll = metrics_safe_multiclass(y_te, P_cal, n_classes=n_classes)
        cm = confusion_matrix(y_te, yhat, labels=list(range(n_classes))).flatten().tolist()

        outrow = dict(
            rep=rep, fold=fold,
            base_seed=row.get("base_seed", np.nan), outer_seed_used=row.get("outer_seed_used", np.nan), outer_tries=row.get("outer_tries", np.nan),
            fixed_epochs=row.get("fixed_epochs", np.nan), tuned_params_json=row.get("tuned_params_json", ""),
            n_train_patients=row.get("n_train_patients", np.nan), n_test_patients=len(idx_te),
            outer_test_counts_json=class_counts_str(y_te_name, classes),
            inner_split_ok=row.get("inner_split_ok", np.nan), inner_split_tries=row.get("inner_split_tries", np.nan),
            inner_val_test_size=row.get("inner_val_test_size", np.nan),
            use_view_fusion=int(use_view_fusion), w_all=w_all, w_fp=w_fp, w_ch=w_ch,
            T=T, inner_val_logloss=row.get("inner_val_logloss", np.nan),
            acc=acc, bal_acc=bal, macro_f1=f1m, auc_ovr=float(auc) if auc == auc else np.nan,
            logloss=ll, cm_flat=",".join(map(str, cm)), replay_source="checkpoint_replay_no_curves_no_figures",
        )
        for v in active_views:
            outrow[f"{v}_checkpoint_rel_file"] = str(row.get(f"{v}_checkpoint_rel_file", ""))
            payload = models[v]["payload"]
            outrow[f"{v}_outer_E_star"] = int(payload.get("E_star", row.get(f"{v}_outer_E_star", -1))) if str(payload.get("E_star", "")).strip() else row.get(f"{v}_outer_E_star", np.nan)
        rows.append(outrow)

        pred_rows = []
        for i, sk in enumerate(idx_te):
            pred = dict(
                SampleKey=str(sk), rep=rep, fold=fold,
                y_true=int(y_te[i]), y_true_name=classes[int(y_te[i])],
                y_pred=int(yhat[i]), y_pred_name=classes[int(yhat[i])],
                correct=int(int(yhat[i]) == int(y_te[i])),
                T=T, w_all=w_all, w_fp=w_fp, w_ch=w_ch,
                entropy=float(_entropy(P_cal[[i], :])[0]), margin=float(_margin_top1_top2(P_cal[[i], :])[0]),
            )
            for j, cname in enumerate(classes):
                pred[f"p_{cname}"] = float(P_cal[i, j])
            for j, cname in enumerate(classes):
                pred[f"praw_{cname}"] = float(P_raw[i, j])
            pred_rows.append(pred)
        pred_df = pd.DataFrame(pred_rows)
        pred_df.to_csv(preds_csv, mode="a", header=not os.path.exists(preds_csv), index=False, encoding=CSV_ENCODING)
        mis = pred_df[pred_df["correct"] == 0].copy()
        if len(mis):
            mis.to_csv(mis_csv, mode="a", header=not os.path.exists(mis_csv), index=False, encoding=CSV_ENCODING)

        split_rows.append({
            "task": "stage1_MIL_disease3", "rep": rep, "fold": fold,
            "base_seed": row.get("base_seed", np.nan), "outer_seed_used": row.get("outer_seed_used", np.nan),
            "outer_tries": row.get("outer_tries", np.nan),
            "inner_split_ok": row.get("inner_split_ok", np.nan),
            "inner_split_tries": row.get("inner_split_tries", np.nan),
            "inner_val_test_size": row.get("inner_val_test_size", np.nan),
            "outer_train_samples_json": row.get("outer_train_samples_json", ""),
            "outer_test_samples_json": row.get("outer_test_samples_json", ""),
            "inner_train_samples_json": row.get("inner_train_samples_json", ""),
            "inner_val_samples_json": row.get("inner_val_samples_json", ""),
        })

        if SAVE_ATTENTION_EXPORTS:
            attn_views = ["all"] if ATTN_EXPORT_VIEWS == "all_only" else list(active_views)
            export_attention_topk_outer_test(
                out_cells_csv=attn_cells_csv,
                out_sum_csv=attn_sum_csv,
                rep=rep, fold=fold, classes=classes,
                y_true=y_te, y_pred=yhat, P_raw=P_raw, P_cal=P_cal,
                sample_ids=idx_te, views_to_export=attn_views,
                view_bags=view_bags, view_cell_ids=view_cell_ids,
                outer_models=models, topk=ATTN_EXPORT_TOPK,
            )

    res = pd.DataFrame(rows)
    to_csv_safe(res, out_csv)
    if split_rows:
        to_csv_safe(pd.DataFrame(split_rows), splits_csv)

    summary = dict(
        task="stage1_MIL_disease3_fixedepochs_balacc_ckpt",
        n_rows=int(len(res)), selected_rep=int(EXPECTED_SELECTED_REP),
        acc_mean=float(res["acc"].mean()), acc_std=float(res["acc"].std()),
        bal_mean=float(res["bal_acc"].mean()), bal_std=float(res["bal_acc"].std()),
        f1_mean=float(res["macro_f1"].mean()), f1_std=float(res["macro_f1"].std()),
        logloss_mean=float(res["logloss"].mean()),
        auc_mean=float(res["auc_ovr"].dropna().mean()) if res["auc_ovr"].notna().any() else float("nan"),
        outer_tries_mean=float(pd.to_numeric(res["outer_tries"], errors="coerce").mean()),
        inner_ok_rate=float(pd.to_numeric(res["inner_split_ok"], errors="coerce").mean()),
        use_view_fusion=int(res["use_view_fusion"].iloc[0]) if len(res) else np.nan,
        classes=classes,
        splits_csv=splits_csv,
        patient_preds_csv=preds_csv,
        misclassified_csv=mis_csv,
        attn_cells_csv=attn_cells_csv if SAVE_ATTENTION_EXPORTS else "",
        attn_patient_summary_csv=attn_sum_csv if SAVE_ATTENTION_EXPORTS else "",
        curves_inner_dir="", curves_outer_dir="", figures_dir="",
        models_outer_dir=os.path.join(OUT_DIR, "models_outer"),
        tuned_params_dir=os.path.join(OUT_DIR, "tuned_params"),
        replay_source="checkpoint_replay_no_curves_no_figures",
    )
    to_csv_safe(pd.DataFrame([summary]), sum_csv)
    print("DONE. Output:", OUT_DIR)
    print("No curves_inner/, curves_outer/, curves/, or figures/ folders were created.")


if __name__ == "__main__":
    main()
