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

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, log_loss, confusion_matrix


RAMAN_CSV = "raman_added2.csv"
OUT_DIR_BASE = "./publication_MIL_binary_reproduction_out_CV4"

LOCKED_INPUT_DIR = "./publication_stage1_binary_MIL_inputs_CV4"
LOCKED_RUNS_CSV = os.path.join(LOCKED_INPUT_DIR, "selected_task_reps_MIL_BINARY_LOCKED_RUNS.csv")

EXPECTED_SELECTED_REP_BY_TASK = {
    "control_vs_Liver": 1,
    "control_vs_Bile": 5,
    "control_vs_Pancreas": 2,
}

CONTROL_LABEL = "Control"
DISEASE_LABELS = ["Liver", "Bile", "Pancreas"]
VIEWS = [("all", "all"), ("fp", "fingerprint"), ("ch", "ch")]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_ENCODING = "utf-8-sig"


SAVE_ATTENTION_EXPORTS = True
ATTN_EXPORT_TOPK = 200
ATTN_EXPORT_VIEWS = "active"  # "active" or "all_only"
ATTN_EPS = 1e-12
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

class AttnMIL(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, attn_dim: int, dropout: float):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.attn = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1))
        self.cls = nn.Linear(embed_dim, 1)

    def forward(self, x_pad: torch.Tensor, mask: torch.Tensor, return_attn: bool = False):
        H = self.embed(x_pad)
        a = self.attn(H).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf"))
        w = torch.softmax(a, dim=1)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        M = torch.sum(H * w.unsqueeze(-1), dim=1)
        logit = self.cls(M).squeeze(-1)
        if return_attn:
            return logit, w
        return logit


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


def apply_temperature_binary(P2: np.ndarray, T: float) -> np.ndarray:
    P2 = np.clip(P2, 1e-12, 1.0)
    logits = np.log(P2) / float(T)
    logits -= logits.max(axis=1, keepdims=True)
    expv = np.exp(logits)
    return expv / (expv.sum(axis=1, keepdims=True) + 1e-12)


def _entropy_binary(P2: np.ndarray) -> np.ndarray:
    P2 = np.clip(P2, 1e-12, 1.0)
    return -(P2 * np.log(P2)).sum(axis=1)


def _margin_binary(P2: np.ndarray) -> np.ndarray:
    return np.abs(P2[:, 1] - P2[:, 0])


def class_counts_str(y_str: np.ndarray, classes: List[str]) -> str:
    d = {c: int(np.sum(y_str == c)) for c in classes}
    return json.dumps(d, ensure_ascii=False)


def build_bags_from_df(df: pd.DataFrame, spec_cols: List[str]) -> Dict[str, np.ndarray]:
    bags = {}
    for sid, g in df.groupby("Sample"):
        X = g[spec_cols].to_numpy(np.float32)
        if X.ndim == 2 and X.shape[0] > 0:
            bags[str(sid)] = X
    return bags


def build_cellmeta_from_df(df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    meta = {}
    has_cell = "Cell" in df.columns
    has_celltype = "CellType" in df.columns
    has_batch = "Batch" in df.columns
    for sid, g in df.groupby("Sample"):
        if has_cell:
            cell_id = g["Cell"].astype(str).to_numpy()
        else:
            cell_id = g.index.astype(str).to_numpy()
        out = {"cell_id": cell_id}
        if has_celltype:
            out["CellType"] = g["CellType"].astype(str).to_numpy()
        if has_batch:
            out["Batch"] = g["Batch"].astype(str).to_numpy()
        meta[str(sid)] = out
    return meta

class MILBagsDataset(Dataset):
    def __init__(self, sample_ids: List[str], bags: Dict[str, np.ndarray], y_by_sample: Dict[str, int],
                 mean: np.ndarray, std: np.ndarray, seed: int = 0, max_instances: Optional[int] = None):
        self.sample_ids = list(sample_ids)
        self.bags = bags
        self.y_by_sample = y_by_sample
        self.mean = mean
        self.std = std
        self.seed = int(seed)
        self.max_instances = max_instances
        self.epoch = 0

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sid = self.sample_ids[idx]
        X = self.bags[sid]
        y = int(self.y_by_sample[sid])
        X = (X - self.mean) / (self.std + 1e-8)
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
def predict_probs_mil(model: nn.Module, bags: Dict[str, np.ndarray], y_by_sample: Dict[str, int],
                      sample_list: List[str], mean: np.ndarray, std: np.ndarray,
                      batch_size: int, device: str) -> Tuple[List[str], np.ndarray]:
    ds = MILBagsDataset(sample_list, bags, y_by_sample, mean, std, seed=0, max_instances=None)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_pad)
    model.eval()
    ids, ps = [], []
    for x_pad, mask, y, sids in dl:
        x_pad = x_pad.to(device)
        mask = mask.to(device)
        logit = model(x_pad, mask)
        p1 = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float64)
        P2 = np.stack([1.0 - p1, p1], axis=1)
        ids.extend(sids); ps.append(P2)
    P = np.concatenate(ps, axis=0) if ps else np.zeros((0, 2), dtype=np.float64)
    return ids, P

@torch.no_grad()
def export_attention_topk_for_samples(model: AttnMIL, bags: Dict[str, np.ndarray], cellmeta_by_sample: Dict[str, Dict[str, np.ndarray]],
                                      y_by_sample: Dict[str, int], sample_list: List[str], mean: np.ndarray, std: np.ndarray,
                                      device: str, topk: int, task_name: str, disease_label: str, rep: int, fold: int,
                                      view: str, y_pred_by_sample: Dict[str, int], prob_by_sample: Dict[str, np.ndarray]):
    model.eval()
    rows_cells, rows_pat = [], []
    topk = int(max(1, topk))
    for sid in sample_list:
        X = bags.get(sid)
        if X is None or X.size == 0:
            continue
        Xn = (X - mean) / (std + 1e-8)
        x = torch.from_numpy(Xn).float().unsqueeze(0).to(device)
        mask = torch.ones((1, Xn.shape[0]), dtype=torch.bool, device=device)
        _, w = model(x, mask, return_attn=True)
        w1 = w.squeeze(0).detach().cpu().numpy().astype(np.float64)
        w1 = np.clip(w1, 0.0, 1.0)
        s = float(w1.sum())
        if not np.isfinite(s) or s <= 0:
            w1 = np.ones_like(w1, dtype=np.float64) / max(1, w1.size)
        else:
            w1 = w1 / (s + ATTN_EPS)
        Tn = int(w1.size)
        y_true = int(y_by_sample.get(sid, -1))
        y_pred = int(y_pred_by_sample.get(sid, -1))
        meta = cellmeta_by_sample.get(sid, {})
        cell_ids = meta.get("cell_id", np.array([str(i) for i in range(Tn)], dtype=object))
        if len(cell_ids) != Tn:
            cell_ids = np.array([str(i) for i in range(Tn)], dtype=object)
        ent = float(-(w1 * np.log(np.clip(w1, ATTN_EPS, 1.0))).sum())
        top1 = float(np.max(w1)) if Tn > 0 else np.nan
        eff_n = float(1.0 / (np.sum(w1 * w1) + ATTN_EPS)) if Tn > 0 else np.nan
        k5 = min(5, Tn)
        top5_sum = float(np.sort(w1)[-k5:].sum()) if Tn > 0 else np.nan
        prow = dict(
            task=task_name, disease_label=str(disease_label), rep=int(rep), fold=int(fold), view=str(view),
            SampleKey=str(sid), y_true=y_true, y_true_name=CONTROL_LABEL if y_true == 0 else disease_label,
            y_pred=y_pred, y_pred_name=CONTROL_LABEL if y_pred == 0 else disease_label,
            correct=int(y_true == y_pred), n_cells=int(Tn), attn_entropy=ent, attn_top1=top1,
            attn_top5_sum=top5_sum, attn_eff_n=eff_n,
        )
        P = prob_by_sample.get(sid)
        if P is not None:
            prow[f"p_{CONTROL_LABEL}"] = float(P[0]); prow[f"p_{disease_label}"] = float(P[1])
        rows_pat.append(prow)
        k = min(topk, Tn)
        for rank, j in enumerate(np.argsort(-w1)[:k], 1):
            r = dict(
                task=task_name, disease_label=str(disease_label), rep=int(rep), fold=int(fold), view=str(view),
                SampleKey=str(sid), y_true=y_true, y_true_name=CONTROL_LABEL if y_true == 0 else disease_label,
                y_pred=y_pred, y_pred_name=CONTROL_LABEL if y_pred == 0 else disease_label,
                correct=int(y_true == y_pred), cell_id=str(cell_ids[j]), attn_weight=float(w1[j]),
                attn_rank=int(rank), n_cells=int(Tn),
            )
            if "CellType" in meta:
                try: r["CellType"] = str(meta["CellType"][j])
                except Exception: pass
            if "Batch" in meta:
                try: r["Batch"] = str(meta["Batch"][j])
                except Exception: pass
            if P is not None:
                r[f"p_{CONTROL_LABEL}"] = float(P[0]); r[f"p_{disease_label}"] = float(P[1])
            rows_cells.append(r)
    return pd.DataFrame(rows_cells), pd.DataFrame(rows_pat)


def metrics_safe_binary(y_true: np.ndarray, P2: np.ndarray):
    yhat = P2.argmax(axis=1)
    acc = float(accuracy_score(y_true, yhat))
    bal = float(balanced_accuracy_score(y_true, yhat))
    f1m = float(f1_score(y_true, yhat, average="macro", labels=[0, 1], zero_division=0))
    auc = np.nan
    try:
        if len(np.unique(y_true)) == 2:
            auc = float(roc_auc_score(y_true, P2[:, 1]))
    except Exception:
        auc = np.nan
    ll = float(log_loss(y_true, P2, labels=[0, 1]))
    return acc, bal, f1m, auc, ll


def torch_load_trusted_checkpoint(path: Path) -> Dict:
    """
    Load a checkpoint bundle produced by our own training script.

    PyTorch 2.6 changed torch.load's default to weights_only=True, which
    intentionally rejects checkpoint bundles containing NumPy arrays / metadata.
    Our binary MIL checkpoints intentionally store model_state + mean/std + meta,
    so public replay must load them with weights_only=False when that argument
    is available. Only use this on trusted checkpoints generated by this project.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support the weights_only keyword.
        return torch.load(path, map_location="cpu")


def load_checkpoint_bundle(row: pd.Series, view: str) -> Dict:
    rel = str(row.get(f"{view}_checkpoint_rel_dir", "")).strip()
    if not rel:
        rep = int(row["rep"]); fold = int(row["fold"]); task_dir = str(row["task_dir"])
        rel = str(Path("checkpoints") / task_dir / f"rep{rep:02d}_fold{fold:02d}_view{view}_FINAL")
    root = Path(LOCKED_INPUT_DIR) / rel
    pt_path = root / "model.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint model.pt for view={view}: {pt_path}")
    payload = torch_load_trusted_checkpoint(pt_path)
    meta = dict(payload.get("meta", {}))
    params_dict = dict(meta.get("params", {}))
    params = MILParams(**{k: v for k, v in params_dict.items() if k in MILParams.__dataclass_fields__}) if params_dict else MILParams()
    in_dim = int(meta.get("in_dim", len(payload["mean"])))
    model = AttnMIL(in_dim=in_dim, embed_dim=int(params.embed_dim), attn_dim=int(params.attn_dim), dropout=float(params.dropout)).to(DEVICE)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return dict(model=model, mean=np.asarray(payload["mean"], dtype=np.float32), std=np.asarray(payload["std"], dtype=np.float32), params=params, meta=meta, root=root)


def copy_locked_audit_artifacts():
    if not COPY_LOCKED_AUDIT_ARTIFACTS_TO_OUT:
        return
    os.makedirs(OUT_DIR_BASE, exist_ok=True)
    for name in [
        "selected_task_reps_MIL_BINARY_LOCKED_RUNS.csv",
        "selected_task_reps_EXPECTED_METRICS.csv",
        "selected_task_reps_EXPECTED_SPLITS.csv",
        "selected_task_reps_EXPECTED_PATIENT_PREDS.csv",
        "selected_task_reps_EXPECTED_ATTN_CELLS.csv",
        "selected_task_reps_EXPECTED_ATTN_PATIENT_SUMMARY.csv",
    ]:
        src = Path(LOCKED_INPUT_DIR) / name
        if src.exists():
            shutil.copy2(src, Path(OUT_DIR_BASE) / name)
    for dname in ["checkpoints", "tuned"]:
        src = Path(LOCKED_INPUT_DIR) / dname
        if src.exists():
            dst = Path(OUT_DIR_BASE) / dname
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    # Copy selected original attention exports for audit under attn_expected/.
    src = Path(LOCKED_INPUT_DIR) / "attn"
    if src.exists():
        dst = Path(OUT_DIR_BASE) / "attn_expected"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def validate_expected_reps(locked: pd.DataFrame):
    for task_short, expected in EXPECTED_SELECTED_REP_BY_TASK.items():
        sub = locked[locked["task_short"].astype(str) == str(task_short)]
        if sub.empty:
            raise ValueError(f"Locked CSV has no rows for expected task {task_short}")
        reps = sorted(pd.to_numeric(sub["rep"], errors="coerce").dropna().astype(int).unique().tolist())
        if reps != [int(expected)]:
            raise ValueError(f"{task_short}: expected selected rep {expected}, but locked CSV contains reps {reps}")


def run_task(df_all: pd.DataFrame, locked_task: pd.DataFrame, disease_label: str) -> pd.DataFrame:
    task_dir = f"CTRL_vs_{disease_label}"
    task_name = f"stage1_MIL_{CONTROL_LABEL}_vs_{disease_label}"
    out_dir = Path(OUT_DIR_BASE) / task_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    attn_dir = out_dir / "attn"
    if SAVE_ATTENTION_EXPORTS:
        attn_dir.mkdir(parents=True, exist_ok=True)

    df = df_all[df_all["Group"].isin([CONTROL_LABEL, disease_label])].copy()
    if df.empty:
        raise ValueError(f"No Raman rows for {CONTROL_LABEL} vs {disease_label}")
    cellmeta_by_sample = build_cellmeta_from_df(df)
    sample_group = df.groupby("Sample")["Group"].first()
    all_samples = sample_group.index.astype(str).values
    y_by_sample = {str(s): int(0 if sample_group.loc[s] == CONTROL_LABEL else 1) for s in all_samples}

    all_cols, wav = get_spec_cols_and_wav(df)
    view_bags = {}
    for vname, region in VIEWS:
        cols = select_region(all_cols, wav, region)
        view_bags[vname] = build_bags_from_df(df, cols)

    rows, pred_rows = [], []
    attn_cells_all, attn_pat_all = [], []
    locked_task = locked_task.copy()
    locked_task["fold"] = pd.to_numeric(locked_task["fold"], errors="coerce").astype(int)
    locked_task = locked_task.sort_values("fold")

    for _, row in locked_task.iterrows():
        rep = int(row["rep"]); fold = int(row["fold"])
        idx_te = parse_json_list(row.get("outer_test_samples_json", ""))
        if not idx_te:
            raise ValueError(f"Missing outer_test_samples_json for {task_name} rep={rep} fold={fold}")
        missing = [s for s in idx_te if s not in y_by_sample]
        if missing:
            raise ValueError(f"Locked outer-test samples not found in current Raman data. Examples: {missing[:10]}")
        y_te = np.array([y_by_sample[s] for s in idx_te], dtype=int)
        y_te_str = np.array([CONTROL_LABEL if y_by_sample[s] == 0 else disease_label for s in idx_te], dtype=object)

        use_view_fusion = bool(int(float(row.get("use_view_fusion", 1))))
        active_views = [v for v, _ in VIEWS] if use_view_fusion else ["all"]
        models = {v: load_checkpoint_bundle(row, v) for v in active_views}

        Pte = {}
        for v in active_views:
            _, P2 = predict_probs_mil(
                model=models[v]["model"], bags=view_bags[v], y_by_sample=y_by_sample,
                sample_list=idx_te, mean=models[v]["mean"], std=models[v]["std"],
                batch_size=max(2, int(models[v]["params"].batch_size)), device=DEVICE,
            )
            Pte[v] = P2

        w_all = float(row.get("w_all", 1.0)); w_fp = float(row.get("w_fp", 0.0)); w_ch = float(row.get("w_ch", 0.0))
        if use_view_fusion:
            P_raw = w_all * Pte["all"] + w_fp * Pte["fp"] + w_ch * Pte["ch"]
        else:
            P_raw = Pte["all"]
        P_raw = P_raw / (P_raw.sum(axis=1, keepdims=True) + 1e-12)
        T = float(row.get("T", 1.0))
        P_cal = apply_temperature_binary(P_raw, T)
        yhat = P_cal.argmax(axis=1)
        acc, bal, f1m, auc, ll = metrics_safe_binary(y_te, P_cal)
        cm = confusion_matrix(y_te, yhat, labels=[0, 1]).flatten().tolist()

        outrow = dict(
            task=task_name, disease_label=disease_label, rep=rep, fold=fold,
            outer_seed_used=row.get("outer_seed_used", np.nan), outer_tries=row.get("outer_tries", np.nan),
            fixed_epochs=row.get("fixed_epochs", np.nan), final_epoch_rule=row.get("final_epoch_rule", ""),
            E_star=row.get("E_star", np.nan), n_test_patients=len(idx_te),
            outer_test_counts_json=class_counts_str(y_te_str, [CONTROL_LABEL, disease_label]),
            use_view_fusion=int(use_view_fusion), w_all=w_all, w_fp=w_fp, w_ch=w_ch, T=T,
            val_logloss=row.get("val_logloss", np.nan), acc=acc, bal_acc=bal, macro_f1=f1m,
            auc=float(auc) if auc == auc else np.nan, logloss=ll, cm_flat=",".join(map(str, cm)),
            replay_source="checkpoint_replay",
        )
        for v in active_views:
            outrow[f"{v}_final_epochs"] = int(models[v]["meta"].get("E_star", row.get("E_star", -1)))
            outrow[f"{v}_checkpoint_rel_dir"] = str(row.get(f"{v}_checkpoint_rel_dir", ""))
        rows.append(outrow)

        for i, sk in enumerate(idx_te):
            pred = dict(
                task=task_name, disease_label=disease_label, SampleKey=str(sk), rep=rep, fold=fold,
                y_true=int(y_te[i]), y_true_name=CONTROL_LABEL if y_te[i] == 0 else disease_label,
                y_pred=int(yhat[i]), y_pred_name=CONTROL_LABEL if int(yhat[i]) == 0 else disease_label,
                correct=int(int(yhat[i]) == int(y_te[i])),
                p_Control=float(P_cal[i, 0]), **{f"p_{disease_label}": float(P_cal[i, 1])},
                praw_Control=float(P_raw[i, 0]), **{f"praw_{disease_label}": float(P_raw[i, 1])},
                T=T, w_all=w_all, w_fp=w_fp, w_ch=w_ch,
                margin=float(abs(P_cal[i, 1] - P_cal[i, 0])), entropy=float(_entropy_binary(P_cal[[i], :])[0]),
            )
            pred_rows.append(pred)

        if SAVE_ATTENTION_EXPORTS:
            prob_map = {sid: P_cal[i].astype(float) for i, sid in enumerate(idx_te)}
            y_pred_map = {sid: int(yhat[i]) for i, sid in enumerate(idx_te)}
            attn_views = ["all"] if ATTN_EXPORT_VIEWS == "all_only" else list(active_views)
            for v in attn_views:
                cells, pats = export_attention_topk_for_samples(
                    model=models[v]["model"], bags=view_bags[v], cellmeta_by_sample=cellmeta_by_sample,
                    y_by_sample=y_by_sample, sample_list=idx_te, mean=models[v]["mean"], std=models[v]["std"],
                    device=DEVICE, topk=ATTN_EXPORT_TOPK, task_name=task_name, disease_label=disease_label,
                    rep=rep, fold=fold, view=v, y_pred_by_sample=y_pred_map, prob_by_sample=prob_map,
                )
                if len(cells): attn_cells_all.append(cells)
                if len(pats): attn_pat_all.append(pats)

    res = pd.DataFrame(rows)
    pred_df = pd.DataFrame(pred_rows)
    to_csv_safe(res, str(out_dir / f"{task_name}_checkpoint_replay.csv"))
    to_csv_safe(pred_df, str(out_dir / f"{task_name}_PATIENT_PREDS.csv"))

    # Copy selected split table into the task folder if available.
    exp_splits = Path(LOCKED_INPUT_DIR) / "selected_task_reps_EXPECTED_SPLITS.csv"
    if exp_splits.exists():
        sp = read_csv_robust(str(exp_splits))
        if "disease_label" in sp.columns:
            sps = sp[sp["disease_label"].astype(str) == str(disease_label)].copy()
        else:
            sps = sp[sp["task"].astype(str) == str(task_name)].copy()
        if len(sps):
            to_csv_safe(sps, str(out_dir / f"{task_name}_SPLITS.csv"))

    if SAVE_ATTENTION_EXPORTS:
        if attn_cells_all:
            to_csv_safe(pd.concat(attn_cells_all, ignore_index=True), str(attn_dir / f"{task_name}_ATTN_CELLS.csv"))
        if attn_pat_all:
            to_csv_safe(pd.concat(attn_pat_all, ignore_index=True), str(attn_dir / f"{task_name}_ATTN_PATIENT_SUMMARY.csv"))

    summary = dict(
        task=task_name, disease_label=disease_label, n_rows=int(len(res)),
        acc_mean=float(res["acc"].mean()), acc_std=float(res["acc"].std()),
        bal_mean=float(res["bal_acc"].mean()), bal_std=float(res["bal_acc"].std()),
        f1_mean=float(res["macro_f1"].mean()), f1_std=float(res["macro_f1"].std()),
        logloss_mean=float(res["logloss"].mean()),
        auc_mean=float(res["auc"].dropna().mean()) if res["auc"].notna().any() else float("nan"),
        use_view_fusion=int(res["use_view_fusion"].iloc[0]) if len(res) else np.nan,
        selected_rep=int(res["rep"].iloc[0]) if len(res) else np.nan,
        curves_dir="", figures_dir="", tuned_dir=os.path.join(OUT_DIR_BASE, "tuned", task_dir),
        checkpoints_dir=os.path.join(OUT_DIR_BASE, "checkpoints", task_dir),
        splits_csv=str(out_dir / f"{task_name}_SPLITS.csv"),
        patient_preds_csv=str(out_dir / f"{task_name}_PATIENT_PREDS.csv"),
        replay_source="checkpoint_replay_no_curves_no_figures",
    )
    to_csv_safe(pd.DataFrame([summary]), str(out_dir / f"{task_name}_checkpoint_replay_SUMMARY.csv"))
    return summary


def main():
    os.makedirs(OUT_DIR_BASE, exist_ok=True)
    locked = read_csv_robust(LOCKED_RUNS_CSV)
    required = ["rep", "fold", "task_short", "task_dir", "disease_label", "outer_test_samples_json"]
    missing = [c for c in required if c not in locked.columns]
    if missing:
        raise ValueError(f"Locked CSV missing columns: {missing}. Found: {locked.columns.tolist()}")
    validate_expected_reps(locked)

    copy_locked_audit_artifacts()

    df = read_csv_robust(RAMAN_CSV, dtype=None)
    for c in ["Batch", "Group", "Sample", "CellType", "Cell"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    if "Sample" not in df.columns or "Group" not in df.columns:
        raise ValueError("Raman CSV must contain at least Sample and Group columns.")

    summaries = []
    for disease in DISEASE_LABELS:
        task_short = f"control_vs_{disease}"
        lt = locked[locked["task_short"].astype(str) == task_short].copy()
        if lt.empty:
            print(f"Skipping {task_short}: no locked rows.")
            continue
        summaries.append(run_task(df, lt, disease_label=disease))

    if summaries:
        to_csv_safe(pd.DataFrame(summaries), os.path.join(OUT_DIR_BASE, "STAGE1_ALL_TASKS_CHECKPOINT_REPLAY_SUMMARY.csv"))
    print("DONE. Output:", OUT_DIR_BASE)
    print("No curves/ or figures/ folders were created.")


if __name__ == "__main__":
    main()
