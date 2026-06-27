#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)



RAMAN_CSV = "raman_added2.csv"
OUT_DIR = "./publication_stage1_binary_fpch_ablation_reproduction_out_CV5"

CONTROL_LABEL = "Control"

LOCKED_INPUT_DIR = "./publication_stage1_binary_ablation_inputs_CV5"
LOCKED_RUNS_CSV = os.path.join(
    LOCKED_INPUT_DIR,
    "selected_task_reps_STAGE1_BINARY_FPCH_LOCKED_RUNS.csv",
)


EXPECTED_SELECTED_REP_BY_TASK = None

RAMAN_SAMPLE_COL = "Sample"
RAMAN_LABEL_COL = "Group"


VIEWS = [("fp", "fingerprint"), ("ch", "ch")]
PROB_EPS = 1e-6
CSV_ENCODING = "utf-8-sig"

ENCODED_CLASS_ORDER_BY_TASK = {
    "control_vs_Liver": ["Control", "Liver"],
    "control_vs_Bile": ["Bile", "Control"],
    "control_vs_Pancreas": ["Control", "Pancreas"],
}

FORBIDDEN_FEATURE_PATTERNS = (
   
)



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


def _norm_feature_name(x: str) -> str:
    return str(x).strip().lower().replace(" ", "").replace("-", "_")


def assert_no_forbidden_feature_names(cols: List[str], where: str):
    bad = []
    for c in cols:
        n = _norm_feature_name(c)
        if any(p in n for p in FORBIDDEN_FEATURE_PATTERNS):
            bad.append(c)
    if bad:
        raise RuntimeError(f"Forbidden cell-count-like feature(s) found in {where}: {bad}")



def encoded_class_order_for_task(task: str, disease_label: str) -> List[str]:
    if task in ENCODED_CLASS_ORDER_BY_TASK:
        classes = list(ENCODED_CLASS_ORDER_BY_TASK[task])
    else:
        # Safe fallback: explicit sklearn LabelEncoder order.
        classes = sorted([str(CONTROL_LABEL), str(disease_label)])
    expected_set = {str(CONTROL_LABEL), str(disease_label)}
    if set(classes) != expected_set or len(classes) != 2:
        raise RuntimeError(
            f"Bad encoded class order for task={task}: {classes}; expected labels={sorted(expected_set)}"
        )
    return classes



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
    assert_no_forbidden_feature_names(cols, "spectral column list")
    if len(cols) == 0:
        raise ValueError("No spectral columns found. Raman wavelength columns must be numeric strings.")
    return cols, wav


def select_region(cols: List[str], wav: np.ndarray, region: str) -> List[str]:
    if region == "fingerprint":
        m = (wav >= 600) & (wav <= 1800)
    elif region == "ch":
        m = (wav >= 2800) & (wav <= 3100)
    else:
        raise ValueError(region)
    out = [c for c, keep in zip(cols, m) if keep]
    if len(out) == 0:
        raise ValueError(f"No spectral columns selected for region={region}. Check wavelength headers.")
    assert_no_forbidden_feature_names(out, f"Raman region={region}")
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
    dfz["Sample"] = sample_ids
    z_mean = dfz.groupby("Sample").mean()
    P = sigmoid(z_mean.values)
    P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    return pd.DataFrame(P, index=z_mean.index, columns=[f"p_{c}" for c in class_names])


def apply_temperature_multiclass(P: np.ndarray, T: float) -> np.ndarray:
    P = np.clip(P, 1e-12, 1.0)
    logits_ = np.log(P) / float(T)
    logits_ -= logits_.max(axis=1, keepdims=True)
    expv = np.exp(logits_)
    return expv / (expv.sum(axis=1, keepdims=True) + 1e-12)


def class_counts_str(y_str: np.ndarray, classes: List[str]) -> str:
    d = {c: int(np.sum(y_str == c)) for c in classes}
    return json.dumps(d, ensure_ascii=False)



def metrics_safe(y_true: np.ndarray, P: np.ndarray, n_classes: int):
    yhat = P.argmax(axis=1)
    acc = float(accuracy_score(y_true, yhat))
    bal = float(balanced_accuracy_score(y_true, yhat))
    f1m = float(f1_score(y_true, yhat, average="macro", labels=list(range(n_classes)), zero_division=0))
    aucv = np.nan
    try:
        if n_classes == 2 and len(np.unique(y_true)) == 2:
            aucv = float(roc_auc_score(y_true, P[:, 1]))
    except Exception:
        aucv = np.nan
    return acc, bal, f1m, aucv


def train_svm_pca(Xtr: np.ndarray, ytr: np.ndarray, best: Dict, seed: int):
    if len(np.unique(ytr)) < 2:
        raise ValueError("Final training set has only one class; check locked split.")
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


def predict_proba(sc: StandardScaler, pca: PCA, svm: SVC, X: np.ndarray) -> np.ndarray:
    return svm.predict_proba(pca.transform(sc.transform(X)))


def train_view_model_locked(
    df: pd.DataFrame,
    spec_cols: List[str],
    classes: List[str],
    train_samples: set,
    seed: int,
    fixed_multi: Dict,
):
    le = LabelEncoder().fit(classes)
    
    if list(le.classes_) != list(classes):
        raise RuntimeError(f"Class order mismatch. classes={classes}, LabelEncoder.classes_={list(le.classes_)}")

    X = df[spec_cols].to_numpy(np.float32)
    tr_idx = np.where(df[RAMAN_SAMPLE_COL].isin(train_samples).values)[0]
    if len(tr_idx) == 0:
        raise RuntimeError("No Raman rows found for locked outer_train patients.")

    Xtr = X[tr_idx]
    ytr = le.transform(df.iloc[tr_idx][RAMAN_LABEL_COL].values)
    sc, pca, svm = train_svm_pca(Xtr, ytr, fixed_multi, seed=seed)
    return dict(sc=sc, pca=pca, svm=svm, spec_cols=spec_cols, classes=classes, le=le, best=fixed_multi)


def predict_patient_probs_view(df: pd.DataFrame, model: Dict, samples_set: set):
    spec_cols = model["spec_cols"]
    X = df[spec_cols].to_numpy(np.float32)

    idx = np.where(df[RAMAN_SAMPLE_COL].isin(samples_set).values)[0]
    if len(idx) == 0:
        return None

    samp = df.iloc[idx][RAMAN_SAMPLE_COL].values
    p_cell = predict_proba(model["sc"], model["pca"], model["svm"], X[idx])
    return agg_probs_to_sample_logit(samp, p_cell, model["classes"])



def parse_json_list(x) -> List[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
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
            s2 = s.strip("[]").strip()
            if not s2:
                return []
            return [t.strip().strip('"').strip("'") for t in s2.split(",") if t.strip()]
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        return [s]


def load_locked_runs(path: str) -> pd.DataFrame:
    df = read_csv_robust(path, dtype=str)
    required = [
        "task", "disease", "selected_rep", "fold",
        "outer_train_json", "outer_test_json",
        "final_seed",
        "w_fp", "w_ch", "T",
        "fp_n_pc", "fp_C", "fp_gamma",
        "ch_n_pc", "ch_C", "ch_gamma",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Locked runs CSV is missing columns: {missing}")

    df = df.copy()
    for c in ["selected_rep", "fold", "final_seed", "fp_n_pc", "ch_n_pc"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
    for c in ["w_fp", "w_ch", "T", "fp_C", "ch_C"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(float)

    df["outer_train_list"] = df["outer_train_json"].apply(parse_json_list)
    df["outer_test_list"] = df["outer_test_json"].apply(parse_json_list)

    if EXPECTED_SELECTED_REP_BY_TASK is not None:
        for task, expected_rep in EXPECTED_SELECTED_REP_BY_TASK.items():
            sub = df[df["task"] == task]
            if sub.empty:
                raise ValueError(f"Expected task not found in locked runs: {task}")
            reps = sorted(sub["selected_rep"].unique().tolist())
            if reps != [int(expected_rep)]:
                raise ValueError(f"Task {task} has selected_rep={reps}, expected {[int(expected_rep)]}")

    for task, sub in df.groupby("task"):
        reps = sorted(sub["selected_rep"].unique().tolist())
        if len(reps) != 1:
            raise ValueError(f"Task {task} has multiple selected reps inside locked file: {reps}")
        if sub["fold"].duplicated().any():
            raise ValueError(f"Task {task} has duplicate fold rows in locked file.")

    return df.sort_values(["task", "fold"]).reset_index(drop=True)


def locked_view_params(row: pd.Series, vname: str) -> Dict:
    return {
        "n_pc": int(row[f"{vname}_n_pc"]),
        "C": float(row[f"{vname}_C"]),
        "gamma": str(row[f"{vname}_gamma"]),
    }



def run_locked_task(df_all: pd.DataFrame, task: str, disease_label: str, locked_rows: pd.DataFrame) -> pd.DataFrame:
    os.makedirs(OUT_DIR, exist_ok=True)

    classes = encoded_class_order_for_task(task, disease_label)
    allowed_labels = {CONTROL_LABEL, disease_label}
    n_classes = 2
    all_cols, wav = get_spec_cols_and_wav(df_all)

    df = df_all[df_all[RAMAN_LABEL_COL].isin(allowed_labels)].copy()
    sample_group = df.groupby(RAMAN_SAMPLE_COL)[RAMAN_LABEL_COL].first()

    out_csv = os.path.join(OUT_DIR, f"stage1_{task}_FPCH.csv")
    out_patient_csv = os.path.join(OUT_DIR, f"stage1_{task}_FPCH_patient_preds.csv")
    out_roc_csv = os.path.join(OUT_DIR, f"stage1_{task}_FPCH_roc_points.csv")
    for f in [out_csv, out_patient_csv, out_roc_csv]:
        if os.path.exists(f):
            os.remove(f)

    le = LabelEncoder().fit(classes)
    if list(le.classes_) != list(classes):
        raise RuntimeError(f"Class order mismatch. classes={classes}, LabelEncoder.classes_={list(le.classes_)}")

    rows, patient_rows, roc_rows = [], [], []

    for _, row_locked in locked_rows.sort_values("fold").iterrows():
        rep = int(row_locked["selected_rep"])
        fold_id = int(row_locked["fold"])
        outer_train = set(map(str, row_locked["outer_train_list"]))
        outer_test = set(map(str, row_locked["outer_test_list"]))
        final_seed = int(row_locked["final_seed"])

        missing_train = sorted([s for s in outer_train if s not in sample_group.index])
        missing_test = sorted([s for s in outer_test if s not in sample_group.index])
        if missing_train or missing_test:
            raise RuntimeError(
                f"[{task}] Locked split contains samples not present in Raman data. "
                f"missing_train={missing_train[:5]}, missing_test={missing_test[:5]}"
            )

        models = {}
        for vname, region in VIEWS:
            cols = select_region(all_cols, wav, region)
            best = locked_view_params(row_locked, vname)
            models[vname] = train_view_model_locked(
                df=df,
                spec_cols=cols,
                classes=classes,
                train_samples=outer_train,
                seed=final_seed,
                fixed_multi=best,
            )

        P_te = {vname: predict_patient_probs_view(df, models[vname], outer_test) for vname, _ in VIEWS}
        if P_te["fp"] is None or P_te["fp"].empty or P_te["ch"] is None or P_te["ch"].empty:
            raise RuntimeError(f"[{task}] Outer test produced empty probabilities.")
        if not P_te["fp"].index.equals(P_te["ch"].index):
            raise RuntimeError(f"[{task}] FP and CH patient indices differ; fusion would be corrupted.")

        idx_te = P_te["fp"].index
        y_te_str = sample_group.loc[idx_te].values
        y_te = le.transform(y_te_str)

        Pte_fp = P_te["fp"].loc[idx_te].to_numpy()
        Pte_ch = P_te["ch"].loc[idx_te].to_numpy()

        w_fp = float(row_locked["w_fp"])
        w_ch = float(row_locked["w_ch"])
        T = float(row_locked["T"])

        P = w_fp * Pte_fp + w_ch * Pte_ch
        P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
        P_T = apply_temperature_multiclass(P, T)

        acc, bal, f1m, aucv = metrics_safe(y_te, P_T, n_classes=n_classes)
        yhat = P_T.argmax(axis=1)
        cm = confusion_matrix(y_te, yhat, labels=np.arange(n_classes)).flatten().tolist()

        out_row = dict(
            selected_rep=rep,
            rep=rep,
            fold=fold_id,
            task=task,
            disease=disease_label,
            ablation="FPCH_only",
            encoded_class_order_json=json.dumps(classes, ensure_ascii=False),
            final_seed=final_seed,
            n_test_patients=int(len(idx_te)),
            outer_test_counts_json=class_counts_str(y_te_str, classes),
            w_fp=w_fp,
            w_ch=w_ch,
            T=T,
            acc=float(acc),
            bal_acc=float(bal),
            macro_f1=float(f1m),
            auc=float(aucv) if aucv == aucv else np.nan,
            cm_flat=",".join(map(str, cm)),
        )
        for vname in ["fp", "ch"]:
            best = locked_view_params(row_locked, vname)
            out_row[f"{vname}_n_pc"] = int(best["n_pc"])
            out_row[f"{vname}_C"] = float(best["C"])
            out_row[f"{vname}_gamma"] = str(best["gamma"])

        for metric_col in ["expected_acc", "expected_bal_acc", "expected_cm_flat"]:
            if metric_col in row_locked.index:
                out_row[metric_col] = row_locked[metric_col]

        rows.append(out_row)

        for i, sk in enumerate([str(s) for s in idx_te.tolist()]):
            yi = int(y_te[i])
            ypi = int(yhat[i])
            patient_rows.append(dict(
                selected_rep=rep,
                rep=rep,
                fold=fold_id,
                task=task,
                disease=disease_label,
                ablation="FPCH_only",
                encoded_class_order_json=json.dumps(classes, ensure_ascii=False),
                SampleKey=str(sk),
                y_true=yi,
                y_true_name=str(classes[yi]),
                y_pred=ypi,
                y_pred_name=str(classes[ypi]),
                correct=int(yi == ypi),
                w_fp=w_fp,
                w_ch=w_ch,
                T=T,
                **{f"p_{classes[0]}": float(P_T[i, 0]), f"p_{classes[1]}": float(P_T[i, 1])},
                **{f"praw_{classes[0]}": float(P[i, 0]), f"praw_{classes[1]}": float(P[i, 1])},
                **{f"fp_p_{classes[0]}": float(Pte_fp[i, 0]), f"fp_p_{classes[1]}": float(Pte_fp[i, 1])},
                **{f"ch_p_{classes[0]}": float(Pte_ch[i, 0]), f"ch_p_{classes[1]}": float(Pte_ch[i, 1])},
            ))

        if len(np.unique(y_te)) == 2:
            fpr, tpr, thr = roc_curve(y_te, P_T[:, 1])
            roc_rows.append(dict(
                selected_rep=rep,
                rep=rep,
                fold=fold_id,
                task=task,
                disease=disease_label,
                positive_class_id=1,
                positive_class_name=classes[1],
                fpr_json=json.dumps(fpr.tolist()),
                tpr_json=json.dumps(tpr.tolist()),
                thr_json=json.dumps(thr.tolist()),
            ))

        print(
            f"[{task} FPCH] selected_rep={rep} fold={fold_id} "
            f"acc={acc:.3f} bal={bal:.3f} f1={f1m:.3f} auc={aucv if aucv==aucv else float('nan'):.3f} "
            f"T={T:.2f} w=(fp={w_fp:.1f},ch={w_ch:.1f}) class_order={classes}"
        )

    res = pd.DataFrame(rows)
    pat = pd.DataFrame(patient_rows)
    roc = pd.DataFrame(roc_rows)
    to_csv_safe(res, out_csv)
    to_csv_safe(pat, out_patient_csv)
    if len(roc):
        to_csv_safe(roc, out_roc_csv)

    summary = dict(
        task=task,
        disease=disease_label,
        ablation="FPCH_only",
        selected_rep=int(res["selected_rep"].iloc[0]),
        encoded_class_order_json=json.dumps(classes, ensure_ascii=False),
        n_rows=int(len(res)),
        acc_mean=float(res["acc"].mean()),
        acc_std=float(res["acc"].std()),
        bal_mean=float(res["bal_acc"].mean()),
        bal_std=float(res["bal_acc"].std()),
        f1_mean=float(res["macro_f1"].mean()),
        auc_mean=float(res["auc"].dropna().mean()) if res["auc"].notna().any() else float("nan"),
    )
    sum_csv = os.path.join(OUT_DIR, f"stage1_{task}_FPCH_SUMMARY.csv")
    to_csv_safe(pd.DataFrame([summary]), sum_csv)
    print("Saved:", out_csv)
    print("Saved:", sum_csv)
    print("Saved:", out_patient_csv)
    if len(roc):
        print("Saved:", out_roc_csv)
    print("SUMMARY:", json.dumps(summary, indent=2, ensure_ascii=False))
    return res


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = read_csv_robust(RAMAN_CSV)
    for c in ["Batch", RAMAN_LABEL_COL, RAMAN_SAMPLE_COL, "CellType", "Cell"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    if RAMAN_LABEL_COL not in df.columns or RAMAN_SAMPLE_COL not in df.columns:
        raise ValueError(f"Raman CSV must contain columns {RAMAN_LABEL_COL!r} and {RAMAN_SAMPLE_COL!r}.")

    locked = load_locked_runs(LOCKED_RUNS_CSV)
    all_results = []
    for task, sub in locked.groupby("task", sort=False):
        disease_label = str(sub["disease"].iloc[0])
        all_results.append(run_locked_task(df, task=task, disease_label=disease_label, locked_rows=sub))

    combined = pd.concat(all_results, ignore_index=True)
    combined_path = os.path.join(OUT_DIR, "stage1_binary_FPCH_selected_task_reps_ALL_TASKS.csv")
    to_csv_safe(combined, combined_path)
    print("Saved combined:", combined_path)


if __name__ == "__main__":
    main()
