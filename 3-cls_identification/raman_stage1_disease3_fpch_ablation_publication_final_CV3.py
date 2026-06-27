#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


RAMAN_CSV = "raman_added2.csv"
OUT_DIR = "./publication_stage1_disease3_fpch_ablation_reproduction_out_CV3"

# Original disease labels for filtering only.
DISEASE_LABELS = ("Liver", "Bile", "Pancreas")



ENCODED_CLASS_ORDER = ("Bile", "Liver", "Pancreas")
TASK_NAME = "disease3"
ABLATION_NAME = "FPCH_only"

LOCKED_INPUT_DIR = "./publication_stage1_disease3_inputs_CV3_Ablation"
LOCKED_RUNS_CSV = os.path.join(
    LOCKED_INPUT_DIR,
    "selected_rep_STAGE1_DISEASE3_FPCH_LOCKED_RUNS.csv",
)


EXPECTED_SELECTED_REP = 4


REQUIRE_LOCKED_SELECTED_REP = True

VIEWS = [("fp", "fingerprint"), ("ch", "ch")]
CSV_ENCODING = "utf-8-sig"
PROB_EPS = 1e-6

SAVE_PATIENT_PREDS = True
SAVE_SPLITS = True
SAVE_ROC_POINTS = True



def read_csv_robust(path: str, dtype=None) -> pd.DataFrame:
    encs = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
    last_err = None
    for enc in encs:
        try:
            return pd.read_csv(path, dtype=dtype, encoding=enc)
        except Exception as e:
            last_err = e
    try:
        return pd.read_csv(path, dtype=dtype)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV {path}. Last errors: {last_err} / {e}")


def to_csv_safe(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)


def write_row_append(path: str, row: Dict):
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False,
        encoding=CSV_ENCODING,
    )


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
        if isinstance(v, (tuple, set)):
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


def json_sorted_list(x) -> str:
    return json.dumps(sorted([str(t) for t in x]), ensure_ascii=False)


def count_cells_per_sample(df: pd.DataFrame, samples: List[str]) -> Dict[str, int]:
    vc = df["Sample"].astype(str).value_counts()
    return {str(s): int(vc.get(str(s), 0)) for s in samples}


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
    if len(cols) == 0:
        raise ValueError("No numeric Raman wavelength columns found.")
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
        raise ValueError(f"No spectral columns selected for region={region}.")
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
    dfz["Sample"] = sample_ids.astype(str)
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
    return json.dumps({c: int(np.sum(y_str == c)) for c in classes}, ensure_ascii=False)


def train_svm_pca(Xtr: np.ndarray, ytr: np.ndarray, best: Dict, seed: int):
    if len(np.unique(ytr)) < 2:
        raise ValueError("Training set has fewer than two classes.")
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)

    n_feat = int(Xtr_s.shape[1])
    n_samp = int(Xtr_s.shape[0])
    max_pc = max(1, min(n_feat, n_samp - 1))
    n_pc = int(min(int(best["n_pc"]), max_pc))

    pca = PCA(n_components=n_pc, random_state=int(seed))
    Ztr = pca.fit_transform(Xtr_s)
    svm = SVC(
        kernel="rbf",
        C=float(best["C"]),
        gamma=str(best["gamma"]),
        probability=True,
        class_weight="balanced",
        cache_size=2000,
        random_state=int(seed),
    )
    svm.fit(Ztr, ytr)
    return sc, pca, svm


def predict_proba_full(sc: StandardScaler, pca: PCA, svm: SVC, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Return predict_proba aligned to encoded classes 0..n_classes-1."""
    P_small = svm.predict_proba(pca.transform(sc.transform(X)))
    P = np.zeros((P_small.shape[0], n_classes), dtype=float)
    for j, cls_id in enumerate(svm.classes_.astype(int)):
        P[:, int(cls_id)] = P_small[:, j]
    return P


def train_view_model_locked(
    df: pd.DataFrame,
    spec_cols: List[str],
    train_samples: set,
    seed: int,
    fixed_multi: Dict,
):
    classes = list(ENCODED_CLASS_ORDER)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    X = df[spec_cols].to_numpy(np.float32)
    tr_idx = np.where(df["Sample"].astype(str).isin(train_samples).values)[0]
    if len(tr_idx) == 0:
        raise RuntimeError("No Raman cells found for locked training samples.")
    Xtr = X[tr_idx]
    y_str = df.iloc[tr_idx]["Group"].astype(str).values
    ytr = np.array([class_to_idx[s] for s in y_str], dtype=int)
    sc, pca, svm = train_svm_pca(Xtr, ytr, fixed_multi, seed=seed)
    return dict(sc=sc, pca=pca, svm=svm, spec_cols=spec_cols, classes=classes, n_classes=len(classes))


def predict_patient_probs_view(df: pd.DataFrame, model: Dict, samples_set: set):
    spec_cols = model["spec_cols"]
    X = df[spec_cols].to_numpy(np.float32)
    idx = np.where(df["Sample"].astype(str).isin(samples_set).values)[0]
    if len(idx) == 0:
        return None
    samp = df.iloc[idx]["Sample"].astype(str).values
    p_cell = predict_proba_full(model["sc"], model["pca"], model["svm"], X[idx], n_classes=model["n_classes"])
    return agg_probs_to_sample_logit(samp, p_cell, model["classes"])



def multiclass_metrics_safe(y_true: np.ndarray, P: np.ndarray, n_classes: int):
    yhat = P.argmax(axis=1)
    acc = float(accuracy_score(y_true, yhat))
    bal = float(balanced_accuracy_score(y_true, yhat))
    f1m = float(f1_score(y_true, yhat, average="macro", labels=list(range(n_classes)), zero_division=0))
    auc = np.nan
    if len(np.unique(y_true)) == n_classes:
        try:
            auc = float(roc_auc_score(y_true, P, multi_class="ovr"))
        except Exception:
            auc = np.nan
    return acc, bal, f1m, auc


def load_locked_runs(path: str, expected_rep: int) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"LOCKED_RUNS_CSV not found: {path}")
    df = read_csv_robust(path, dtype=str)
    need = [
        "rep", "fold", "task", "outer_train_json", "outer_test_json",
        "w_fp", "w_ch", "T", "final_seed",
        "fp_n_pc", "fp_C", "fp_gamma",
        "ch_n_pc", "ch_C", "ch_gamma",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Locked runs CSV missing required columns: {missing}")

    df = df.copy()
    df["rep"] = df["rep"].astype(int)
    df["fold"] = df["fold"].astype(int)
    df = df[df["task"].astype(str) == TASK_NAME].copy()
    if REQUIRE_LOCKED_SELECTED_REP:
        df = df[df["rep"] == int(expected_rep)].copy()
    if df.empty:
        raise ValueError(f"No locked rows found for task={TASK_NAME}, rep={expected_rep}")
    if df["fold"].duplicated().any():
        raise ValueError("Locked runs CSV has duplicated fold rows for the selected repetition.")

    df["outer_train_list"] = df["outer_train_json"].apply(parse_json_list)
    df["outer_test_list"] = df["outer_test_json"].apply(parse_json_list)
    for c in ["w_fp", "w_ch", "T", "fp_C", "ch_C"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    for c in ["final_seed", "fp_n_pc", "ch_n_pc"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
    return df.sort_values("fold").reset_index(drop=True)


def view_params_from_row(row: pd.Series, vname: str) -> Dict:
    return {
        "n_pc": int(row[f"{vname}_n_pc"]),
        "C": float(row[f"{vname}_C"]),
        "gamma": str(row[f"{vname}_gamma"]),
    }


def run_locked_disease3_fpch():
    os.makedirs(OUT_DIR, exist_ok=True)
    classes = list(ENCODED_CLASS_ORDER)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    df = read_csv_robust(RAMAN_CSV)
    for c in ["Batch", "Group", "Sample", "CellType", "Cell"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    if "Group" not in df.columns or "Sample" not in df.columns:
        raise ValueError("Raman CSV must contain 'Sample' and 'Group' columns.")

    df = df[df["Group"].astype(str).isin(set(DISEASE_LABELS))].copy()
    if df.empty:
        raise ValueError("No disease-only rows found after filtering Raman CSV by DISEASE_LABELS.")

    sample_group = df.groupby("Sample")["Group"].first()
    missing = set(classes) - set(sample_group.values.tolist())
    if missing:
        raise ValueError(f"Missing disease labels in Raman CSV: {missing}")

    all_cols, wav = get_spec_cols_and_wav(df)
    locked = load_locked_runs(LOCKED_RUNS_CSV, EXPECTED_SELECTED_REP)

    out_csv = os.path.join(OUT_DIR, f"stage1_{TASK_NAME}_FPCH.csv")
    sum_csv = os.path.join(OUT_DIR, f"stage1_{TASK_NAME}_FPCH_SUMMARY.csv")
    patient_csv = os.path.join(OUT_DIR, f"stage1_{TASK_NAME}_FPCH_patient_preds.csv")
    splits_csv = os.path.join(OUT_DIR, f"stage1_SPLITS_{TASK_NAME}_FPCH.csv")
    roc_points_csv = os.path.join(OUT_DIR, f"stage1_{TASK_NAME}_FPCH_roc_points_ovr.csv")

    for f in [out_csv, sum_csv, patient_csv, splits_csv, roc_points_csv]:
        if os.path.exists(f):
            os.remove(f)

    rows = []

    for _, row in locked.iterrows():
        rep = int(row["rep"])
        fold_id = int(row["fold"])
        outer_train = set(row["outer_train_list"])
        outer_test = set(row["outer_test_list"])
        final_seed_base = int(row["final_seed"])

        missing_train = sorted([s for s in outer_train if s not in sample_group.index])
        missing_test = sorted([s for s in outer_test if s not in sample_group.index])
        if missing_train or missing_test:
            raise RuntimeError(
                f"Locked split contains samples not found in Raman CSV. "
                f"missing_train={missing_train[:5]}, missing_test={missing_test[:5]}"
            )

        models = {}
        for vname, region in VIEWS:
            cols = select_region(all_cols, wav, region)
            models[vname] = train_view_model_locked(
                df=df,
                spec_cols=cols,
                train_samples=outer_train,
                seed=final_seed_base,
                fixed_multi=view_params_from_row(row, vname),
            )

        P_te = {vname: predict_patient_probs_view(df, models[vname], outer_test) for vname, _ in VIEWS}
        if any(P_te[v] is None for v in ["fp", "ch"]):
            raise RuntimeError(f"rep={rep}, fold={fold_id}: a view returned empty outer-test probabilities.")
        if not P_te["fp"].index.equals(P_te["ch"].index):
            raise RuntimeError(f"rep={rep}, fold={fold_id}: FP/CH patient indices mismatch on outer test.")

        idx_te = P_te["fp"].index
        y_te_str = sample_group.loc[idx_te].astype(str).values
        y_te = np.array([class_to_idx[s] for s in y_te_str], dtype=int)

        Pte_fp = P_te["fp"].loc[idx_te].to_numpy()
        Pte_ch = P_te["ch"].loc[idx_te].to_numpy()

        w_fp = float(row["w_fp"])
        w_ch = float(row["w_ch"])
        T = float(row["T"])
        P = w_fp * Pte_fp + w_ch * Pte_ch
        P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
        P_T = apply_temperature_multiclass(P, T)

        acc, bal, f1m, auc = multiclass_metrics_safe(y_te, P_T, n_classes=n_classes)
        yhat = P_T.argmax(axis=1)
        cm = confusion_matrix(y_te, yhat, labels=np.arange(n_classes)).flatten().tolist()

        out_row = dict(
            rep=rep,
            fold=fold_id,
            task=TASK_NAME,
            ablation=ABLATION_NAME,
            final_seed=int(final_seed_base),
            n_test_patients=int(len(idx_te)),
            outer_test_counts_json=class_counts_str(y_te_str, classes),
            w_fp=w_fp,
            w_ch=w_ch,
            T=T,
            acc=float(acc),
            bal_acc=float(bal),
            macro_f1=float(f1m),
            auc_ovr=float(auc) if auc == auc else np.nan,
            cm_flat=",".join(map(str, cm)),
        )
        for vname in ["fp", "ch"]:
            params = view_params_from_row(row, vname)
            out_row[f"{vname}_n_pc"] = int(params["n_pc"])
            out_row[f"{vname}_C"] = float(params["C"])
            out_row[f"{vname}_gamma"] = str(params["gamma"])
        rows.append(out_row)
        write_row_append(out_csv, out_row)

        if SAVE_SPLITS:
            write_row_append(splits_csv, dict(
                rep=rep,
                fold=fold_id,
                task=TASK_NAME,
                ablation=ABLATION_NAME,
                outer_train_samples_json=json_sorted_list(outer_train),
                outer_test_samples_json=json_sorted_list(outer_test),
            ))

        if SAVE_PATIENT_PREDS:
            idx_list = [str(s) for s in idx_te.tolist()]
            cell_counts = count_cells_per_sample(df, idx_list)
            for i, sk in enumerate(idx_list):
                prow = dict(
                    rep=rep,
                    fold=fold_id,
                    task=TASK_NAME,
                    ablation=ABLATION_NAME,
                    SampleKey=str(sk),
                    n_cells=int(cell_counts.get(sk, 0)),
                    y_true=int(y_te[i]),
                    y_true_name=str(classes[int(y_te[i])]),
                    y_pred=int(yhat[i]),
                    y_pred_name=str(classes[int(yhat[i])]),
                    correct=int(int(y_te[i]) == int(yhat[i])),
                    w_fp=w_fp,
                    w_ch=w_ch,
                    T=T,
                )
                for k, cname in enumerate(classes):
                    prow[f"p_{cname}"] = float(P_T[i, k])
                    prow[f"praw_{cname}"] = float(P[i, k])
                    prow[f"fp_p_{cname}"] = float(Pte_fp[i, k])
                    prow[f"ch_p_{cname}"] = float(Pte_ch[i, k])
                write_row_append(patient_csv, prow)

        if SAVE_ROC_POINTS:
            for k, cname in enumerate(classes):
                y_bin = (y_te == k).astype(int)
                if len(np.unique(y_bin)) < 2:
                    continue
                fpr, tpr, thr = roc_curve(y_bin, P_T[:, k])
                write_row_append(roc_points_csv, dict(
                    rep=rep,
                    fold=fold_id,
                    task=TASK_NAME,
                    ablation=ABLATION_NAME,
                    class_id=int(k),
                    class_name=str(cname),
                    fpr_json=json.dumps(fpr.tolist()),
                    tpr_json=json.dumps(tpr.tolist()),
                    thr_json=json.dumps(thr.tolist()),
                ))

        print(
            f"[{TASK_NAME} FPCH] rep {rep} fold {fold_id} "
            f"acc={acc:.3f} bal={bal:.3f} f1={f1m:.3f} "
            f"auc={auc if auc == auc else float('nan'):.3f} "
            f"T={T:.2f} w=(fp={w_fp:.1f},ch={w_ch:.1f})"
        )

    res = pd.DataFrame(rows)
    summary = dict(
        task=TASK_NAME,
        ablation=ABLATION_NAME,
        selected_rep=int(EXPECTED_SELECTED_REP),
        n_rows=int(len(res)),
        acc_mean=float(res["acc"].mean()),
        acc_std=float(res["acc"].std()),
        bal_mean=float(res["bal_acc"].mean()),
        bal_std=float(res["bal_acc"].std()),
        f1_mean=float(res["macro_f1"].mean()),
        auc_mean=float(res["auc_ovr"].dropna().mean()) if res["auc_ovr"].notna().any() else float("nan"),
    )
    to_csv_safe(pd.DataFrame([summary]), sum_csv)

    print("Saved:", out_csv)
    print("Saved:", sum_csv)
    if SAVE_PATIENT_PREDS:
        print("Saved:", patient_csv)
    if SAVE_SPLITS:
        print("Saved:", splits_csv)
    if SAVE_ROC_POINTS:
        print("Saved:", roc_points_csv)
    print("SUMMARY:", json.dumps(summary, indent=2, ensure_ascii=False))
    return res


if __name__ == "__main__":
    run_locked_disease3_fpch()
