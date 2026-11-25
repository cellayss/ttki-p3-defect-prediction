"""Memuat, mengaudit, dan membersihkan data NASA MDP dari Kaggle.

Semua fungsi di sini tidak melihat data test: pembersihan (hapus duplikat dan
kasus inkonsisten) dilakukan SEBELUM split, lalu transformasi fitur (log + scaler)
di-fit hanya pada data train lewat Pipeline sklearn.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Nama kolom label dan LOC berbeda antar-versi (PROMISE vs MDP). Urutan = prioritas.
LABEL_CANDIDATES = ["defects", "Defective", "defective", "label", "bug", "bugs",
                    "problems", "c", "class"]
LOC_CANDIDATES = ["LOC_TOTAL", "loc", "LOC", "loc_total", "LOC_EXECUTABLE", "lOCode"]
# Kolom yang tidak boleh jadi fitur (ID atau informasi setelah kejadian).
FORBIDDEN = {"id", "ID", "module_id", "name", "error_count", "ERROR_COUNT",
             "ERROR_DENSITY", "error_density"}

POSITIVE = {"true", "yes", "y", "1", "t", "buggy", "defective"}


def load_file(path: str | Path) -> pd.DataFrame:
    """Baca satu file proyek (.csv atau .arff)."""
    path = Path(path)
    if path.suffix.lower() == ".arff":
        data, _ = arff.loadarff(path)
        df = pd.DataFrame(data)
        for c in df.columns:  # ARFF menyimpan teks sebagai bytes
            if df[c].dtype == object:
                df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
        return df
    return pd.read_csv(path)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def to_binary(s: pd.Series) -> pd.Series:
    """Ubah label (True/False, Y/N, jumlah bug) menjadi 0/1."""
    if pd.api.types.is_numeric_dtype(s):
        return (s > 0).astype(int)
    return s.astype(str).str.strip().str.lower().isin(POSITIVE).astype(int)


def audit(df: pd.DataFrame, label_col: str, loc_col: str | None) -> dict:
    """Ringkasan kualitas data mentah (sebelum dibersihkan)."""
    feats = [c for c in df.columns if c != label_col]
    num = df[feats].select_dtypes(include="number")
    y = to_binary(df[label_col])
    dup_exact = int(df.duplicated(keep="first").sum())
    grp = df.assign(_y=y).groupby(feats, dropna=False)["_y"].nunique()
    inconsistent_groups = int((grp > 1).sum())
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "label_col": label_col,
        "loc_col": loc_col,
        "missing_cells": int(df.isna().sum().sum()),
        "inf_cells": int(np.isinf(num.to_numpy(dtype=float)).sum()),
        "negative_cells": int((num < 0).sum().sum()),
        "duplicate_rows_exact": dup_exact,
        "inconsistent_feature_groups": inconsistent_groups,
        "defective": int(y.sum()),
        "defect_ratio": round(float(y.mean()), 4),
        "loc_missing_or_zero": (int((df[loc_col].isna() | (df[loc_col] <= 0)).sum())
                                if loc_col else None),
        "forbidden_cols_present": sorted(set(df.columns) & FORBIDDEN),
    }


def clean(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, dict]:
    """Pembersihan ala Shepperd et al. (2013) versi D'':
    buang baris dengan label kosong, nilai inf/kosong, baris identik,
    dan SEMUA baris yang fiturnya sama tetapi labelnya berbeda."""
    log = {"start": len(df)}
    df = df.drop(columns=[c for c in df.columns if c in FORBIDDEN], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    log["after_missing"] = len(df)
    df = df.drop_duplicates()
    log["after_dedup"] = len(df)
    feats = [c for c in df.columns if c != label_col]
    y = to_binary(df[label_col])
    n_labels = df.assign(_y=y).groupby(feats, dropna=False)["_y"].transform("nunique")
    df = df[n_labels == 1]
    log["after_inconsistent"] = len(df)
    return df.reset_index(drop=True), log


def split(X: pd.DataFrame, y: pd.Series, seed: int, val=0.2, test=0.2):
    """Stratified train/validation/test 60/20/20."""
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=test,
                                                stratify=y, random_state=seed)
    rel = val / (1 - test)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=rel,
                                              stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def _log1p_clip(a):
    return np.log1p(np.clip(a, 0, None))


def make_pipeline(model) -> Pipeline:
    """log(1+x) lalu StandardScaler; di-fit hanya pada data yang diberikan ke .fit()."""
    return Pipeline([
        ("log", FunctionTransformer(_log1p_clip, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def list_project_files(raw_dir: str | Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    return sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in {".csv", ".arff"})
