"""Metrik klasifikasi dan metrik berbasis anggaran inspeksi (effort-aware).

"Modul cacat ditemukan" = jumlah MODUL berlabel cacat yang ikut terinspeksi,
bukan jumlah bug, karena data NASA hanya punya label biner.
Effort = LOC modul (proxy; lihat keterbatasan di Lavazza & Rotoloni, 2025).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def classification_metrics(y, prob, threshold: float = 0.5) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, prob),
        "roc_auc": roc_auc_score(y, prob) if len(set(y)) > 1 else np.nan,
    }


# ---------- urutan inspeksi ----------
def order_by_prob(prob, loc):
    """Probabilitas tertinggi dulu; seri dipecah dengan LOC terkecil."""
    return np.lexsort((np.asarray(loc), -np.asarray(prob)))


def order_by_density(prob, loc):
    """Probabilitas ÷ LOC tertinggi dulu (Skenario B2). LOC+1 agar tidak bagi nol."""
    dens = np.asarray(prob) / (np.asarray(loc) + 1.0)
    return np.lexsort((np.asarray(loc), -dens))


def order_manual_up(loc):
    """ManualUp: modul terkecil dulu, tanpa model (Zhou et al., 2018)."""
    return np.argsort(np.asarray(loc), kind="stable")


def order_random(n, seed):
    return np.random.default_rng(seed).permutation(n)


# ---------- metrik anggaran ----------
def _curve(y_sorted, loc_sorted):
    x = np.concatenate([[0], np.cumsum(loc_sorted) / loc_sorted.sum()])
    yy = np.concatenate([[0], np.cumsum(y_sorted) / max(y_sorted.sum(), 1)])
    return x, yy


def _area(x, y):
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def effort_metrics(y, loc, order, budget: float = 0.20) -> dict:
    """Inspeksi modul sesuai `order` sampai total LOC yang dibuka <= budget × total LOC."""
    y = np.asarray(y)[order]
    loc = np.asarray(loc, dtype=float)[order]
    loc = np.where(loc <= 0, 1.0, loc)  # modul tanpa LOC dianggap 1 baris
    cum = np.cumsum(loc) / loc.sum()
    inspected = cum <= budget + 1e-12
    k = int(inspected.sum())
    found = int(y[:k].sum())
    total_def = int(y.sum())
    first = np.flatnonzero(y == 1)
    ifa = int(first[0]) if len(first) else len(y)

    # Popt (Mende & Koschke, 2010; Kamei et al.): 1 - (A_opt - A_m)/(A_opt - A_worst)
    x_m, c_m = _curve(y, loc)
    dens = y / loc
    opt = np.lexsort((loc, -dens))
    worst = np.lexsort((-loc, dens))
    a_m = _area(x_m, c_m)
    a_opt = _area(*_curve(y[opt], loc[opt]))
    a_worst = _area(*_curve(y[worst], loc[worst]))
    popt = 1 - (a_opt - a_m) / (a_opt - a_worst) if a_opt > a_worst else np.nan

    return {
        "recall_at_20loc": found / total_def if total_def else np.nan,
        "precision_at_20loc": found / k if k else 0.0,
        "pct_modules_opened": k / len(y),
        "defective_found": found,
        "ifa": ifa,
        "popt": popt,
    }
