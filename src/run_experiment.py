"""Minggu 3: eksperimen terkontrol.

Desain (sama untuk setiap proyek dan seed; split identik dengan baseline):
  model      : LR, RF
  weighting  : none (= baseline)  | balanced  (Skenario A: class_weight="balanced")
  ranking    : prob (Skenario B)  | density   (Skenario B2: probabilitas ÷ LOC)
  anggaran   : inspeksi sampai 20% total LOC (Skenario C)
  pembanding : Random, ManualUp

Jalankan dari root repositori:
    python -m src.run_experiment
Output:
    results/metrics_experiment.csv   semua metrik (validation & test)
    results/summary_test.csv         rata-rata ± std per konfigurasi (test)
    results/stats_wilcoxon.csv       uji Wilcoxon berpasangan antar-skenario (test)
    results/run_log_experiment.txt   versi, seed, ringkasan
    figures/recall20_test.png, figures/ifa_test.png
"""
from __future__ import annotations

import platform
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import evaluation as ev
from src import preprocessing as pp
from src.run_baseline import PROJECTS, SEEDS  # protokol yang sama dengan Minggu 2

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT, FIG = ROOT / "data" / "raw", ROOT / "results", ROOT / "figures"
METRICS_A = ["precision", "recall", "f1", "pr_auc", "roc_auc", "recall_at_20loc", "popt"]
METRICS_B = ["recall_at_20loc", "precision_at_20loc", "pct_modules_opened", "ifa", "popt"]


def make_model(name, weighting, seed):
    cw = "balanced" if weighting == "balanced" else None
    if name == "LR":
        return LogisticRegression(max_iter=2000, class_weight=cw, random_state=seed)
    return RandomForestClassifier(n_estimators=300, class_weight=cw,
                                  random_state=seed, n_jobs=-1)


def run():
    files = pp.list_project_files(RAW)
    if PROJECTS:
        files = [f for f in files if f.stem.upper() in {p.upper() for p in PROJECTS}]
    if not files:
        sys.exit(f"Tidak ada data di {RAW}.")
    rows = []
    for f in files:
        name = f.stem
        df = pp.load_file(f)
        label = pp.find_column(df, pp.LABEL_CANDIDATES)
        loc = pp.find_column(df, pp.LOC_CANDIDATES)
        df, _ = pp.clean(df, label)
        y = pp.to_binary(df[label])
        X = df.drop(columns=[label]).select_dtypes(include="number")
        for seed in SEEDS:
            X_tr, X_va, X_te, y_tr, y_va, y_te = pp.split(X, y, seed)
            for mname, weighting in product(["LR", "RF"], ["none", "balanced"]):
                pipe = pp.make_pipeline(make_model(mname, weighting, seed)).fit(X_tr, y_tr)
                for split_name, Xs, ys in (("validation", X_va, y_va), ("test", X_te, y_te)):
                    prob = pipe.predict_proba(Xs)[:, 1]
                    locs = Xs[loc].to_numpy()
                    cls = ev.classification_metrics(ys, prob)
                    for ranking, order in (("prob", ev.order_by_prob(prob, locs)),
                                           ("density", ev.order_by_density(prob, locs))):
                        rows.append({"project": name, "seed": seed, "set": split_name,
                                     "model": mname, "weighting": weighting, "ranking": ranking,
                                     "n_eval": len(ys), "pos_eval": int(ys.sum()),
                                     **cls, **ev.effort_metrics(ys, locs, order)})
            for split_name, Xs, ys in (("validation", X_va, y_va), ("test", X_te, y_te)):
                locs = Xs[loc].to_numpy()
                base = {"project": name, "seed": seed, "set": split_name, "weighting": "-",
                        "n_eval": len(ys), "pos_eval": int(ys.sum())}
                rows.append({**base, "model": "Random", "ranking": "random",
                             **ev.effort_metrics(ys, locs, ev.order_random(len(ys), seed))})
                rows.append({**base, "model": "ManualUp", "ranking": "loc_asc",
                             **ev.effort_metrics(ys, locs, ev.order_manual_up(locs))})
        print(f"[OK] {name}")
    return pd.DataFrame(rows)


def config_label(r):
    if r["model"] in ("Random", "ManualUp"):
        return r["model"]
    w = "bal" if r["weighting"] == "balanced" else "none"
    return f"{r['model']}-{w}-{r['ranking']}"


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok]); m = ok.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]).clip(max=1)
    tmp = np.empty(m); tmp[idx] = adj; out[ok] = tmp
    return out


def paired(test, a, b, metric, project, family):
    x = a.set_index("seed")[metric]; z = b.set_index("seed")[metric]
    d = (x - z).dropna()
    if len(d) < 5 or np.allclose(d, 0):
        p = np.nan
    else:
        p = wilcoxon(d, zero_method="zsplit").pvalue
    return {"family": family, "project": project, "comparison": test, "metric": metric,
            "mean_a": x.mean(), "mean_b": z.mean(), "median_diff": d.median(), "p": p}


def stats(m):
    t = m[m["set"] == "test"].copy()
    t["config"] = t.apply(config_label, axis=1)
    res = []
    for proj, g in t.groupby("project"):
        G = lambda c: g[g["config"] == c]
        for mdl, rk in product(["LR", "RF"], ["prob"]):
            for met in METRICS_A:   # Skenario A: balanced vs none
                res.append(paired(f"{mdl}: bal vs none ({rk})", G(f"{mdl}-bal-{rk}"),
                                  G(f"{mdl}-none-{rk}"), met, proj, "A_class_weighting"))
        for mdl, w in product(["LR", "RF"], ["none", "bal"]):
            for met in METRICS_B:   # Skenario B2: density vs prob
                res.append(paired(f"{mdl}-{w}: density vs prob", G(f"{mdl}-{w}-density"),
                                  G(f"{mdl}-{w}-prob"), met, proj, "B2_ranking"))
        for c in sorted(x for x in g["config"].unique() if x not in ("Random", "ManualUp")):
            for ref in ("Random", "ManualUp"):
                for met in ("recall_at_20loc", "ifa"):
                    res.append(paired(f"{c} vs {ref}", G(c), G(ref), met, proj, "vs_reference"))
    s = pd.DataFrame(res)
    s["p_holm"] = np.nan
    for _, idx in s.groupby(["family", "project", "metric"]).groups.items():
        s.loc[idx, "p_holm"] = holm(s.loc[idx, "p"])
    return s, t


def figures(t):
    FIG.mkdir(exist_ok=True)
    order = ["Random", "ManualUp"] + [f"{m}-{w}-{r}" for m in ("LR", "RF")
                                      for w in ("none", "bal") for r in ("prob", "density")]
    projects = sorted(t["project"].unique())
    for metric, fname, ylabel in (("recall_at_20loc", "recall20_test.png",
                                   "Recall@20% LOC (modul cacat ditemukan)"),
                                  ("ifa", "ifa_test.png", "IFA (modul sehat sebelum cacat pertama)")):
        fig, axes = plt.subplots(1, len(projects), figsize=(4 * len(projects), 4.2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, p in zip(axes, projects):
            g = t[t["project"] == p].groupby("config")[metric].agg(["mean", "std"]).reindex(order)
            colors = ["#9e9e9e", "#9e9e9e"] + ["#4c72b0", "#7fa3d6", "#dd8452", "#f0b48a"] * 2  # biru=none, oranye=balanced; muda=density
            ax.barh(range(len(order)), g["mean"], xerr=g["std"], color=colors,
                    alpha=0.85,
                    error_kw={"elinewidth": .8, "capsize": 2})
            ax.set_yticks(range(len(order)), order, fontsize=8)
            ax.invert_yaxis(); ax.set_title(p); ax.set_xlabel(ylabel, fontsize=8)
            if metric == "recall_at_20loc":
                ax.axvline(0.2, color="k", lw=.6, ls=":")
        fig.suptitle(f"{ylabel} — data test, rata-rata ± std dari {len(SEEDS)} seed", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=200)
        plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    m = run()
    m.to_csv(OUT / "metrics_experiment.csv", index=False)
    s, t = stats(m)
    s.round(4).to_csv(OUT / "stats_wilcoxon.csv", index=False)
    cols = ["f1", "recall", "precision", "pr_auc", "recall_at_20loc",
            "pct_modules_opened", "ifa", "popt"]
    summ = t.groupby(["project", "config"])[cols].agg(["mean", "std"]).round(3)
    summ.to_csv(OUT / "summary_test.csv")
    figures(t)
    with open(OUT / "run_log_experiment.txt", "w") as fh:
        fh.write(f"waktu run      : {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"python         : {platform.python_version()} ({platform.system()})\n")
        fh.write(f"numpy/pandas   : {np.__version__} / {pd.__version__}\n")
        fh.write(f"scikit-learn   : {sklearn.__version__}   scipy: {scipy.__version__}\n")
        fh.write(f"seed           : {SEEDS}\nproyek         : {PROJECTS}\n")
        fh.write("konfigurasi    : model{LR,RF} × weighting{none,balanced} × ranking{prob,density}"
                 " + Random + ManualUp\n")
        fh.write("uji statistik  : Wilcoxon signed-rank berpasangan per seed, koreksi Holm per proyek × keluarga × metrik\n\n")
        fh.write("Ringkasan TEST (rata-rata):\n")
        fh.write(t.groupby(["project", "config"])[["f1", "recall", "precision", "recall_at_20loc",
                                                    "ifa", "popt"]].mean().round(3).to_string())
    print(f"Selesai. Lihat {OUT/'run_log_experiment.txt'} dan folder figures/")


if __name__ == "__main__":
    main()
