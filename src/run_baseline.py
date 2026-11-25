"""Minggu 2: reproduksi baseline (tanpa class weighting) pada data Kaggle.

Jalankan dari root repositori:
    python -m src.run_baseline
Output:
    results/audit.csv        ringkasan audit tiap proyek (data mentah)
    results/cleaning.csv     jumlah baris sebelum/sesudah pembersihan
    results/metrics.csv      metrik per proyek × model × seed × set
    results/run_log.txt      versi Python & paket, seed, waktu run
"""
from __future__ import annotations

import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import evaluation as ev
from src import preprocessing as pp

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results"
SEEDS = list(range(10))          # dibekukan: jangan diubah setelah melihat hasil test
PROJECTS = ["KC1", "CM1", "JM1", "PC1"]


def models(seed):
    return {
        "LR": LogisticRegression(max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
    }


def main():
    files = pp.list_project_files(RAW)
    if PROJECTS:
        files = [f for f in files if f.stem.upper() in {p.upper() for p in PROJECTS}]
    if not files:
        sys.exit(f"Tidak ada .csv/.arff di {RAW}. Lihat data/README.md untuk cara mengunduh.")

    OUT.mkdir(exist_ok=True)
    audits, cleans, rows = [], [], []
    for f in files:
        name = f.stem
        df = pp.load_file(f)
        label = pp.find_column(df, pp.LABEL_CANDIDATES)
        loc = pp.find_column(df, pp.LOC_CANDIDATES)
        if label is None or loc is None:
            print(f"[LEWATI] {name}: kolom label={label} / LOC={loc} tidak ditemukan")
            continue
        audits.append({"project": name, "file": f.name, **pp.audit(df, label, loc)})
        df, log = pp.clean(df, label)
        cleans.append({"project": name, **log})

        y = pp.to_binary(df[label])
        X = df.drop(columns=[label]).select_dtypes(include="number")
        if y.sum() < 10:
            print(f"[LEWATI] {name}: modul cacat < 10 setelah dibersihkan")
            continue

        for seed in SEEDS:
            X_tr, X_va, X_te, y_tr, y_va, y_te = pp.split(X, y, seed)
            for mname, model in models(seed).items():
                pipe = pp.make_pipeline(model).fit(X_tr, y_tr)
                for split_name, Xs, ys in (("validation", X_va, y_va), ("test", X_te, y_te)):
                    prob = pipe.predict_proba(Xs)[:, 1]
                    locs = Xs[loc].to_numpy()
                    rec = {"project": name, "seed": seed, "set": split_name,
                           "n_train": len(y_tr), "n_eval": len(ys), "pos_eval": int(ys.sum())}
                    rows.append({**rec, "model": mname, "ranking": "prob",
                                 **ev.classification_metrics(ys, prob),
                                 **ev.effort_metrics(ys, locs, ev.order_by_prob(prob, locs))})
            # pembanding tanpa model, hanya pada test
            locs = X_te[loc].to_numpy()
            base = {"project": name, "seed": seed, "set": "test",
                    "n_train": len(y_tr), "n_eval": len(y_te), "pos_eval": int(y_te.sum())}
            rows.append({**base, "model": "Random", "ranking": "random",
                         **ev.effort_metrics(y_te, locs, ev.order_random(len(y_te), seed))})
            rows.append({**base, "model": "ManualUp", "ranking": "loc_asc",
                         **ev.effort_metrics(y_te, locs, ev.order_manual_up(locs))})
        print(f"[OK] {name}: {len(df)} baris bersih, {int(y.sum())} cacat")

    pd.DataFrame(audits).to_csv(OUT / "audit.csv", index=False)
    pd.DataFrame(cleans).to_csv(OUT / "cleaning.csv", index=False)
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "metrics.csv", index=False)

    with open(OUT / "run_log.txt", "w") as fh:
        fh.write(f"waktu run      : {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"python         : {platform.python_version()} ({platform.system()})\n")
        fh.write(f"numpy/pandas   : {np.__version__} / {pd.__version__}\n")
        fh.write(f"scikit-learn   : {sklearn.__version__}\n")
        fh.write(f"seed           : {SEEDS}\n")
        fh.write("split          : stratified 60/20/20 setelah hapus duplikat & kasus inkonsisten\n")
        fh.write(f"file data      : {[f.name for f in files]}\n\n")
        if len(m):
            summ = (m[m["set"] == "test"]
                    .groupby(["project", "model"])[["f1", "pr_auc", "recall_at_20loc", "ifa", "popt"]]
                    .agg(["mean", "std"]).round(3))
            fh.write("Ringkasan TEST (rata-rata ± std dari 10 seed):\n")
            fh.write(summ.to_string())
    print(f"Selesai. Lihat {OUT/'run_log.txt'}")


if __name__ == "__main__":
    main()
