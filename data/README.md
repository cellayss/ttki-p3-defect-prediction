# Data

Data **tidak disimpan** di repositori ini. Unduh sendiri dari Kaggle.

| Item | Isi |
|---|---|
| Dataset | NASA Software Defect Prediction |
| Pemilik | aczy156 |
| URL | https://www.kaggle.com/datasets/aczy156/nasa-software-defect-prediction |
| Tanggal unduh | 12 November 2026 |
| Versi / ukuran | 1.79MB |
| Lisensi | Unknown — tidak dicantumkan oleh pengunggah Kaggle |
| Asal data primer | NASA Metrics Data Program (MDP), via PROMISE |

**Catatan lisensi.** Halaman Kaggle tidak mencantumkan lisensi. Data primer berasal dari
NASA Metrics Data Program yang dipublikasikan untuk riset melalui repositori PROMISE.
Karena status lisensi salinan Kaggle tidak jelas, data hanya dipakai untuk keperluan
akademik non-komersial dan **tidak didistribusikan ulang** di repositori ini;
pembaca diminta mengunduh sendiri dari tautan Kaggle di atas.

## Cara mengunduh

**Cara 1 – manual:** buka URL di atas → tombol *Download* → ekstrak ZIP →
salin semua file `.arff` ke folder `data/raw/`.

**Cara 2 – Kaggle CLI:**
```bash
pip install kaggle
# letakkan kaggle.json di ~/.kaggle/ (JANGAN di folder repo)
kaggle datasets download -d aczy156/nasa-software-defect-prediction -p data/raw --unzip
```

## Catatan audit

- **File tersedia:** 13 file ARFF — CM1, JM1, KC1, KC3, KC4, MC1, MC2, MW1, PC1, PC2, PC3, PC4, PC5.
- **Proyek yang dipakai:** KC1 (utama), CM1, JM1, PC1 (replikasi).
- **Unit observasi:** satu modul (satu baris).
- **Varian:** D'' (versi bersih Shepperd et al., 2013). Pembersihan ulang
  (nilai kosong, duplikat, kasus inkonsisten) tidak membuang baris satu pun —
  lihat `results/cleaning.csv`.

| Proyek | Baris | Kolom | Kolom label | Kolom LOC | Modul cacat | Rasio cacat |
|---|---|---|---|---|---|---|
| CM1 | 327 | 38 | Defective | LOC_TOTAL | 42 | 12,8% |
| JM1 | 7.782 | 22 | label | LOC_TOTAL | 1.672 | 21,5% |
| KC1 | 1.183 | 22 | Defective | LOC_TOTAL | 314 | 26,5% |
| PC1 | 705 | 38 | Defective | LOC_TOTAL | 61 | 8,7% |

- **Catatan:** jumlah dan jenis metrik berbeda antar-proyek (22 vs 38 kolom);
  model dilatih per proyek, jadi ini tidak memengaruhi perbandingan dalam satu proyek.
- **LOC = 0:** 2 modul di PC1; dalam perhitungan anggaran inspeksi dianggap 1 baris.
- **Kolom terlarang (ID / informasi setelah kejadian):** tidak ditemukan.