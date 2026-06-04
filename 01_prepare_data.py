import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "macro_data.xlsx"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

COLS = [
    "ipp", "retail", "freight", "services", "unemployment",
    "cpi", "core_cpi", "ppi", "key_rate", "m2",
    "credit_firms", "credit_households", "real_wages",
    "reserves", "exports", "imports", "usd_rub",
]

df = pd.read_excel(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")

data_raw = df[COLS].copy()

missing_before = data_raw.isna().sum()

print("Пропуски до заполнения")
print(missing_before[missing_before > 0] if missing_before.any() else "Пропусков нет")
print(f"Всего пропусков: {int(missing_before.sum())}")

missing_before.to_excel(os.path.join(OUTPUT_DIR, "missing_before.xlsx"))

# Внутренние пропуски — линейная интерполяция
# Крайние пропуски — ближайшее доступное значение
data_filled = data_raw.interpolate(method="linear").ffill().bfill()

missing_after = data_filled.isna().sum()

print("Пропуски после заполнения")
print(missing_after[missing_after > 0] if missing_after.any() else "Пропусков нет")
print(f"Всего пропусков после заполнения: {int(missing_after.sum())}")

missing_after.to_excel(os.path.join(OUTPUT_DIR, "missing_after.xlsx"))

data_filled.to_excel(os.path.join(OUTPUT_DIR, "macro_data_filled.xlsx"))

hist_dir = os.path.join(OUTPUT_DIR, "histograms")
os.makedirs(hist_dir, exist_ok=True)

for col in data_filled.columns:
    plt.figure(figsize=(7, 4))
    plt.hist(data_filled[col].dropna(), bins=20, edgecolor="black")
    plt.title(f"Распределение показателя: {col}")
    plt.xlabel(col)
    plt.ylabel("Частота")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(hist_dir, f"hist_{col}.png"), dpi=300)
    plt.close()

n_cols = 3
n_rows = int(np.ceil(len(COLS) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(COLS):
    axes[i].hist(data_filled[col].dropna(), bins=20, edgecolor="black")
    axes[i].set_title(col)
    axes[i].grid(True, alpha=0.3)

for j in range(len(COLS), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "all_histograms.png"), dpi=300)
plt.close()
