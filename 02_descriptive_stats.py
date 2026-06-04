import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, kpss

INPUT_FILE = "outputs/macro_data_filled.xlsx"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

data = pd.read_excel(INPUT_FILE, index_col=0)
data.index = pd.to_datetime(data.index)

print(f"Период: {data.index.min().strftime('%Y-%m')} — {data.index.max().strftime('%Y-%m')}")
print(f"Размерность: T = {data.shape[0]}, N = {data.shape[1]}")

desc = data.describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
desc.columns = ["N", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max"]

print("Описательная статистика преобразованных рядов")
print(desc.round(3).to_string())

desc.round(3).to_excel(os.path.join(OUTPUT_DIR, "table_descriptive_statistics.xlsx"))

scaler = StandardScaler()

data_scaled = pd.DataFrame(
    scaler.fit_transform(data),
    index=data.index,
    columns=data.columns,
)

data_scaled.to_excel(os.path.join(OUTPUT_DIR, "macro_data_scaled.xlsx"))

# Проверка стандартизации
standardization_check = pd.DataFrame({
    "Mean_after_scaling": data_scaled.mean(),
    "Std_after_scaling": data_scaled.std(ddof=0),
})

standardization_check.to_excel(os.path.join(OUTPUT_DIR, "standardization_check.xlsx"))

print("Проверка стандартизации")
print(standardization_check.round(4).to_string())

corr = data_scaled.corr()
corr.to_excel(os.path.join(OUTPUT_DIR, "correlation_matrix.xlsx"))

print("Корреляционная матрица")
print(corr.round(2).to_string())

plt.figure(figsize=(11, 9))
plt.imshow(corr.values, aspect="auto")
plt.colorbar(label="Корреляция")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Корреляционная матрица стандартизированных рядов")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"), dpi=300)
plt.close()

blocks = {
    "real_sector": ["ipp", "retail", "freight", "services"],
    "labor_income": ["unemployment", "real_wages"],
    "prices": ["cpi", "core_cpi", "ppi"],
    "money_credit": ["key_rate", "m2", "credit_firms", "credit_households"],
    "external_fx": ["reserves", "exports", "imports", "usd_rub"],
}

block_dir = os.path.join(OUTPUT_DIR, "standardized_series_by_blocks")
os.makedirs(block_dir, exist_ok=True)

for block_name, block_cols in blocks.items():
    available_cols = [col for col in block_cols if col in data_scaled.columns]

    plt.figure(figsize=(10, 5))
    for col in available_cols:
        plt.plot(data_scaled.index, data_scaled[col], label=col)

    plt.axhline(0, linewidth=1)
    plt.title(f"Стандартизированные ряды: {block_name}")
    plt.xlabel("Дата")
    plt.ylabel("Стандартизированное значение")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(block_dir, f"{block_name}.png"), dpi=300)
    plt.close()

T_obs = len(data)
maxlag_adf = int(np.floor(12 * (T_obs / 100) ** 0.25))


def fmt_p_kpss(p: float) -> str:
    if p >= 0.10:
        return ">0.10"
    if p <= 0.01:
        return "<0.01"
    return str(round(p, 3))


def run_adf(series: pd.Series, regression: str) -> tuple:
    stat, p, *_ = adfuller(
        series,
        maxlag=maxlag_adf,
        autolag="AIC",
        regression=regression,
    )
    return round(stat, 3), round(p, 3)


def run_kpss(series: pd.Series, regression: str) -> tuple:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, *_ = kpss(
            series,
            regression=regression,
            nlags="auto",
        )
    return round(stat, 3), p


def is_stationary(adf_p: float, kpss_p: float) -> str:
    """
    ADF:  H0 = единичный корень.
          p < 0.05 => стационарность.
    KPSS: H0 = стационарность.
          p >= 0.05 => стационарность.
    """
    adf_stationary = adf_p < 0.05
    kpss_stationary = kpss_p >= 0.05

    if adf_stationary and kpss_stationary:
        return "yes"
    if not adf_stationary and not kpss_stationary:
        return "no"
    return "conflict"


rows = []

for col in data.columns:
    level = data[col].dropna()
    diff1 = level.diff().dropna()

    adf_ct_stat, adf_ct_p = run_adf(level, "ct")
    kpss_ct_stat, kpss_ct_p = run_kpss(level, "ct")
    res_ct = is_stationary(adf_ct_p, kpss_ct_p)

    adf_c_stat, adf_c_p = run_adf(level, "c")
    kpss_c_stat, kpss_c_p = run_kpss(level, "c")
    res_c = is_stationary(adf_c_p, kpss_c_p)

    adf_d_stat, adf_d_p = run_adf(diff1, "c")
    kpss_d_stat, kpss_d_p = run_kpss(diff1, "c")
    res_d = is_stationary(adf_d_p, kpss_d_p)

    if res_ct == "yes" or res_c == "yes":
        integration = "I(0)"
    elif res_d == "yes":
        integration = "I(1)"
    elif res_d == "conflict":
        integration = "I(1)?"
    else:
        integration = "I(2)+?"

    rows.append({
        "Variable": col,
        "ADF(ct) stat": adf_ct_stat,
        "ADF(ct) p": adf_ct_p,
        "KPSS(ct) stat": kpss_ct_stat,
        "KPSS(ct) p": fmt_p_kpss(kpss_ct_p),
        "Level with trend": res_ct,
        "ADF(c) stat": adf_c_stat,
        "ADF(c) p": adf_c_p,
        "KPSS(c) stat": kpss_c_stat,
        "KPSS(c) p": fmt_p_kpss(kpss_c_p),
        "Level": res_c,
        "ADF(diff) stat": adf_d_stat,
        "ADF(diff) p": adf_d_p,
        "KPSS(diff) stat": kpss_d_stat,
        "KPSS(diff) p": fmt_p_kpss(kpss_d_p),
        "Diff": res_d,
        "Integration": integration,
    })

stationarity = pd.DataFrame(rows).set_index("Variable")
stationarity.to_excel(os.path.join(OUTPUT_DIR, "stationarity_tests_full.xlsx"))

summary = (
    stationarity["Integration"]
    .value_counts()
    .rename_axis("Integration")
    .reset_index(name="Count")
)

summary.to_excel(os.path.join(OUTPUT_DIR, "stationarity_summary.xlsx"), index=False)

print("ADF / KPSS ТЕСТЫ")
print(f"T = {T_obs}, maxlag ADF = {maxlag_adf}")
print(stationarity[["ADF(c) p", "KPSS(c) p", "ADF(diff) p", "KPSS(diff) p", "Integration"]].to_string())

print("Сводка по порядку интегрированности")
print(summary.to_string(index=False))

for integ in ["I(0)", "I(1)", "I(1)?", "I(2)+?"]:
    subset = stationarity[stationarity["Integration"] == integ].index.tolist()
    if subset:
        print(f"{integ}: {', '.join(subset)}")
