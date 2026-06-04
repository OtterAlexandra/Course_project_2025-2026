import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

INPUT_FILE = "outputs/macro_data_scaled.xlsx"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

data_scaled = pd.read_excel(INPUT_FILE, index_col=0)
data_scaled.index = pd.to_datetime(data_scaled.index)

X = data_scaled.values
T, N = X.shape

print(f"Период: {data_scaled.index.min().strftime('%Y-%m')} — {data_scaled.index.max().strftime('%Y-%m')}")
print(f"Размерность: T = {T}, N = {N}")

pca_full = PCA()
pca_full.fit(data_scaled)

explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

ev_table = pd.DataFrame({
    "PC": np.arange(1, len(explained_var) + 1),
    "Explained var": np.round(explained_var, 4),
    "Explained var, %": np.round(explained_var * 100, 2),
    "Cumulative var": np.round(cumulative_var, 4),
    "Cumulative var, %": np.round(cumulative_var * 100, 2),
})

ev_table.to_excel(os.path.join(OUTPUT_DIR, "pca_explained_variance.xlsx"), index=False)

print("Объясненная дисперсия")
print(ev_table.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(explained_var) + 1), explained_var * 100, marker="o")
plt.xlabel("Главная компонента")
plt.ylabel("Объясненная дисперсия, %")
plt.title("Scree plot")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_scree_plot.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(cumulative_var) + 1), cumulative_var * 100, marker="o")
plt.axhline(50, linestyle="--", linewidth=1)
plt.axhline(70, linestyle="--", linewidth=1)
plt.xlabel("Число компонент")
plt.ylabel("Накопленная объясненная дисперсия, %")
plt.title("Накопленная объясненная дисперсия")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_cumulative_variance.png"), dpi=300)
plt.close()

corr_matrix = np.corrcoef(X.T)
eigenvalues = np.linalg.eigvalsh(corr_matrix)[::-1]
eigenvalues = np.maximum(eigenvalues, 0)

r_max_ah = min(10, N - 1)
r_range = np.arange(1, r_max_ah + 1)

er_values = []
gr_values = []

for r in r_range:
    mu_r = eigenvalues[r - 1]
    mu_r1 = eigenvalues[r]

    tail = eigenvalues[r:]
    V_r = tail.mean() if len(tail) > 0 else 1e-10

    ER = mu_r / mu_r1 if mu_r1 > 1e-10 else np.nan

    if V_r > 1e-10 and mu_r1 > 1e-10:
        GR = np.log(1 + mu_r / V_r) / np.log(1 + mu_r1 / V_r)
    else:
        GR = np.nan

    er_values.append(ER)
    gr_values.append(GR)

er_values = np.array(er_values)
gr_values = np.array(gr_values)

optimal_er = int(r_range[np.nanargmax(er_values)])
optimal_gr = int(r_range[np.nanargmax(gr_values)])

ah_table = pd.DataFrame({
    "r": r_range,
    "ER": np.round(er_values, 4),
    "GR": np.round(gr_values, 4),
})

ah_table.to_excel(os.path.join(OUTPUT_DIR, "ahn_horenstein_er_gr.xlsx"), index=False)

print("\n" + "=" * 80)
print("AHN-HORENSTEIN ER / GR")
print("=" * 80)
print(ah_table.to_string(index=False))
print(f"\nОптимум ER: r = {optimal_er}")
print(f"Оптимум GR: r = {optimal_gr}")

# График ER / GR
plt.figure(figsize=(8, 5))
plt.plot(r_range, er_values, marker="o", label="ER")
plt.plot(r_range, gr_values, marker="o", label="GR")
plt.xlabel("r")
plt.ylabel("Значение критерия")
plt.title("Ahn–Horenstein ER и GR")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ahn_horenstein_er_gr.png"), dpi=300)
plt.close()

n_components = optimal_gr

print("Финальное число компонент")
print(f"Выбрано n_components = {n_components}")

pca_final = PCA(n_components=n_components)
scores_array = pca_final.fit_transform(data_scaled)

component_names = [f"PC{i + 1}" for i in range(n_components)]

scores = pd.DataFrame(
    scores_array,
    index=data_scaled.index,
    columns=component_names,
)

scores.to_excel(os.path.join(OUTPUT_DIR, "pca_scores.xlsx"))

# Веса переменных в линейных комбинациях
weights = pd.DataFrame(
    pca_final.components_.T,
    index=data_scaled.columns,
    columns=component_names,
)

weights.to_excel(os.path.join(OUTPUT_DIR, "pca_weights.xlsx"))

corr_loadings = pd.DataFrame(
    pca_final.components_.T * np.sqrt(pca_final.explained_variance_),
    index=data_scaled.columns,
    columns=component_names,
)

corr_loadings.to_excel(os.path.join(OUTPUT_DIR, "pca_correlation_loadings.xlsx"))

print("Веса переменных в главных компонентах")
print(weights.round(4).to_string())

print("Корреляционные нагрузки")
print(corr_loadings.round(4).to_string())

print("Доля объясненной дисперсии финальной модели")
print(f"{pca_final.explained_variance_ratio_.sum() * 100:.2f}%")

plt.figure(figsize=(10, 5))
for col in scores.columns:
    plt.plot(scores.index, scores[col], label=col)

plt.axhline(0, linewidth=1)
plt.xlabel("Дата")
plt.ylabel("Значение компоненты")
plt.title("Динамика выбранных главных компонент")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_scores_dynamics.png"), dpi=300)
plt.close()

if n_components >= 2:
    plt.figure(figsize=(8, 7))
    plt.scatter(corr_loadings["PC1"], corr_loadings["PC2"])

    for var in corr_loadings.index:
        plt.text(
            corr_loadings.loc[var, "PC1"],
            corr_loadings.loc[var, "PC2"],
            var,
            fontsize=8,
        )

    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Корреляционные нагрузки переменных на PC1 и PC2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_loadings_pc1_pc2.png"), dpi=300)
    plt.close()

for pc in component_names:
    sorted_loadings = corr_loadings[pc].sort_values()

    plt.figure(figsize=(8, 6))
    plt.barh(sorted_loadings.index, sorted_loadings.values)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Корреляционная нагрузка")
    plt.title(f"Нагрузки переменных на {pc}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"loadings_{pc}.png"), dpi=300)
    plt.close()

print("Напоминание")
print("""Знак главных компонент условен: компоненту и ее нагрузки можно одновременно 
умножить на -1 без изменения смысла PCA. Интерпретация должна основываться 
на структуре нагрузок, а не на абсолютном знаке."""
)
