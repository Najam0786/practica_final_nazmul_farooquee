import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# 1. Setup
# -----------------------------
np.random.seed(42)

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# -----------------------------
# 2. Load dataset
# -----------------------------
df = pd.read_csv("data/AirQualityUCI.csv", sep=";", decimal=",")

# -----------------------------
# 3. Data cleaning
# -----------------------------

# Remove empty columns
df = df.dropna(axis=1, how='all')

# Replace -200 with NaN (IMPORTANT)
df = df.replace(-200, np.nan)

# Fix time format (very important)
df['Time'] = df['Time'].str.replace('.', ':', regex=False)

# Create Datetime column
df['Datetime'] = pd.to_datetime(
    df['Date'] + ' ' + df['Time'],
    format='%d/%m/%Y %H:%M:%S',
    errors='coerce'
)

# Drop original Date and Time
df = df.drop(columns=['Date', 'Time'])

# -----------------------------
# 4. Basic info
# -----------------------------
print("\n--- INFO ---")
print(df.info())

print("\n--- SHAPE ---")
print(df.shape)

print("\n--- MISSING VALUES (%) ---")
missing = df.isnull().mean() * 100
print(missing)

# -----------------------------
# 4b. Treat missing values
# -----------------------------

# Drop NMHC(GT): 90% missing, not usable
df = df.drop(columns=['NMHC(GT)'])

# Drop rows where target is NaN
target = "CO(GT)"
df = df.dropna(subset=[target])

# Fill remaining NaN with column median
num_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target]
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

print("\n--- MISSING VALUE TREATMENT ---")
print("NMHC(GT) dropped: ~90% missing values — column not usable")
print("Target rows with NaN dropped: cannot impute the target variable")
print("Remaining missing values imputed using median to preserve distribution")

print("\n--- SHAPE AFTER CLEANING ---")
print(df.shape)
print("\n--- REMAINING NaN ---")
print(df.isnull().sum().sum(), "total NaN remaining")
print("\nMemory usage (MB):", round(df.memory_usage(deep=True).sum() / 1e6, 3))

# -----------------------------
# 6. Descriptive statistics 
# -----------------------------
desc = df.describe()

# Add extra stats: variance, IQR, skewness, kurtosis, mode
skewness = df[num_cols].skew().rename('skewness')
kurtosis = df[num_cols].kurt().rename('kurtosis')
iqr_row  = (df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25)).rename('IQR')
variance = df[num_cols].var().rename('variance')
mode_row = df[num_cols].mode().iloc[0].rename('mode')

enhanced = pd.concat([desc, variance.to_frame().T,
                       skewness.to_frame().T,
                       kurtosis.to_frame().T,
                       iqr_row.to_frame().T,
                       mode_row.to_frame().T])
enhanced.to_csv("output/ej1_descriptivo_enhanced.csv")  
desc.to_csv("output/ej1_descriptivo.csv")                

print(f"\nTarget skewness: {df[target].skew():.4f}")
print(f"Target kurtosis: {df[target].kurt():.4f}")
print(f"Target IQR:      {df[target].quantile(0.75) - df[target].quantile(0.25):.4f}")

# -----------------------------
# 7. Histograms with KDE
# -----------------------------
plot_cols = num_cols + [target]
n_cols_plot = 4
n_rows_plot = (len(plot_cols) + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                          figsize=(16, n_rows_plot * 3))
axes = axes.flatten()
for i, col in enumerate(plot_cols):
    sns.histplot(df[col], ax=axes[i], kde=True, bins=40, color='steelblue')
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel("")
for j in range(len(plot_cols), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Distribuciones de Variables Numéricas (con KDE)", fontsize=12)
plt.tight_layout()
plt.savefig("output/ej1_histogramas.png")
plt.close()

# -----------------------------
# 8. Outlier detection (IQR) — all numeric variables
# -----------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("\n--- OUTLIERS PER VARIABLE ---")
outlier_lines = []
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    line = f"  {col}: {n_out} outliers  (IQR bounds: [{lo:.2f}, {hi:.2f}])"
    print(line)
    outlier_lines.append(line)

# Save outliers to TXT
with open("output/ej1_outliers.txt", "w", encoding="utf-8") as f:
    f.write("Deteccion de Outliers — Metodo IQR (1.5 x IQR)\n")
    f.write("=" * 55 + "\n")
    for line in outlier_lines:
        f.write(line.strip() + "\n")

print(f"\nOutliers in '{target}' detected using IQR but NOT removed or capped.")
print("Decision: retained as they represent real environmental extremes.")

# Append decision justification to outliers file
with open("output/ej1_outliers.txt", "a", encoding="utf-8") as f:
    f.write("\nDecision: outliers RETAINED (not capped or removed).\n")
    f.write("They represent real environmental extremes; removing them would distort the data.\n")

# -----------------------------
# 9. Correlation heatmap
# -----------------------------
plt.figure(figsize=(10, 8))
# Pearson correlation used to measure linear relationships between variables
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.savefig("output/ej1_heatmap_correlacion.png")
plt.close()

# -----------------------------
# 10. Top 3 correlations + multicollinearity
# -----------------------------
target_corr = corr[target].abs().sort_values(ascending=False)
top3 = target_corr[1:4]

print("\n--- TOP 3 CORRELATIONS ---")
print(top3)

print("\nStrong correlations (|r| > 0.8) with target:")
print(target_corr[target_corr > 0.8])

# Detect multicollinearity: pairs with |r| > 0.9
print("\n--- MULTICOLLINEARITY (|r| > 0.9) ---")
multi_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.9:
            c1, c2 = corr.columns[i], corr.columns[j]
            r_val = corr.iloc[i, j]
            print(f"  {c1} vs {c2}: r={r_val:.3f}")
            multi_pairs.append((c1, c2, r_val))
if not multi_pairs:
    print("  No multicollinearity detected.")

# Save multicollinearity to TXT
with open("output/ej1_multicollinearity.txt", "w", encoding="utf-8") as f:
    f.write("Multicollinearity Analysis — Pairs with |r| > 0.9\n")
    f.write("=" * 50 + "\n")
    if multi_pairs:
        for c1, c2, r_val in multi_pairs:
            f.write(f"{c1} vs {c2}: r={r_val:.3f}\n")
    else:
        f.write("No multicollinearity detected.\n")

# -----------------------------
# 11. Categorical variables: hour and day_of_week
# -----------------------------
df['hour']        = df['Datetime'].dt.hour
df['day_of_week'] = df['Datetime'].dt.dayofweek  # 0=Monday, 6=Sunday

day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Absolute and relative frequencies
for cat_col in ['hour', 'day_of_week']:
    freq_abs = df[cat_col].value_counts().sort_index()
    freq_rel = (freq_abs / freq_abs.sum() * 100).round(2)
    print(f"\n--- {cat_col.upper()} FREQUENCY ---")
    freq_df = pd.DataFrame({'absolute': freq_abs, 'relative_%': freq_rel})
    print(freq_df.to_string())
    dominant = freq_abs.idxmax()
    print(f"  Dominant category: {dominant} ({freq_rel[dominant]:.1f}%)")

# Combined categorical chart: 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

hour_abs = df['hour'].value_counts().sort_index()
hour_rel = hour_abs / hour_abs.sum() * 100
axes[0].bar(hour_abs.index, hour_rel.values, color='steelblue', edgecolor='black')
axes[0].set_title("Distribución por Hora del Día", fontsize=11)
axes[0].set_xlabel("Hora")
axes[0].set_ylabel("Frecuencia Relativa (%)")
axes[0].grid(axis='y', alpha=0.3)

dow_abs = df['day_of_week'].value_counts().sort_index()
dow_rel = dow_abs / dow_abs.sum() * 100
axes[1].bar(range(7), dow_rel.values, color='darkorange', edgecolor='black')
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(day_names)
axes[1].set_title("Distribución por Día de la Semana", fontsize=11)
axes[1].set_xlabel("Día")
axes[1].set_ylabel("Frecuencia Relativa (%)")
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle("Variables Categóricas (Frecuencia Relativa)", fontsize=12)
plt.tight_layout()
plt.savefig("output/ej1_categoricas.png")
plt.close()

# -----------------------------
# 12. Boxplots by both categoricals
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 5))
sns.boxplot(x=df['hour'], y=df[target], ax=axes[0])
axes[0].set_title("CO(GT) por Hora del Día")
axes[0].set_xlabel("Hora")
sns.boxplot(x=df['day_of_week'], y=df[target], ax=axes[1])
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(day_names)
axes[1].set_title("CO(GT) por Día de la Semana")
axes[1].set_xlabel("Día")
plt.tight_layout()
plt.savefig("output/ej1_boxplots.png")
plt.close()

# -----------------------------
# 13. Datetime check
# -----------------------------
print("\n--- DATETIME VALID ---")
print("Valid Datetime rows:", df['Datetime'].notnull().sum())
print("Datetime missing values (~1.2%) due to parsing issues were retained but not critical for analysis.")