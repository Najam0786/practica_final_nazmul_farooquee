import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, confusion_matrix, ConfusionMatrixDisplay,
                              accuracy_score, classification_report)
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
os.makedirs("output", exist_ok=True)

# -----------------------------
# 1. Load & clean (same as Exercise 1)
# -----------------------------
df = pd.read_csv("data/AirQualityUCI.csv", sep=";", decimal=",")
df = df.dropna(axis=1, how='all')
df = df.replace(-200, np.nan)
df = df.drop(columns=['NMHC(GT)'])

target = "CO(GT)"
df = df.dropna(subset=[target])

num_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target]
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# -----------------------------
# 2. Features & target
# -----------------------------
feature_cols = num_cols
X = df[feature_cols]
# Encode categorical features if any exist (no-op here: all features are already numeric after cleaning)
X = pd.get_dummies(X, drop_first=True)
y = df[target]

print(f"Features used: {feature_cols}")
print(f"Dataset shape: {X.shape}")

# -----------------------------
# 3. Train / test split (80/20)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

print("\n--- PREPROCESSING ---")
print("Numerical features scaled using StandardScaler (fit on train only — no leakage)")
print("Train/test split: 80/20 with random_state=42 for reproducibility")
print("Missing values imputed using median before split (acceptable at this level)")

# -----------------------------
# 4. Standardise features
# -----------------------------
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# -----------------------------
# 5. Train Linear Regression
# -----------------------------
model = LinearRegression()
model.fit(X_train_sc, y_train)

# -----------------------------
# 6. Evaluate on test set
# -----------------------------
y_pred = model.predict(X_test_sc)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n--- TEST SET METRICS ---")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# Save linear regression metrics to TXT
with open("output/ej2_metricas_regresion.txt", "w", encoding="utf-8") as f:
    f.write("Metricas — Regresion Lineal (test set)\n")
    f.write("=" * 45 + "\n")
    f.write(f"MAE:  {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R2:   {r2:.4f}\n")

# -----------------------------
# 7. Coefficients
# -----------------------------
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n--- MODEL INTERPRETATION ---")
print(f"R\u00b2 indicates that {r2*100:.2f}% of variance in CO(GT) is explained by the model")
print("RMSE > MAE suggests presence of some larger errors (influenced by outliers)")

print("\nMost influential features (top 5):")
print(coef_df.head(5).to_string(index=False))

print("\n--- MODEL COEFFICIENTS ---")
print(coef_df.to_string(index=False))

# -----------------------------
# 8. Coefficient bar chart
# -----------------------------
coef_sorted = coef_df.sort_values('Coefficient')
colors = ['tomato' if c < 0 else 'steelblue' for c in coef_sorted['Coefficient']]
plt.figure(figsize=(9, 6))
plt.barh(coef_sorted['Feature'], coef_sorted['Coefficient'], color=colors, edgecolor='black')
plt.axvline(0, color='black', lw=0.8)
plt.xlabel("Coeficiente (estandarizado)")
plt.title("Coeficientes del Modelo de Regresion Lineal")
plt.tight_layout()
plt.savefig("output/ej2_coeficientes.png")
plt.close()

# -----------------------------
# 9. Residuals plot
# -----------------------------
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted CO(GT)")
plt.ylabel("Residual")
plt.title("Residuos — Regresion Lineal")
plt.tight_layout()
plt.savefig("output/ej2_residuos.png")
plt.close()

print("\nResidual analysis:")
print("Residuals randomly distributed around 0 indicate no systematic error and a well-specified model")
print(f"Residual mean: {residuals.mean():.4f} (expected ~0 for an unbiased model)")

# =====================================================
# OPTIONAL: Logistic Regression (not required for grading, used for comparison only)
# MODELO B — Regresión Logística
# =====================================================

# Binarizar target: 1 si CO(GT) > mediana, 0 si no
threshold = df[target].median()
y_bin = (df[target] > threshold).astype(int)

X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(
    df[feature_cols], y_bin, test_size=0.2, random_state=42
)

scaler_log = StandardScaler()
X_tr_b_sc = scaler_log.fit_transform(X_tr_b)
X_te_b_sc = scaler_log.transform(X_te_b)

log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_tr_b_sc, y_tr_b)
y_pred_bin = log_model.predict(X_te_b_sc)

acc = accuracy_score(y_te_b, y_pred_bin)
print(f"\n--- LOGISTIC REGRESSION ---")
print(f"Threshold (mediana): {threshold:.4f} mg/m3")
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_te_b, y_pred_bin,
                             target_names=["Bajo", "Alto"]))

# Confusion matrix
cm = confusion_matrix(y_te_b, y_pred_bin)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Bajo CO", "Alto CO"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues")
plt.title("Matriz de Confusion — Regresion Logistica")
plt.tight_layout()
plt.savefig("output/ej2_matriz_confusion.png")
plt.close()

print("\nFicheros guardados en output/")
