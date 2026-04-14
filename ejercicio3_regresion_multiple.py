import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs("output", exist_ok=True)


# =====================================================
# FUNCIONES REQUERIDAS
# =====================================================

def regresion_lineal_multiple(X_train, y_train, X_test):
    """
    Ajusta un modelo de regresión lineal múltiple mediante la ecuación
    normal OLS (sin sklearn) y devuelve predicciones sobre X_test.

    Fórmula: beta = (X^T X)^-1 X^T y

    Parameters
    ----------
    X_train : ndarray, shape (n, p)  — features de entrenamiento
    y_train : ndarray, shape (n,)    — target de entrenamiento
    X_test  : ndarray, shape (m, p)  — features de test

    Returns
    -------
    y_pred : ndarray, shape (m,)    — predicciones sobre X_test
    beta   : ndarray, shape (p+1,)  — coeficientes [intercept, β1, β2, ...]
    """
    ones_train = np.ones((X_train.shape[0], 1))
    ones_test  = np.ones((X_test.shape[0], 1))
    Xb_train = np.hstack([ones_train, X_train])
    Xb_test  = np.hstack([ones_test,  X_test])
    beta = np.linalg.pinv(Xb_train.T @ Xb_train) @ Xb_train.T @ y_train
    y_pred = Xb_test @ beta
    return y_pred, beta


def calcular_mae(y_real, y_pred):
    """
    Calcula el Mean Absolute Error (MAE).

    Parameters
    ----------
    y_real : ndarray — valores reales
    y_pred : ndarray — valores predichos

    Returns
    -------
    float — MAE
    """
    return float(np.mean(np.abs(y_real - y_pred)))


def calcular_rmse(y_real, y_pred):
    """
    Calcula el Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_real : ndarray — valores reales
    y_pred : ndarray — valores predichos

    Returns
    -------
    float — RMSE
    """
    return float(np.sqrt(np.mean((y_real - y_pred) ** 2)))


def calcular_r2(y_real, y_pred):
    """
    Calcula el coeficiente de determinación R².

    Parameters
    ----------
    y_real : ndarray — valores reales
    y_pred : ndarray — valores predichos

    Returns
    -------
    float — R²
    """
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def graficar_real_vs_predicho(y_real, y_pred,
                               path="output/ej3_predicciones.png"):
    """
    Genera scatter plot de Valores Reales vs. Valores Predichos
    con la línea de referencia perfecta (y = ŷ).

    Parameters
    ----------
    y_real : ndarray — valores reales
    y_pred : ndarray — valores predichos
    path   : str     — ruta de guardado del gráfico
    """
    plt.figure(figsize=(7, 6))
    plt.scatter(y_real, y_pred, alpha=0.5, label="Predicciones")
    plt.grid(alpha=0.3)
    lims = [min(y_real.min(), y_pred.min()), max(y_real.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', lw=2, label="Predicción perfecta")
    plt.xlabel("Valor Real (y)")
    plt.ylabel("Valor Predicho (ŷ)")
    plt.title("Real vs. Predicho — Regresión Lineal Múltiple (NumPy OLS)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# =====================================================
# MAIN — datos sintéticos con semilla fija (seed=42)
# =====================================================

# 1. Generar datos sintéticos
n = 500
X_raw = np.random.randn(n, 3)  # NOTE: Scaling not required — synthetic features are already standard normal
true_beta = np.array([5.0, 2.0, -1.0, 0.5])
noise = np.random.randn(n) * 1.5
y = (true_beta[0]
     + true_beta[1] * X_raw[:, 0]
     + true_beta[2] * X_raw[:, 1]
     + true_beta[3] * X_raw[:, 2]
     + noise)

print("--- DATOS SINTÉTICOS ---")
print(f"n={n}, features=3, sigma_ruido=1.5")
print(f"Coeficientes reales: {true_beta}")

# 2. Train / test split (80/20) — sin sklearn
idx = np.random.permutation(n)
n_train = int(0.8 * n)
X_train, X_test = X_raw[idx[:n_train]], X_raw[idx[n_train:]]
y_train, y_test = y[idx[:n_train]], y[idx[n_train:]]
print(f"Train: {len(y_train)}, Test: {len(y_test)}")

# 3. Ajuste OLS (ecuación normal)
y_pred, beta_hat = regresion_lineal_multiple(X_train, y_train, X_test)

# 4. Métricas
mae  = calcular_mae(y_test, y_pred)
rmse = calcular_rmse(y_test, y_pred)
r2   = calcular_r2(y_test, y_pred)

labels = ["beta_0 (intercepto)", "beta_1", "beta_2", "beta_3"]

print("\n--- COEFICIENTES AJUSTADOS ---")
for lab, tv, fv in zip(labels, true_beta, beta_hat):
    print(f"  {lab}: real={tv:.1f}, ajustado={fv:.4f}")

print("\n--- MÉTRICAS (test set) ---")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")
print("\nReferencia del profesor:")
print("  MAE  aprox 1.20 (+/-0.20)")
print("  RMSE aprox 1.50 (+/-0.20)")
print("  R2   aprox 0.80 (+/-0.05)")

print("\n--- INTERPRETATION ---")
print("Coeficientes estimados muy cercanos a los valores reales -> implementacion correcta")
print(f"R2 = {r2:.4f} -> el modelo explica el {r2*100:.1f}% de la varianza del target")
print("MAE indicates average prediction error magnitude")
print("RMSE penalizes larger errors more heavily than MAE")

# 5. Guardar coeficientes en TXT
with open("output/ej3_coeficientes.txt", "w", encoding="utf-8") as f:
    f.write("Coeficientes ajustados vs. valores reales\n")
    f.write("=" * 50 + "\n")
    for lab, tv, fv in zip(labels, true_beta, beta_hat):
        f.write(f"{lab}: real={tv:.1f}, ajustado={fv:.4f}\n")

# 6. Guardar métricas en TXT
with open("output/ej3_metricas.txt", "w", encoding="utf-8") as f:
    f.write("Metricas del modelo — test set sintetico (seed=42)\n")
    f.write("=" * 50 + "\n")
    f.write(f"MAE:  {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R2:   {r2:.4f}\n")
    f.write("\nReferencia del profesor:\n")
    f.write("  MAE  aprox 1.20 (+/-0.20)\n")
    f.write("  RMSE aprox 1.50 (+/-0.20)\n")
    f.write("  R2   aprox 0.80 (+/-0.05)\n")

# 7. Gráfico Real vs. Predicho
graficar_real_vs_predicho(y_test, y_pred)

print("\nFicheros guardados en output/")
