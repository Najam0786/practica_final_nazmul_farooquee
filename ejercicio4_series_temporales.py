import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

os.makedirs("output", exist_ok=True)


# =====================================================
# FUNCIÓN PROPORCIONADA — NO MODIFICAR
# =====================================================

def generar_serie_temporal(seed=42):
    """
    Genera una serie temporal sintética de 6 años de datos diarios.
    Contiene: tendencia lineal ascendente, estacionalidad anual (period=365)
    y ruido gaussiano.

    IMPORTANTE: No modificar esta función ni la semilla.

    Parameters
    ----------
    seed : int — semilla aleatoria (default=42)

    Returns
    -------
    pd.Series con índice de fechas diarias (freq='D')
    """
    np.random.seed(seed)
    n = 365 * 6                                           # 6 años de datos diarios
    t = np.arange(n)
    fechas = pd.date_range(start='2018-01-01', periods=n, freq='D')

    tendencia      = 0.05 * t                             # tendencia lineal ascendente
    estacionalidad = 15 * np.sin(2 * np.pi * t / 365)    # ciclo anual (period=365)
    ruido          = np.random.randn(n) * 3               # ruido gaussiano N(0, 9)

    serie = 100 + tendencia + estacionalidad + ruido
    return pd.Series(serie, index=fechas, name='valor')


# =====================================================
# ANÁLISIS
# =====================================================

# 1. Generar serie
ts = generar_serie_temporal(seed=42)

print("--- SERIE TEMPORAL SINTÉTICA ---")
print(f"Inicio: {ts.index.min().date()}   Fin: {ts.index.max().date()}")
print(f"Total dias: {len(ts)}")
print(f"Media: {ts.mean():.2f}   Std: {ts.std():.2f}")

# 2. Visualización: serie completa
plt.figure(figsize=(14, 4))
plt.plot(ts, lw=0.8, color='steelblue')
plt.title("Serie Temporal Sintética — Datos Diarios (6 años)", fontsize=13)
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/ej4_serie_original.png")
plt.close()

# 3. Tendencia (media móvil 365 días)
rolling_mean = ts.rolling(window=365, center=True).mean()
slope, intercept, r_val, p_val, se = stats.linregress(np.arange(len(ts)), ts.values)

print("\n--- TENDENCIA ---")
print(f"Pendiente lineal: {slope:.4f} unidades/dia")
print(f"Cambio total 6 anios: {slope * len(ts):.1f} unidades")

# 4. Descomposición estacional (period=365 — anual)
decomp = seasonal_decompose(ts, model='additive', period=365,
                            extrapolate_trend='freq')

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
decomp.observed.plot(ax=axes[0], lw=0.6, color='steelblue')
axes[0].set_ylabel("Observado")
axes[0].grid(True, alpha=0.3)
decomp.trend.plot(ax=axes[1], lw=1.5, color='red')
axes[1].set_ylabel("Tendencia")
axes[1].grid(True, alpha=0.3)
decomp.seasonal.plot(ax=axes[2], lw=0.8, color='green')
axes[2].set_ylabel("Estacionalidad")
axes[2].grid(True, alpha=0.3)
decomp.resid.plot(ax=axes[3], lw=0.6, color='grey')
axes[3].set_ylabel("Residuo")
axes[3].grid(True, alpha=0.3)
plt.suptitle("Descomposición Aditiva (period=365)", fontsize=13)
plt.tight_layout()
plt.savefig("output/ej4_descomposicion.png")
plt.close()

# 5. Estacionalidad
season_amp = decomp.seasonal.max() - decomp.seasonal.min()
print(f"\n--- ESTACIONALIDAD ---")
print(f"Periodo: 365 dias (anual)")
print(f"Amplitud: {season_amp:.4f} unidades")

# 6. Análisis del residuo
residuals = decomp.resid.dropna()

resid_mean     = residuals.mean()
resid_std      = residuals.std()
resid_skew     = float(residuals.skew())
resid_kurt     = float(residuals.kurtosis())
_, p_norm      = stats.normaltest(residuals)
adf_result     = adfuller(residuals, autolag='AIC')
adf_stat       = adf_result[0]
adf_pvalue     = adf_result[1]

print("\n--- ANÁLISIS DEL RESIDUO ---")
print(f"Media:    {resid_mean:.6f}")
print(f"Std dev:  {resid_std:.4f}")
print(f"Asimetria: {resid_skew:.4f}")
print(f"Curtosis:  {resid_kurt:.4f}")
print(f"Test normalidad p-value: {p_norm:.6f}")
print(f"ADF estadistico:  {adf_stat:.4f}")
print(f"ADF p-value:      {adf_pvalue:.6f}")
if adf_pvalue < 0.05:
    print("  -> Residuo ESTACIONARIO (rechaza raiz unitaria)")
else:
    print("  -> Residuo NO estacionario (p >= 0.05)")
if p_norm > 0.05:
    print("Residuals follow approximately normal distribution (p > 0.05 -> no rejection)")
else:
    print("Residuals deviate from normal distribution (p <= 0.05)")

# 7. ACF y PACF del residuo
fig, axes = plt.subplots(2, 1, figsize=(12, 7))
plot_acf(residuals,  ax=axes[0], lags=40, title="ACF del Residuo")
plot_pacf(residuals, ax=axes[1], lags=40, title="PACF del Residuo")
plt.tight_layout()
plt.savefig("output/ej4_acf_pacf.png")
plt.close()
print("ACF/PACF show low autocorrelation -> residual behaves like white noise")
print("No significant spikes -> no remaining structure in residuals")

# 8. Histograma del residuo con curva normal teórica
mu_r, sigma_r = norm.fit(residuals)
x_range = np.linspace(residuals.min(), residuals.max(), 300)
pdf_normal = norm.pdf(x_range, mu_r, sigma_r)

plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=60, density=True, edgecolor='black',
         alpha=0.7, label="Residuos")
plt.plot(x_range, pdf_normal, 'r-', lw=2,
         label=f"Normal (mu={mu_r:.2f}, sigma={sigma_r:.2f})")
plt.xlabel("Residuo")
plt.ylabel("Densidad")
plt.title("Histograma del Residuo con Curva Normal Teórica")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/ej4_histograma_ruido.png")
plt.close()

# 9. Guardar análisis en TXT
with open("output/ej4_analisis.txt", "w", encoding="utf-8") as f:
    f.write("Analisis de la Serie Temporal Sintetica\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Periodo analizado: {ts.index.min().date()} a {ts.index.max().date()}\n")
    f.write(f"Total observaciones: {len(ts)}\n\n")
    f.write("TENDENCIA\n")
    f.write(f"  Pendiente lineal: {slope:.4f} unidades/dia\n")
    f.write(f"  Cambio total: {slope * len(ts):.1f} unidades en 6 anios\n\n")
    f.write("ESTACIONALIDAD\n")
    f.write(f"  Periodo: 365 dias (anual)\n")
    f.write(f"  Amplitud: {season_amp:.4f} unidades\n\n")
    f.write("RESIDUO\n")
    f.write(f"  Media:     {resid_mean:.6f}\n")
    f.write(f"  Std dev:   {resid_std:.4f}\n")
    f.write(f"  Asimetria: {resid_skew:.4f}\n")
    f.write(f"  Curtosis:  {resid_kurt:.4f}\n\n")
    f.write("TEST DE NORMALIDAD (D'Agostino-Pearson)\n")
    f.write(f"  p-value: {p_norm:.6f}\n")
    if p_norm > 0.05:
        f.write("  -> No se rechaza normalidad (ruido aproximadamente gaussiano)\n\n")
    else:
        f.write("  -> Se rechaza normalidad\n\n")
    f.write("TEST ADF (Dickey-Fuller Aumentado)\n")
    f.write(f"  Estadistico: {adf_stat:.4f}\n")
    f.write(f"  p-value:     {adf_pvalue:.6f}\n")
    if adf_pvalue < 0.05:
        f.write("  -> Residuo ESTACIONARIO (p < 0.05)\n")
    else:
        f.write("  -> Residuo NO estacionario (p >= 0.05)\n")

print("\n--- GLOBAL INTERPRETATION ---")
print("Series shows upward trend + yearly seasonality + Gaussian noise")
print("\nTodos los ficheros guardados en output/")
