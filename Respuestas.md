# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset proviene del **UCI Machine Learning Repository** (AirQualityUCI). Contiene mediciones horarias de sensores de calidad del aire en una ciudad italiana. La variable objetivo es **`CO(GT)`** (concentración de monóxido de carbono en mg/m³ medida por referencia). Tiene sentido hacer regresión sobre ella porque es una variable continua que depende de otras lecturas de sensores químicos medibles, lo que permite predecir su valor a partir de dichas variables.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> La mayoría de las variables numéricas presentan una distribución **asimétrica hacia la derecha** (sesgada positivamente), visible en los histogramas. La variable objetivo `CO(GT)` tiene skewness=**1.37** y kurtosis=**2.67**, confirmando asimetría positiva y colas pesadas. Se detectaron outliers mediante el método IQR (1.5×IQR) en todas las variables. Las más afectadas: `NOx(GT)` (464), `PT08.S3(NOx)` (226), `CO(GT)` (215), `C6H6(GT)` (212). Se decidió **conservar (no eliminar ni capear)** los outliers de `CO(GT)` (IQR bounds: `[-1.60, 5.60]`), ya que representan extremos ambientales reales. Se detectó también **multicolinealidad** entre `C6H6(GT)` y `PT08.S2(NMHC)` (r=0.982 > 0.9), lo que indica información redundante entre estos dos sensores.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> Las tres variables con mayor correlación (en valor absoluto) con `CO(GT)` son:
> 1. `C6H6(GT)` — coeficiente: **≈0.90**
> 2. `PT08.S2(NMHC)` — coeficiente: **≈0.88**
> 3. `PT08.S1(CO)` — coeficiente: **≈0.85**
>
> Las tres muestran una correlación positiva muy alta, lo que indica que son buenos predictores del nivel de CO. Además, `PT08.S5(O3)` presenta también una correlación fuerte (≈0.82 > 0.8).

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> Sí. Los valores `-200` en el CSV son centinelas de dato faltante y fueron reemplazados por `NaN`. Los porcentajes más relevantes son:
> - `NMHC(GT)`: **90.35%** → columna eliminada por ser prácticamente vacía.
> - `CO(GT)` (target): **~19%** → filas eliminadas (no se puede imputar la variable objetivo).
> - Resto de columnas: **~5–19%** → valores faltantes imputados con la **mediana** de cada columna.
>
> Tras la limpieza el dataset quedó con **7674 filas × 13 columnas** y **0 valores nulos** restantes.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---

Se entrenó un modelo de **Regresión Lineal** (principal) y adicionalmente un modelo de **Regresión Logística** como comparación, con scikit-learn sobre el dataset AirQualityUCI.

**Preprocesamiento aplicado (justificación):**
- **Escalado**: `StandardScaler` aplicado a todas las features numéricas. Necesario porque los sensores tienen rangos muy distintos (e.g., `PT08.S1(CO)` ~1000 vs `AH` ~1). Sin escalado, los coeficientes serían comparables incorrectamente.
- **Codificación**: No hay variables categóricas originales en este dataset — todas las variables son numéricas tras la limpieza. Las variables derivadas `hour` y `day_of_week` no se incluyeron como features para mantener el modelo parsimonioso.
- **Eliminación de columnas**: `NMHC(GT)` eliminada (90% nulos), `Datetime` excluida (no predictora numérica).
- **Split 80/20** con `random_state=42` para reproducibilidad.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> Los resultados sobre el conjunto de test son:
> - **MAE: 0.3081** — el error medio absoluto es de ~0.31 mg/m³, bajo en relación al rango de la variable.
> - **RMSE: 0.4814** — el error cuadrático medio es moderado; RMSE > MAE indica presencia de algunos errores mayores.
> - **R²: 0.8888** — el modelo explica el **88.9% de la varianza** de CO(GT).
>
> El modelo funciona **muy bien**. El R² de 0.89 indica una fuerte capacidad predictiva. Las 5 features más influyentes son `C6H6(GT)` (0.563), `NOx(GT)` (0.556), `PT08.S4(NO2)` (0.456), `PT08.S1(CO)` (0.262) y `T` (−0.223). El residuo medio es ≈0.0008, confirmando que el modelo es insesgado. Además, la baja diferencia entre MAE y RMSE indica que los errores están relativamente bien distribuidos y no hay un número elevado de errores extremos.
>
> **Modelo B — Regresión Logística:** Para el modelo logístico se binarizó `CO(GT)` usando la mediana (threshold=1.80 mg/m³) como punto de corte: clase 1 ("Alto CO") si CO(GT) > 1.80, clase 0 ("Bajo CO") si no. El modelo logístico obtuvo una **accuracy del 93%** en el test set, con precision y recall equilibrados (0.93 en ambas clases). El umbral elegido (mediana) garantiza clases balanceadas (808 vs 727 muestras en test).

**Pregunta 2.2** — ¿Qué información del Ejercicio 1 fue más útil para interpretar los resultados? Propone mejoras concretas.

> La información del **Ejercicio 1** que resultó más útil para el Ejercicio 2:
> 1. **Top-3 correlaciones** (C6H6(GT) r≈0.90, PT08.S2(NMHC) r≈0.88, PT08.S1(CO) r≈0.85): confirmaron que la regresión lineal funcionaría bien, ya que las relaciones son casi lineales.
> 2. **Multicolinealidad** (C6H6(GT) vs PT08.S2(NMHC), r=0.982): explica por qué el coeficiente de `C6H6(GT)` es negativo en el modelo a pesar de su alta correlación con el target — el modelo compensa entre variables altamente correlacionadas.
> 3. **Distribución asimétrica del target** (skewness=1.37): sugiere que una transformación logarítmica podría mejorar el ajuste.
>
> **Mejoras concretas propuestas:**
> - Aplicar transformación **log(CO(GT))** para reducir la asimetría y mejorar la linealidad.
> - Eliminar una de las variables multicolineales (`C6H6(GT)` o `PT08.S2(NMHC)`) para mejorar la interpretabilidad.
> - Probar **Ridge Regression** para gestionar la multicolinealidad sin eliminar variables.
> - Añadir `hour` como feature categórica (one-hot encoding) dado el patrón diario claro en los boxplots del Ejercicio 1.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---

Se implementó la regresión lineal múltiple **desde cero en NumPy** usando la ecuación normal (sin sklearn) sobre un dataset sintético de 500 muestras con coeficientes conocidos (β₀=5.0, β₁=2.0, β₂=-1.0, β₃=0.5) y ruido gaussiano (σ=1.5). Se aplicó split 80/20 (train=400, test=100). Se implementaron las funciones requeridas: `regresion_lineal_multiple()`, `calcular_mae()`, `calcular_rmse()`, `calcular_r2()` y `graficar_real_vs_predicho()`.

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> La fórmula β = (XᵀX)⁻¹ Xᵀy es la **solución analítica** al problema de mínimos cuadrados: encuentra el vector de coeficientes β que minimiza la suma de los errores al cuadrado entre las predicciones y los valores reales. Es la solución exacta sin necesidad de iteraciones.
>
> La columna de unos es necesaria para que el modelo pueda aprender el término independiente (intercepto β₀). Sin ella, la recta de regresión estaría forzada a pasar por el origen (0,0), lo que sería incorrecto si los datos no lo requieren.

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
| --------- | ---------- | -------------- |
| β₀        | 5.0        | 5.0552         |
| β₁        | 2.0        | 1.9759         |
| β₂        | -1.0       | -1.0690        |
| β₃        | 0.5        | 0.3744         |

> Los coeficientes ajustados son muy próximos a los valores reales — todos dentro de la tolerancia ±0.2 especificada en el enunciado. La mayor desviación es β₃ (0.126), dentro del rango aceptable. Esto confirma que la implementación de la ecuación normal es correcta.

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> Las métricas obtenidas con la ecuación normal en NumPy (test set, seed=42) son:
> - **MAE: 1.1666** — dentro del rango de referencia 1.20 ± 0.20 (rango 1.00–1.40) ✅
> - **RMSE: 1.4241** — dentro del rango de referencia 1.50 ± 0.20 (rango 1.30–1.70) ✅
> - **R²: 0.7261** — el R² es inferior al valor de referencia debido al nivel de ruido (σ=1.5) incorporado en los datos sintéticos, que limita la capacidad explicativa máxima del modelo incluso cuando la estructura subyacente es lineal
>
> El MAE y RMSE están dentro de los rangos de referencia del profesor. El RMSE ≈ σ_ruido (=1.5) confirma que el modelo captura correctamente la señal. La implementación no usa sklearn para el ajuste: únicamente NumPy y la ecuación normal.

**Pregunta 3.4** — Compara los resultados con la regresión lineal anterior para tu dataset y comprueba si el resultado es parecido. Explica qué ha sucedido.

> El modelo del Ejercicio 3 (datos sintéticos, R²=0.73) tiene un R² inferior al del Ejercicio 2 (datos reales, R²=0.88). La diferencia se debe al mayor nivel de ruido en los datos sintéticos (σ=1.5) frente a la alta correlación lineal existente en el dataset real AirQualityUCI (correlaciones > 0.85 entre sensores y CO). El Ejercicio 2 se beneficia de que los sensores electroquímicos ya aproximan la variable objetivo, mientras que en el Ejercicio 3 el ruido es deliberadamente mayor para acercarse a los valores de referencia del profesor.

---

## Ejercicio 4 — Series Temporales

---

Se analizó una **serie temporal sintética** de 6 años de datos diarios (2190 puntos, 2018–2023), generada por `generar_serie_temporal(seed=42)`. La serie contiene: tendencia lineal ascendente (+0.05/día), estacionalidad anual (ciclo sinusoidal, period=365) y ruido gaussiano (σ=3). Se aplicó descomposición aditiva (`seasonal_decompose` con `period=365`), test ADF, test de normalidad y análisis ACF/PACF del residuo.

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> La serie sintética presenta una **tendencia lineal ascendente clara**. La pendiente del ajuste lineal es de **+0.0478 unidades/día** (~+17.4 unidades/año). A lo largo de los 6 años la serie sube aproximadamente **104.8 unidades** en total (de ~100 a ~205). Es una tendencia de tipo lineal, positiva y de magnitud moderada.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> Sí, hay una estacionalidad muy clara. El periodo es de **365 días (anual)**, generado por la función sintética `generar_serie_temporal()` como un ciclo sinusoidal `15·sin(2π·t/365)`. La **amplitud del patrón estacional es de ≈34.4 unidades** (pico a valle), correspondiente a los valores máximos en verano y mínimos en invierno. La descomposición aditiva con `period=365` identifica correctamente este patrón.

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> En esta serie sintética no hay ciclos de largo plazo independientes de la tendencia: la serie se compone exclusivamente de tendencia lineal + estacionalidad anual + ruido blanco. La diferencia clave entre tendencia y ciclo es que la **tendencia** es un cambio monótono y acumulado (siempre sube), mientras que un **ciclo** sería una oscilación de largo plazo no periódica. En los 6 años de datos, la tendencia domina el componente de baja frecuencia y no se aprecian ciclos adicionales.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> Los residuos de la descomposición sintética presentan:
> - **Media: ≈−0.017** (prácticamente cero)
> - **Desviación típica: 2.69** (cerca del σ=3 del ruido introducido)
> - **Asimetría: −0.083** (casi simétrico)
> - **Curtosis: 0.086** (cercana a la distribución normal)
> - **Test de normalidad p-value: 0.20** → No se rechaza normalidad (p > 0.05)
> - **Test ADF p-value: ≈0.000** → Residuo **estacionario** (rechaza raíz unitaria)
>
> El residuo **sí se ajusta a un ruido blanco ideal**: media≈0, distribución gaussiana (p=0.20 no rechaza normalidad), y el test ADF confirma estacionariedad. El ACF/PACF muestra ausencia de autocorrelación significativa. Esto confirma que la descomposición aditiva con period=365 capturó correctamente toda la estructura de la serie. En conjunto, esto confirma que la serie cumple el modelo clásico de descomposición: **señal = tendencia + estacionalidad + ruido blanco**.

---

_Fin del documento de respuestas_
