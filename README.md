# Práctica Final — Análisis y Modelado de Datos
**Módulo:** Estadística para Data Science  
**Autor:** Nazmul Farooquee  
**Dataset:** AirQualityUCI (UCI Machine Learning Repository)

---

## Estructura del repositorio

```
├── data/
│   └── AirQualityUCI.csv          # Dataset original
├── output/                        # Todos los ficheros generados
│   ├── ej1_descriptivo.csv
│   ├── ej1_descriptivo_enhanced.csv
│   ├── ej1_histogramas.png
│   ├── ej1_heatmap_correlacion.png
│   ├── ej1_categoricas.png
│   ├── ej1_boxplots.png
│   ├── ej1_outliers.txt
│   ├── ej1_multicollinearity.txt
│   ├── ej2_metricas_regresion.txt
│   ├── ej2_coeficientes.png
│   ├── ej2_residuos.png
│   ├── ej2_matriz_confusion.png
│   ├── ej3_coeficientes.txt
│   ├── ej3_metricas.txt
│   ├── ej3_predicciones.png
│   ├── ej4_serie_original.png
│   ├── ej4_descomposicion.png
│   ├── ej4_acf_pacf.png
│   ├── ej4_histograma_ruido.png
│   └── ej4_analisis.txt
├── ejercicio1_descriptivo.py      # Análisis estadístico descriptivo
├── ejercicio2_inferencia.py       # Regresión lineal con scikit-learn
├── ejercicio3_regresion_multiple.py  # OLS desde cero en NumPy
├── ejercicio4_series_temporales.py   # Análisis de series temporales
├── Respuestas.md                  # Respuestas a todas las preguntas
└── Practica_Final_Enunciado.pdf   # Enunciado original
```

## Ejecución

```bash
python ejercicio1_descriptivo.py
python ejercicio2_inferencia.py
python ejercicio3_regresion_multiple.py
python ejercicio4_series_temporales.py
```

Todos los scripts generan sus ficheros de salida en la carpeta `output/`.

## Resumen de ejercicios

| Ejercicio | Descripción | Métricas clave |
|-----------|-------------|----------------|
| 1 | Análisis descriptivo, outliers, correlaciones | Skewness=1.37, Top-3 r≈0.90/0.88/0.85 |
| 2 | Regresión lineal (scikit-learn) | MAE=0.308, RMSE=0.481, R²=0.889 |
| 3 | OLS desde cero (NumPy) | MAE=1.167, RMSE=1.424, R²=0.726 |
| 4 | Descomposición serie temporal | ADF p≈0, Normalidad p=0.20 |
