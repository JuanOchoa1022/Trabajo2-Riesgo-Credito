# Comparación de Modelos Supervisados con Validación Cruzada

Proyecto de ciencia de datos orientado a público general. El objetivo es diseñar y ejecutar un experimento de aprendizaje supervisado que compare el rendimiento de modelos clásicos de clasificación usando validación cruzada como herramienta principal de evaluación.

## Resumen
- Problema: predicción de morosidad seria en 2 años (riesgo de crédito).
- Dataset: Give Me Some Credit (Kaggle, competitions/GiveMeSomeCredit).
- Etiqueta: `SeriousDlqin2yrs` (binaria: 1 = morosidad seria, 0 = no).
- Predictores: >10 variables numéricas (edad, ingresos, utilización de crédito, atrasos, etc.).
- Métrica principal: ROC AUC (robusta ante desbalance de clases).
- Partición: 70% entrenamiento, 15% validación, 15% prueba, estratificada.
- Modelos comparados: LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier.

El notebook principal es `Trabajo2.ipynb` y está diseñado con celdas “Qué/Por qué/Cómo interpretar” para que cualquier lector entienda el proceso y los resultados.

## Dataset
- Origen: Kaggle → competitions/GiveMeSomeCredit.
- Archivo esperado: `data/cs-training.csv`.
- Variables destacadas:
  - `RevolvingUtilizationOfUnsecuredLines`: utilización de líneas de crédito no garantizadas.
  - `age`: edad.
  - `DebtRatio`: pagos mensuales de deuda y gastos / ingreso mensual.
  - `MonthlyIncome`: ingreso mensual (con faltantes).
  - `NumberOfTimes90DaysLate`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfTime30-59DaysPastDueNotWorse`: historial de atrasos.
  - `NumberOfOpenCreditLinesAndLoans`, `NumberRealEstateLoansOrLines`, `NumberOfDependents`.
- Particularidades: desbalance de clases (pocos positivos), faltantes en `MonthlyIncome` y `NumberOfDependents`.

## Metodología
1. Diagnóstico inicial y EDA breve para comprender desbalance y rangos de variables.
2. Partición en tres subconjuntos (train/val/test) con estratificación para preservar proporciones de clase.
3. Pipeline de preprocesamiento (scikit‑learn) aplicado consistentemente a todos los modelos:
   - Imputación de faltantes (mediana) para numéricas.
   - Estandarización (StandardScaler) para métodos sensibles a escala (p. ej., KNN).
4. Entrenamiento y evaluación base en train y val.
5. Validación cruzada (StratifiedKFold, K=5) sobre el conjunto de entrenamiento:
   - Se registran AUC por fold, promedio y desviación estándar.
6. Comparación y selección del modelo con mejor AUC promedio y menor variabilidad.
7. Reentrenamiento del mejor modelo con train+val y evaluación final en test (AUC, curva ROC y matriz de confusión como apoyo visual).

## ¿Por qué ROC AUC?
- Resume la capacidad de discriminación para todos los umbrales posibles.
- Es menos sensible al desbalance que la exactitud (accuracy).
- Permite comparar modelos sin fijar un punto de corte específico.

## Interpretación de Resultados
- `train_auc` vs `val_auc`: grandes diferencias sugieren sobreajuste.
- `cv_mean_auc` y `cv_std_auc`: desempeño promedio y estabilidad entre folds; preferimos alta media y baja desviación.
- AUC en prueba: debería ser coherente con la validación cruzada (cerca de la media; dentro de ~1–2 desviaciones estándar).
- Curva ROC: más cerca de la esquina superior izquierda implica mejor discriminación. La diagonal indica azar.
- Matriz de confusión (umbral 0.5): sólo ilustrativa; en problemas desbalanceados los umbrales deben ajustarse según costos de negocio.

## Requisitos y Ejecución
- Python 3.x
- Paquetes: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
- Estructura esperada:
  - `Trabajo2.ipynb`
  - `data/cs-training.csv`

Pasos:
1. Abrir `Trabajo2.ipynb` en Jupyter.
2. Ejecutar todas las celdas en orden. El notebook verificará la existencia de `data/cs-training.csv`.

## Hallazgos esperables (guía)
- La utilización de crédito y el historial de atrasos aportan señal predictiva fuerte hacia mayor riesgo.
- La regresión logística suele ofrecer buen equilibrio entre rendimiento y estabilidad; KNN necesita estandarización y puede degradarse con ruido; árboles pueden sobreajustar si no se podan.
- La validación cruzada típicamente produce AUC moderada‑alta y varianza moderada por el desbalance.

## Limitaciones
- Desbalance de clases: requiere estratificación y métricas apropiadas; un AUC alto no implica buen desempeño en un umbral específico.
- Costos de error no modelados: para aplicaciones reales se debe optimizar el umbral según costos de falsos positivos/negativos.
- Datos anónimos y acotados: no todas las variables relevantes de negocio están disponibles.

## Próximos pasos sugeridos
- Búsqueda de hiperparámetros (GridSearchCV/RandomizedSearchCV) y `class_weight='balanced'`.
- Curva Precision‑Recall y análisis de umbrales orientado a costos.
- Calibración de probabilidades (Platt/Isotonic) y evaluación de estabilidad temporal si se dispone de datos por períodos.

---
Este repositorio incluye el notebook autoexplicativo para facilitar la comunicación de hallazgos a audiencias no técnicas y técnicas por igual.
