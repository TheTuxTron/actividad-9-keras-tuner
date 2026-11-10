# Actividad 9 – Optimización de una red neuronal con KerasTuner

Notebook: [`Predicting diabetes with a Keras NN.ipynb`](Predicting%20diabetes%20with%20a%20Keras%20NN.ipynb)

## Contenido

1. **Ejemplo de clase:** predicción de diabetes (Pima Indians) con una red Keras, TensorBoard y *early stopping*.
2. **Actividad (individual):** clasificación de tumores con el dataset **Breast Cancer Wisconsin (Diagnostic)**
   (569 muestras, 30 características), comparando una red base (12-8-1) con una red cuyos hiperparámetros
   se optimizaron con **KerasTuner (`BayesianOptimization`, 30 pruebas)**.

## Resultados principales (conjunto de prueba, 86 muestras)

| Métrica | Base 12-8-1 | Optimizada |
|---|---|---|
| Accuracy | 0.9884 | 0.9884 |
| Recall (maligno) | 0.9688 | 0.9688 |
| ROC-AUC | 1.0000 | 1.0000 |
| Pérdida (BCE) | 0.0486 | **0.0372** |

Mejor configuración: 2 capas ocultas ReLU (8 y 120 neuronas), dropout 0.4 en la primera, L2 = 1e-5,
Adam con *learning rate* 0.01 y *batch size* 32. El análisis completo y las respuestas están al final del notebook.

## Archivos

| Archivo | Descripción |
|---|---|
| `diabetes.csv` | Dataset del ejemplo de clase |
| `modelo_diabetes.keras` | Modelo entrenado del ejemplo de diabetes |
| `modelo_cancer_optimizado.keras` | Mejor modelo encontrado con KerasTuner |
| `scaler_cancer.joblib` | `StandardScaler` necesario para predecir con el modelo optimizado |

Las carpetas `logs/` (TensorBoard) y `kt_tuning/` (pruebas de KerasTuner) no se suben; se regeneran al ejecutar el notebook.

## Cómo ejecutarlo

```bash
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
jupyter notebook
```
