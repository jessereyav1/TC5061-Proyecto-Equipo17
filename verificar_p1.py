import mlflow.sklearn
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model_uri = "models:/mibici-equipo17-p1-demanda@champion"

print(f"Cargando el modelo desde: {model_uri}")
loaded_model = mlflow.sklearn.load_model(model_uri)
print("¡Modelo cargado exitosamente en producción!")
print("Tipo de modelo:", type(loaded_model))