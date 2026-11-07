from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import os
import mlflow
import mlflow.sklearn

# Inicializar FastAPI
app = FastAPI(
    title="API MiBici - Clasificación P2 (Producción)",
    description="API REST para servir el modelo de clasificación de riesgo de la Pregunta 2 desde MLflow Model Registry.",
    version="1.0.0"
)

# ---------------------------------------------------------
# 1. Esquema de Pydantic alineado con P2_FEATURES
# ---------------------------------------------------------
class ModelInput(BaseModel):
    hora_dia: int = Field(..., description="Hora del día (0 a 23)", ge=0, le=23)
    dia_semana: int = Field(..., description="Día de la semana (0=Lunes a 6=Domingo)", ge=0, le=6)
    fin_de_semana: int = Field(..., description="Indicador de fin de semana (0 o 1)", ge=0, le=1)
    capacidad: float = Field(..., description="Capacidad total de anclajes de la estación")
    pct_ocupacion: float = Field(..., description="Porcentaje de ocupación estimado de la estación (0.0 a 1.0)")
    riesgo_ahora: int = Field(..., description="Riesgo actual al cierre de la hora (0 o 1)", ge=0, le=1)
    neto: int = Field(..., description="Flujo neto de la hora actual (llegadas - salidas)")
    neto_ult_3h: float = Field(..., description="Suma del flujo neto de la hora actual y las dos anteriores")
    
    class Config:
        json_schema_extra = {
            "example": {
                "hora_dia": 8,
                "dia_semana": 1,
                "fin_de_semana": 0,
                "capacidad": 30.0,
                "pct_ocupacion": 0.15,
                "riesgo_ahora": 0,
                "neto": -2,
                "neto_ult_3h": -5.0
            }
        }

class ModelOutput(BaseModel):
    prediccion: int = Field(..., description="Clase predicha de riesgo para la siguiente hora (0 o 1)")
    probabilidad: float = Field(..., description="Probabilidad asociada a la clase de riesgo predicha")
    status: str = Field(default="success")


# ---------------------------------------------------------
# 2. Carga del Modelo desde MLflow Model Registry (Carpeta local ./mlruns)
# ---------------------------------------------------------
NOMBRE_MODELO = "mibici-equipo17-p2-riesgo"
ALIAS_PRODUCCION = "champion"

model = None

@app.on_event("startup")
def load_model_from_registry():
    global model
    try:
        model_uri = f"models:/{NOMBRE_MODELO}@{ALIAS_PRODUCCION}"
        print(f"Intentando cargar el modelo desde el registro local: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("¡Modelo cargado exitosamente desde MLflow Registry!")
    except Exception as e:
        print(f"[ADVERTENCIA] No se pudo cargar el modelo en el arranque: {e}")
        model = None


# ---------------------------------------------------------
# 3. Endpoints de la API
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "mensaje": "Servicio de API REST activo para el modelo de clasificación P2 (MiBici)",
        "modelo_cargado": model is not None,
        "documentacion": "/docs"
    }

@app.post("/predict", response_model=ModelOutput)
def predict(data: ModelInput):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="El modelo no se encuentra cargado en el servidor."
        )
    
    try:
        features_ordenadas = [
            "hora_dia",
            "dia_semana",
            "fin_de_semana",
            "capacidad",
            "pct_ocupacion",
            "riesgo_ahora",
            "neto",
            "neto_ult_3h"
        ]
        
        input_data_dict = data.dict()
        input_df = pd.DataFrame([input_data_dict])[features_ordenadas]
        
        pred = model.predict(input_df)[0]
        
        prob = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)
            prob = float(probs[0][int(pred)])
            
        return ModelOutput(
            prediccion=int(pred),
            probabilidad=prob,
            status="success"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Error durante la inferencia del modelo: {str(e)}"
        )