from fastapi import FastAPI, HTTPException
from typing import Literal
import math

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
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
# 1. Esquemas de entrada/salida (Pydantic v2) alineados con P2_FEATURES
#    - extra="forbid": rechaza campos desconocidos (errores de tipeo en el cliente)
#    - strict en enteros: "8" o 8.5 no se aceptan como hora_dia
#    - rangos por campo (ge/le) + validaciones cruzadas entre variables
#    Cualquier violación devuelve HTTP 422 con el detalle del campo inválido.
# ---------------------------------------------------------
class ModelInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "hora_dia": 8,
                "dia_semana": 1,
                "fin_de_semana": 0,
                "capacidad": 30.0,
                "pct_ocupacion": 0.15,
                "riesgo_ahora": 0,
                "neto": -2,
                "neto_ult_3h": -5.0,
            }
        },
    )

    hora_dia: StrictInt = Field(..., ge=0, le=23, description="Hora del día (0 a 23)")
    dia_semana: StrictInt = Field(..., ge=0, le=6, description="Día de la semana (0=Lunes a 6=Domingo)")
    fin_de_semana: StrictInt = Field(..., ge=0, le=1, description="Indicador de fin de semana (0 o 1)")
    capacidad: float = Field(..., gt=0, le=200, description="Capacidad total de anclajes de la estación (> 0)")
    pct_ocupacion: float = Field(..., ge=0.0, le=1.0, description="Ocupación estimada de la estación (0.0 a 1.0)")
    riesgo_ahora: StrictInt = Field(..., ge=0, le=1, description="Riesgo actual al cierre de la hora (0 o 1)")
    neto: StrictInt = Field(..., description="Flujo neto de la hora actual (llegadas - salidas)")
    neto_ult_3h: float = Field(..., description="Suma del flujo neto de la hora actual y las dos anteriores")

    @field_validator("capacidad", "pct_ocupacion", "neto_ult_3h")
    @classmethod
    def valor_finito(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("debe ser un número finito (no NaN ni infinito)")
        return v

    @model_validator(mode="after")
    def validaciones_cruzadas(self) -> "ModelInput":
        # fin_de_semana debe ser coherente con dia_semana (5=Sábado, 6=Domingo)
        esperado = 1 if self.dia_semana >= 5 else 0
        if self.fin_de_semana != esperado:
            raise ValueError(
                f"fin_de_semana={self.fin_de_semana} no coincide con dia_semana={self.dia_semana} "
                f"(se esperaba {esperado})"
            )
        # El flujo neto de una hora no puede superar la capacidad física de la estación
        if abs(self.neto) > self.capacidad:
            raise ValueError("|neto| no puede ser mayor que la capacidad de la estación")
        # El acumulado de 3 horas no puede exceder 3 veces la capacidad
        if abs(self.neto_ult_3h) > 3 * self.capacidad:
            raise ValueError("|neto_ult_3h| no puede ser mayor que 3 x capacidad")
        return self


class ModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediccion: Literal[0, 1] = Field(..., description="Clase predicha de riesgo para la siguiente hora (0 o 1)")
    probabilidad: float = Field(..., ge=0.0, le=1.0, description="Probabilidad de la clase predicha (0.0 a 1.0)")
    status: Literal["success"] = Field(default="success")


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
        
        input_data_dict = data.model_dump()
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