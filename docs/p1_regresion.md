### 1. ¿Qué se predice o se agrupa, con qué granularidad y horizonte?
* **Qué se predice:** Se predice la **demanda futura de viajes iniciados** (variable objetivo: conteo de viajes / salidas por unidad de tiempo).
* **Granularidad espacial:** 
  * **Nivel Estación o Zona:** Predicción individual para cada `Origen_Id` (estación).
* **Granularidad temporal:** **Horaria**. La granularidad horaria es clave para el rebalanceo reactivo continuo.
* **Horizonte de predicción:** Corto plazo, **las próximas 24 horas**.

---

### 2. ¿Por qué es relevante para la operación de MiBici?
El principal desafío operativo de los sistemas de bicicletas compartidas es la **asimetría de la demanda** (las bicicletas tienden a acumularse en centros de atracción de viajes durante ciertas horas, dejando vacías las zonas residenciales de origen).

La relevancia de este modelo radica en:
* **Evitar estaciones vacías (desabasto):** Garantizar que un usuario que requiera iniciar un trayecto siempre encuentre unidades disponibles.
* **Evitar estaciones llenas (saturación):** Prevenir que un usuario llegue a su destino y no encuentre puertos libres (`dpcapacity`) para anclar y finalizar su viaje.
* **Optimización de recursos de rebalanceo:** Permitir planificar con anticipación las rutas de los vehículos encargados de mover físicamente las bicicletas entre estaciones antes de que ocurran eventos de desabasto.

---

### 3. ¿Quién usaría el resultado y qué decisión tomaría con él?
* **Usuario Principal: El Equipo de Operaciones y Logística de MiBici.**
  * **Decisión que toma:** Planificar y optimizar de forma proactiva las rutas de rebalanceo de unidades (camionetas de traslado). El modelo les indicará de qué estaciones con excedente retirar bicicletas y hacia cuáles estaciones críticas trasladarlas antes de las horas pico de demanda.
* **Usuario Secundario: El Planificador Urbano / Dirección de Expansión.**
  * **Decisión que toma:** Utilizar las predicciones a mediano/largo plazo y el análisis de saturación constante para decidir en qué zonas ampliar la infraestructura física (añadir más anclajes en estaciones existentes o abrir nuevas estaciones en puntos de alta demanda recurrente).
* **Usuario Terciario: Desarrolladores de la App del Usuario (mediante API de datos abiertos).**
  * **Decisión que toma:** Informar de forma preventiva al usuario final en la aplicación móvil mediante alertas (*e.g., "Esta estación suele quedarse sin bicicletas en los próximos 30 minutos, te sugerimos la estación alterna X"*) o incentivar dinámicas para que los mismos usuarios rebalanceen el sistema a cambio de beneficios.