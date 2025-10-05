import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# Datos de la corredora
datos = {
    "Superficie_m2": [50, 70, 65, 90, 45],
    "Num_Habitaciones": [1, 2, 2, 3, 1],
    "Distancia_Metro_km": [0.5, 1.2, 0.8, 0.2, 2.0],
    "Precio_UF": [2500, 3800, 3500, 5200, 2100]
}

df_viviendas = pd.DataFrame(datos)


variable_x = df_viviendas[["Superficie_m2", "Num_Habitaciones", "Distancia_Metro_km"]]
variable_y = df_viviendas["Precio_UF"]

# Crear modelo de regresión lineal con sklearn
modelo_regresion = LinearRegression()
modelo_regresion.fit(variable_x, variable_y)

# Predicciones con el modelo de arriba
y_predicha = modelo_regresion.predict(variable_x)

# Calcular valores de la prediccion
mae = mean_absolute_error(variable_y, y_predicha)
r2 = r2_score(variable_y, y_predicha)


print("R²:", round(r2, 2))
print("MAE:", round(mae, 2))

# Comparar valores reales vs predichos
print(pd.DataFrame({"Valores originales": variable_y, "Valores predichos": y_predicha}))

