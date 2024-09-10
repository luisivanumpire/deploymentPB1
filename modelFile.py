import streamlit as st # type: ignore
import pandas as pd # type: ignore
import joblib
import numpy as np
import xgboost as xgb

# Inyectar CSS personalizado para los estilos
BACKGROUND_COLOR = '#0D1B2A'
LETRAS = '#A9D6E5'
COLOR_BUTTOM = '#415A77'
COLOR_LOADFILE = '#1B263B'
        


# Título de la aplicación
st.title("PREDICTBUILD")
st.header("Predicción de Presupuestos con IA")

# Instrucciones para el usuario
st.write("Subir un archivo CSV con las características de las viviendas para predecir el presupuesto.")

# Cargar el archivo CSV
uploaded_file = st.file_uploader("Subir un archivo CSV", type=["csv"])

# Botón para ejecutar el modelo
if st.button("Predecir"):
    if uploaded_file is not None:
        # Cargar los datos del archivo CSV
        data = pd.read_csv(uploaded_file, sep=',')
        #data = pd.read_excel(uploaded_file)
        
        # Cargar el modelo y los parámetros de normalización guardados
        scaler = joblib.load('scaler.pkl')
        model = joblib.load('xgboost_model.pkl')
        
        # Mostrar las primeras filas del DataFrame cargado
        st.write("Datos cargados:")
        st.write(data.head())
        
        # Verificar que el número de columnas coincida con las características originales
        if data.shape[1] != scaler.n_features_in_:
            st.error(f"El archivo CSV cargado tiene {data.shape[1]} características, pero se esperaban {scaler.n_features_in_} características.")
        else:
            # Normalizar los datos
            data_scaled = scaler.transform(data.values)
            
            # Realizar predicciones con el modelo XGBoost
            predicciones = model.predict(data_scaled)
            
            # Mostrar las predicciones
            st.write("Predicciones del presupuesto estimado:")
            st.write(predicciones)
    else:
        st.error("Por favor, carga un archivo CSV.")
