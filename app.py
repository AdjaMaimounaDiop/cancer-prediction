import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Prédiction Tumeur", layout="centered")
st.title("🔬 Prédiction de Tumeur Mammaire")
st.write("Modèle MLP entraîné sur le dataset Wisconsin.")

@st.cache_resource
def load_model_and_scaler():
    # Chargement du modèle .keras
    model = tf.keras.models.load_model('modele_cancer.keras')

    # Recalcul du scaler
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    column_names = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
    df = pd.read_csv(url, names=column_names)
    X = df.drop(columns=['ID', 'Diagnosis']).values

    scaler = StandardScaler()
    scaler.fit(X)
    moyennes = X.mean(axis=0)

    return model, scaler, moyennes

model, scaler, moyennes = load_model_and_scaler()

st.subheader("Entrez les caractéristiques de la tumeur :")

col1, col2 = st.columns(2)
with col1:
    radius = st.number_input("Rayon moyen", value=12.0)
    texture = st.number_input("Texture moyenne", value=15.0)
    perimeter = st.number_input("Périmètre moyen", value=75.0)

with col2:
    area = st.number_input("Surface moyenne", value=450.0)
    smoothness = st.number_input("Lissage moyen", value=0.08)

if st.button("Prédire"):
    input_data = np.copy(moyennes).reshape(1, 30)
    input_data[0, 0] = radius
    input_data[0, 1] = texture
    input_data[0, 2] = perimeter
    input_data[0, 3] = area
    input_data[0, 4] = smoothness

    input_scaled = scaler.transform(input_data)
    prob = model.predict(input_scaled, verbose=0)[0][0]

    st.write("---")
    if prob > 0.5:
        st.error(f"⚠️ **Prédiction : Maligne** (Probabilité : {prob:.2%})")
    else:
        st.success(f"✅ **Prédiction : Bénigne** (Probabilité : {1 - prob:.2%})")
