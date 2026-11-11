import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Prédiction Tumeur", layout="wide")
st.title("🔬 Prédiction de Tumeur Mammaire")
st.write("Ce modèle utilise un réseau de neurones (MLP) entraîné sur le dataset Wisconsin (30 caractéristiques).")

@st.cache_resource
def load_model_and_scaler():
    # Chargement du modèle .keras
    model = tf.keras.models.load_model('modele_cancer.keras')
    
    # Chargement des données pour calculer le scaler et les moyennes
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    column_names = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
    df = pd.read_csv(url, names=column_names)
    X = df.drop(columns=['ID', 'Diagnosis']).values
    
    scaler = StandardScaler()
    scaler.fit(X)
    moyennes = X.mean(axis=0)
    
    # On renvoie aussi les noms des colonnes pour l'affichage
    feature_names = [f'Feature_{i}' for i in range(1, 31)]
    
    return model, scaler, moyennes, feature_names

model, scaler, moyennes, feature_names = load_model_and_scaler()

# On crée une copie des moyennes pour stocker les valeurs choisies par l'utilisateur
valeurs_utilisateur = np.copy(moyennes)

st.subheader("Caractéristiques principales")
st.write("Modifiez les 5 valeurs les plus influentes :")

col1, col2 = st.columns(2)
with col1:
    valeurs_utilisateur[0] = st.number_input("Rayon moyen", value=float(moyennes[0]))
    valeurs_utilisateur[1] = st.number_input("Texture moyenne", value=float(moyennes[1]))
    valeurs_utilisateur[2] = st.number_input("Périmètre moyen", value=float(moyennes[2]))

with col2:
    valeurs_utilisateur[3] = st.number_input("Surface moyenne", value=float(moyennes[3]))
    valeurs_utilisateur[4] = st.number_input("Lissage moyen", value=float(moyennes[4]))

# --- LA PARTIE AJOUTÉE : Le menu déroulant pour les 25 autres features ---
with st.expander("⚙️ Afficher les 30 caractéristiques (Paramètres avancés)"):
    st.write("Les valeurs non modifiées restent à leur moyenne par défaut.")
    
    # On crée 3 colonnes pour que ce soit plus compact
    col_a, col_b, col_c = st.columns(3)
    
    for i in range(5, 30):
        # On distribue les champs dans les 3 colonnes
        with [col_a, col_b, col_c][i % 3]:
            valeurs_utilisateur[i] = st.number_input(
                f"{feature_names[i]}", 
                value=float(moyennes[i])
            )

st.write("---")

if st.button("🔍 Prédire", type="primary"):
    # On prépare les données sous forme (1, 30)
    input_data = valeurs_utilisateur.reshape(1, 30)
    
    # Standardisation
    input_scaled = scaler.transform(input_data)
    
    # Prédiction
    prob = model.predict(input_scaled, verbose=0)[0][0]
    
    if prob > 0.5:
        st.error(f"⚠️ **Prédiction : Maligne** (Probabilité : {prob:.2%})")
    else:
        st.success(f"✅ **Prédiction : Bénigne** (Probabilité : {1 - prob:.2%})")
