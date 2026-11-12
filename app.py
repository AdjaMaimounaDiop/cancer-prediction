import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Prédiction Tumeur", layout="wide")
st.title("🔬 Prédiction de Tumeur Mammaire")
st.write("Ce modèle utilise un réseau de neurones (MLP : 30 -> 16 -> 8 -> 1) entraîné sur le dataset Wisconsin.")

@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model('modele_cancer.keras')
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    column_names = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
    df = pd.read_csv(url, names=column_names)
    X = df.drop(columns=['ID', 'Diagnosis']).values
    
    scaler = StandardScaler()
    scaler.fit(X)
    moyennes = X.mean(axis=0)
    
    feature_names = [
        "Rayon moyen", "Texture moyenne", "Périmètre moyen", "Surface moyenne", "Lissage moyen",
        "Compacité moyenne", "Concavité moyenne", "Points concaves moyens", "Symétrie moyenne", "Dimension fractale moyenne",
        "Erreur standard Rayon", "Erreur standard Texture", "Erreur standard Périmètre", "Erreur standard Surface", "Erreur standard Lissage",
        "Erreur standard Compacité", "Erreur standard Concavité", "Erreur standard Points concaves", "Erreur standard Symétrie", "Erreur standard Dimension fractale",
        "Rayon extrême", "Texture extrême", "Périmètre extrême", "Surface extrême", "Lissage extrême",
        "Compacité extrême", "Concavité extrême", "Points concaves extrêmes", "Symétrie extrême", "Dimension fractale extrême"
    ]
    
    # On récupère un patient sain (B) et un patient malade (M) du dataset pour les boutons de test
    idx_benin = df[df['Diagnosis'] == 'B'].index[0]
    idx_malin = df[df['Diagnosis'] == 'M'].index[0]
    
    patient_benin = df.iloc[idx_benin].drop(['ID', 'Diagnosis']).values.astype(float)
    patient_malin = df.iloc[idx_malin].drop(['ID', 'Diagnosis']).values.astype(float)
    
    return model, scaler, moyennes, feature_names, patient_benin, patient_malin

model, scaler, moyennes, feature_names, patient_benin, patient_malin = load_model_and_scaler()

# --- BOUTONS DE TEST RAPIDE ---
st.subheader("Test rapide")
st.write("Chargez un profil type pour tester le modèle en 1 clic :")
col_btn1, col_btn2 = st.columns(2)

# On utilise st.session_state pour garder les valeurs en mémoire quand on clique
if 'valeurs_utilisateur' not in st.session_state:
    st.session_state.valeurs_utilisateur = np.copy(moyennes)

if col_btn1.button("🟢 Charger un cas Bénin (Sain)"):
    st.session_state.valeurs_utilisateur = np.copy(patient_benin)
    
if col_btn2.button("🔴 Charger un cas Malin (Tumeur)"):
    st.session_state.valeurs_utilisateur = np.copy(patient_malin)

st.write("---")

# --- FORMULAIRE PRINCIPAL ---
valeurs_utilisateur = np.copy(st.session_state.valeurs_utilisateur)

st.subheader("Caractéristiques principales")
st.write("Modifiez les 5 valeurs les plus influentes :")

col1, col2 = st.columns(2)
with col1:
    valeurs_utilisateur[0] = st.number_input(feature_names[0], value=float(valeurs_utilisateur[0]))
    valeurs_utilisateur[1] = st.number_input(feature_names[1], value=float(valeurs_utilisateur[1]))
    valeurs_utilisateur[2] = st.number_input(feature_names[2], value=float(valeurs_utilisateur[2]))

with col2:
    valeurs_utilisateur[3] = st.number_input(feature_names[3], value=float(valeurs_utilisateur[3]))
    valeurs_utilisateur[4] = st.number_input(feature_names[4], value=float(valeurs_utilisateur[4]))

with st.expander("⚙️ Afficher les 30 caractéristiques (Paramètres avancés)"):
    col_a, col_b, col_c = st.columns(3)
    for i in range(5, 30):
        with [col_a, col_b, col_c][i % 3]:
            valeurs_utilisateur[i] = st.number_input(f"{feature_names[i]}", value=float(valeurs_utilisateur[i]))

st.write("---")

# --- PRÉDICTION ---
if st.button("🔍 Prédire", type="primary"):
    input_data = valeurs_utilisateur.reshape(1, 30)
    input_scaled = scaler.transform(input_data)
    prob = model.predict(input_scaled, verbose=0)[0][0]
    
    if prob > 0.5:
        st.error(f"⚠️ **Prédiction : Maligne** (Probabilité : {prob:.2%})")
    else:
        st.success(f"✅ **Prédiction : Bénigne** (Probabilité : {1 - prob:.2%})")
