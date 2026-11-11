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
    
    # Noms des caractéristiques traduits en français
    feature_names = [
        # 1. Moyennes (Mean)
        "Rayon moyen", "Texture moyenne", "Périmètre moyen", "Surface moyenne", "Lissage moyen",
        "Compacité moyenne", "Concavité moyenne", "Points concaves moyens", "Symétrie moyenne", "Dimension fractale moyenne",
        # 2. Erreurs standard (SE)
        "Erreur standard Rayon", "Erreur standard Texture", "Erreur standard Périmètre", "Erreur standard Surface", "Erreur standard Lissage",
        "Erreur standard Compacité", "Erreur standard Concavité", "Erreur standard Points concaves", "Erreur standard Symétrie", "Erreur standard Dimension fractale",
        # 3. Valeurs extrêmes (Worst)
        "Rayon extrême", "Texture extrême", "Périmètre extrême", "Surface extrême", "Lissage extrême",
        "Compacité extrême", "Concavité extrême", "Points concaves extrêmes", "Symétrie extrême", "Dimension fractale extrême"
    ]
    
    return model, scaler, moyennes, feature_names

model, scaler, moyennes, feature_names = load_model_and_scaler()

# On crée une copie des moyennes pour stocker les valeurs choisies par l'utilisateur
valeurs_utilisateur = np.copy(moyennes)

st.subheader("Caractéristiques principales")
st.write("Modifiez les 5 valeurs les plus influentes :")

col1, col2 = st.columns(2)
with col1:
    valeurs_utilisateur[0] = st.number_input(feature_names[0], value=float(moyennes[0]))
    valeurs_utilisateur[1] = st.number_input(feature_names[1], value=float(moyennes[1]))
    valeurs_utilisateur[2] = st.number_input(feature_names[2], value=float(moyennes[2]))

with col2:
    valeurs_utilisateur[3] = st.number_input(feature_names[3], value=float(moyennes[3]))
    valeurs_utilisateur[4] = st.number_input(feature_names[4], value=float(moyennes[4]))

# --- Menu déroulant pour les 25 autres features ---
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
