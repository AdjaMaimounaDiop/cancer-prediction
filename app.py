# ============================================================
# 1. IMPORTS DES BIBLIOTHÈQUES
# ============================================================
import streamlit as st          # Framework pour créer l'interface web interactive
import numpy as np              # Calculs numériques (tableaux, moyennes, reshape...)
import pandas as pd             # Manipulation de données (lecture CSV, DataFrame)
import tensorflow as tf         # Chargement du modèle de réseau de neurones
import joblib                   # Chargement du scaler sauvegardé (.pkl)
import os                       # Vérification de l'existence des fichiers

# ============================================================
# 2. CONFIGURATION DE LA PAGE STREAMLIT
# ============================================================
# layout="wide" permet d'utiliser toute la largeur de l'écran
st.set_page_config(page_title="Prédiction Tumeur", layout="wide")

# Titre principal et description du modèle affichés en haut de la page
st.title("🔬 Prédiction de Tumeur Mammaire")
st.write("Ce modèle utilise un réseau de neurones (MLP : 30 -> 16 -> 8 -> 1) entraîné sur le dataset Wisconsin.")


# ============================================================
# 3. FONCTION DE CHARGEMENT DU MODÈLE, DU SCALER ET DES DONNÉES
# ============================================================
# @st.cache_resource : Streamlit garde le résultat en mémoire pour ne pas 
# recharger le modèle (lourd) à chaque interaction de l'utilisateur.
@st.cache_resource
def load_model_and_scaler():
    # --- 3.1 Chargement du modèle Keras ---
    if not os.path.exists('modele_cancer.keras'):
        st.error("❌ Fichier 'modele_cancer.keras' introuvable.")
        st.stop()  # Stoppe l'exécution de l'app si le fichier manque
    model = tf.keras.models.load_model('modele_cancer.keras')

    # --- 3.2 Chargement du scaler (StandardScaler sauvegardé) ---
    if not os.path.exists('scaler.pkl'):
        st.error("❌ Fichier 'scaler.pkl' introuvable. Il doit être dans le même dossier.")
        st.stop()
    scaler = joblib.load('scaler.pkl')

    # --- 3.3 Chargement du dataset pour récupérer les moyennes et exemples ---
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    # On nomme les colonnes : ID, Diagnosis (B/M) et 30 features (Feature_1 à Feature_30)
    column_names = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
    df = pd.read_csv(url, names=column_names)
    
    # On extrait uniquement les 30 features (on enlève ID et Diagnosis)
    X_raw = df.drop(columns=['ID', 'Diagnosis']).values
    # On calcule la moyenne de chaque feature (utilisée pour pré-remplir le formulaire)
    moyennes = X_raw.mean(axis=0) 
    
    # --- 3.4 Noms lisibles des 30 features pour l'affichage ---
    feature_names = [
        "Rayon moyen", "Texture moyenne", "Périmètre moyen", "Surface moyenne", "Lissage moyen",
        "Compacité moyenne", "Concavité moyenne", "Points concaves moyens", "Symétrie moyenne", "Dimension fractale moyenne",
        "Erreur standard Rayon", "Erreur standard Texture", "Erreur standard Périmètre", "Erreur standard Surface", "Erreur standard Lissage",
        "Erreur standard Compacité", "Erreur standard Concavité", "Erreur standard Points concaves", "Erreur standard Symétrie", "Erreur standard Dimension fractale",
        "Rayon extrême", "Texture extrême", "Périmètre extrême", "Surface extrême", "Lissage extrême",
        "Compacité extrême", "Concavité extrême", "Points concaves extrêmes", "Symétrie extrême", "Dimension fractale extrême"
    ]
    
    # --- 3.5 Extraction d'un patient bénin et d'un patient malin du dataset ---
    # On cherche le premier index où Diagnosis = 'B' (bénin) et 'M' (malin)
    idx_benin = df[df['Diagnosis'] == 'B'].index[0]
    idx_malin = df[df['Diagnosis'] == 'M'].index[0]
    
    # On récupère les valeurs des features pour ces deux patients (converties en float)
    patient_benin = df.iloc[idx_benin].drop(['ID', 'Diagnosis']).values.astype(float)
    patient_malin = df.iloc[idx_malin].drop(['ID', 'Diagnosis']).values.astype(float)
    
    # On retourne tout ce qu'il faut à l'application
    return model, scaler, moyennes, feature_names, patient_benin, patient_malin

# Exécution de la fonction de chargement et récupération des objets
model, scaler, moyennes, feature_names, patient_benin, patient_malin = load_model_and_scaler()


# ============================================================
# 4. BOUTONS DE TEST RAPIDE
# ============================================================
st.subheader("Test rapide")
st.write("Chargez un profil type pour tester le modèle en 1 clic :")

# On crée 2 colonnes pour placer les boutons côte à côte
col_btn1, col_btn2 = st.columns(2)

# Initialisation : si aucune valeur n'existe encore, on met les moyennes par défaut
if 'valeurs_utilisateur' not in st.session_state:
    st.session_state.valeurs_utilisateur = np.copy(moyennes)

# Bouton 1 : charge les valeurs d'un patient bénin
if col_btn1.button("🟢 Charger un cas Bénin (Sain)"):
    st.session_state.valeurs_utilisateur = np.copy(patient_benin)
    
# Bouton 2 : charge les valeurs d'un patient malin
if col_btn2.button("🔴 Charger un cas Malin (Tumeur)"):
    st.session_state.valeurs_utilisateur = np.copy(patient_malin)

st.write("---")  # Ligne de séparation visuelle


# ============================================================
# 5. FORMULAIRE PRINCIPAL (SAISIE DES VALEURS)
# ============================================================
# On récupère les valeurs actuelles (moyennes ou celles d'un patient chargé)
valeurs_utilisateur = np.copy(st.session_state.valeurs_utilisateur)

st.subheader("Caractéristiques principales")
st.write("Modifiez les 5 valeurs les plus influentes :")

# 2 colonnes pour les 5 inputs principaux
col1, col2 = st.columns(2)
with col1:
    # Les 3 premières features (Rayon, Texture, Périmètre)
    valeurs_utilisateur[0] = st.number_input(feature_names[0], value=float(valeurs_utilisateur[0]))
    valeurs_utilisateur[1] = st.number_input(feature_names[1], value=float(valeurs_utilisateur[1]))
    valeurs_utilisateur[2] = st.number_input(feature_names[2], value=float(valeurs_utilisateur[2]))

with col2:
    # Les 2 features suivantes (Surface, Lissage)
    valeurs_utilisateur[3] = st.number_input(feature_names[3], value=float(valeurs_utilisateur[3]))
    valeurs_utilisateur[4] = st.number_input(feature_names[4], value=float(valeurs_utilisateur[4]))

# Expander : section repliable pour ne pas surcharger l'écran avec les 25 autres features
with st.expander("⚙️ Afficher les 30 caractéristiques (Paramètres avancés)"):
    # 3 colonnes pour répartir les 25 champs restants
    col_a, col_b, col_c = st.columns(3)
    for i in range(5, 30):
        # On alterne entre les 3 colonnes (i % 3) pour un affichage équilibré
        with [col_a, col_b, col_c][i % 3]:
            valeurs_utilisateur[i] = st.number_input(f"{feature_names[i]}", value=float(valeurs_utilisateur[i]))

st.write("---")


# ============================================================
# 6. PRÉDICTION
# ============================================================
# Bouton déclencheur de la prédiction (type="primary" = bouton bleu mis en avant)
if st.button("🔍 Prédire", type="primary"):
    # --- 6.1 Préparation des données ---
    # reshape(1, 30) : transforme le vecteur (30,) en matrice (1, 30) 
    # car le modèle et le scaler attendent un batch de 1 échantillon
    input_data = valeurs_utilisateur.reshape(1, 30)
    
    # --- 6.2 Standardisation ---
    # On applique le même scaler que celui utilisé à l'entraînement
    input_scaled = scaler.transform(input_data)
    
    # --- 6.3 Prédiction ---
    # model.predict renvoie une matrice (1, 1), on récupère la probabilité (0 à 1)
    # verbose=0 supprime les logs de TensorFlow
    prob = model.predict(input_scaled, verbose=0)[0][0]
    
    # --- 6.4 Affichage du résultat ---
    # Si prob > 0.5 : le modèle prédit la classe 1 = Malin
    if prob > 0.5:
        st.error(f"⚠️ **Prédiction : Maligne** (Probabilité : {prob:.2%})")
    # Sinon : classe 0 = Bénin (on affiche 1 - prob = probabilité d'être bénin)
    else:
        st.success(f"✅ **Prédiction : Bénigne** (Probabilité : {1 - prob:.2%})")
