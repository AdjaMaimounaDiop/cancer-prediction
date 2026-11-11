# 🔬 Prédiction de Tumeur Mammaire

Cette application web permet de prédire si une tumeur mammaire est **bénigne** ou **maligne** à l'aide d'un réseau de neurones.

## 🧠 Le Modèle
* **Dataset** : Breast Cancer Wisconsin (UCI Machine Learning Repository).
* **Architecture** : Réseau de neurones Multi-Layer Perceptron (MLP) avec une architecture `30 -> 16 -> 8 -> 1`.
* **Framework** : TensorFlow / Keras.
* **Prétraitement** : Standardisation des données (StandardScaler) pour normaliser les échelles.

## 🚀 Démo en ligne
L'application est déployée sur Render. Vous pouvez la tester ici :
👉 **https://cancer-prediction-2-1z8a.onrender.com/** *(Remplace ce lien par ton vrai lien Render)*

## ⚙️ Fonctionnalités
* Saisie des 5 caractéristiques principales (Rayon, Texture, Périmètre, Surface, Lissage).
* Menu déroulant pour ajuster les 25 autres caractéristiques.
* Boutons de **test rapide** (charger un patient sain ou malade en 1 clic).
* Graphique explicatif montrant les 10 features les plus importantes pour le réseau de neurones (XAI).

## 🛠️ Installation locale (Optionnel)
Si vous souhaitez exécuter ce projet sur votre machine :
1. Cloner le dépôt : `git clone https://github.com/AdjaMaimounaDiop/cancer-prediction.git`
2. Installer les dépendances : `pip install -r requirements.txt`
3. Lancer l'application : `streamlit run app.py`
