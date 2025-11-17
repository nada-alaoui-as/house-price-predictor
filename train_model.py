import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Fixer la seed pour la reproductibilité
np.random.seed(42)
n_samples = 1000

# 1. GÉNÉRATION DES DONNÉES SYNTHÉTIQUES
print("Génération des données synthétiques...")

# Générer les 7 caractéristiques
square_feet = np.random.uniform(800, 4000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age_years = np.random.uniform(0, 50, n_samples)
lot_size = np.random.uniform(2000, 10000, n_samples)
garage_spaces = np.random.randint(0, 4, n_samples)
neighborhood_score = np.random.uniform(1, 10, n_samples)

# Créer le DataFrame
data = pd.DataFrame({
    'square_feet': square_feet,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age_years': age_years,
    'lot_size': lot_size,
    'garage_spaces': garage_spaces,
    'neighborhood_score': neighborhood_score
})

# 2. GÉNÉRATION DES PRIX (variable cible)
# Formule : prix_base + (surface * 150) + (chambres * 20000) + (salles_de_bain * 15000) 
#           - (âge * 2000) + autres_facteurs + bruit_aléatoire
prix_base = 100000
prix = (prix_base + 
        (square_feet * 150) + 
        (bedrooms * 20000) + 
        (bathrooms * 15000) - 
        (age_years * 2000) + 
        (lot_size * 10) + 
        (garage_spaces * 10000) + 
        (neighborhood_score * 5000) + 
        np.random.normal(0, 50000, n_samples))

data['price'] = prix

print(f"Dataset créé : {data.shape[0]} maisons avec {data.shape[1]-1} caractéristiques")
print("\nAperçu des données :")
print(data.head())
print("\nStatistiques descriptives :")
print(data.describe())

# 3. DIVISION DES DONNÉES (80% train, 20% test)
X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nDonnées d'entraînement : {X_train.shape[0]} échantillons")
print(f"Données de test : {X_test.shape[0]} échantillons")

# 4. NORMALISATION DES FEATURES
print("\nNormalisation des features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. ENTRAÎNEMENT DU MODÈLE RANDOM FOREST
print("\nEntraînement du modèle Random Forest...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1  # Utilise tous les CPU disponibles
)

model.fit(X_train_scaled, y_train)
print("Modèle entraîné avec succès !")

# 6. ÉVALUATION DES PERFORMANCES
print("\n=== ÉVALUATION DU MODÈLE ===")

# Prédictions sur l'ensemble d'entraînement
y_train_pred = model.predict(X_train_scaled)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"\nPerformances sur l'ensemble d'entraînement :")
print(f"  MAE (Mean Absolute Error) : ${train_mae:,.2f}")
print(f"  R² Score : {train_r2:.4f}")

# Prédictions sur l'ensemble de test
y_test_pred = model.predict(X_test_scaled)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nPerformances sur l'ensemble de test :")
print(f"  MAE (Mean Absolute Error) : ${test_mae:,.2f}")
print(f"  R² Score : {test_r2:.4f}")

# 7. SAUVEGARDE DU MODÈLE, SCALER ET FEATURE NAMES
print("\n=== SAUVEGARDE DES FICHIERS ===")

# Sauvegarder le modèle
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Modèle sauvegardé : models/model.pkl")

# Sauvegarder le scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler sauvegardé : models/scaler.pkl")

# Sauvegarder les noms des features
feature_names = X.columns.tolist()
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✓ Feature names sauvegardés : models/feature_names.pkl")

# Sauvegarder aussi les données pour référence (optionnel)
data.to_csv('data/housing_data.csv', index=False)
print("✓ Données sauvegardées : data/housing_data.csv")

print("\n✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")