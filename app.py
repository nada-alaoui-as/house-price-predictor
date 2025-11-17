import gradio as gr
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration du style pour les graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Créer le dossier plots s'il n'existe pas
Path("plots").mkdir(exist_ok=True)

# 1. CHARGEMENT DU MODÈLE, SCALER ET FEATURE NAMES
print("Chargement du modèle et des composants...")

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("✓ Modèle chargé avec succès !")
print(f"✓ Features : {feature_names}")

# 2. FONCTION DE VISUALISATION 1 : IMPORTANCE DES FEATURES
def plot_feature_importance():
    """Affiche l'importance des features du modèle Random Forest"""
    
    feature_importance = model.feature_importances_
    
    # Créer le DataFrame pour le tri
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title('Importance des Caractéristiques dans la Prédiction du Prix', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarder le graphique
    filepath = 'plots/feature_importance.png'
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filepath

# 3. FONCTION DE VISUALISATION 2 : RÉSUMÉ DES INPUTS
def plot_input_summary(square_feet, bedrooms, bathrooms, age_years, 
                        lot_size, garage_spaces, neighborhood_score):
    """Affiche un résumé visuel des caractéristiques saisies"""
    
    # Préparer les données
    features = {
        'Surface (sq ft)': square_feet,
        'Chambres': bedrooms,
        'Salles de bain': bathrooms,
        'Âge (années)': age_years,
        'Terrain (sq ft)': lot_size,
        'Garage': garage_spaces,
        'Score quartier': neighborhood_score
    }
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Barplot horizontal
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    bars = ax.barh(list(features.keys()), list(features.values()), color=colors)
    
    # Ajouter les valeurs sur les barres
    for i, (bar, value) in enumerate(zip(bars, features.values())):
        ax.text(value + max(features.values()) * 0.02, i, f'{value:,.0f}', 
                va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Valeur', fontsize=12, fontweight='bold')
    ax.set_title('Résumé des Caractéristiques de la Maison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarder
    filepath = 'plots/input_summary.png'
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filepath

# 4. FONCTION DE VISUALISATION 3 : PRÉDICTION AVEC INTERVALLE DE CONFIANCE
def plot_prediction_confidence(predicted_price, lower_bound, upper_bound):
    """Affiche la prédiction avec son intervalle de confiance"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Créer le graphique
    y_pos = 0
    
    # Intervalle de confiance (barre horizontale)
    ax.barh(y_pos, upper_bound - lower_bound, left=lower_bound, 
            height=0.3, color='lightblue', alpha=0.6, label='Intervalle de confiance 95%')
    
    # Prédiction centrale (ligne verticale)
    ax.axvline(predicted_price, color='darkblue', linewidth=3, 
               label=f'Prédiction : ${predicted_price:,.0f}')
    
    # Bornes de l'intervalle
    ax.axvline(lower_bound, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(upper_bound, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Annotations
    ax.text(predicted_price, y_pos + 0.2, f'${predicted_price:,.0f}', 
            ha='center', fontsize=14, fontweight='bold', color='darkblue')
    ax.text(lower_bound, y_pos - 0.2, f'${lower_bound:,.0f}', 
            ha='center', fontsize=10, color='gray')
    ax.text(upper_bound, y_pos - 0.2, f'${upper_bound:,.0f}', 
            ha='center', fontsize=10, color='gray')
    
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Prix ($)', fontsize=12, fontweight='bold')
    ax.set_title('Prédiction du Prix avec Intervalle de Confiance (95%)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarder
    filepath = 'plots/prediction_confidence.png'
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filepath

# 5. FONCTION PRINCIPALE DE PRÉDICTION
def predict_price(square_feet, bedrooms, bathrooms, age_years, 
                  lot_size, garage_spaces, neighborhood_score):
    """
    Prédit le prix d'une maison et génère les visualisations
    """
    
    # Préparer les données d'entrée
    input_data = pd.DataFrame({
        'square_feet': [square_feet],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'age_years': [age_years],
        'lot_size': [lot_size],
        'garage_spaces': [garage_spaces],
        'neighborhood_score': [neighborhood_score]
    })
    
    # Normaliser les données
    input_scaled = scaler.transform(input_data)
    
    # Faire la prédiction
    predicted_price = model.predict(input_scaled)[0]
    
    # Calculer l'intervalle de confiance à 95%
    # On utilise les prédictions de tous les arbres du Random Forest
    tree_predictions = np.array([tree.predict(input_scaled)[0] 
                                  for tree in model.estimators_])
    std_dev = np.std(tree_predictions)
    
    # Intervalle de confiance : + ou - 1.96 * écart-type (pour 95%)
    confidence_interval = 1.96 * std_dev
    lower_bound = predicted_price - confidence_interval
    upper_bound = predicted_price + confidence_interval
    
    # Générer les visualisations
    plot1 = plot_feature_importance()
    plot2 = plot_input_summary(square_feet, bedrooms, bathrooms, age_years,
                               lot_size, garage_spaces, neighborhood_score)
    plot3 = plot_prediction_confidence(predicted_price, lower_bound, upper_bound)
    
    # Préparer le texte de résultat
    result_text = f"""
    ## Résultat de la Prédiction
    
    **Prix estimé : ${predicted_price:,.2f}**
    
    ### Intervalle de confiance (95%) :
    - **Borne inférieure :** ${lower_bound:,.2f}
    - **Borne supérieure :** ${upper_bound:,.2f}
    
    ### Caractéristiques de la maison :
    - Surface : {square_feet:,.0f} sq ft
    - Chambres : {bedrooms}
    - Salles de bain : {bathrooms}
    - Âge : {age_years:.0f} ans
    - Terrain : {lot_size:,.0f} sq ft
    - Garage : {garage_spaces} place(s)
    - Score du quartier : {neighborhood_score:.1f}/10
    """
    
    return result_text, plot1, plot2, plot3

# 6. CRÉATION DE L'INTERFACE GRADIO
with gr.Blocks(title="Prédiction de Prix Immobiliers", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # Prédicteur de Prix Immobiliers
        ### Estimez le prix d'une maison en fonction de ses caractéristiques
        
        *Modèle : Random Forest Regressor avec intervalle de confiance à 95%*
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Caractéristiques de la maison")
            
            square_feet = gr.Slider(
                minimum=800, maximum=4000, value=2000, step=50,
                label="Surface habitable (sq ft)"
            )
            
            bedrooms = gr.Slider(
                minimum=1, maximum=5, value=3, step=1,
                label="Nombre de chambres"
            )
            
            bathrooms = gr.Slider(
                minimum=1, maximum=3, value=2, step=1,
                label="Nombre de salles de bain"
            )
            
            age_years = gr.Slider(
                minimum=0, maximum=50, value=10, step=1,
                label="Âge de la maison (années)"
            )
            
            lot_size = gr.Slider(
                minimum=2000, maximum=10000, value=5000, step=100,
                label="Taille du terrain (sq ft)"
            )
            
            garage_spaces = gr.Slider(
                minimum=0, maximum=3, value=1, step=1,
                label="Places de garage"
            )
            
            neighborhood_score = gr.Slider(
                minimum=1, maximum=10, value=5, step=0.1,
                label="Score du quartier (1-10)"
            )
            
            predict_btn = gr.Button("Prédire le Prix", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("## Résultats")
            
            result_text = gr.Markdown()
            
            with gr.Row():
                plot1 = gr.Image(label="Importance des Caractéristiques", type="filepath")
                plot2 = gr.Image(label="Résumé des Inputs", type="filepath")
            
            plot3 = gr.Image(label="Prédiction avec Intervalle de Confiance", type="filepath")
    
    # Connecter le bouton à la fonction de prédiction
    predict_btn.click(
        fn=predict_price,
        inputs=[square_feet, bedrooms, bathrooms, age_years, 
                lot_size, garage_spaces, neighborhood_score],
        outputs=[result_text, plot1, plot2, plot3]
    )
    
    gr.Markdown(
        """
        ---
        ###  À propos de ce modèle
        
        Ce modèle utilise un **Random Forest Regressor** entraîné sur 1000 maisons synthétiques.
        L'intervalle de confiance est calculé à partir de l'écart-type des prédictions de tous les arbres.
        
        **Développé dans le cadre du TP2 - Virtualisation & Cloud Computing**
        """
    )

# 7. LANCEMENT DE L'APPLICATION
if __name__ == "__main__":
    print("\nLancement de l'application Gradio...")
    print("L'application sera accessible sur : http://127.0.0.1:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)