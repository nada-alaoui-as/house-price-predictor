# Utiliser Python 3.11 slim comme image de base
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt en premier (pour le cache Docker)
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers nécessaires dans le conteneur
COPY app.py .
COPY train_model.py .
COPY models/ ./models/
COPY data/ ./data/

# Créer le dossier plots
RUN mkdir -p plots

# Exposer le port 7860 (port par défaut de Gradio)
EXPOSE 7860

# Configurer les variables d'environnement pour Gradio
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Commande pour démarrer l'application
CMD ["python", "app.py"]
