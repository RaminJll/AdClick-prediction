# Prédiction de Clic sur une Annonce (Ad Click Prediction)

Application web (API Flask) intégrant un modèle de Deep Learning pour prédire la probabilité qu'un utilisateur clique sur une annonce publicitaire.

Le modèle (`model.keras`) et les paramètres de normalisation des données (`scaler_mean.npy`, `scaler_scale.npy`) sont inclus pour assurer 
une prédiction cohérente entre l'entraînement et la production.

---

## Technologies Clés

* **Langage :** Python 3.9+
* **Framework Web :** Flask
* **Modélisation :** Keras / TensorFlow
* **Déploiement :** Docker, Gunicorn

## Structure du Projet

```text
C:.
|   ad_mini.csv
|   app.py
|   Dockerfile
|   model.keras
|   model.py
|   requirements.txt
|   scaler_mean.npy
|   scaler_scale.npy
|   structure.txt
|   
\---templates
        index.html
```
---

## Lancement Local

Pour développer et tester l'application rapidement :

1.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Lancer l'application (en mode développement) :**
    ```bash
    python app.py
    ```

L'application sera accessible sur : [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Lancement via Docker

Utilisez cette méthode pour un environnement de production fiable et reproductible :

1.  **Construire l'image :**
    ```bash
    docker build -t adclick-app .
    ```
2.  **Lancer le conteneur :**
    ```bash
    docker run -it --rm -p 5000:5000 adclick-app
    ```

L'application web sera disponible à l'adresse `http://localhost:5000`.
