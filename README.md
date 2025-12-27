#  Prédiction de Souscription à une Assurance Automobile

### Application Data Science & Deep Learning -- Streamlit

------------------------------------------------------------------------

##  Contexte et motivation

Dans le secteur bancaire et assurantiel, les campagnes de prospection
téléphonique représentent un coût important. Appeler tous les clients
sans distinction conduit à une faible efficacité commerciale et à une
mauvaise allocation des ressources.

Ce projet s'inscrit dans ce contexte et vise à **exploiter les données
historiques de campagnes téléphoniques** afin de : - prédire la
probabilité qu'un client souscrive à une assurance automobile ; - aider
les décideurs à **prioriser les clients à fort potentiel** ; - démontrer
l'apport des **réseaux de neurones profonds** dans un cas d'usage métier
réel.

Le projet a été réalisé dans un cadre **académique (Master Data Science
& Intelligence Artificielle)** et met l'accent sur la **rigueur
méthodologique**, la **qualité du code** et la **déployabilité**.

------------------------------------------------------------------------

## Objectifs du projet

### Objectif général

Développer un système complet d'aide à la décision basé sur le Deep
Learning pour la prédiction de la souscription à une assurance
automobile.

### Objectifs spécifiques

-   Comprendre les facteurs influençant la décision de souscription
-   Réaliser une analyse exploratoire approfondie des données
-   Construire un modèle de classification binaire performant
-   Déployer un dashboard analytique interactif
-   Fournir un outil de prédiction individuel simple et interprétable
-   Garantir la reproductibilité et la robustesse du pipeline ML

------------------------------------------------------------------------

##  Données utilisées

Les données proviennent de campagnes de marketing direct menées par une
banque.

### Types de variables

-   **Sociodémographiques** : âge, profession, statut marital, niveau
    d'éducation
-   **Financières** : solde moyen du client
-   **Historique de contact** :
    -   nombre d'appels
    -   jour et mois de contact
    -   durée des appels
    -   résultat de la campagne précédente
-   **Canal de communication** : téléphone, mobile, etc.
-   **Variable cible** :
    -   `CarInsurance`
        -   1 : souscription\
        -   0 : non-souscription

### Prétraitements effectués

-   Suppression des variables non informatives (ID, horodatages bruts)
-   Création de variables dérivées (durée d'appel)
-   Traitement des valeurs manquantes
-   Encodage One-Hot des variables catégorielles
-   Normalisation des variables numériques

------------------------------------------------------------------------

##  Méthodologie

### 1. Analyse exploratoire (EDA)

-   Analyse de la distribution de la variable cible
-   Étude du taux de souscription par profession, mois et canal
-   Analyse de la saisonnalité
-   Calcul et visualisation des corrélations
-   Extraction d'indicateurs clés (KPIs)

### 2. Modélisation

-   Modèle basé sur un **réseau de neurones profonds (Keras /
    TensorFlow)**
-   Architecture adaptée à la classification binaire
-   Séparation entraînement / validation
-   Optimisation des performances
-   Évaluation via des métriques adaptées (accuracy, rappel, etc.)

### 3. Déploiement

-   Application interactive développée avec **Streamlit**
-   Séparation claire entre :
    -   analyse descriptive (dashboard)
    -   prédiction individuelle
-   Visualisations interactives avec Plotly
-   Gestion rigoureuse du preprocessing pour éviter toute fuite de
    données

------------------------------------------------------------------------

##  Architecture du projet

    📁 Projet_FRN_Ousmane_Faye/
    │
    ├── Dashboard.py                  # Application Streamlit
    ├── Projet_FRN_Ousmane_Faye.ipynb  # Notebook d’analyse et d’entraînement
    ├── modele_assurance_auto.h5      # Modèle Deep Learning entraîné
    ├── preprocessor_ct.pkl           # Préprocesseur (ColumnTransformer)
    ├── scaler.pkl                    # StandardScaler entraîné
    ├── carInsurance_2024 (3).csv     # Jeu de données
    └── README.md                     # Documentation du projet

------------------------------------------------------------------------

##  Fonctionnalités de l'application

###  Dashboard analytique

-   Indicateurs clés de performance (KPIs)
-   Taux de souscription global et filtré
-   Analyses par :
    -   profession
    -   mois
    -   canal de communication
    -   résultat de campagne précédente
-   Graphiques interactifs
-   Matrice de corrélation

###  Module de prédiction

-   Formulaire interactif pour un nouveau client
-   Calcul automatique de la probabilité de souscription
-   Recommandation métier :
    -   prioriser ou non l'appel
-   Jauge visuelle de probabilité

------------------------------------------------------------------------

##  Technologies utilisées

### Langages et bibliothèques

-   **Python**
-   **Streamlit**
-   **TensorFlow / Keras**
-   **Scikit-learn**
-   **Pandas / NumPy**

### Visualisation

-   **Plotly Express**
-   **Plotly Graph Objects**

------------------------------------------------------------------------

##  Installation et exécution

### Prérequis

-   Python ≥ 3.9
-   pip

### Installation des dépendances

``` bash
pip install streamlit pandas numpy scikit-learn tensorflow plotly joblib
```

### Lancement de l'application

``` bash
streamlit run Dashboard.py
```

L'application est accessible par défaut à l'adresse :

    http://localhost:8501

------------------------------------------------------------------------

##  Résultats et apports

-   Amélioration de l'efficacité des campagnes marketing
-   Réduction des coûts liés aux appels inutiles
-   Illustration concrète de l'apport du Deep Learning
-   Projet complet couvrant tout le cycle de vie d'un modèle ML

------------------------------------------------------------------------

##  Limites et perspectives

### Limites

-   Données issues d'un contexte spécifique
-   Modèle statique (pas de réentraînement automatique)

### Perspectives

-   Intégration d'un pipeline sklearn complet
-   Ajout de méthodes d'explicabilité (SHAP, LIME)
-   Déploiement cloud (Docker, AWS, GCP)
-   Mise à jour dynamique du modèle

------------------------------------------------------------------------

##  Auteur

**Ousmane Faye**\
Master Data Science & Intelligence Artificielle\
Projet académique -- Réseaux de Neurones Profonds\
Décembre 2025

— Ousmane Faye —

  Un outil au service de l'intelligence commerciale
------------------------------------------------------------------------

##  Licence

Projet réalisé dans un cadre académique et pédagogique.\
Toute utilisation commerciale nécessite une autorisation préalable.
