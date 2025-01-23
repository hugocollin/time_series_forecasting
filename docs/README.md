# README : Time Series Forecasting

## Table des matières
- [Description](#description)
- [Fonctionnalités principales](#fonctionnalités-principales)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Contribution](#contribution)
- [Auteurs](#auteurs)

## Description

Ce projet vise à prédire des données de consommation électrique en utilisant différentes approches de modélisation (lissage exponentiel, ARIMA, réseaux de neurones...).

### Fonctionnalités principales

1. **Chargement et préparation des données**  
   - Conversion de la variable temps en POSIXct.  
   - Découpage en séries d’entraînement, de validation et de test.  
   - Gestion d’une fréquence de 96 observations par jour (pas de 15 minutes).

2. **Analyse exploratoire**  
   - Visualisation des séries et identification de tendances et de saisons (season plot).  
   - Décomposition additive montrant des pics de consommation matin/soir et des creux la nuit.  
   - Étude des autocorrélations (ACF, PACF) pour caractériser la dépendance temporelle.

3. **Modélisation**  
   - Modèles de lissage exponentiel (non adaptés à ces données très saisonnières).  
   - Modèles ARIMA (avec différenciation pour rendre la série stationnaire).  
   - Modèles de réseaux de neurones (NNAR), Random Forest, Gradient Boosting, SVM.  
   - Choix des hyperparamètres (validation croisée) et comparaison des performances (RMSE, MAPE).

4. **Conclusion**  
   - Les meilleurs résultats sont obtenus avec NNAR (prédiction univariée) et Gradient Boosting (prédiction intégrant la température).  
   - Les approches classiques (ARIMA) sont moins performantes que les réseaux de neurones. 

### Structure du projet

```bash
├── data
│   ├── COLLIN_Hugo.xlsx
│   └── Elec-train.xlsx
├── docs
│   ├── COLLIN_Hugo.pdf
│   ├── Projet.pdf
│   └── README.md
└── main.r
```

## Installation

Pour installer ce projet, clonez le dépôt sur votre machine locale, en utilisant la commande suivante :

```bash
git clone https://github.com/hugocollin/time_series_forecasting
```

## Utilisation

Pour utiliser ce projet :

1. Ouvrez le fichier `main.r` avec RStudio.

2. Exécutez le code.

*Assurez-vous de spécifier le chemin du dossier racine lors du chargement des données, ainsi que le chemin du dossier dexportation lors de la génération du fichier Excel contenant les prédictions.*


## Contribution

Toutes les contributions sont les bienvenues ! Voici comment vous pouvez contribuer :

1. Forkez le projet.
2. Créez votre branche de fonctionnalité  (`git checkout -b feature/AmazingFeature`).
3. Commitez vos changements (`git commit -m 'Add some AmazingFeature'`).
4. Pushez sur la branche (`git push origin feature/AmazingFeature`).
5. Ouvrez une Pull Request. 

## Auteurs

Ce projet a été développée par [COLLIN Hugo](https://github.com/hugocollin), dans le cadre du Master 2 SISE.