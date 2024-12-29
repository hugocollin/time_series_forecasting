# Liste des packages requis
packages <- c("readxl", "writexl", "forecast", "tseries", "dplyr", "randomForest", "xgboost", "e1071", "caret")

# Installer les packages manquants
installed_packages <- packages %in% rownames(installed.packages())
if(any(!installed_packages)){
  install.packages(packages[!installed_packages])
}

# Chargement des librairies nécessaires
library(readxl)
library(writexl)
library(forecast)
library(tseries)
library(dplyr)
library(randomForest)
library(xgboost)
library(e1071)
library(caret)

# Création de la fonction pour le calcul des métriques d'évaluation
calculate_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mape <- mean(abs((actual - predicted) / actual)) * 100
  return(list(RMSE = rmse, MAPE = mape))
}

# -----/ I. Mise en forme des données et visualisation /-----

# Chargement des données
setwd ("chemin vers le dossier racine du projet") # À changer selon le chemin de votre machine
data_path <- file.path("data", "Elec-train.xlsx")
data <- read_excel(data_path)

# Changement du format de la colonne temps
data$Timestamp <- as.POSIXct(data$Timestamp, format = "%m/%d/%Y %H:%M")

# Renommage des colonnes
data <- data %>%
  rename(Power = `Power (kW)`,
         Temp = `Temp (C°)`)

# Affichage du jeu de données
head(data)

# Séparation des données en jeu d'entraînement, validation et test
train_data <- subset(data, Timestamp < as.POSIXct("2010-02-16 00:00:00"))
validate_data <- subset(data, Timestamp >= as.POSIXct("2010-02-16 00:00:00") & Timestamp < as.POSIXct("2010-02-17 00:00:00"))
test_data  <- subset(data, Timestamp >= as.POSIXct("2010-02-17 00:00:00"))

# Transformation des données en série temporelle multivariée (96 observations par jour)
train_data_ts <- ts(train_data[, c("Power", "Temp")],
                    frequency = 96)

validate_data_ts <- ts(validate_data[, c("Power", "Temp")],
                       frequency = 96)

test_data_ts <- ts(test_data[, c("Power", "Temp")],
                   frequency = 96)

# Affichage du graphique de la série temporelle de train
plot(train_data_ts, main = "Consommation électrique et température dans le temps sur les données de train", xlab = "Temps", ylab = "",  lty = c(1, 2), lwd = c(1, 1))

# Affichage du graphique de la série temporelle de validation
plot(validate_data_ts, main = "Consommation électrique et température dans le temps sur les données de validation", xlab = "Temps", ylab = "",  lty = c(1, 2), lwd = c(1, 1))

# Préparation des données pour les modèles à réseaux de neurones
nn_train_data <- train_data %>%
  mutate(Hour = as.numeric(format(Timestamp, "%H")),
         Day = as.numeric(format(Timestamp, "%j")))

nn_validate_data <- validate_data %>%
  mutate(Hour = as.numeric(format(Timestamp, "%H")),
         Day = as.numeric(format(Timestamp, "%j")))

nn_test_data <- test_data %>%
  mutate(Hour = as.numeric(format(Timestamp, "%H")),
         Day = as.numeric(format(Timestamp, "%j")))

# Affichage des statistiques des données
summary(nn_train_data)
summary(nn_validate_data)
summary(nn_test_data)

# -----/ II. Analyse exploratoire des données /-----

# Affichage d'un season plot manuel pour la consommation électrique
power_vec <- train_data_ts[, 1]
n <- length(power_vec)

# Calcul du nombre de jours complets
nb_days <- floor(n / 96)
n_complet <- nb_days * 96 

# Tronquage du vecteur à nb_days * 96
power_vec_mod <- power_vec[1:n_complet]

# Mise en forme du jeu de données en matrice : nb_days lignes, 96 colonnes
power_mat <- matrix(power_vec_mod, nrow = nb_days, ncol = 96, byrow = TRUE)

# Affichage sans l'axe X automatique
matplot(t(power_mat), type = "l", lty = 1, col = 1:nb_days,
        xaxt = "n",
        xlab = "Heure de la journée", ylab = "Consommation (kW)",
        main = "Season plot de la consommation journalière")

# Positions et labels pour l'axe X
x_ticks <- seq(1, 96, by = 4)  # 24 positions
x_labels <- seq(0, 23, by = 1) # 24 labels, de 0 à 23

# Affichage de l'axe X personnalisé
axis(1, at = x_ticks, labels = x_labels)

# Décomposition additive de la consommation électrique
autoplot(decompose(train_data_ts[, 1], type = "additive"), main = "Décomposition additive - Consommation électrique")

# Décomposition additive de la température
autoplot(decompose(train_data_ts[, 2], type = "additive"), main = "Décomposition additive - Température")

# Réalisation d'une ACF
acf(train_data_ts[, 1], lag.max = 5000, main = "ACF - Consommation électrique")

# Réalisation d'une PACF
pacf(train_data_ts[, 1], lag.max = 5000, main = "PACF - Consommation électrique")

# -----/ III. Utilisation des modèles de prédictions /-----
# -----/ 1. Modèles de lissage exponentiel /-----

# Modèle du Lissage Exponentiel Simple : Simple Exponential Smoothing Method
ses_model <- ses(train_data_ts[, 1], h = 96)

# Affichage des performances
summary(ses_model)

# Enregistrement des prédictions
ses_predictions <- ses_model$mean

# Calcul des métriques pour le modèle SES
ses_metrics <- calculate_metrics(validate_data$Power, ses_predictions)

# Affiche des prédictions
plot(ses_model, main = "Prédictions du modèle de Lissage Exponentiel Simple", xlab = "Temps", ylab = "Consommation (kW)", xlim = c(46, 48))

# Modèle du Lissage Exponentiel Double : Holt Linear Trend Method
# Ajustement du modèle Holt
holt_model <- holt(train_data_ts[, 1], h = 96)

# Affichage du résumé du modèle
summary(holt_model)

# Enregistrement des prédictions
holt_predictions <- holt_model$mean

# Calcul des métriques
holt_metrics <- calculate_metrics(validate_data$Power, holt_predictions)

# Affichage des prédictions
plot(holt_model, main = "Prédictions du modèle de Lissage Exponentiel Double", xlab = "Temps", ylab = "Consommation (kW)", xlim = c(46, 48))

# Affichage des prédictions sur le jeu de validation
plot(validate_data$Timestamp, validate_data$Power, type = "l", col = "black", lwd = 2,
     xlab = "Temps", ylab = "Consommation (kW)", main = "Prédictions avec les modèles de lissage exponentiel")
lines(validate_data$Timestamp, ses_predictions, col = "blue", lwd = 2, lty = 2)
lines(validate_data$Timestamp, holt_predictions, col = "red", lwd = 2, lty = 2)
legend("topleft", legend = c("Données réelles", "Données prédites par le modèle SES", "Données prédites par le modèle Holt"),
       col = c("black", "blue", "red"), lty = c(1, 2, 2), lwd = 2)

# -----/ 2. Modèles ARIMA /-----
# Application d'une différenciation pour enlever la tendance et la saisonnalité
train_data_power_diff <- diff(train_data_ts[, 1], lag = 96, differences = 1)
train_data_temp_diff <- diff(train_data_ts[, 2], lag = 96, differences = 1)

# Décomposition additive de la série temporelle différenciée pour la consommation électrique
autoplot(decompose(train_data_power_diff, type = "additive"), main = "Décomposition additive sur la série différenciée - Consommation électrique")

# Réalisation d'une ACF
acf(train_data_power_diff, lag.max = 5000, main = "ACF sur la série différenciée - Consommation électrique")

# Réalisation d'une PACF
pacf(train_data_power_diff, lag.max = 5000, main = "PACF sur la série différenciée - Consommation électrique")

# Test de Ljung-Box
Box.test(train_data_power_diff, lag = 10, type = "Ljung-Box")

# Test Augmented Dickey-Fuller
adf.test(train_data_power_diff, alternative = "stationary")

# Modèle Auto-Régressif Intégré Moyenne Mobile : ARIMA
# Ajustement de plusieurs modèles ARIMA
arima_model <- auto.arima(train_data_power_diff,
                          stepwise = FALSE)

# Affichage du résumé du modèle
summary(arima_model)

# Génération des prévisions pour les 96 prochaines observations
forecast_arima <- forecast(arima_model, h = 96)

# Affichage des prévisions
plot(forecast_arima, main = "Prévisions avec le modèle ARIMA", xlab = "Temps", ylab = "Consommation (kW)", xlim = c(46.7, 48))

# Obtention des prévisions sur l'échelle originale
last_observations <- tail(train_data_ts[, 1], 96)
forecast_arima_mean <- as.numeric(forecast_arima$mean)
last_observations_num <- as.numeric(last_observations)

# Calcul des prévisions sur l'échelle originale
forecast_arima_original <- forecast_arima_mean + last_observations_num

# Calcul des métriques
arima_metrics <- calculate_metrics(validate_data$Power, forecast_arima_original)

# Affichage des prédictions sur le jeu de validation
plot(validate_data$Timestamp, validate_data$Power, type = "l", col = "black", lwd = 2,
     xlab = "Temps", ylab = "Consommation (kW)", main = "Prédictions avec le modèle ARIMA")
lines(validate_data$Timestamp, forecast_arima_original, col = "blue", lwd = 2, lty = 2)
legend("topleft", legend = c("Données réelles", "Données prédites par le modèle ARIMA"),
       col = c("black", "blue"), lty = c(1, 2), lwd = 2)

# Vérification des résidus
checkresiduals(arima_model)

# Modèle Auto-Régressif Intégré Moyenne Mobile avec Saisonnalité : SARIMA
# Ajustement du modèle SARIMA
sarima_model <- auto.arima(train_data_power_diff,
                           seasonal = TRUE,
                           stepwise = FALSE)

# Affichage du résumé du modèle
summary(sarima_model)

# Génération des prévisions pour les 96 prochaines observations
forecast_sarima <- forecast(sarima_model, h = 96)

# Affichage des prévisions
plot(forecast_sarima, main = "Prévisions avec le modèle SARIMA", xlab = "Temps", ylab = "Consommation (kW)", xlim = c(46.7, 48))

# Obtention des prévisions sur l'échelle originale
last_observations_sarima <- tail(train_data_ts[, 1], 96)
forecast_sarima_mean <- as.numeric(forecast_sarima$mean)
last_observations_sarima_num <- as.numeric(last_observations_sarima)

# Calcul des prévisions sur l'échelle originale
forecast_sarima_original <- forecast_sarima_mean + last_observations_sarima_num

# Calcul des métriques
sarima_metrics <- calculate_metrics(validate_data$Power, forecast_sarima_original)

# Affichage des prédictions sur le jeu de validation
plot(validate_data$Timestamp, validate_data$Power, type = "l", col = "black", lwd = 2,
     xlab = "Temps", ylab = "Consommation (kW)", 
     main = "Prédictions avec le modèle SARIMA")
lines(validate_data$Timestamp, forecast_sarima_original, col = "blue", lwd = 2, lty = 2)
legend("topleft", legend = c("Données réelles", "Données prédites par le modèle SARIMA"),
       col = c("black", "blue"), lty = c(1, 2), lwd = 2)

# Vérification des résidus
checkresiduals(sarima_model)

# Modèle Auto-Régressif Intégré Moyenne Mobile avec Exogène : ARIMAX 
# Ajustement du modèle ARIMAX
arimax_model <- auto.arima(train_data_power_diff,
                           xreg = train_data_temp_diff,
                           seasonal = TRUE,
                           stepwise = FALSE)

# Affichage du résumé du modèle
summary(arimax_model)

# Génération des prévisions pour les 96 prochaines observations
forecast_arimax <- forecast(arimax_model, 
                            xreg = validate_data_ts[, 2],
                            h = 96)

# Affichage des prévisions
plot(forecast_arimax, main = "Prévisions avec le modèle ARIMAX", xlab = "Temps", ylab = "Différence de Consommation (kW)", xlim = c(46.7, 48))

# Obtention des prévisions sur l'échelle originale
last_observations_arimax <- tail(train_data_ts[, "Power"], 96)
forecast_arimax_mean <- as.numeric(forecast_arimax$mean)
last_observations_arimax_num <- as.numeric(last_observations_arimax)

# Calcul des prévisions sur l'échelle originale
forecast_arimax_original <- forecast_arimax_mean + last_observations_arimax_num

# Calcul des métriques
arimax_metrics <- calculate_metrics(validate_data$Power, forecast_arimax_original)

# Affichage des prédictions sur le jeu de validation
plot(validate_data$Timestamp, validate_data$Power, type = "l", col = "black", lwd = 2,
     xlab = "Temps", ylab = "Consommation (kW)", 
     main = "Prédictions avec le modèle ARIMAX")
lines(validate_data$Timestamp, forecast_arimax_original, col = "blue", lwd = 2, lty = 2)
legend("topleft", legend = c("Données réelles", "Données prédites par le modèle ARIMAX"),
       col = c("black", "blue"), lty = c(1, 2), lwd = 2)

# Vérification des résidus
checkresiduals(arimax_model)

# -----/ 3. Modèles à réseaux de neuronnes /-----
# Modèle de réseaux de neurones auto-régressifs : NNAR
# Ajustement du modèle NNAR
nnar_model <- nnetar(train_data_ts[, 1])

# Affichage du résumé du modèle
summary(nnar_model)

# Génération des prévisions pour les 96 prochaines observations
forecast_nnar <- forecast(nnar_model, h = 96)

# Calcul des métriques
nnar_metrics <- calculate_metrics(validate_data$Power, forecast_nnar$mean)

# Affichage des prédictions sur le jeu de validation
plot(validate_data$Timestamp, validate_data$Power, type = "l", col = "black", lwd = 2,
     xlab = "Temps", ylab = "Consommation (kW)", 
     main = "Prédictions avec le modèle NNAR")
lines(validate_data$Timestamp, forecast_nnar$mean, col = "blue", lwd = 2, lty = 2)
legend("topleft", legend = c("Données réelles", "Données prédites par le modèle NNAR"),
       col = c("black", "blue"), lty = c(1, 2), lwd = 2)

# Modèle Random Forest
# Définition de la grille d'hyperparamètres
rf_grid <- expand.grid(mtry = c(2, 3))

# Ajustement du modèle Random Forest
rf_tune <- train(Power ~ Temp + Hour + Day, 
                 data = nn_train_data, 
                 method = "rf", 
                 tuneGrid = rf_grid,
                 trControl = trainControl(method = "cv", number = 5))

# Affichage des résultats
print(rf_tune)

# Récupéaration des meilleurs paramètres
final_rf_model <- rf_tune$finalModel

# Création du modèle avec les meilleurs paramètres
rf_model <- randomForest(Power ~ Temp + Hour + Day,
                         data = nn_train_data, ntree = final_rf_model$ntree, mtry = final_rf_model$mtry)

# Affichage du résumé du modèle
summary(rf_model)

# Génération des prédictions pour les 96 prochaines observations
rf_predictions <- predict(rf_model, newdata = nn_validate_data)

# Calcul des métriques
rf_metrics <- calculate_metrics(validate_data$Power, rf_predictions)

# Affichage des prédictions sur le jeu de validation
plot(validate_data$Timestamp, validate_data$Power, type = "l", col = "black", lwd = 2,
     xlab = "Temps", ylab = "Consommation (kW)", 
     main = "Prédictions avec le modèle Random Forest")
lines(validate_data$Timestamp, rf_predictions, col = "blue", lwd = 2, lty = 2)
legend("topleft", legend = c("Données réelles", "Données prédites par le modèle Random Forest"),
       col = c("black", "blue"), lty = c(1, 2), lwd = 2)

# Modèle Gradient Boosting
# Définition de la grille d'hyperparamètres
gb_grid <- expand.grid(
  nrounds          = c(100, 200, 300),
  max_depth        = c(3, 4, 5),
  eta              = c(0.05, 0.1, 0.2),
  gamma            = c(0, 0.3, 0.5),
  colsample_bytree = c(1),
  min_child_weight = c(1),
  subsample        = c(1)
)
gb_train_control <- trainControl(method = "cv", number = 5)

# Ajustement du modèle
gb_tune <- train(
  Power ~ Temp + Hour + Day,
  data       = nn_train_data,
  method     = "xgbTree",
  trControl  = gb_train_control,
  tuneGrid   = gb_grid
)

# Récupéaration des meilleurs paramètres
best_xgb <- gb_tune$finalModel

# Création des matrices
train_matrix <- model.matrix(Power ~ Temp + Hour + Day, data = nn_train_data)
validate_matrix <- model.matrix(Power ~ Temp + Hour + Day, data = nn_validate_data)

# Création du modèle avec les meilleurs paramètres
gb_model <- xgboost(data = train_matrix, label = nn_train_data$Power, nrounds = best_xgb$tuneValue$nrounds, max_depth = best_xgb$tuneValue$max_depth, eta = best_xgb$tuneValue$eta, gamma = best_xgb$tuneValue$gamma, colsample_bytree = best_xgb$tuneValue$colsample_bytree, min_child_weight = best_xgb$tuneValue$min_child_weight, subsample = best_xgb$tuneValue$subsample)

# Affichage du résumé du modèle
summary(gb_model)

# Génération des prédictions pour les 96 prochaines observations
gb_predictions <- predict(gb_model, newdata = validate_matrix)

# Calcul des métriques
gb_metrics <- calculate_metrics(validate_data$Power, gb_predictions)

# Affichage des prédictions sur le jeu de validation
plot(validate_data$Timestamp, validate_data$Power, type = "l", col = "black", lwd = 2,
     xlab = "Temps", ylab = "Consommation (kW)", 
     main = "Prédictions avec le modèle Gradient Boosting")
lines(validate_data$Timestamp, gb_predictions, col = "blue", lwd = 2, lty = 2)
legend("topleft", legend = c("Données réelles", "Données prédites par le modèle Gradient Boosting"),
       col = c("black", "blue"), lty = c(1, 2), lwd = 2)

# Modèle SVM
# Définition de la grille d'hyperparamètres
svm_grid <- expand.grid(
  C = c(0.1, 1, 10, 100),
  sigma = c(0.001, 0.01, 0.1, 1)
)
svm_train_control <- trainControl(method = "cv", number = 5)

# Ajustement du modèle
svm_tune <- train(
  Power ~ Temp + Hour + Day,
  data       = nn_train_data,
  method     = "svmRadial",
  trControl  = svm_train_control,
  tuneGrid   = svm_grid,
  preProcess = c("center", "scale")
)

# Récupération des meilleurs paramètres
best_svm <- svm_tune$bestTune

# Création du modèle avec les meilleurs paramètres
svm_model <- svm(Power ~ Temp + Hour + Day,
                 data = nn_train_data, kernel = "radial", cost = best_svm$C, gamma = best_svm$sigma)

# Affichage du résumé du modèle
summary(svm_model)

# Génération des prédictions pour les 96 prochaines observations
svm_predictions <- predict(svm_model, newdata = nn_validate_data)

# Calcul des métriques
svm_metrics <- calculate_metrics(validate_data$Power, svm_predictions)

# Affichage des prédictions sur le jeu de validation
plot(validate_data$Timestamp, validate_data$Power, type = "l", col = "black", lwd = 2,
     xlab = "Temps", ylab = "Consommation (kW)", 
     main = "Prédictions avec le modèle SVM")
lines(validate_data$Timestamp, svm_predictions, col = "blue", lwd = 2, lty = 2)
legend("topleft", legend = c("Données réelles", "Données prédites par le modèle SVM"),
       col = c("black", "blue"), lty = c(1, 2), lwd = 2)

# Modèle LM
# Ajustement du modèle LM
lm_model <- lm(Power ~ Temp + Hour + Day, data = nn_train_data)

# Affichage du résumé du modèle
summary(lm_model)

# Génération des prédictions pour les 96 prochaines observations
lm_predictions <- predict(lm_model, newdata = nn_validate_data)

# Calcul des métriques
lm_metrics <- calculate_metrics(validate_data$Power, lm_predictions)

# Affichage des prédictions sur le jeu de validation
plot(validate_data$Timestamp, validate_data$Power, type = "l", col = "black", lwd = 2,
     xlab = "Temps", ylab = "Consommation (kW)", 
     main = "Prédictions avec le modèle LM")
lines(validate_data$Timestamp, lm_predictions, col = "blue", lwd = 2, lty = 2)
legend("topleft", legend = c("Données réelles", "Données prédites par le modèle LM"),
       col = c("black", "blue"), lty = c(1, 2), lwd = 2)

# -----/ IV. Prédiction de la consommation sur le jeu de test /-----

# Métriques de performance pour tous les modèles
all_metrics <- data.frame(
  Model = c("ARIMA", "SARIMA", "ARIMAX", "NNAR", "Random Forest", "Gradient Boosting", "SVM", "LM"),
  RMSE  = c(arima_metrics$RMSE, sarima_metrics$RMSE, arimax_metrics$RMSE, nnar_metrics$RMSE,
            rf_metrics$RMSE, gb_metrics$RMSE, svm_metrics$RMSE, lm_metrics$RMSE),
  MAPE  = c(arima_metrics$MAPE, sarima_metrics$MAPE, arimax_metrics$MAPE, nnar_metrics$MAPE,
            rf_metrics$MAPE, gb_metrics$MAPE, svm_metrics$MAPE, lm_metrics$MAPE)
)

cat("\nRécapitulatif des métriques de performance :\n")
print(all_metrics)

# Prédictions sur le jeu de test sans la température (NNAR)
nnar_test_forecast <- forecast(nnar_model, h = nrow(test_data))

# Affichage des prédictions sur le jeu de test
plot(1:nrow(test_data), nnar_test_forecast$mean, type = "l", col = "blue", lwd = 2,
     xlab = "Temps (index)", ylab = "Consommation (kW)",
     main = "Prédictions sur le jeu de test avec le modèle NNAR (sans température)")
legend("topleft", legend = c("Données prédites par le modèle NNAR"),
       col = c("blue"), lty = 1, lwd = 2)

# Prédictions sur le jeu de test avec la température (Gradient Boosting)
# Préparation des données pour le jeu de test
test_matrix <- model.matrix(~ Temp + Hour + Day, data = nn_test_data)

# Génération des prédictions
gb_test_predictions <- predict(gb_model, newdata = test_matrix)

# Affichage des prédictions
plot(1:nrow(nn_test_data), gb_test_predictions, type = "l", col = "blue", lwd = 2,
     xlab = "Temps (index)", ylab = "Consommation (kW)",
     main = "Prédictions sur le jeu de test avec le modèle Gradient Boosting (avec température)")
legend("topleft", legend = c("Données prédites par le modèle Gradient Boosting"),
       col = c("blue"), lty = 1, lwd = 2)

# Sauvegarde des prédictions dans un DataFrame
predictions_df <- data.frame(
  Predictions_without_temp = as.numeric(nnar_test_forecast$mean),
  Predictions_with_temp = as.numeric(gb_test_predictions)
)

# Sauvegarde des prédictions dans un fichier Excel
write_xlsx(predictions_df, "chemin d'enregistrement du fichier") # À changer selon le chemin souhaité