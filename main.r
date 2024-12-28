# Chargement des librairies nécessaires
library(readxl)
library(forecast)
library(tseries)
library(dplyr)
library(randomForest)
library(xgboost)
library(e1071)

# Création de la fonction pour le calcul des métriques d'évaluation
calculate_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mape <- mean(abs((actual - predicted) / actual)) * 100
  return(list(RMSE = rmse, MAPE = mape))
}

# -----/ I. Mise en forme des données et visualisation /-----

# Chargement des données
setwd ("C:/Users/hugoc/Documents/GitHub/time_series_forecasting")
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
plot(train_data_ts, main = "Consommation électrique et température dans le temps", xlab = "Temps", ylab = "",  lty = c(1, 2), lwd = c(1, 1))

# Affichage du graphique de la série temporelle de validation
plot(validate_data_ts, main = "Consommation électrique et température dans le temps", xlab = "Temps", ylab = "",  lty = c(1, 2), lwd = c(1, 1))

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

# [COMMENTAIRE]
# 1. Répétition quotidienne :
# On observe une structure répétitive dans la consommation électrique au fil de la journée. Les pics de consommation apparaissent systématiquement le matin vers 7h-8h pour se stabiliser jusqu’à 15h-16h. Puis un second pic survient vers 16h-17h jusqu’à 21h-22h, ce qui reflète les habitudes humaines, comme le début de la journée de travail et les activités domestiques en soirée.
# 2. Creux durant la nuit :
# La consommation est minimale pendant la nuit, entre 23h et 5h, période où l'activité est généralement plus faible.
# 3. Variabilité inter-journalière :
# Bien que la forme globale reste constante d’un jour à l’autre, il existe une variabilité notable d’une journée à l’autre. Cela pourrait être lié à des facteurs externes comme la température, les jours de semaine (travail vs week-end).

# Décomposition additive de la consommation électrique
autoplot(decompose(train_data_ts[, 1], type = "additive"), main = "Décomposition additive - Consommation électrique")

# Décomposition additive de la température
autoplot(decompose(train_data_ts[, 2], type = "additive"), main = "Décomposition additive - Température")

# [COMMENTAIRE]
# 1. Tendance :
# On observe une tendance globale avec des fluctuations significatives à la baisse au début puis une stabilisation. Cela indique une dynamique de diminution générale de la consommation électrique, et nous pouvons voir qu’à l’inverse la trend de la température augmente, ce qui est logique car plus les températures augmentent moins le besoin de chauffer est nécessaire.
# 2. Saisonnalité :
# Au niveau de la composante saisonnière montrent des cycles réguliers très marqués que ce soit pour la consommation ou pour la température. Cela suggère une forte dépendance temporelle, probablement sur des cycles quotidiens (par exemple, variation jour/nuit ou jours ouvrés vs week-end).

# Réalisation d'une ACF
acf(train_data_ts[, 1], lag.max = 5000, main = "ACF - Consommation électrique")

# [COMMENTAIRE]
# L'ACF révèle un décalage significatif, un déclin progressif et des pics périodiques, témoignant d'une forte dépendance temporelle. Cela est cohérent avec la saisonnalité observée dans le season plot. Le graphique ACF présente une configuration sinusoïdale marquée et périodique, confirmant un comportement saisonnier prononcé. De plus, la lente décroissance des autocorrélations suggère que la série est non stationnaire et caractérisée par des fluctuations périodiques.

# Réalisation d'une PACF
pacf(train_data_ts[, 1], lag.max = 5000, main = "PACF - Consommation électrique")

# [COMMENTAIRE]
# Le PACF confirme et précise les observations précédentes, une forte dépendance à court terme et une forte saisonnalité quotidienne.

# -----/ III. Prédiction de la consommation électrique /-----
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

# [COMMENTAIRE]
# Les modèles prédisent une valeur constante après la dernière observation (ligne bleue horizontale). Nous obtenons de mauvais résultats car les modèles ne prennent pas en compte les tendances ni la saisonnalité, ce qui explique l'absence de variation dans la prévision. Il n’est donc pas adapté pour faire une prédiction sur notre jeu de données.
# Concernant les lissages exponentiels triples : Holt-Winters Method - Additive Seasonal et Multiplicative seasonal il est impossible de les utiliser car la fréquence des données est trop élevée.
# De manière générale, les modèles de lissage exponentiel ne sont pas adaptés pour prédire ces données, car elles contiennent une tendance et une saisonnalité significatives.

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

# [COMMENTAIRE]
# 1. Data :
# On observe une série plus stationnaire en apparence, avec des fluctuations autour de zéro. Les pics et les creux importants qui reflétaient la saisonnalité journalière ont disparu.
# 2. Trend :
# On observe des fluctuations, mais sans tendance globale. Cela suggère que la différenciation a permis de supprimer la tendance linéaire.
# 3. Saisonalité :
# Après la différenciation saisonnière avec un lag de 96, la composante saisonnière devrait être proche de zéro. C'est ce que l'on observe sur le graphique. La différenciation a effectivement retiré la saisonnalité journalière.

# Réalisation d'une ACF
acf(train_data_power_diff, lag.max = 5000, main = "ACF sur la série différenciée - Consommation électrique")

# [COMMENTAIRE]
# L'ACF obtenue montre encore des pics significatifs (beaucoup moins nombreux qu'avant la différenciation) notamment aux premiers lags, qui décroissent progressivement vers zéro. Cela suggère la présence d'une composante MA (Moyenne Mobile).

# Réalisation d'une PACF
pacf(train_data_power_diff, lag.max = 5000, main = "PACF sur la série différenciée - Consommation électrique")

# [COMMENTAIRE]
# Le PACF possède un pic important au début, puis des pics plus faibles et une décroissance vers zéro. Cela suggère la présence d'une composante AR (Autorégressive).

# Test de Ljung-Box
Box.test(train_data_power_diff, lag = 10, type = "Ljung-Box")

# Test Augmented Dickey-Fuller
adf.test(train_data_power_diff, alternative = "stationary")

# [COMMENTAIRE]
# Le test de Ljung-Box rend un résultat où X-squared = 5495.8 et p-value < 2.2e-16. Ici, la p-value extrêmement faible indique que les résidus de la série différenciée présentent encore des autocorrélations significatives, ce qui suggère qu'un modèle ARIMA pourrait capturer ces corrélations.
# Pour le test Augmented Dickey-Fuller, nous obtenons un Dickey-Fuller = -13.498 et p-value = 0.01. La série différenciée est donc stationnaire (l’hypothèse nulle de non-stationnarité est rejetée) et cela confirme que la différenciation a correctement éliminé la tendance et la saisonnalité initiales.
# Une deuxième différenciation n'est donc pas nécessaire car le test Augmented Dickey-Fuller montre que la série est stationnaire après la première différenciation saisonnière avec un lag de 96 (p-value < 0.01). Faire une deuxième différenciation non saisonnière risquerait de sur-différencier la série, ce qui peut compliquer le modèle ARIMA sans améliorer la qualité des prévisions.

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
# Ajustement du modèle Random Forest
rf_model <- randomForest(Power ~ Temp + Hour + Day,
                         data = nn_train_data, ntree = 100)

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
# Création des matrices
train_matrix <- model.matrix(Power ~ Temp + Hour + Day, data = nn_train_data)
validate_matrix <- model.matrix(Power ~ Temp + Hour + Day, data = nn_validate_data)

# Ajustement du modèle
gb_model <- xgboost(data = train_matrix, label = nn_train_data$Power, nrounds = 100)

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
# Ajustement du modèle SVM
svm_model <- svm(Power ~ Temp + Hour + Day,
                 data = nn_train_data, kernel = "radial")

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