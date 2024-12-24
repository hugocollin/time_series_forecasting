# Chargement des librairies nécessaires
library(readxl)
library(forecast)
library(tseries)

# -----/ I. Mise en forme des données et visualisation /-----

# Chargement des données
data_path <- file.path("data", "Elec-train.xlsx")
data <- read_excel(data_path)

# Mise en forme des données
data$Timestamp <- as.POSIXct(data$Timestamp, format = "%m/%d/%Y %H:%M")

# Affichage du jeu de données
head(data)

# Séparation des données en jeu d'entraînement et jeu de test (entre les données exitantes et les données futures à prédire)
train_data <- subset(data, Timestamp < as.POSIXct("2010-02-17 00:00:00"))
test_data  <- subset(data, Timestamp >= as.POSIXct("2010-02-17 00:00:00"))

# Transformation des données en série temporelle multivariée (96 observations par jour)
train_data_ts <- ts(train_data[, c("Power (kW)", "Temp (C°)")],
                    start = c(2010, 1),
                    frequency = 96)

test_data_ts <- ts(test_data[, c("Power (kW)", "Temp (C°)")],
                   frequency = 96)

# Affichage des premières valeurs des séries temporelles
head(train_data_ts)
head(test_data_ts)

# Affichage du graphique de la série temporelle
plot(train_data_ts, main = "Consommation électrique et température dans le temps", xlab = "Temps", ylab = "",  lty = c(1, 2), lwd = c(1, 1))

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

# Réalisation d'un Augmented Dickey-Fuller Test
adf.test(train_data_ts[, 1], alternative = "stationary")

# [COMMENTAIRE]
# Étant donné que la p-value (0.01) est inférieure au seuil de significativité de 0.05 et que la statistique de test Dickey-Fuller = -14.169, nous rejetons l'hypothèse nulle de non-stationnarité. Cela veut donc dire que la série temporelle est stationnaire ce qui rentre en contradiction avec les observations précédentes. Nous essayerons donc d'appliquer par la suite une différenciation pour rendre la série stationnaire.

# -----/ III. Prédiction de la consommation électrique /-----
# -----/ 1. Modèles de lissage exponentiel /-----

# Lissage exponentiel simple : Simple Exponential Smoothing Method
ses_model <- ses(train_data_ts[, 1], h = 96)

# Affichage des performances
summary(ses_model)

# Enregistrement des prédictions
predictions <- ses_model$mean

# Affiche de la prédiction
plot(ses_model, xaxt = "n")

# Zoom de l'affichage sur la partie prédite
last_timestamp <- max(train_data$Timestamp)
forecast_time <- seq(from = last_timestamp + 15 * 60, by = "15 mins", length.out = 96)

# Extraction des 96 dernières observations
last_96_power <- tail(train_data_ts[, 1], 96)
last_96_time <- tail(train_data$Timestamp, 96)

# Tracage de graphique
plot(last_96_time, last_96_power, type = "l",
     col = "black", lwd = 2,
     xlim = c(min(last_96_time), max(forecast_time)),
     ylim = range(c(last_96_power, predictions), na.rm = TRUE),
     main = "Prédiction du modèle de Lissage Exponentiel Simple",
     xlab = "Temps", ylab = "Consommation (kW)")
lines(forecast_time, predictions, col = "blue", lwd = 2, lty = 2)
legend("topright", legend = c("Consommation réelle", "Consommation prédite"),
       col = c("black", "blue"), lty = c(1, 2), lwd = 2)

# Lissage exponentiel double : Holt Linear Trend Method
# Ajustement du modèle Holt
holt_model <- holt(train_data_ts[, 1], h = 96)

# Affichage du résumé du modèle
summary(holt_model)

# Enregistrement des prédictions
predictions_holt <- holt_model$mean

# Affichage des prédictions
plot(holt_model, xaxt = "n")

# Création d'un vecteur de timestamps pour les prévisions
last_timestamp <- max(train_data$Timestamp)
forecast_time_holt <- seq(from = last_timestamp + 15 * 60, by = "15 mins", length.out = 96)

# Extraction des 96 dernières observations du jeu d'entraînement
last_96_power_holt <- tail(train_data_ts[, 1], 96)
last_96_time_holt <- tail(train_data$Timestamp, 96)

# Tracé manuel des 96 derniers points et des 96 prédictions Holt
plot(last_96_time_holt, last_96_power_holt, type = "l",
     col = "black", lwd = 2,
     xlim = c(min(last_96_time_holt), max(forecast_time_holt)),
     ylim = range(c(last_96_power_holt, predictions_holt), na.rm = TRUE),
     main = "Prédiction du modèle de Lissage Exponentiel Double",
     xlab = "Timestamp", ylab = "Consommation (kW)")
lines(forecast_time_holt, predictions_holt, col = "blue", lwd = 2, lty = 2)
legend("topright", legend = c("Consommation réelle", "Consommation prédite"),
       col = c("black", "blue"), lty = c(1, 2), lwd = 2)

# [COMMENTAIRE]
# Les modèles prédisent une valeur constante après la dernière observation (ligne bleue horizontale). Nous obtenons de mauvais résultats car les modèles ne prennent pas en compte les tendances ni la saisonnalité, ce qui explique l'absence de variation dans la prévision. Il n’est donc pas adapté pour faire une prédiction sur notre jeu de données.
# Concernant les lissages exponentiels triples : Holt-Winters Method - Additive Seasonal et Multiplicative seasonal il est impossible de les utiliser car la fréquence des données est trop élevée.
# De manière générale, les modèles de lissage exponentiel ne sont pas adaptés pour prédire ces données, car elles contiennent une tendance et une saisonnalité significatives.

# -----/ 2. Modèles ARIMA /-----
# Application d'une différenciation pour enlever la tendance et la saisonnalité pour la consommation électrique
train_power_diff <- diff(train_data_ts[, 1], lag = 96, differences = 1)

# Affichage de la décomposition de la série temporelle différenciée
autoplot(decompose(train_power_diff, type = "additive"), main = "Décomposition additive - Série différenciée de la consommation électrique")

# Affichage de l'ACF de la série temporelle différenciée
acf(train_power_diff, lag.max = 5000, main = "ACF - Série différenciée de la consommation électrique")

# Affichage de la PACF de la série temporelle différenciée
pacf(train_power_diff, lag.max = 5000, main = "PACF - Série différenciée de la consommation électrique")

# Test de Ljung-Box
lb_test <- Box.test(train_power_diff, lag = 96, type = "Ljung-Box")
print(lb_test)