# SDANN-and-RFGB-for-SI-forecasting
Sequential Deep Artificial Neural Network (SDANN) and the Deep Hybrid Random Forest Gradient Boosting (RFGB) for Solar Irradiance forecasting
# Solar Forecasting Models

## Introduction
Effective solar energy utilization demands improvements in forecasting due to the unpredictable nature of solar irradiance (SI). This study introduces and rigorously tests two innovative forecasting models across different locations: the Sequential Deep Artificial Neural Network (SDANN) and the Deep Hybrid Random Forest Gradient Boosting (RFGB).

## Background
The anticipated rise in global energy demand necessitates a significant shift towards renewable sources. Solar energy, particularly effective in regions with high solar exposure, is emerging as a crucial renewable contender. Advanced solar forecasting is recognized for its substantial benefits in the renewable energy sector.

## Contributions
This research integrates sophisticated AI and machine learning for SI forecasting, addressing challenges like balancing computational efficiency with accuracy and refining the accuracy and adaptability of our models. Our study explores solar energy potential using innovative deep learning techniques, focusing on the SDANN and Hybrid RFGB models.

## Methodology
### Data Sites
We selected the cities of Hyderabad, Sukkur, and Turbat in Pakistan for their advantageous meteorological conditions for solar energy potential forecasting.

### Data Collection
The dataset was sourced from NASA's Langley Research Center's POWER Project, providing a comprehensive foundation for our analysis.

### Model Architecture
Our models leverage advanced AI techniques to forecast solar irradiance, utilizing a deep learning approach with SDANN and a hybrid method combining Random Forest and Gradient Boosting in RFGB. The coding is done using Google Colab.

###Import Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE


from keras.layers import Dense
from keras.models import Sequential
### Import Sukkur Dataset
from google.colab import drive
drive.mount('/content/drive')
#@title
# Display first 5 rows

df_t = pd.read_csv("link for csv file.csv")
df_t[:5]
# Data Information

df_t.info()
# Remove null values and describe statistics

df_t = df_t.dropna()
df_t.describe()
### Features and Target
# Features
X = df_t[df_t.columns.drop(["Date", "SI"])]

# Target
y = df_t["SI"]
# Split Data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 42)
# Scale features
scaled = StandardScaler()

X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)

### Sequential Deep Artificial Neural Network (Dataset Name)
# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(500, activation = 'relu', input_dim = 8))

# Adding the second hidden layer
model.add(Dense(units = 400, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units =300, activation = 'relu'))

# Adding the fourth hidden layer
model.add(Dense(units =200, activation = 'relu'))

# Adding the fifth hidden layer
model.add(Dense(units =100,activation = 'relu'))

# Adding the sixth hidden layer
model.add(Dense(units =25, activation = 'relu'))

# Adding the output layer
model.add(Dense(units = 1, activation = 'relu'))

model.compile(optimizer = 'adam',loss = 'mean_squared_error')

model.fit(X_train_scaled, y_train, batch_size = 10, epochs = 100)
### Actual and Predicted Values SDANN Comparison
# Display few values to get an idea 

pred_ta = model.predict(X_test_scaled)
X_ANN_Suk=pd.DataFrame({"Actual": y_test,
              "Predicted": pred_ta.flatten()})[:10000]
X_ANN_Suk[:10]
X_ANN_Suk.to_csv("X_ANN_Suk.csv")
### Error Values SDANN Dataset
# R Squared 
ta_r2 = r2_score(y_test, pred_ta)
print("R Squared =", ta_r2, "\n")

# Mean Biased Error
ta_mbe = np.mean(y_test - pred_ta.flatten())
print("Mean Biased Error =", ta_mbe, "\n")

# Mean Absolute Error
ta_mae = MAE(y_test, pred_ta)
print("Mean Absolute Error =", ta_mae, "\n")

# Mean Squared Error
ta_mse = MSE(y_test, pred_ta)
print("Mean Squared Error =", ta_mse, "\n")

# Root Mean Squared Error
ta_rmse = MSE(y_test, pred_ta)**0.5
print("Root Mean Squared Error =", ta_rmse, "\n")
### Actual vs Predicted Values Comparison SDANN by KDE Plot
from matplotlib.font_manager import FontProperties
plt.figure(figsize = (12, 7))
sns.kdeplot(x = y_test, label = "Actual", linewidth = 10, color = "yellow")
sns.kdeplot(pred_ta.flatten(), label = "Predicted", linewidth = 10, color = "b")
plt.xlabel("SI", size = 25, weight='bold')
plt.ylabel("Density", size = 25, weight='bold')
plt.xticks(size = 25, weight='bold')
plt.yticks(size = 25, weight='bold')
font = FontProperties(family= 'Comic Sans MS',  # 'Times new roman', 
                                   weight='bold',
                                   style='normal', size=25)
plt.legend(["Actual", "Predicted"], loc='upper left', prop=font)
sns.set_context("talk")
plt.savefig("KDE_ANN_Suk.png",dpi=2400, facecolor="w")
plt.show()
### Actual vs Predicted Values Comparison SDANN by Scatter Plot
plt.figure(figsize = (12, 7))
sns.regplot(y_test, pred_ta.flatten(), scatter_kws={"color": "blue"}, line_kws={"color": "green"})
plt.title("Sukkur SI Actual and ANN Predicted Values", size = 20, weight='bold')
plt.xlabel("Actual SI", size = 15, weight='bold')
plt.ylabel("Density", size = 15, weight='bold')
plt.xticks(size = 15, weight='bold')
plt.yticks(size = 15, weight='bold')
sns.set_context("talk")
plt.savefig("Scat_ANN_Suk.png",dpi=2400, facecolor="w")
plt.show()
### Custom Prediction (SDANN Sukkur)
custom_ta = model.predict(scaled.transform([[35.77, 33.42, 18.05, 43.01, 8.42, 24.44, 93.94, 4.94]]))
print("Custom Prediction of Irradiance on Sukkur Dataset using ANN =", custom_ta)


### Random Forest
rf_t = RandomForestRegressor(random_state = 42)


parameters = {"n_estimators": [70, 140, 210, 280],
              "max_depth": np.arange(1, 10),
              "min_samples_leaf": [0.1, 0.2, 0.3, 0.4, 0.5]}

cv = GridSearchCV(rf_t, parameters, cv = 3)

cv.fit(X_train, y_train)

# Best Parameters and Accuracy
print("Best Parameters =", cv.best_params_)
print("Best Accuracy =", cv.best_score_)

### Gradient Boosting Regressor
gbr_s = GradientBoostingRegressor(random_state = 42)

parameters = {"n_estimators": [70, 90, 110, 130],
              "max_depth": np.arange(1, 10),
              "min_samples_leaf": [0.1, 0.2, 0.3, 0.4, 0.5]}

cv = GridSearchCV(gbr_s, parameters, cv = 3)

cv.fit(X_train, y_train)

# Best Parameters and Accuracy
print("Best Parameters =", cv.best_params_)
print("Best Accuracy =", cv.best_score_)


### Hybrid Model RFGB
rfs_o = RandomForestRegressor(max_depth = 2, min_samples_leaf = 0.1, n_estimators = 280, random_state = 42)

gbr_o = GradientBoostingRegressor(init = rfs_o,
                                  max_depth = 4,
                                  min_samples_leaf = 0.1,
                                  n_estimators = 110)

gbr_o.fit(X_train, y_train)

pred_th = gbr_o.predict(X_test)

X_RFGB_Suk=pd.DataFrame({"Actual": y_test,
              "Predicted": pred_th})[:10000]
X_RFGB_Suk[:10]              

X_RFGB_Suk.to_csv("X_RFGB_Suk.csv")

### Error Values Hybrid RFGB Sukkur

# R Squared 
th_r2 = r2_score(y_test, pred_th)
print("R Squared =", th_r2, "\n")

# Mean Biased Error
th_mbe = np.mean(y_test - pred_th.flatten())
print("Mean Biased Error =", th_mbe, "\n")

# Mean Absolute Error
th_mae = MAE(y_test, pred_th)
print("Mean Absolute Error =", th_mae, "\n")

# Mean Squared Error
th_mse = MSE(y_test, pred_th)
print("Mean Squared Error =", th_mse, "\n")

# Root Mean Squared Error
th_rmse = MSE(y_test, pred_th)**0.5
print("Root Mean Squared Error =", th_rmse, "\n")

### Actual vs Predicted Values Comparison Hybrid RFGB by KDE Plot
plt.figure(figsize = (12, 7))
sns.kdeplot(x = y_test, label = "Actual", linewidth = 10, color = "yellow")
sns.kdeplot(pred_th.flatten(), label = "Predicted", linewidth = 10, color = "r")
plt.xlabel("SI", size = 25, weight='bold')
plt.ylabel("Density", size = 25, weight='bold')
plt.xticks(size = 25, weight='bold')
plt.yticks(size = 25, weight='bold')
font = FontProperties(family= 'Comic Sans MS',  # 'Times new roman', 
                                   weight='bold',
                                   style='normal', size=25)
plt.legend(["Actual", "Predicted"], loc='upper left', prop=font)
sns.set_context("poster")
plt.savefig("KDE_RFGB_Suk.png",dpi=2400, facecolor="w")
plt.show()
### Suctom Predictions RFGB
custom_th = gbr_o.predict(scaled.transform([[35.77, 33.42, 18.05, 43.01, 8.42, 24.44, 93.94, 4.94]]))
print("Custom Prediction of Irradiance on Sukkur Dataset using Hybrid Model =", custom_th)

### Actual vs Predicted Values Comparison Hybrid RFGB by Scatter Plot
plt.figure(figsize = (12, 7))
sns.regplot(y_test, pred_th.flatten() , scatter_kws={"color": "green"}, line_kws={"color": "blue"})
plt.title("Sukkur SI Actual and RFGB Predicted Values", size = 20, weight='bold')
plt.xlabel("Actual SI", size = 15, weight='bold')
plt.ylabel("Density", size = 15, weight='bold')
plt.xticks(size = 15, weight='bold')
plt.yticks(size = 15, weight='bold')
sns.set_context("talk")
plt.savefig("Scat_RFGB_Suk.png",dpi=2400, facecolor="w")
plt.show()
### Actual vs Predicted Values Comparison Hybrid RFGB and SDANN by Scatter Plot
plt.figure(figsize = (12, 7))
plt.scatter(y_test, pred_ta.flatten(), c="b")
plt.scatter(y_test, pred_th.flatten(), c="orange")
plt.title("b. Scatter Plot of ANN & RFGB for Sukkur", c= "Black", size = 20, weight='bold')
plt.xlabel("SI", size = 15, weight='bold')
plt.ylabel("Density", size = 15, weight='bold')
plt.xticks(size = 15, weight='bold')
plt.yticks(size = 15, weight='bold')
font = FontProperties( weight='bold',
                                   style='normal', size=15)
plt.legend(["SDANN Predicted", "RFGB Predicted"], loc='upper left', prop=font)
sns.set_context("talk")
plt.savefig("Comp-ANN_RFGB_Suk.png",dpi=2400, facecolor="w")
plt.show()

## Findings
Our research not only advances the field of SI forecasting but also conducts detailed ablation studies to assess meteorological feature impacts on model performance.

## Conclusion
By integrating cutting-edge AI in SI forecasting, this research not only advances the field but also sets the stage for future renewable energy strategies and global policy-making.

## Acknowledgments
https://power.larc.nasa.gov/beta/data-access-viewer/
https://colab.research.google.com/
