from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Charger les données
df = pd.read_csv("ad_mini.csv")
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

X = df.drop(columns=["Clicked on Ad"])
y = df["Clicked on Ad"]

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation en train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Définition du modèle
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)), 
    Dropout(0.3),
    keras.layers.Dense(32, activation="relu"),
    Dropout(0.3),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

# Callback pour sauvegarder le meilleur modèle basé sur val_loss
checkpoint = ModelCheckpoint("model.keras", monitor="val_loss", save_best_only=True)

# Compilation et entraînement du modèle
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Sauvegarde du scaler
np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)
print("Modèle entraîné et sauvegardé avec succès.")
