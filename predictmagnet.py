from tensorflow import keras
import sklearn.preprocessing as pp
import joblib
import numpy as np
#load model
model_Bx = keras.models.load_model("Model_Bx.h5",compile=False)
model_By = keras.models.load_model("Model_By.h5",compile=False)
model_Bz = keras.models.load_model("Model_Bz.h5",compile=False)
#load scaler
scaler_waktu = joblib.load("scaler_Waktu.pkl")
scaler_Bx = joblib.load("scaler_Bx.pkl")
scaler_By = joblib.load("scaler_By.pkl")
scaler_Bz = joblib.load("scaler_Bz.pkl")
#input
while True:
    try:
        waktu = float(input("Masukkan waktu dalam jam desimal (0-23.5): "))
    except ValueError:
        print("Masukkan waktu dengan benar!")
    else:
        if((waktu < 0) or (waktu > 23.5)):
            print("Waktu di luar rentang yang diberikan!")
        else:
            break

#predict
waktu_scaled = scaler_waktu.transform([[waktu]])
Bx_scaled = model_Bx.predict(waktu_scaled)
By_scaled = model_By.predict(waktu_scaled)
Bz_scaled = model_Bz.predict(waktu_scaled)
#inverse transform
Bx = scaler_Bx.inverse_transform(Bx_scaled)
By = scaler_By.inverse_transform(By_scaled)
Bz = scaler_Bz.inverse_transform(Bz_scaled)
B = np.sqrt((Bx*Bx)+(By*By)+(Bz*Bz))
#show values
print("=== MEDAN MAGNET BUMI ===")
print(f"waktu = {waktu}")
print(f"Bx = {Bx[[0]]} nT")
print(f"By = {By[[0]]} nT")
print(f"Bz = {Bz[[0]]} nT")
print(f"B = {B[[0]]} nT")