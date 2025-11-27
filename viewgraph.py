from tensorflow import keras
import joblib
import sklearn.preprocessing as pp
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('datamagnet.csv')
waktu = df[['Waktu']]
scl_waktu = joblib.load('scaler_Waktu.pkl')
waktu_scaled = scl_waktu.transform(waktu)
ModelName = input("Masukkan nama model = ")
if(ModelName in ["Bx","By","Bz"]):
    scl_model = joblib.load(f"scaler_{ModelName}.pkl")
    model = keras.models.load_model(f"Model_{ModelName}.h5",compile=False)
    y_test = df[[ModelName]]
else:
    print("Model tidak ditemukan")
    quit

y_pred_scaled = model.predict(waktu_scaled)
y_pred = scl_model.inverse_transform(y_pred_scaled)
plt.plot(waktu,y_pred)
plt.scatter(waktu,y_test)
plt.xlabel("Waktu")
plt.ylabel(ModelName)
plt.title("Grafik Medan Magnet Bumi")
plt.show()