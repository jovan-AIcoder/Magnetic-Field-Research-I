from tensorflow import keras
import joblib
import sklearn.preprocessing as pp
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('datamagnet.csv')
time_col = df[['Time']]
scl_time = joblib.load('scaler_Time.pkl')
time_scaled = scl_time.transform(time_col)
ModelName = input("Enter model name (Bx/By/Bz): ")
if(ModelName in ["Bx","By","Bz"]):
    scl_model = joblib.load(f"scaler_{ModelName}.pkl")
    model = keras.models.load_model(f"Model_{ModelName}.h5",compile=False)
    y_test = df[[ModelName]]
else:
    print("Model not found")
    exit()

y_pred_scaled = model.predict(time_scaled)
y_pred = scl_model.inverse_transform(y_pred_scaled)
plt.plot(time_col,y_pred)
plt.scatter(time_col,y_test)
plt.xlabel("Time")
plt.ylabel(ModelName)
plt.title("Earth's Magnetic Field Plot")
plt.show()