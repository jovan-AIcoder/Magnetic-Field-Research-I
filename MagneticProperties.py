from tensorflow import keras
import sklearn.preprocessing as pp
import joblib
import numpy as np
#load model
model_Bx = keras.models.load_model("Model_Bx.h5",compile=False)
model_By = keras.models.load_model("Model_By.h5",compile=False)
model_Bz = keras.models.load_model("Model_Bz.h5",compile=False)
#load scaler
scaler_time = joblib.load("scaler_Time.pkl")
scaler_Bx = joblib.load("scaler_Bx.pkl")
scaler_By = joblib.load("scaler_By.pkl")
scaler_Bz = joblib.load("scaler_Bz.pkl")
#predict at reference time (0.0 decimal hours)
time_val = 0.0
time_scaled = scaler_time.transform([[time_val]])
Bx_scaled = model_Bx.predict(time_scaled)
By_scaled = model_By.predict(time_scaled)
Bz_scaled = model_Bz.predict(time_scaled)
#inverse transform
Bx = scaler_Bx.inverse_transform(Bx_scaled)
By = scaler_By.inverse_transform(By_scaled)
Bz = scaler_Bz.inverse_transform(Bz_scaled)
#calculate all properties
B = np.sqrt((Bx*Bx)+(By*By)+(Bz*Bz))
dec = np.atan(By/Bx)
dec = np.degrees(dec)
inc = np.atan(Bz/np.sqrt((Bx*Bx)+(By*By)))
inc = np.degrees(inc)
mu_0 = 4e-7 * np.pi
Bh = np.sqrt(Bx**2 + By**2)
M = (3/mu_0)*np.sqrt(Bh**2+((1/4)*Bz**2))
m = M * (4/3) * np.pi * ((6371e3)**3)
theta = np.asin((3*Bh)/(mu_0*M))
theta = np.degrees(theta)
#show values
print("=== REFERENCE MAGNETIC PROPERTIES ===")
print(f"Bx = {Bx[[0]]} nT")
print(f"By = {By[[0]]} nT")
print(f"Bz = {Bz[[0]]} nT")
print(f"B = {B[[0]]} nT")
print(f"Declination = {dec[[0]]} degrees")
print(f"Inclination = {inc[[0]]} degrees")
print(f"Magnetization = {M[[0]]} A/m")
print(f"Magnetic Moment = {m[[0]]} A.m^2")
print(f"Polar angle of observation (reference: magnetic dipole) = {theta[[0]]} degrees")