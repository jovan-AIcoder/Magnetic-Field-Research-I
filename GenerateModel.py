from tensorflow import keras
import pandas as pd
import sklearn.preprocessing as pp
import joblib
df = pd.read_csv('datamagnet.csv')
X = df[['Waktu']]
scl_Waktu = pp.MinMaxScaler()
X_scaled = scl_Waktu.fit_transform(X)
joblib.dump(scl_Waktu,"scaler_Waktu.pkl")
ModelName = input("Masukkan komponen medan magnet = ")
if(ModelName in ['Bx','By','Bz']):
    y_train = df[[ModelName]]
    scl = pp.MinMaxScaler()
    y_scaled = scl.fit_transform(y_train)
    joblib.dump(scl,f"scaler_{ModelName}.pkl")
    
else:
    print(f"Tidak ada komponen magnet {ModelName}")
    quit

model = keras.Sequential([
    keras.layers.Dense(64,activation='mish',input_shape=(1,)),
    keras.layers.Dense(32,activation='mish'),
    keras.layers.Dense(16,activation='mish'),
    keras.layers.Dense(8,activation='mish'),
    keras.layers.Dense(4,activation='mish'),
    keras.layers.Dense(2,activation='mish'),
    keras.layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
model.fit(X_scaled,y_scaled,epochs=300,batch_size=24)
model.save(f"Model_{ModelName}.h5")