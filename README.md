(This project is made in Indonesian language by Jovan, a physicist)
Prediksi Variasi Harian Medan Magnet Bumi Menggunakan Deep Neural Networks

Repositori ini berisi implementasi model Deep Learning berbasis TensorFlow untuk memprediksi komponen-komponen medan magnet bumi (Bx, By, Bz) berdasarkan waktu dalam sehari. Model ini dikembangkan untuk mengeksplorasi dinamika variasi harian medan magnet bumi yang dipengaruhi oleh interaksi ionosfer, arus global, dan proses-proses elektromagnetik lainnya.

Latar Belakang

Medan magnet bumi mengalami variasi harian dengan pola yang relatif teratur, terutama dipengaruhi oleh arus ionosfer (Sq currents), radiasi matahari, dan interaksi plasma magnetosfer. Variasi ini sering berada pada orde ribuan nanoTesla (nT) dan dapat dimodelkan secara empiris dengan pendekatan machine learning.

Model dalam repositori ini menggunakan dataset berdimensi kecil (48 sampel per hari), dengan resolusi waktu 30 menit, untuk mempelajari pola periodik harian pada tiga komponen medan magnet bumi.

Struktur Model

Model dibangun menggunakan arsitektur jaringan saraf multilapis dengan karakteristik berikut:

Arsitektur:
64 → 32 → 16 → 8 → 4 → 2 → 1

Activation function:
Semua layer kecuali output menggunakan Mish/Swish, yang memberikan sifat smooth non-monotonic dan sering lebih stabil dibanding ReLU.

Output layer: Linear (tanpa aktivasi), cocok untuk regresi.

Scale data: MinMaxScaler untuk menstabilkan training.

Epoch: 300

Optimizer:
Adamax atau Nadam, dipilih karena stabil untuk dataset kecil.

Loss function:
Mean Squared Error (MSE)

Rata-rata nilai loss berada pada kisaran:

0.0021 – 0.0041 (scaled space)


yang menunjukkan fitting yang cukup baik untuk ukuran dataset yang terbatas.

Dataset

Dataset yang digunakan berisi:

48 sampel

Resolusi waktu 30 menit (0.5 jam)

24 jam penuh

Nilai komponen medan magnet dalam satuan nanoTesla (nT)

Sumber data berasal dari observasi magnetometer (lokasi tidak dinyatakan)

Tujuan Proyek

Menguji apakah jaringan saraf sederhana dapat mempelajari variasi harian geomagnetik.

Mengevaluasi performa deep learning untuk pemodelan geofisika dengan dataset kecil.

Mengembangkan baseline untuk penelitian lanjutan, seperti:

prediksi badai geomagnetik

korelasi dengan indeks Kp/Dst

perbandingan model fisik vs model data-driven

File dalam Repositori
File	Deskripsi
GenerateModel.py	Script utama untuk training model
Model_Bx.h5	Model terlatih untuk komponen Bx
Model_By.h5	Model terlatih untuk komponen By
Model_Bz.h5	Model terlatih untuk komponen Bz
README.md	Dokumentasi proyek
Cara Menggunakan
1. Install dependensi
pip install tensorflow numpy scikit-learn

2. Jalankan training
python GenerateModel.py

3. Memuat model
from tensorflow import keras

model = keras.models.load_model("Model_Bx.h5")
pred = model.predict([[jam_dalam_decimal]])

Potensi Pengembangan:

Menambah data dari observatorium geomagnetik internasional (INTERMAGNET).

Menambahkan fitur:

tanggal

indeks aktivitas matahari

kondisi ionosfer

Menggunakan arsitektur LSTM/Transformer untuk menangkap dinamika temporal lebih dalam.

Membandingkan model dengan simulasi fisika (IGRF, CHAOS geomagnetic model).

Lisensi:

Repositori ini menggunakan lisensi bebas-modifikasi (silakan tambahkan MIT/Apache/CC sesuai keinginanmu).

Kontributor:

Dikembangkan oleh: Jovan Alcoder
Topik: AI untuk Geofisika / Pemodelan Medan Magnet Bumi