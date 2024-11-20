import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import requests
from io import StringIO

# Pobranie danych z GitHuba (temperatura w Nowym Jorku)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
response = requests.get(url)
data = pd.read_csv(StringIO(response.text))

# Przygotowanie danych - wybór kolumny Temp
df = data[['Date', 'Temp']].copy().dropna()  # Wybierz datę i temperaturę, usuń wiersze z brakującymi danymi
df.reset_index(drop=True, inplace=True)

# Normalizacja danych
scaler = MinMaxScaler(feature_range=(0, 1))
df['Temp_scaled'] = scaler.fit_transform(df[['Temp']])

# Funkcja pomocnicza do przygotowania danych do uczenia
def prepare_data(series, n_steps):
    X, y = [], []
    for i in range(len(series)-n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    return np.array(X), np.array(y)

# Wybór liczby kroków czasowych (np. 20)
n_steps = 20
X, y = prepare_data(df['Temp_scaled'].values, n_steps)

# Podział na dane treningowe i testowe
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Dodanie dodatkowego wymiaru dla LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Definicja modelu sieci neuronowej z LSTM
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 1), return_sequences=True),
    LSTM(50, activation='relu'),
    Dense(1)
])

# Kompilacja modelu
model.compile(loss=mean_squared_error, optimizer=Adam(learning_rate=0.001))

# Trenowanie modelu
epochs = 50
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Predykcja
y_pred = model.predict(X_test)

# Odtworzenie oryginalnej skali
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
y_pred_inv = scaler.inverse_transform(y_pred).ravel()

# Wykres przebiegu uczenia
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Strata (MSE) na danych treningowych', color='green')
plt.plot(history.history['val_loss'], label='Strata (MSE) na danych walidacyjnych', color='pink')
plt.title('Przebieg uczenia modelu')
plt.xlabel('Epoki')
plt.ylabel('Strata (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# Wykres porównania danych rzeczywistych z predykcją modelu
plt.figure(figsize=(14, 7))
plt.plot(np.arange(len(y_test_inv)), y_test_inv, label='Dane rzeczywiste', color='green')
plt.plot(np.arange(len(y_test_inv)), y_pred_inv, label='Predykcja modelu', color='pink')
plt.title('Porównanie danych rzeczywistych z predykcją modelu')
plt.xlabel('Indeks próbki')
plt.ylabel('Temperatura')
plt.legend()
plt.grid(True)
plt.show()