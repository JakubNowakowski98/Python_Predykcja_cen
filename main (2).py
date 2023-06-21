import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Activation, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import datetime as dt
import urllib.request

def projekt():
    # Funkcja sprawdzająca, czy jest połączenie internetowe
    def connect(host='http://google.com'):
        try:
            urllib.request.urlopen(host)
            return True
        except:
            return False

    if connect():
        data = yf.download("CL=F", start=(dt.date.today() - dt.timedelta(days=90)).strftime('%Y-%m-%d'),
                           end=dt.date.today().strftime('%Y-%m-%d'))
    else:
        data = pd.read_csv('prices.csv')

    # Pobierz dane o cenach ropy ze zbioru yfinance
    data = yf.download("CL=F", start=(dt.date.today() - dt.timedelta(days=90)).strftime('%Y-%m-%d'), end=dt.date.today().strftime('%Y-%m-%d'))

    # Close = na koniec dnia, Open na początek dnia, tak to rozumiem
    price_column = 'Close'

    # Wybierz kolumnę z cenami ropy 'Close'
    price = data[price_column].values

    # Normalizacja cen
    price = price.reshape(-1, 1)

    # Ustaw wartości cen na zakres 0 1, żeby model się lepiej uczył
    scaler = MinMaxScaler(feature_range=(0, 1))
    price = scaler.fit_transform(price)

    # Podział danych na zestawy uczące i testowe
    train_size = int(len(price) * 0.8)
    test_size = len(price) - train_size
    train_data, test_data = price[0:train_size,:], price[train_size:len(price),:]

    # Przygotuj dane dla sieci LSTM
    def prepare_data(data, look_back=10):
        X, Y = [], []
        for i in range(len(data)-look_back-1):
            a = data[i:(i+look_back), 0]
            X.append(a)
            Y.append(data[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 7
    train_X, train_Y = prepare_data(train_data,
                                    look_back)
    test_X, test_Y = prepare_data(test_data, look_back)

    train_X = np.reshape(train_X,
                         (train_X.shape[0],
                          1,
                          train_X.shape[1]))
    test_X = np.reshape(test_X,
                        (test_X.shape[0],
                         1,
                         test_X.shape[1]))

    # Stwórz sieć neuronową
    model = Sequential()
    model.add(LSTM(256, input_shape=(1, look_back), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='RMSprop')

    # Trenuj sieć neuronową
    # Dla 200 epok wychodzą mi najlepsze wyniki, było testowane dla 100, 300, 500 i 1000 epok.
    model.fit(train_X, train_Y,
              epochs=200,
              batch_size=1,
              verbose=2)

    # Dokonaj predykcji cen ropy naftowej
    predicted_price = model.predict(test_X)

    # Wymiary o wartości 1 robi tak, że to nie jest osobny wymiar, lub tablica w tablicy traktuje jako jedna tablica.
    y_pred = np.squeeze(predicted_price)

    # Przywracam wartości cen z zakresu 0 1 na stan początkowy
    predicted_price = scaler.inverse_transform(predicted_price)
    predicted_price = np.squeeze(predicted_price)
    test_Y = scaler.inverse_transform([test_Y])
    y_pred = scaler.inverse_transform([y_pred])
    test_Y = np.squeeze(test_Y)
    y_pred = np.squeeze(y_pred)

    from matplotlib import pyplot as plt
    x = np.arange(y_pred.shape[0])
    plt.plot(x, y_pred, label = 'Predykcje')
    plt.plot(x, test_Y, label = 'Wartości prawdziwe')
    plt.legend()
    plt.savefig('predicitions3months.png')
    plt.show()


    diff_pred = np.diff(y_pred.squeeze())
    diff_true = np.diff(test_Y)
    diff_pred = np.sign(diff_pred)
    diff_true = np.sign(diff_true)

    # Oceń wyniki predykcji
    from sklearn.metrics import confusion_matrix, accuracy_score

    print(confusion_matrix(diff_true, diff_pred))
    print("Dokładność:", accuracy_score(diff_true, diff_pred))

    # Ocena dokładności predykcji na podstawie wybranych metryk z ostatnich 90 dni

    print("------ Ocena dokładności predykcji na podstawie wybranych metryk ------")

    sredni_blad_kwadratowy = mean_squared_error(np.squeeze(test_Y),
                                                predicted_price)
    print("Średni błąd kwadratowy(MSE) wynosi: ", sredni_blad_kwadratowy)

    sredni_blad_bezwzgledny = mean_absolute_error(np.squeeze(test_Y),
                                                  predicted_price)
    print("Średni błąd bezwzględny(MAE) wynosi: ", sredni_blad_bezwzgledny)

    from scipy.stats import pearsonr
    wsp_korelacji, _ = pearsonr(np.squeeze(test_Y),
                                predicted_price)
    print("Współczynnik korelacji(r) wynosi: ", wsp_korelacji)

    sredni_blad_procentowy = mean_absolute_percentage_error(np.squeeze(test_Y),
                                                            predicted_price)
    print("Średni błąd procentowy(MAPE) wynosi: ", sredni_blad_procentowy)

    wsp_determinacji = r2_score(np.squeeze(test_Y),
                                predicted_price)
    print("Współczynnik determinacji(R^2) wynosi: ", wsp_determinacji)

    #dla nasteopnych 7 dni

    week_X = []
    last = test_X[-1]

    for i in range(7):
        next_price = model.predict(np.reshape(last, (1, 1, look_back)))
        week_X.append(next_price)
        last = np.concatenate((last[:, 1:], next_price), axis=1)


    week_X = np.squeeze(np.array(week_X)).reshape(-1, 1)

    week_X = scaler.inverse_transform(week_X).squeeze()

    week_Y = pd.date_range(start=dt.date.today(),
                           end=dt.date.today() +
                           dt.timedelta(days=6),
                           freq='D')

    week = pd.DataFrame({ 'Data': week_Y, 'Cena': week_X })
    week = week.set_index('Data')

    week.to_csv('predicitionsNext7.csv')

    x = np.arange(week_X.shape[0])
    plt.plot(x, week_X, label = 'Predykcje')
    plt.legend()
    plt.savefig('predicitionsNext7.png')
    plt.show()
    return 0

projekt()

# Ustawiam, aby program wykonywał się codziennie o 10:00, co zpowoduje zaktualizowanie bazy danych i modelu
from apscheduler.schedulers.blocking import BlockingScheduler
harmonogram = BlockingScheduler()
harmonogram.add_job(projekt, 'cron', hour = '10', minute = '00')
harmonogram.start()