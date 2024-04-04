import yfinance as yf
from finta import TA
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from tensorflow.keras.metrics import MeanAbsoluteError
import joblib
from keras.models import load_model
import numpy as np


loaded_model = load_model('model_linear.keras')
scaler = joblib.load('btc_price_scaler.pkl')
stock = 'BTC-USD'
start = '2014-09-17'
end = '2024-03-07'
df = yf.download(stock, start, end)
df.index = pd.to_datetime(df.index)
df.fillna(method='ffill', inplace=True)
date_str = str(input("Input data format-YYYY-MM-DD: "))
n = int(input("Days after: "))
def predict_next_day_price(date_str, loaded_model, df, scaler, sequence_length=100):
    # Преобразование введенной даты в формат datetime
    date = pd.to_datetime(date_str)

    # Находим индекс даты в DataFrame
    date_index = df.index.get_loc(date)

    # Выборка последовательности цен закрытия за последние sequence_length дней перед указанной датой
    start_index = date_index - sequence_length
    end_index = date_index
    last_sequence = df['Close'][start_index:end_index].values

    # Нормализация последовательности
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))

    # Изменение формы последовательности для предсказания
    last_sequence_scaled = last_sequence_scaled.reshape((1, sequence_length, 1))

    # Предсказание цены на следующий день
    predicted_price_scaled = loaded_model.predict(last_sequence_scaled)

    # Обратное масштабирование предсказанной цены
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    return predicted_price[0][0]



def get_price_for_date_and_ndays_later(df, date_str, n):
    """
    Возвращает данные за введенную дату и за n дней после нее.

    :param df: DataFrame с данными, индексированными датами.
    :param date_str: Строка с датой в формате, совместимом с pd.to_datetime.
    :param n: Количество дней после введенной даты.
    :return: Данные за указанную дату и за n дней после нее.
    """
    # Преобразование строки в дату
    date = pd.to_datetime(date_str)

    # Вычисление даты n дней после введенной даты
    date_n_days_later = date + pd.Timedelta(days=n)

    # Извлечение данных за обе даты
    data_for_date = df.loc[df.index == date]
    data_for_date_n_days_later = df.loc[df.index == date_n_days_later]

    return data_for_date, data_for_date_n_days_later


data_for_date, data_for_date_n_days_later = get_price_for_date_and_ndays_later(df, date_str, n)
predicted_next_day_price = predict_next_day_price(date_str, loaded_model, df, scaler, sequence_length=100)
print(f'Predicted price for {date_str}: {predicted_next_day_price}')
print(f"Данные за {n} дней после указанной даты: {data_for_date_n_days_later}")