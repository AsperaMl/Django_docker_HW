import lightgbm as lgb
import pandas as pd
# Загрузка модели
lgb_model = lgb.Booster(model_file='lgb_model1.txt')

columns_after_encoding = [
    'age',  # числовая переменная
    'gender_female', 'gender_male', 'gender_non-binary',  # предположим, есть также non-binary
    'platform_Facebook', 'platform_Instagram', 'platform_YouTube'
]
def make_prediction(model):
    age = 10 #input("Введите возраст:")
    gender = 'male' #input("Введите пол (male/female/none-binary) : ")
    platform = 'Facebook' #input("Введите платформу (например, Instagram, Facebook, Youtube): ")

    # Создание DataFrame на основе введенных данных
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'platform': [platform]
    })

    # Преобразование категориальных переменных с помощью one-hot кодирования
    # Важно использовать тот же метод кодирования, что и при обучении модели

    input_data_processed = pd.get_dummies(input_data)


    # Убедитесь, что входные данные содержат все необходимые столбцы, которые были в обучающем наборе
    # Возможно, потребуется добавить столбцы с нулями для категорий, которых нет среди введенных данных
    # Например:
    for col in columns_after_encoding:
        if col not in input_data_processed.columns:
            input_data_processed[col] = 0
    input_data_processed = input_data_processed[columns_after_encoding]
    input_data_processed = input_data_processed.reindex(columns=columns_after_encoding)
    # Предсказание с помощью модели
    if input_data_processed.shape[1] == 7:  # expected_number_of_features = 7 в вашем случае
        # Теперь можно делать предсказание
        prediction = model.predict(input_data_processed)
    else:
        print(
            f"Ошибка: количество признаков в данных для предсказания ({input_data_processed.shape[1]}) не соответствует ожидаемому ({7}).")
    # Вывод результата
    print(f"Предсказанное время, проведенное на платформе: {prediction[0]}")


# Вызов функции предсказания с передачей обученной модели
make_prediction(lgb_model)