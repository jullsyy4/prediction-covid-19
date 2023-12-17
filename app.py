from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from markupsafe import Markup
from sklearn.linear_model import LinearRegression
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import mean_squared_error, mean_absolute_error

matplotlib.use('agg')

app = Flask(__name__)

# Завантаження даних
data = pd.read_csv('D:\STUDY\Kr\covid_19_data1.csv', parse_dates=['ObservationDate'])

# Вибір потрібних стовпців
data = data[['ObservationDate', 'Province/State', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered']]

# Конвертація стовпця "ObservationDate" в формат дати
data['ObservationDate'] = pd.to_datetime(data['ObservationDate'], format='%m/%d/%Y')

# Тимчасове значення
test = data.copy()
predictions = data.copy()
data_grouped = data.copy()

def calc_prediction():
    global test
    global predictions

    # Розбиття даних на навчальний та тестовий набори
    train_size = int(len(data_grouped) * 0.8)
    train, test = data_grouped[:train_size], data_grouped[train_size:]

    # Побудова моделі регресії
    model = LinearRegression()

    # Перетворення range в NumPy array та решейп
    X_train, y_train = np.array(range(len(train))), train['Confirmed'].values.reshape(-1, 1)
    X_train = X_train.reshape(-1, 1)

    # Fit the model
    model.fit(X_train, y_train)

    # Перетворення range в NumPy array та решейп
    X_test, y_test = np.array(range(len(train), len(data_grouped))), test['Confirmed'].values.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    # Прогнозування на тестовому наборі
    predictions = model.predict(X_test)

calc_prediction()

# Функція оцінки якості прогнозування
def evaluate_model(y_true, y_pred):
    # Обчислення RMSE та MAE
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)

    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')

    return rmse, mae

evaluate_model(test['Confirmed'], predictions)

@app.route('/')
def index():
    # Отримання списку унікальних країн
    unique_countries = data['Country/Region'].unique()

    return render_template('index.html', unique_countries=unique_countries)

@app.route('/predict', methods=['POST'])
def predict():
    return render_template('result.html', prediction=predictions[0][0])

global_stats_all = data.groupby('ObservationDate').sum().reset_index()

@app.route('/show_forecast', methods=['GET'])
def show_forecast():
    global data_grouped
    selected_country = request.args.get('country', 'all')

    # Завантаження даних з урахуванням вибраної країни
    if selected_country == 'all':
        data_grouped = data.groupby('ObservationDate').sum().reset_index()
    else:
        data_grouped = data[data['Country/Region'] == selected_country].groupby('ObservationDate').sum().reset_index()

    # Перетворення даних для графіка
    plt.figure(figsize=(12, 6))
    plt.plot(data_grouped['ObservationDate'], data_grouped['Confirmed'], label='Confirmed Cases')

    # Додавання прогнозованих значень, якщо вони доступні
    calc_prediction()
    plt.plot(test['ObservationDate'], predictions, label='Predictions', linestyle='--')

    plt.title(f'COVID-19 Confirmed Cases - {selected_country if selected_country != "all" else "Worldwide"}')
    plt.xlabel('Date')
    plt.ylabel('Confirmed Cases')
    plt.legend()

    # Збереження графіка в байтовий об'єкт
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plt.close()

    # Кодування графіка у base64
    img_str = "data:image/png;base64," + base64.b64encode(img_data.read()).decode()

    # Використання Markup для безпечного вставлення HTML-коду у шаблон
    img_markup = Markup('<img src="{}" alt="Forecast Chart" class="center">'.format(img_str))

    return render_template('result.html',
                           prediction=predictions[0][0] if 'predictions' in globals() else None,
                           forecast_chart=img_markup, global_stats=global_stats_all)


@app.route('/show_evaluation', methods=['GET'])
def show_evaluation():
    # Виклик функції для оцінки якості прогнозування
    rmse, mae = evaluate_model(test['Confirmed'], predictions)

    return render_template('evaluation.html', rmse=rmse, mae=mae)

@app.route('/show_global_stats', methods=['GET'])
def show_global_stats():
    global_stats = data.groupby(['ObservationDate', 'Country/Region']).sum().reset_index()
    unique_countries = data['Country/Region'].unique()

    # Отримання вибраної країни з параметрів запиту
    selected_country = request.args.get('country', 'Worldwide')

    if selected_country == 'Worldwide':
        # Виведення глобальної статистики, якщо не вибрано конкретну країну
        global_stats_all = data.groupby('ObservationDate').sum().reset_index()
        return render_template('global_stats.html',
                               global_stats=global_stats_all, unique_countries=unique_countries, selected_country=selected_country)
    else:
        # Виведення статистики для вибраної країни
        selected_country_stats = global_stats[global_stats['Country/Region'] == selected_country]
        return render_template('global_stats.html',
                               global_stats=selected_country_stats, unique_countries=unique_countries, selected_country=selected_country)

if __name__ == '__main__':
    app.run(debug=True)