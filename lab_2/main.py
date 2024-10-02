import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

def main():
    
    # Загрузка данных
    data = pd.read_csv('lab_2/10_Гостиницы Чикаго CMAHS Average Daily Rate/Chicago_hotels.csv', delimiter=';')
    
    # Очистка данных: замена запятых на точки и удаление пробелов
    data['x4'] = data['x4'].str.replace(',', '.').str.strip()

    # Преобразование в числовой формат и пропуск некорректных значений
    data['x4'] = pd.to_numeric(data['x4'], errors='coerce')
    data = data.dropna(subset=['x4'])  # Удаление строк с NaN

    # Подготовка временного ряда
    start_date = '1994-01'
    end_date = pd.date_range(start=start_date, periods=len(data), freq='M')  # Создание временного индекса

    # Присваивание временного индекса
    data['date'] = end_date
    data.set_index('date', inplace=True)

    time_series = data['x4']

    # Построение графика исходного временного ряда
    plt.figure(figsize=(12, 6))
    plt.plot(time_series, label='Средняя цена в Metropolitan Area')
    plt.title('Исходный временной ряд: Средняя цена в Metropolitan Area')
    plt.xlabel('Год')
    plt.ylabel('Средняя цена ($)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Декомпозиция ряда на тренд, сезонность и остатки
    result = seasonal_decompose(time_series, model='additive', period=12)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(result.trend)
    plt.title('Тренд')
    plt.xlabel('Год')
    plt.ylabel('Значение')
    plt.grid(True)
    
    plt.subplot(212)
    plt.plot(result.seasonal)
    plt.title('Сезонность')
    plt.xlabel('Год')
    plt.ylabel('Значение')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # SARIMA модель
    p, d, q = 1, 1, 1
    P, D, Q, s = 1, 1, 1, 12
    model = SARIMAX(time_series, order=(p, d, q), seasonal_order=(P, D, Q, s))
    results = model.fit()
    
    # Прогноз на 8 месяцев вперед
    forecast_steps = 8
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_conf_int = forecast.conf_int()
    
    # Построение графика прогноза
    plt.figure(figsize=(12, 6))
    plt.plot(time_series, label='Исходный ряд', color='blue')
    plt.plot(forecast.predicted_mean, label='Прогноз', color='red')
    plt.fill_between(forecast_conf_int.index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3)
    
    plt.title('Прогноз SARIMA на 8 месяцев вперед')
    plt.xlabel('Год')
    plt.ylabel('Средняя цена ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Вывод спрогнозированных значений
    forecast_values = forecast.predicted_mean[-forecast_steps:]
    print('Спрогнозированные значения на 8 месяцев вперед:')
    print(forecast_values)

if __name__ == '__main__':
    main()
