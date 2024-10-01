import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram, linkage

def main():
    # Чтение данных digit.dat
    data = pd.read_csv('10_Цифры/digit.dat', sep=';')

    # Очистка данных от лишних пробелов
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Преобразование категориальных данных (замена текста на цифры)
    data.replace({
        'ZERO': 0,
        'ONE': 1,
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9
    }, inplace=True)

    # Преобразование столбца 'A' в числовой тип данных
    data['A'] = pd.to_numeric(data['A'], errors='coerce')

    # Выбор переменных
    selected_columns = ['B', 'C', 'D', 'E', 'F', 'G', 'H']  # Var1,..., Var7

    data_X = data[selected_columns]
    data_Y = data['A']  # Digit (зависимая переменная)

    # Стандартизация данных
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_X)

    # Иерархическая кластеризация
    linkage_matrix = linkage(data_scaled, method='ward', metric='euclidean')

    # Визуализация дендрограммы
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix)
    plt.title('Дендрограмма')
    plt.xlabel('Точки данных')
    plt.ylabel('Расстояние')
    plt.show()

    # Определение оптимального числа кластеров (метод "локоть")
    distances = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=0, n_init=12, algorithm='lloyd')
        kmeans.fit(data_scaled)
        distances.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), distances, marker='o', linestyle='-')
    plt.title('Метод локтя')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Сумма квадратов внутри кластера')
    plt.show()

    # Кластеризация методом k средних для 5 кластеров
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, algorithm='lloyd')
    kmeans.fit(data_scaled)

    # Многомерное шкалирование для визуализации
    mds = MDS(n_components=2, random_state=0)
    data_2d = mds.fit_transform(data_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=kmeans.labels_, cmap='rainbow')
    plt.title('Визуализация кластеризации (Многомерное шкалирование)')
    plt.xlabel('Количество зажженных горизонтальных сегментов')
    plt.ylabel('Количество зажженных вертикальных сегментов')
    plt.show()

    # Интерпретация результатов кластеризации
    data['Cluster'] = kmeans.labels_
    numeric_columns = selected_columns + ['A']
    cluster_means = data.groupby('Cluster')[numeric_columns].mean()
    print(cluster_means)

if __name__ == "__main__":
    main()

