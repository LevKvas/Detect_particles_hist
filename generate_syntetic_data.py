import pandas as pd
import numpy as np
from scipy import stats
import random


def generate_synthetic_distributions(num_distributions, base_pressure=3):
    """
    Генерирует синтетические распределения частиц по эксцентриситетам и площадям

    Parameters:
    num_distributions (int): количество распределений для генерации
    base_pressure (int): базовое давление для именования файлов

    Returns:
    dict: словарь с данными для каждого распределения
    """

    distributions = {}

    for i in range(num_distributions):
        pressure = base_pressure + i * 20  # Увеличиваем давление для каждого распределения

        # Генерация распределения по эксцентриситетам
        eccentricity_data = generate_eccentricity_distribution()

        # Генерация распределения по площадям (зависит от давления)
        square_data = generate_square_distribution(pressure)

        distributions[f"{pressure}_torr"] = {
            'eccentricity': eccentricity_data,
            'square': square_data
        }

    return distributions


def generate_eccentricity_distribution():
    """Генерирует синтетическое распределение по эксцентриситетам"""

    # Базовые центры бинов (как в исходных данных)
    bin_centers = [0.0333, 0.1000, 0.1667, 0.2333, 0.3000, 0.3667, 0.4333,
                   0.5000, 0.5667, 0.6333, 0.7000, 0.7667, 0.8333, 0.9000, 0.9667]

    # Создаем бета-распределение (подходит для значений 0-1)
    # Параметры подобраны для получения пика в районе 0.7-0.9
    x = np.linspace(0, 1, 1000)
    pdf = stats.beta.pdf(x, 2, 1.5) + 0.5 * stats.beta.pdf(x, 5, 3)

    # Нормализуем и масштабируем
    pdf_normalized = pdf / pdf.max()

    # Сэмплируем значения для каждого бина
    particle_counts = []
    total_particles = random.randint(20, 40)  # Общее количество частиц

    for center in bin_centers:
        # Находим ближайшую точку в PDF
        idx = np.argmin(np.abs(x - center))
        probability = pdf_normalized[idx]

        # Добавляем случайный шум
        noise = random.uniform(0.7, 1.3)
        count = int(probability * total_particles * noise / 3)

        # Гарантируем, что счетчик неотрицательный
        count = max(0, count)
        particle_counts.append(count)

    # Нормализуем общее количество
    current_total = sum(particle_counts)
    if current_total > 0:
        scale_factor = total_particles / current_total
        particle_counts = [int(count * scale_factor) for count in particle_counts]

    return list(zip(bin_centers, particle_counts))


def generate_square_distribution(pressure):
    """Генерирует синтетическое распределение по площадям"""

    # Базовые центры бинов зависят от давления
    # При более высоком давлении ожидаем большие площади
    base_area = 100 + pressure * 5

    # Создаем центры бинов
    num_bins = 15
    bin_centers = []
    for i in range(num_bins):
        center = base_area + i * (150 + pressure * 3)
        bin_centers.append(round(center, 2))

    # Создаем логнормальное распределение (типично для размеров частиц)
    shape, location, scale = 0.8, 0, base_area / 3
    samples = stats.lognorm.rvs(shape, location, scale, size=1000)

    # Создаем гистограмму
    counts, bin_edges = np.histogram(samples, bins=num_bins,
                                     range=(bin_centers[0] - 50, bin_centers[-1] + 50))

    particle_counts = []
    total_particles = random.randint(25, 35)

    for i in range(num_bins):
        # Масштабируем и добавляем шум
        count = int(counts[i] * total_particles / len(samples) * random.uniform(0.5, 2.0))

        # Гарантируем, что большинство частиц в первых нескольких бинах
        if i > 3:
            count = random.randint(0, 2)

        particle_counts.append(count)

    # Убеждаемся, что есть хотя бы одна частица в первом бине
    if sum(particle_counts[:3]) == 0:
        particle_counts[0] = random.randint(1, 3)

    # Добавляем редкие большие частицы
    if random.random() < 0.7:  # 70% chance
        large_bin = random.randint(10, 14)
        particle_counts[large_bin] = 1

    return list(zip(bin_centers, particle_counts))


def save_to_excel(distributions, filename="synthetic_particle_distributions.xlsx"):
    """Сохраняет распределения в Excel файл"""

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for pressure, data in distributions.items():
            # Создаем DataFrame для эксцентриситетов
            ecc_df = pd.DataFrame(data['eccentricity'],
                                  columns=['Bin_Center', 'Particle_Count'])

            # Создаем DataFrame для площадей
            sq_df = pd.DataFrame(data['square'],
                                 columns=['Bin_Center', 'Particle_Count'])

            # Сохраняем в разные листы
            ecc_df.to_excel(writer, sheet_name=f'{pressure}_eccentricity', index=False)
            sq_df.to_excel(writer, sheet_name=f'{pressure}_square', index=False)

    print(f"Данные сохранены в файл: {filename}")


def main():
    """Основная функция"""

    print("Генератор синтетических распределений частиц")
    print("=" * 50)

    # Запрашиваем количество распределений у пользователя
    try:
        num_distributions = int(input("Введите количество распределений для генерации: "))
        base_pressure = int(input("Введите начальное давление (torr) [по умолчанию 3]: ") or "3")
    except ValueError:
        print("Ошибка: введите целое число")
        return

    if num_distributions <= 0:
        print("Количество распределений должно быть положительным числом")
        return

    # Генерируем распределения
    print(f"\nГенерация {num_distributions} распределений...")
    distributions = generate_synthetic_distributions(num_distributions, base_pressure)

    # Сохраняем в Excel
    filename = input(
        "Введите имя файла для сохранения [по умолчанию: synthetic_distributions.xlsx]: ") or "synthetic_distributions.xlsx"
    if not filename.endswith('.xlsx'):
        filename += '.xlsx'

    save_to_excel(distributions, filename)

    # Выводим статистику
    print(f"\nСтатистика сгенерированных распределений:")
    print("-" * 40)
    for pressure, data in distributions.items():
        total_ecc = sum(count for _, count in data['eccentricity'])
        total_sq = sum(count for _, count in data['square'])
        print(f"{pressure}: {total_ecc} частиц (эксцентриситет), {total_sq} частиц (площадь)")


if __name__ == "__main__":
    main()