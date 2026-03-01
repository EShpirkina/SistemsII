import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(42)  # Для воспроизводимости

# Зададим параметры двумерного нормального распределения
true_mean = np.array([1.0, 2.0])  # Вектор средних (μ)
true_cov = np.array([[2.0, 0.8],  # Ковариационная матрица (Σ)
                     [0.8, 1.0]])


def conditional_distribution_gaussian(x_fixed, fixed_index, true_mean, true_cov):
    """
    Вычисляет параметры условного одномерного нормального распределения
    p(x_i | x_j), где j = fixed_index.
    Параметры:
        x_fixed (float): Значение зафиксированной переменной x_j.
        fixed_index (int): Индекс зафиксированной переменной (0 или 1).
        true_mean (np.array): Вектор средних исходного 2D распределения.
        true_cov (np.array): Ковариационная матрица исходного 2D распределения.
    Возвращает:
        tuple: (cond_mean, cond_var) - среднее и дисперсия условного распределения.
    """
    # Индекс переменной, которую мы семплируем
    target_index = 1 - fixed_index

    # Разбиваем параметры на блоки
    # Переменная 1 (целевая), Переменная 2 (зафиксированная)
    mu_i = true_mean[target_index]
    mu_j = true_mean[fixed_index]
    sigma_ii = true_cov[target_index, target_index]
    sigma_ij = true_cov[target_index, fixed_index]
    sigma_jj = true_cov[fixed_index, fixed_index]

    # Формулы для условного распределения в нормальном случае
    cond_mean = mu_i + (sigma_ij / sigma_jj) * (x_fixed - mu_j)
    cond_var = sigma_ii - (sigma_ij ** 2) / sigma_jj

    return cond_mean, cond_var

def gibbs_sampler_gaussian(true_mean, true_cov, init_state, n_samples, burn_in=500):
    """
    Генерирует выборку из 2D нормального распределения с помощью семплирования по Гиббсу.

    Параметры:
        true_mean, true_cov: Параметры целевого распределения.
        init_state (np.array): Начальное состояние цепи (например, [0, 0]).
        n_samples (int): Количество сохраняемых сэмплов после "прогрева".
        burn_in (int): Длина "прогрева" цепи (отбрасывается).

    Возвращает:
        np.array: Массив сэмплов формы (n_samples, 2).
    """
    samples = np.zeros((n_samples + burn_in, 2))
    current_state = init_state.copy()

    for t in range(n_samples + burn_in):
        # Обновляем x1, условное относительно x2
        cond_mean_1, cond_var_1 = conditional_distribution_gaussian(
            current_state[1], fixed_index=1, true_mean=true_mean, true_cov=true_cov
        )
        current_state[0] = np.random.normal(cond_mean_1, np.sqrt(cond_var_1))
        # Обновляем x2, условное относительно нового x1
        cond_mean_2, cond_var_2 = conditional_distribution_gaussian(
            current_state[0], fixed_index=0, true_mean=true_mean, true_cov=true_cov
        )
        current_state[1] = np.random.normal(cond_mean_2, np.sqrt(cond_var_2))
        samples[t] = current_state

    # Отбрасываем первые burn_in сэмплов ("прогрев")
    return samples[burn_in:]

# Параметры семплирования
init_state = np.array([0.0, 0.0])
n_samples = 5000
burn_in = 1000

# Запуск семплировщика Гиббса
gibbs_samples = gibbs_sampler_gaussian(true_mean, true_cov, init_state, n_samples, burn_in)
# Генерация эталонной выборки прямой функцией
direct_samples = multivariate_normal.rvs(mean=true_mean, cov=true_cov, size=n_samples)

# Построение графиков
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Сравнение сэмплов на плоскости
axes[0, 0].scatter(direct_samples[:, 0], direct_samples[:, 1], alpha=0.5, s=10, label='Прямая выборка', color='blue')
axes[0, 0].scatter(gibbs_samples[:, 0], gibbs_samples[:, 1], alpha=0.5, s=10, label='Выборка Гиббса', color='red',
                   marker='x')
axes[0, 0].set_xlabel('x1')
axes[0, 0].set_ylabel('x2')
axes[0, 0].set_title('Сравнение выборок на плоскости')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Маргинальное распределение по x1
axes[0, 1].hist(direct_samples[:, 0], bins=40, density=True, alpha=0.7, label='Прямая', color='blue')
axes[0, 1].hist(gibbs_samples[:, 0], bins=40, density=True, alpha=0.7, label='Гиббс', color='red', histtype='step',
                linewidth=2)
axes[0, 1].set_xlabel('x1')
axes[0, 1].set_ylabel('Плотность')
axes[0, 1].set_title('Маргинальное распределение: x1')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Маргинальное распределение по x2
axes[1, 0].hist(direct_samples[:, 1], bins=40, density=True, alpha=0.7, label='Прямая', color='blue')
axes[1, 0].hist(gibbs_samples[:, 1], bins=40, density=True, alpha=0.7, label='Гиббс', color='red', histtype='step',
                linewidth=2)
axes[1, 0].set_xlabel('x2')
axes[1, 0].set_ylabel('Плотность')
axes[1, 0].set_title('Маргинальное распределение: x2')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Траектория цепи для первых 100 сэмплов
axes[1, 1].plot(gibbs_samples[:100, 0], gibbs_samples[:100, 1], 'o-', markersize=4, linewidth=0.8, alpha=0.7)
axes[1, 1].scatter(gibbs_samples[0, 0], gibbs_samples[0, 1], color='green', s=100, zorder=5, label='Начало')
axes[1, 1].scatter(true_mean[0], true_mean[1], color='black', s=150, marker='*', zorder=5, label='Среднее (μ)')
axes[1, 1].set_xlabel('x1')
axes[1, 1].set_ylabel('x2')
axes[1, 1].set_title('Траектория цепи Гиббса (первые 100 точек)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


