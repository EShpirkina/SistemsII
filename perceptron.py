import numpy as np
import matplotlib.pyplot as plt


class SingleLayerPerceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs #кол-во проходов
        self.weights = None # входной_сигнал = x₁·w₁ + x₂·w₂ + ... + xₙ·wₙ + b
        self.bias = None #свободный член b в уравнении
        self.errors_history = []

    def activation(self, x):
        """Функция активации (ступенчатая функция)"""
        return 1 if x >= 0 else 0

    def predict(self, X):
        """Предсказание для одного образца"""
        linear_output = np.dot(X, self.weights) + self.bias # входной_сигнал = x₁·w₁ + x₂·w₂ + ... + xₙ·wₙ + b
        return self.activation(linear_output)

    def predict_batch(self, X):
        """Предсказание для нескольких образцов"""
        return np.array([self.predict(x) for x in X])

    def train(self, X, y):
        """Обучение перцептрона"""
        n_samples, n_features = X.shape

        # Инициализация весов маленькими случайными значениями
        np.random.seed(42)  # для воспроизводимости результатов
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0

        self.errors_history = []

        for epoch in range(self.epochs):
            errors = 0

            for idx, (x_i, target) in enumerate(zip(X, y)):
                # Прямой проход
                linear_output = np.dot(x_i, self.weights) + self.bias
                prediction = self.activation(linear_output)

                # Вычисление ошибки
                error = target - prediction

                # Обновление весов (правило Хебба с учителем)
                if error != 0:
                    self.weights += self.lr * error * x_i
                    self.bias += self.lr * error
                    errors += 1

            # Сохраняем количество ошибок на этой эпохе
            self.errors_history.append(errors)

            # Ранняя остановка, если ошибок нет
            if errors == 0:
                print(f"Обучение завершено на эпохе {epoch + 1}")
                break

        return self.errors_history

    def accuracy(self, X, y):
        """Вычисление точности модели"""
        predictions = self.predict_batch(X)
        return np.mean(predictions == y)


# Создаем данные для XOR
def create_xor_data():
    """Создает датасет для функции XOR"""
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 0])  # XOR truth table
    return X, y


# Визуализация результатов
def plot_decision_boundary(perceptron, X, y, title="Decision Boundary"):
    """Визуализация разделяющей поверхности"""
    # Создаем сетку для визуализации
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Предсказания для всех точек сетки
    Z = np.array([perceptron.predict(np.array([x, y]))
                  for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    # Рисуем контур
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, s=100, edgecolors='black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.grid(True, alpha=0.3)


# Основная программа
def main():
    print("=" * 60)
    print("Однослойный перцептрон для задачи XOR")
    print("=" * 60)

    # Создаем данные
    X, y = create_xor_data()

    print("\nВходные данные:")
    for i in range(len(X)):
        print(f"  XOR({X[i][0]}, {X[i][1]}) = {y[i]}")

    # Создаем и обучаем перцептрон
    perceptron = SingleLayerPerceptron(learning_rate=0.1, epochs=100)
    errors_history = perceptron.train(X, y)

    # Результаты обучения
    print(f"  Финальные веса: [{perceptron.weights[0]:.4f}, {perceptron.weights[1]:.4f}]")
    print(f"  Финальный bias: {perceptron.bias:.4f}")
    print(f"  Точность на обучающих данных: {perceptron.accuracy(X, y) * 100:.1f}%")

    # Проверка предсказаний
    print("\n" + "=" * 60)
    print("Предсказания модели:")
    for x in X:
        pred = perceptron.predict(x)
        print(f"  XOR{tuple(x)} = {pred} (ожидалось: {y[np.where((X == x).all(axis=1))[0][0]]})")

    # Визуализация
    plt.figure(figsize=(12, 5))

    # График разделяющей поверхности
    plt.subplot(1, 2, 1)
    plot_decision_boundary(perceptron, X, y, "Разделяющая поверхность")

    # График ошибок по эпохам
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(errors_history) + 1), errors_history, 'b-', linewidth=2)
    plt.xlabel('Эпоха')
    plt.ylabel('Количество ошибок')
    plt.title('Ошибки по эпохам обучения')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


# Дополнительная демонстрация проблемы XOR
def demonstrate_xor_problem():
    """Демонстрация того, почему XOR не решается однослойным перцептроном"""

    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ ПРОБЛЕМЫ XOR ДЛЯ ОДНОСЛОЙНОГО ПЕРЦЕПТРОНА")
    print("=" * 60)

    X, y = create_xor_data()

    # Попробуем разные начальные условия
    print("\nПопытка 1: Случайная инициализация")
    p1 = SingleLayerPerceptron(learning_rate=0.1, epochs=50)
    p1.train(X, y)
    print(f"  Точность: {p1.accuracy(X, y) * 100:.1f}%")

    print("\nПопытка 2: Другая скорость обучения")
    p2 = SingleLayerPerceptron(learning_rate=0.5, epochs=50)
    p2.train(X, y)
    print(f"  Точность: {p2.accuracy(X, y) * 100:.1f}%")

    print("\nПопытка 3: Больше эпох")
    p3 = SingleLayerPerceptron(learning_rate=0.1, epochs=200)
    p3.train(X, y)
    print(f"  Точность: {p3.accuracy(X, y) * 100:.1f}%")

    print("\n" + "=" * 60)
    print("ВЫВОД: Однослойный перцептрон НЕ может решить XOR,")
    print("так как XOR не является линейно разделимой функцией.")
    print("Для решения XOR требуется минимум двухслойная сеть.")

# Раскомментируйте для дополнительной демонстрации
demonstrate_xor_problem()