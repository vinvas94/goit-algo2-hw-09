import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Визначення функції Сфери
def sphere_function(x):
    return sum(xi ** 2 for xi in x)


# Hill Climbing
def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    current_solution = [random.uniform(low, high) for low, high in bounds]
    current_value = func(current_solution)

    for _ in range(iterations):
        neighbor_solution = [max(min(xi + random.uniform(-epsilon, epsilon), bounds[i][1]), bounds[i][0])
                             for i, xi in enumerate(current_solution)]
        neighbor_value = func(neighbor_solution)

        if neighbor_value < current_value:
            current_solution = neighbor_solution
            current_value = neighbor_value

        if current_value < epsilon:
            break

    return current_solution, current_value


# Random Local Search
def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    best = [random.uniform(b[0], b[1]) for b in bounds]
    best_value = func(best)

    for _ in range(iterations):
        candidate = [random.uniform(b[0], b[1]) for b in bounds]
        candidate_value = func(candidate)

        if candidate_value < best_value:
            best = candidate
            best_value = candidate_value

        if best_value < epsilon:
            break

    return best, best_value


# Simulated Annealing
def simulated_annealing(func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6):
    current = [random.uniform(b[0], b[1]) for b in bounds]
    current_value = func(current)

    for _ in range(iterations):
        temp *= cooling_rate
        if temp < epsilon:
            break

        neighbor = [max(min(current[i] + random.uniform(-0.1, 0.1), bounds[i][1]), bounds[i][0])
                    for i in range(len(bounds))]
        neighbor_value = func(neighbor)

        if neighbor_value < current_value or random.uniform(0, 1) < math.exp((current_value - neighbor_value) / temp):
            current, current_value = neighbor, neighbor_value

    return current, current_value


if __name__ == "__main__":
    # Межі для функції (двовимірний випадок)
    bounds = [(-5, 5), (-5, 5)]  

    # Виконання алгоритмів
    algorithms = {
        "Hill Climbing": hill_climbing,
        "Random Local Search": random_local_search,
        "Simulated Annealing": simulated_annealing
    }

    results = {}

    for name, algorithm in algorithms.items():
        solution, value = algorithm(sphere_function, bounds)
        results[name] = (solution, value)
        print(f"\n=== {name} ===")
        print(f"Оптимальне значення: {solution}, Функція: {value:.6f}")

    # Створення сітки значень для побудови поверхні
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    Z = X**2 + Y**2

    # Побудова графіка
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    # Додавання знайдених точок
    colors = {"Hill Climbing": "red", "Random Local Search": "blue", "Simulated Annealing": "purple"}
    
    for name, (solution, _) in results.items():
        ax.scatter(solution[0], solution[1], sphere_function(solution), color=colors[name], s=100, label=name)

    # Налаштування графіка
    ax.set_title("Функція Сфери з позначеними точками оптимізації")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")
    ax.legend()

    # Показати графік
    plt.show()
