import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


def approximate_precision_curve(x, y):
    """Аппроксимирует кривую Precision параболой y = ax^2 + bx + c."""
    coeffs = np.polyfit(x, y, deg=3)  # Найти коэффициенты a, b, c
    poly_func = np.poly1d(coeffs)  # Создать полином

    return poly_func, coeffs


model = YOLO('dotav1_det/weights/best.pt')
res = model.val(data='DOTAv1.yaml')

confidence = res.curves_results[2][0]
precision = res.curves_results[2][1][0]

# Пример данных (замените на реальные значения)
# x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  # Например, Recall
# y = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4])  # Precision

x = confidence
y = precision

# Аппроксимация
poly_func, coeffs = approximate_precision_curve(x, y)

# Визуализация
x_fit = np.linspace(min(x), max(x), 100)
y_fit = poly_func(x_fit)

plt.scatter(x, y, label="Исходные данные")
plt.plot(x_fit, y_fit, label="Аппроксимация", color="red")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid()
# plt.show()
plt.savefig('show.jpg')

print(f"Коэффициенты аппроксимации: a={coeffs[0]}, b={coeffs[1]}, c={coeffs[2]}")
