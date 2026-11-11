import matplotlib.pyplot as plt
import numpy as np


# Dữ liệu
x = np.array([
    [73.5, 75, 76.5, 79, 81.5, 82.5, 84,
     85, 86.5, 87.5, 89, 90, 91.5]
]).T

y = np.array([
    [1.49, 1.50, 1.51, 1.54, 1.58, 1.59, 1.60,
     1.62, 1.63, 1.64, 1.66, 1.67, 1.68]
]).T


# =========================================================
# Tính b1 và b0
# =========================================================
def calculateb1b0(x, y):

    # Tính trung bình
    xbar = np.mean(x)
    ybar = np.mean(y)

    x2bar = np.mean(x ** 2)
    xybar = np.mean(x * y)

    # Tính b1
    b1 = (xbar * ybar - xybar) / (xbar ** 2 - x2bar)

    # Tính b0
    b0 = ybar - b1 * xbar

    return b1, b0


# Tính b1, b0
b1, b0 = calculateb1b0(x, y)

print("Trung bình X =", np.mean(x))
print("Trung bình Y =", np.mean(y))

print("Độ lệch chuẩn X =", np.std(x))
print("Độ lệch chuẩn Y =", np.std(y))

print("B1 =", b1)
print("B0 =", b0)


# =========================================================
# Hệ số tương quan
# =========================================================
correlation = np.corrcoef(
    x.flatten(),
    y.flatten()
)[0, 1]

print("Tương quan X và Y =", correlation)


# =========================================================
# Dự đoán Y
# =========================================================
y_predicted = b0 + b1 * x

print("\nY predicted:")
print(y_predicted)


# =========================================================
# Visualize data
# =========================================================
def showGraph(x, y, y_predicted,
              title="",
              xlabel="",
              ylabel=""):

    plt.figure(figsize=(14, 8))

    # Giá trị thực tế
    plt.plot(
        x,
        y,
        'r-o',
        label="value sample"
    )

    # Giá trị dự đoán
    plt.plot(
        x,
        y_predicted,
        'b-x',
        label="predicted value"
    )

    x_min = np.min(x)
    x_max = np.max(x)

    y_min = np.min(y)
    y_max = np.max(y)

    # Trung bình Y
    ybar = np.mean(y)

    plt.axhline(
        ybar,
        linestyle='--',
        linewidth=4,
        label="mean"
    )

    plt.axis([
        x_min * 0.95,
        x_max * 1.05,
        y_min * 0.95,
        y_max * 1.05
    ])

    plt.xlabel(
        xlabel,
        fontsize=16
    )

    plt.ylabel(
        ylabel,
        fontsize=16
    )

    plt.text(
        x_min,
        ybar * 1.01,
        "mean",
        fontsize=16
    )

    plt.legend(
        fontsize=15
    )

    plt.title(
        title,
        fontsize=20
    )

    plt.show()


showGraph(
    x,
    y,
    y_predicted,
    title="Mô hình hồi quy tuyến tính đơn biến",
    xlabel="Giá trị X",
    ylabel="Giá trị Y"
)