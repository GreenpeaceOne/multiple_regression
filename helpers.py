import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def init_data(filename):
    reader = csv.reader(open(filename, "r"), delimiter=",")
    x = list(reader)
    result = np.matrix(x).astype("float")

    Y = result[:, 0]
    X = result.copy()
    X[:, 0] = 1

    count = X.shape[0]
    x_count = X.shape[1] - 1

    return X, Y, count, x_count


# Получить значения для уравнения регрессии. Сначала возвращается Y, X1, X2 ... Xn
def get_coefs(X, Y):
    Xt = X.copy().transpose()

    # XTX примет вид
    mul = Xt * X

    inverse = np.linalg.inv(mul.copy())

    return inverse * (Xt * Y)


# векторы прогноза и остатков регрессии
def get_forecast_and_residuals(X, Y, coefs):
    forecast = []
    prod = coefs.item(0, 0)
    for row in X:
        coef_row = np.array(coefs.copy().transpose()[:, 1:])
        row_of_x = np.array(row[:, 1:])
        s = coef_row * row_of_x
        forecast.append(sum(s[0]) + prod)

    forecast = np.array(forecast)

    Y_arr = np.array(Y.transpose())
    errors = Y_arr - forecast

    return np.matrix(forecast).transpose(), np.matrix(errors).transpose()


# Рассчитаем стандартную ошибка регрессии S
def get_standard_error(residuals, count, x_count):
    residuals_arr = np.array(residuals)
    sq_residuals_arr = residuals_arr ** 2
    s = sum(sq_residuals_arr)[0]

    return np.abs(np.sqrt(s / (count - x_count - 1)))


# Рассчитаем стандартные ошибки коэффициентов регрессии
def get_gerress_coefs_errors(X, S, count, x_count):
    inverse = np.abs(np.linalg.inv(X.copy().transpose() * X))
    D = S * np.sqrt(inverse)

    return np.matrix(np.diagonal(D)).transpose()


def get_t_stat(coefs, Sa):
    t_stats = [np.abs(coefs.item(i, 0)) / Sa.item(i, 0) for i in range(coefs.shape[0])]

    return np.matrix(t_stats).transpose()


def get_confidence_interval_regression_coefs(coefs, Sa, alpha):
    # Считаем доверительные интервалы
    coef_av_right = []
    coef_av_left = []
    for i in range(coefs.shape[0]):
        coef_av_right.append(coefs.item(i, 0) + Sa.item(i, 0) * alpha)
        coef_av_left.append(coefs.item(i, 0) - Sa.item(i, 0) * alpha)

    return np.matrix(coef_av_right).transpose(), np.matrix(coef_av_left).transpose()


def get_confidence_interval_Y(X, forecast, S, alpha):
    inverse = np.linalg.inv(X.copy().transpose() * X)

    # Считаем стандартные ошибки среднего значения Y
    errors_y = [S * np.sqrt(((row * inverse) * row.transpose()).item(0, 0)) for row in X]
    errors_y = np.matrix(errors_y).transpose()

    # Считаем доверительные интервалы
    Y_av_p = []
    Y_av_m = []
    for i in range(errors_y.shape[0]):
        Y_av_p.append(forecast.item(i, 0) + errors_y.item(i, 0) * alpha)
        Y_av_m.append(forecast.item(i, 0) - errors_y.item(i, 0) * alpha)

    return np.matrix(Y_av_p).transpose(), np.matrix(Y_av_m).transpose()


def plot_predicted_data(Y, forecast, Y_av_p, Y_av_m):
    p0 = (np.array(Y.transpose())[0])
    p1 = (np.array(forecast.transpose())[0])
    p2 = (np.array(Y_av_p.transpose())[0])
    p3 = (np.array(Y_av_m.transpose())[0])

    fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(range(len(p1)), p0, 'b', label='Настоящее значение', marker ='o')
    ax.plot(range(len(p1)), p1, 'r', label='Предсказанное значения')
    ax.plot(range(len(p2)), p2, 'k', label='Верхний доверительный интервал', linestyle='--')
    ax.plot(range(len(p3)), p3, 'k', label='Нижний доверительный интервал', linestyle='--')
    plt.title("Предсказанное значение с доверительными интервалами")
    plt.xlabel("Порядковый номер наблюдения")
    plt.ylabel("Цена")
    # show a legend on the plot
    plt.legend()
    plt.show()


def corrcoef_loop(matrix):
    rows, cols = matrix.shape[0], matrix.shape[1]
    r = np.ones(shape=(rows, rows))
    p = np.ones(shape=(rows, rows))
    for i in range(rows):
        for j in range(i + 1, rows):
            r_, p_ = pearsonr(matrix[i], matrix[j])  # Линейный коэффициент корреляции
            r[i, j] = r[j, i] = r_
            p[i, j] = p[j, i] = p_
    return r, p


def pair_correlation_matrix(X, Y):
    dframe = X.copy()
    dframe[:, 0] = Y
    matrix, _ = corrcoef_loop(np.array(dframe.transpose()))

    return matrix
