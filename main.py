from helpers import *
from scipy.stats import t, f


def main():
    X, Y, count, x_count = init_data("data/test.csv")

    cor_matrix = pair_correlation_matrix(X, Y)

    # Коэффициенты уравнения регрессии
    coefs = get_coefs(X, Y)

    # векторы прогноза и остатков регрессии
    forecast, residuals = get_forecast_and_residuals(X, Y, coefs)

    # Матрица отклонений Y от своего среднего значения
    y = np.matrix(np.array(Y.transpose()) - np.mean(Y)).transpose()

    # Вычислим коэффициент детерминации (Тесность определим благодоря шкале Чеддока)
    Rp2 = 1 - (residuals.transpose()*residuals).item(0, 0)/(y.transpose()*y).item(0, 0)

    # критерий Фишера или F- статистика
    F = (Rp2*(count-x_count-1))/((1-Rp2)*x_count)
    Fcrit = f.ppf(q=1-0.05, dfn=x_count, dfd=count-x_count-1)

    # Рассчитаем стандартную ошибку регрессии S
    S = get_standard_error(residuals, count, x_count)

    # Рассчитаем стандартные ошибки коэффициентов регрессии
    Sa = get_gerress_coefs_errors(X, S, count, x_count)

    # Критерий стьюдента
    t_stats = get_t_stat(coefs, Sa)
    # Находим распределение Стьюдента
    alpha = t.ppf(1 - 0.05, count - x_count - 1)

    # Доверительные интервалы для коэффициентов регрессии
    coef_av_right, coef_av_left = get_confidence_interval_regression_coefs(coefs, Sa, alpha)

    # Доверительные интервалы для всех наблюдений
    Y_av_p, Y_av_m = get_confidence_interval_Y(X, forecast, S, alpha)

    plot_predicted_data(Y, forecast, Y_av_p, Y_av_m)



if __name__ == '__main__':
    main()
