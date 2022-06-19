"""
Лабораторна робота №1
Виконав: Волошенюк Юрій (КМ-93)
Алгоритми однофакторного і двох факторного дисперсійного аналізу.
Мета роботи – здобути практичні навички проведення і аналізу даних однофакторного та двохфакторного дисперсійного аналізу.

Завдання.
Провести дисперсійний аналіз даних, відповідно до варіанту. Визначити при якій довірчій ймовірності виконуються необхідні умови.
Провести двох факторний дисперсійний аналіз даних, відповідно до варіанту.
Визначити при якій довірчій ймовірності виконуються необхідні умови.
За результатами оформити звіт. В звіті відобразити особливості реалізації алгоритмів.
"""
# імпортуємо потрібні для виконання бібліотеки
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f
from scipy import stats
from numpy import mean, square
from prettytable import PrettyTable

df = pd.read_csv('A4.txt', header=None)
df.columns = [f'{i}' for i in range(12)]
print(df)


def graphs_for_parameters():
    figure, ax = plt.subplots(nrows=6, ncols=2, figsize=(35, 15))
    ax = ax.flatten()

    for i, value in df.iteritems():
        ax[int(i)].plot(value, color='red')
        ax[int(i)].set_ylabel(f'A{int(i) + 1}')

    figure.tight_layout()
    plt.show()

# Для всіх параметрів оцінимо основні статистичні параметри - середнє значення параметру,
# дисперсію параметру, моду параметру, медіану параметру,
# коефіцієнт асиметрії параметру, коефіцієнт ексцесу параметру та перевіряємо гіпотезу


def parameter():
    fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(12, 14))
    ax = ax.flatten()
    for idx, value in df.iteritems():
        print(40*'*')
        print(f'Parameter:{int(idx) + 1}')
        print(40*'*')
        value1 = value.mean()
        print("Середнє значення:", round(value1, 4))
        value_disp = value.std()
        print("Дисперсія параметру:", round(value_disp, 4))
        mode = value.mode()[0]
        print("Мода параметру:", round(mode, 4))
        median = value.median()
        print("Медіана параметру:", round(median, 4))
        asymmetric = value.skew()
        print("Коефіцієнт асиметрії:", round(asymmetric, 4))
        excess = value.kurtosis()
        print("Коефіцієнт ексцесу:", round(excess, 4))
        test = stats.shapiro(value)[1]

        if test > 1e-5:
            print("Розподіл нормальний")

        else:
            print("Розподіл не нормальний")

        value.hist(ax=ax[int(idx)], bins=20, color='m')
        ax[int(idx)].set(title=f'A{int(idx) + 1}')

    fig.tight_layout()
    plt.show()


# Побудуємо таблицю однофакторного експерименту, для кажного фактору
# За припущенням дисперсійного аналізу - повинна мати місце рівність дисперсій
# Знайдемо оцінку дисперсії, що характеризує зміни параметра, пов'язані з фактором
def factor1(n=df.shape[0], k=df.shape[1], g_alpha=0.153, alpha=0.95, db_sum=0, db_sum_2=0, sum_2=0):
    table = PrettyTable()
    table.add_column('factor', ['Si^2'])
    s_list, x_i = [], []
    for i in df.columns:
        si_2 = (1 / (n - 1) * (pow(df[i], 2).agg('sum') - (1 / n) * pow(df[i].agg('sum'), 2)))
        s_list.append(si_2)
        table.add_column(f'A{str(int(i) + 1)}', [round(si_2, 2)])
    print(table)
    g = max(s_list) / sum(s_list)
    if g >= g_alpha:
        print("Нульова гіпотеза про рівність дисперсій відхиляється.")
    else:
        print("Нульова гіпотеза про рівність дисперсій приймається.")

    for i in df.columns:
        db_sum += pow(df[i], 2).agg('sum')
        db_sum_2 += pow(df[i].agg('sum'), 2)
        sum_2 += df[i].agg('sum')

        value = df[i].agg('mean')
        x_i.append(value)

    s_0 = (1 / (k * (n - 1)) * (db_sum - (1 / n) * db_sum_2))
    print("Оцінка дисперсії, що характеризує розсіювання поза фактором: ", round(s_0, 4))
    s_2 = (db_sum - (pow(sum_2, 2)) / (k * n)) / (k * n - 1)
    print("Вибіркова дисперсія: ", round(s_2, 4))
    line = mean(x_i)
    s_a_2 = ((n * (pow(x_i - line, 2).sum())) / (k - 1))
    print("Оцінка дисперсії, що характеризує зміни параметра: ", round(s_a_2, 4), '\n')

    if (s_a_2 / s_0) > f.ppf(alpha, (k - 1), k * (n - 1)):
        print("Фактор значущий")
    else:
        print("Фактора незначущий")


# Обчислимо основні показник та знайдемо оцінки дисперсій:
def factor2(n=1000, m=5, k=df.shape[1], alpha=0.95):
    factor_2 = df.groupby(df.index // n).agg(list)
    factor_2_mean = factor_2.applymap(mean)
    u = factor_2_mean
    factor_2_squared = factor_2.applymap(square).applymap(sum)

    q1 = pow(u.values, 2).sum()
    print("Q1 =", round(q1, 4))
    q2 = pow(u.sum(axis=0), 2).sum() / m
    print("Q2 =", round(q2, 4))
    q3 = pow(u.sum(axis=1), 2).sum() / k
    print("Q3 =", round(q3, 4))
    q4 = pow(u.sum(axis=1).sum(), 2) / (m * k)
    print("Q4 =", round(q4, 4))
    print("\nОцінки дисперсій:")
    s_0 = (q1 + q4 - q2 - q3) / ((k - 1) * (m - 1))
    print("S_0^2 =", round(s_0, 4))
    s_a = (q2 - q4) / (k - 1)
    print("S_a^2 =", round(s_a, 4))
    s_b = (q3 - q4) / (m - 1)
    print("S_b^2 =", round(s_b, 4))

    if (s_a / s_0) > f.ppf(alpha, (k - 1), (k - 1) * (m - 1)):
        print("Фактор А є значущим")
    else:
        print("Фактор А є незначущим")

    if (s_b / s_0) > f.ppf(alpha, (m - 1), (k - 1) * (m - 1)):
        print("Фактор B є значущим")
    else:
        print("Фактор B є незначущим")

    if (s_a / s_0) > f.ppf(alpha, (k - 1), (k - 1) * (m - 1)) and (s_b / s_0) > f.ppf(alpha, (m - 1),
                                                                                      (k - 1) * (m - 1)):
        q5 = factor_2_squared.to_numpy().sum()
        print("Q5 =", q5)
        s_ab = (q5 - n * q1) / (m * k * (n - 1))
        print("S_ab^2 =", s_ab)

        if (n * s_0 / s_ab) > f.ppf(alpha, (k - 1) * (m - 1), m * k * (n - 1)):
            print("Вплив факторів - значущий ")
        else:
            print("Вплив факторів - незначущий ")


graphs_for_parameters()
parameter()
factor1()
factor2()
# З результатів роботи програми, можна побачити,
# що вплив фактору є значним, згідно однофакторного та двофакторного аналізіу.
