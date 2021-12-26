import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


def plot_emirical_cdf(sample):
    hist, edges = np.histogram(sample, bins=len(sample))
    y = hist.cumsum()
    y = y/50
    for i in range(len(y)):
        plt.plot([edges[i], edges[i+1]], [y[i], y[i]], c="green")
    plt.show()


df = pd.read_csv('Dataframe.csv', sep=' ')
df = df.drop(df[df.iris_species != 'virginica'].index)
df = df.reset_index()


# 2.1
# находим выборочное среднее
mean = (df['petal_length'].sum()) / df.shape[0]
# находим выборочную дисперсию
dispersion = 0
for i in range(df.shape[0]):
    dispersion += (mean - df['petal_length'][i])**2
dispersion = dispersion / df.shape[0]
# находим несмещённую выборочную дисперсию
unbiased_variance = dispersion * (df.shape[0] / (df.shape[0] - 1))
# вариационный ряд
df = df.sort_values("petal_length")
# вычисляем медиану
n = 0
p = df.shape[0] % 2
if p == 0:
    n = df.shape[0] / 2
else:
    n = df.shape[0] // 2 + 1
print("2.1"
          "\nВыборочное среднее: ", mean, "\nВыборочная дисперсия: ", dispersion, "\nНесмещенная выборочная дисперсия: ", unbiased_variance,
          "\nМинимальная порядковая статистика: ", df['petal_length'][0], "\nМаксимальная порядковая статистика: ", df['petal_length'][df.shape[0]-1],
          "\nРазмах: ", df['petal_length'][df.shape[0]-1] - df['petal_length'][0], "\nМедиана: ", df['petal_length'][n])


# 2.2
print("2.2")
sample = list(np.reshape(df['petal_length'].values, -1, order='F'))
plot_emirical_cdf(sample)
print("валидационный ряд для эмпирической функции распр-я: ")
# валидационный ряд отсортированный по возрастанию
print(sample)
# строим гистограмму распределения
plt.hist(sample, density=True, bins=3)
hist = {}
buf = df['petal_length'][0]
n = 1
for i in range(1, df['petal_length'].shape[0]):
    if df['petal_length'][i] == buf:
        n += 1
    else:
        hist[buf] = n
        buf = df['petal_length'][i]
        n = 1
    if i == df['petal_length'].shape[0] - 1:
        hist[buf] = n
print("сведения для гистограммы")
print(hist)
# находим ядерную оценку
sb.distplot(df['petal_length'], hist=False)
plt.show()


# 2.3
# таблицы Стьюдента
u = 2.57
# находим доверительный интервал при известном значении дисперсии для математического ожидания
n = (dispersion * u) / (df.shape[0])**0.5
# находим доверительный интервал при известном значении математического ожидания для дисперсии
v = dispersion * df.shape[0]
# из таблицы Хи-квадрат
q1 = 76.15
q2 = 29.71
print("2.3", '\nдоверительный интервал для математического ожидания: [', mean - n, ';', mean + n, ']',
      '\nдоверительный интервал для дисперсии: [', v/q1, ';', v/q2, ']')


#2.4
print("2.4")
# α = 0.01
# значение из таблицы
t = 1.6276
fx = [0.1, 0.68, 0.82, 0.96, 0.98, 1]
# находим наибольшее отклонение и значение критерия
f_x = [0.5 + 0.4177, 0.5 + 0.1644, 0.5 + 0.1951, 0.5 + 0.4249, 0.5 + 0.4920, 0.5 + 0.4995]
# ф-ция лапласа
l = []
for i in range(len(fx)):
    if fx[i] > f_x[i]:
        l.append((fx[i] - f_x[i]))
    else:
        l.append(f_x[i] - fx[i])
print("F*(x)    F(x)    |F(x) - F*(x)|")
for i in range(len(fx)):
    print(f'{f_x[i]:.4f}', " ", f'{fx[i]:.2f}', "    ", f'{l[i]:.4f}')
maxx = max(l)
print("следовательно, λ = ", maxx)
if maxx < t:
    print(maxx, "<", t, " => распределение является нормальным для уровня значимости 0,01")
else:
    print(maxx, ">", t, " => распределение не является нормальным")


#3 пункт
# 3.1
