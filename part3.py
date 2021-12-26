import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Dataframe.csv', sep=' ')
df = df.drop(df[df.iris_species != 'virginica'].index)
min1 = min(df['sepal_width'])
max1 = max(df['sepal_width'])
min2 = min(df['petal_length'])
max2 = max(df['petal_length'])

# 3.1
print('3.1')
# количество интервалов по формуле Стерджесса
leni = 6
# X [4.5;4.9], (4.9;5.3], (5.3;5.7], (5.7;6.1], (6.1;6.5], (6.5;6.9]
# Y (2.1;2.4], (2.4;2.7], (2.7;3], (3;3.3], (3.3;3.6], (3.6;3.9)
x = pd.Series(df['sepal_width'])
y = pd.Series(df['petal_length'])
l = []
for i in range(6):
    l.append([0, 0, 0, 0, 0, 0])
for n1, n2 in zip([2.1, 2.41, 2.71, 3.01, 3.31, 3.61], [2.4, 2.7, 3, 3.3, 3.6, 3.9]):
    s = x.between(n1, n2)
    for m1, m2 in zip([4.5, 4.91, 5.31, 5.71, 6.11, 6.51], [4.9, 5.3, 5.7, 6.1, 6.5, 6.9]):
        k = y.between(m1, m2)
        df2 = pd.concat([s, k], axis=1)
        for i in range(100, 150):
            if df2['sepal_width'][i] == True and df2['petal_length'][i] == True:
                ind1 = int((n1 - 2.1) / 0.3)
                ind2 = int((m1 - 4.5) / 0.4)
                l[ind1][ind2] += 1
df4 = pd.DataFrame(l, index=["(2.1;2.4]", "(2.4;2.7]", "(2.7;3]", "(3;3.3]", "(3.3;3.6]", "(3.6;3.9)"],
                   columns=['[4.5;4.9]', '(4.9;5.3]', '(5.3;5.7]', '(5.7;6.1]', '(6.1;6.5]', '(6.5;6.9]']).T
df4['i'] = df4.sum(axis=1)
df4.loc['j'] = df4.sum()
print(df4)
n = 0
for i, j in zip(["(2.1;2.4]", "(2.4;2.7]", "(2.7;3]", "(3;3.3]", "(3.3;3.6]", "(3.6;3.9)"], range(6)):
    n += df4[i][j]**2 / (df4[i][6] * df4["i"][j])
n -= 1
n *= len(df['sepal_width'])
# по таблице хи-квадрата
xi = 37.6
if n > xi:
    print('Гипотеза о независимости ширины чашелистика и длины лепестка отвергается, так как', n, '>', xi)
else:
    print('Гипотеза о независимости ширины чашелистика и длины лепестка принимается, так как', n, '<', xi)


#3.2
print('3.2')
# выборочное среднее
meanx = sum(df['sepal_width']) / len(df['sepal_width'])
meany = sum(df['petal_length']) / len(df['petal_length'])
subx = [i - meanx for i in df['sepal_width']]
suby = [i - meany for i in df['petal_length']]
# оценка ковариации
cov = (sum([subx[i] * suby[i] for i in range(len(subx))])) / (len(df['petal_length']) - 1)
print("Оценка ковариации коэффициента корреляции: ", cov)
# выборочный коэффициент корреляции
meanxy = sum((df['sepal_width'] * df['petal_length']) / len(df['sepal_width']))
sxy = meanxy - meany * meanx
meanx2 = sum(i**2 for i in df['sepal_width']) / len(df['sepal_width'])
meany2 = sum(i**2 for i in df['petal_length']) / len(df['petal_length'])
sx = (meanx2 - meanx**2)**0.5
sy = (meany2 - meany**2)**0.5
# коэффициент корреляции
rxy = sxy/(sx*sy)
t = (rxy * (len(df['petal_length']) - 2)**0.5) / (1 - rxy**2)**0.5
# alpha = 0.01
st = 2.57
print("Коэффициент корреляции: ", rxy)
if st > t:
    print("Гипотеза о незначимости коэффициента корреляции подтверждается, так как ", st, ">", t)
else:
    print("Гипотеза о незначимости коэффициента корреляции на уровне значимости 0.99 отвергается, так как ", st, "<", t)


#3.3
print('3.3')
x = df['sepal_width'].values.reshape(-1, 1)
y = df['petal_width'].values.reshape(-1, 1)
reg = LinearRegression()
reg.fit(x, y)
plt.scatter(df['sepal_width'], df['petal_width'])
plt.plot(df['sepal_width'], reg.predict(x), color='green', linewidth=2)
plt.show()
print("уравнение линейной регрессии: Y={:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))
print("коэффициент детерминации: ", rxy**2)
m = 1
f = (rxy**2 / (1 - rxy**2) * (len(df['petal_width'] - m - 1))) / m
ff = 4.08
if f > ff:
    print("Гипотеза о значимости критерия Фишера отвергается, так как ", f, ">", ff)
else:
    print("Гипотеза о значимости критерия Фишера принимается, так как ", f, "<", ff)
