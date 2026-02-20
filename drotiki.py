import math as m
import random
def integral(x_0, x_n, n):
    y_0 = 0
    y_1 = 1
    S_pr = (x_n - x_0)*(y_1 - y_0)
    kol_popodaniy = 0
    for i in range(0, n, 1):
        x = random.uniform(x_0, x_n)
        y = random.uniform(y_0, y_1)
        if y < m.sin(x): kol_popodaniy += 1
    ver_pop = kol_popodaniy / n
    integ = ver_pop * S_pr
    return integ
n = 9
I = integral(0, m.pi, n)
I1 = integral(0, m.pi, n+1)
while abs(I-I1) > 0.001:
    n += 1
    I = integral(0, m.pi, n)
    I1 = integral(0, m.pi, n + 1)
print("значение интеграла: ", I)
print("количество случайных точек: ", n)
