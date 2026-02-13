import math as m
def integral(x_0, x_n, n):
    sum = 0
    d = (x_n - x_0)/n
    for i in range (1, n, 1):
        sum += d * m.sin(d*i+x_0)
    return sum
n = 1
I = integral(0, m.pi, n)
I1 = integral(0, m.pi, n+1)
while abs(I-I1) > 0.0001:
    n += 1
    I = integral(0, m.pi, n)
    I1 = integral(0, m.pi, n + 1)

print("количество прямоугольников: ", n)
print("значение интеграла: ", I)
