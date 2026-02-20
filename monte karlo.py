import math as m
import random
def integral(x_0, x_n, n):
    sum=0
    for i in range (0,n-1,1):
        x = random.uniform(x_0, x_n)
        sum+=m.sin(x)
    return sum/n*(x_n-x_0)

n=1
I = integral(0, m.pi, n)
I1 = integral(0, m.pi, n+1)
while abs(I-I1) > 0.0001:
    n += 1
    I = integral(0, m.pi, n)
    I1 = integral(0, m.pi, n + 1)


print("значение интеграла: ", I)
print("количество случайных чисел: ", n)