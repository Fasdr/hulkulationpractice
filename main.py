import matplotlib.pyplot as plt
import numpy as np
from math import exp

"Задаем начальные параметры"

h = 0.002  # шаг по x
t = 0.002  # шаг по t
a = 0.022
r = a*t/(2*(h**2))

nx = int(1/h)+1
nt = int(1/t)+1

u = np.zeros((nt, nx))  # решение разностной схемы
f = np.zeros((nt, nx))
ru = np.zeros((nt, nx))  # точное решение уравнения

for i in range(0, nx):
    for j in range(0, nt):
        ru[j][i] = -(i*h)**4+(i*h)**2+i*j*t*h+(j*t)**2-(t*j)*exp(i*h)

for i in range(0, nx):
    u[0][i] = -(i*h)**4+(i*h)**2

for j in range(0, nt):
    u[j][0] = (t*j)**2-(t*j)
    u[j][nx-1] = (t*j)+(t*j)**2-(t*j)*exp(1)

for i in range(0, nx):
    for j in range(0, nt):
        f[j][i] = (i*h)+2*(t*j)-exp(i*h)+a*(12*((i*h)**2)-2+(t*j)*exp(i*h))

"Решаем разностную схему"

ll = np.zeros(nx-1)
kk = np.zeros(nx-1)

for j in range(1, nt):
    ll[0] = 0
    kk[0] = u[j][0]
    for i in range(1, nx-1):
        ll[i] = r/(1+2*r-r*ll[i-1])
        kk[i] = ((r*u[j-1][i-1]+(1-2*r)*u[j-1][i]+r*u[j-1][i+1])+r*kk[i-1])/(1+2*r-r*ll[i-1])
    for i in range(nx-2, -1, -1):
        u[j][i] = ll[i]*u[j][i+1]+kk[i]

"""
plt.matshow(u)
plt.show()
"""
