import matplotlib.pyplot as plt
import numpy as np
from math import exp
"""
Шаг первый создать массив значений функций на сетке
u[x][t]
"""
h = 0.01 #шаг по x
t = 0.01 #шаг по t
a = 0.022

nx = int(1/h)
nt = int(1/t)

u = np.zeros((nx, nt))
f = np.zeros((nx, nt))

for i in range(0, nx):
    u[i][0] = -(i*h)**4+(i*h)**2

for j in range(0, nt):
    u[0][j] = (t*j)**2-(t*j)
    u[nx-1][j] = (t*j)+(t*j)**2-t*exp(1)

for i in range(0, nx):
    for j in range(0, nt):
        f[i][j] = (i*h)+2*(t*j)-exp(i*h)+a*(12*((i*h)**2)-2+t*exp(i*h))

plt.matshow(f)
plt.show()

