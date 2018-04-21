import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def poly_func(x,a,b,c,d,e,f):
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x**1 + f

def func(x,a,b):
    return b*x**1 + a

xdata = np.linspace(0, 4, 50)
y     = func(xdata, 2.5, 1.3)
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata   = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')

popt, pcov = curve_fit(func, xdata, ydata)
print popt
plt.plot(xdata, func(xdata, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
