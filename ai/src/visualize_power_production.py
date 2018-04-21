import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import datetime
import re,sys


def parse_data(data_file):
	file = open(data_file, "r")
	lst = []
	for line in file.readlines():
   		lst.append(line.split())
   	return lst

time_power = parse_data('../data.netzsin.us/20180420.txt')

print time_power

"""

for ts in time_power[]
def timestamp2date(ts):
	print(datetime.datetime.fromtimestamp(int("1284101485")).strftime('%Y-%m-%d %H:%M:%S'))

plt.plot(xdata, ydata, 'b-', label='data')

popt, pcov = curve_fit(func, xdata, ydata)
print popt
plt.plot(xdata, func(xdata, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
"""
