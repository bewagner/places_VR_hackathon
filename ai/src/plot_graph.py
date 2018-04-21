import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
data = pd.read_csv('electricity_data.csv', infer_datetime_format=True, index_col=0, parse_dates=True, header=None,
                   dtype=float)
data.columns = ['Power']
data.index.name = 'Date'
data.fillna(inplace=True, method='ffill')


#data = pd.read_csv('stromverbrauch_heute.csv', infer_datetime_format=True, parse_dates=True, index_col=0)

data = data.loc['2017-04-27 00:00':'2017-04-28 00:00']

data = data.rolling(window=10).mean()



data.fillna(inplace=True, method='bfill')
data = data.resample('H').resample('min')
data = data.interpolate(method='cubic')
data.to_csv('stromverbrauch_heute.csv')




np.savetxt('stromverbrauch_heute.txt', X=data.values.T, fmt="%.6f", delimiter=", ")
print(len(data))
print(", ".join(map(str,np.squeeze(data.values))))
plt.savefig('plot.pdf')
plt.show()
