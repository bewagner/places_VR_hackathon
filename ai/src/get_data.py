import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import datetime
import re,sys
import pandas

year =2018
month=4
day  =19
file_dir = '../data.netzsin.us/'

def timestamp2date(ts):
	return(datetime.datetime.fromtimestamp(int(ts)).strftime('%Y-%m-%d %H:%M:%S'))

def parse_data(y,m,d,data_file ):

	file_str = data_file+str(y)+str(m).zfill(2)+str(d).zfill(2)+'.txt'
	data_frame = pandas.read_csv(file_str,skiprows=13,infer_datetime_format=True,parse_dates=True,sep="\t", header=None)
	
	data_frame.columns = ['Date', 'Power']
		
	data_frame.index = pandas.to_datetime(map(timestamp2date,data_frame['Date']))
	data_frame = pandas.DataFrame(data_frame['Power'],index=data_frame.index, columns=['Power'])
	data_frame.index.name = 'Date'	
	data_frame_reduced = data_frame.resample('15Min').median()
	data_frame_reduced.to_csv('electricity_data.csv', header=False, mode='a')



while True:
	if day == 0:
		day=30
		month = month-1
		if month == 0:
			month = 12
			year = year-1
	parse_data(year,month,day,file_dir)
	day = day -1
	"""
	power = map(lambda x: x[1], time_power)
	time  = map(lambda x: x[0], time_power)
	time_lst = []
	for ts in time_power:
		time_lst.append(timestamp2date(ts[0]))
	new_file= file_dir+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_neu.txt
	day = dy
	
	dic = {'date': time_lst, 'power': power}
	
	time_power_df = pandas.DataFrame(dic)
	len(time_power_df)

	file = open(new_file,"w") 
	
 
	file.close() 


	dic = {'date': time_lst, 'power': power}
	time_power_df = pandas.DataFrame(dic)

	print time_power_df['power']
	"""

