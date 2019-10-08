#file that gets ozone data from the last 24 hours

from flask import render_template
from flaskexample import app
import pandas as pd
from flask import request
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from datetime import date, datetime, timezone, timedelta
import pytz
import calendar
import requests


#get current time to be able to properly request the correct data files from Air Quality API
time_request = datetime.utcnow()

#Air Quality API for today's hourly updated weather/ozone measurements
front_url = 'https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/today/HourlyData_'
back_url = ".dat"
past_list = []

#loop through all 24 hours of previous data
for i in range(2, 25):
    #data sometimes slow to be available from the site and that causes errors - 
    #begin with deleting two hours and work up to 24 hours back
    time_change = time_request - timedelta(hours=i)
    time_url = (time_change.strftime('%Y%m%d%H'))
    url_combo = front_url + time_url + back_url
    data_file = pd.read_csv(url_combo, sep = '|',
                       names = ['Date', 'Time', 'Site num', 'Site name', 'GMT offset', 'Parameter name', 
                               'Units', 'Value', 'Agency' ])
    la = data_file[data_file['Site name'] == 'Los Angeles - N. Mai'] 
    la_need = la[(la['Parameter name'] == 'OZONE') | (la['Parameter name'] == 'TEMP')| 
             (la['Parameter name'] == 'WS')| (la['Parameter name'] == 'BARPR') |  
             (la['Parameter name'] == 'RHUM')]
    past_list.append([la_need.iloc[1,0], la_need.iloc[0,1], la_need.iloc[0,7],la_need.iloc[1,7],
                      la_need.iloc[2,7],la_need.iloc[3,7], la_need.iloc[4,7]])

#convert list of lists to dataframe and name columns
past_df = pd.DataFrame.from_records(past_list, columns = ['date', 'time', 'pressure', 'ozone', 'humidity', 'temp', 'wind'])

#combine date and time, convert to datetime object, and set as index
past_df['Time'] = pd.to_datetime(past_df['date'] + ' ' + past_df['time'])
past_df.set_index('Time', inplace = True)

#remove old date and time columns
past_df.drop(['date', 'time'], axis = 1, inplace = True)

#parse out hour to set the base for 8 hour resampling
date_base = past_df.index[0]
hour_base = int(date_base.strftime('%H'))
past_df_8hr = past_df.resample('8H', base = hour_base).max()

#convert to correct units that match model input 
past_df_8hr['ozone'] = past_df_8hr['ozone']/1000
past_df_8hr['temp'] = past_df_8hr['temp']*9/5 + 32
past_df_8hr['wind'] = past_df_8hr['wind']*1.94384

#rearrange column order to make it easier to format input arry
past_df_8hr= past_df_8hr[['temp', 'ozone', 'pressure', 'humidity', 'wind']]
past_df_8hr

### Need to have AWS run this every hour so it's pre-loaded because it's very slow
# then export in csv of little, formatted, df. right now use csv placeholder
past_df_8hr.to_csv('past_df_8hr.csv')