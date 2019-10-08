""" This file largely follows the steps outlined in the Insight Flask tutorial, except data is stored in a
flat csv (./assets/births2012_downsampled.csv) vs. a postgres database. If you have a large database, or
want to build experience working with SQL databases, you should refer to the Flask tutorial for instructions on how to
query a SQL database from here instead.

May 2019, Donald Lee-Brown
"""

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


@app.route('/', methods=['GET', 'POST'])
def birthmodel_input():

#set default input for ozone, it will update after button is pushed
   ozone = [{'time': '+8 hours', 'level': 0.00, 'risk': 'low'},
 {'time': '+16 hours', 'level': 0.00, 'risk': 'low'},
 {'time': '+24 hours', 'level': 0.00, 'risk': 'low'},
 {'time': '+32 hours', 'level': 0.00, 'risk': 'low'},
 {'time': '+40 hours', 'level': 0.00, 'risk': 'low'},
 {'time': '+48 hours', 'level': 0.00, 'risk': 'low'}]
 
   response=requests.get('https://api.darksky.net/forecast/**** API KEY ****/34.066,-118.22688?exclude=[currently,minutely,daily,alerts,flags]')
   date_json = response.json()
   date_json['hourly']['data'][0]

   #extract the hourly data only and parse through to find the values needed for model
   day = date_json['hourly']['data']
   all_hours=[]

   for hour in day:
      t= hour['time']
      temp=hour['temperature'] 
      press=hour['pressure']
      wind=hour['windSpeed']
      humidity=hour['humidity']
        
      all_hours.append([t, temp, press, wind, humidity])

   #convert utc time to local time in LA and format nicely
   for row in all_hours:
       row[0] = datetime.fromtimestamp(row[0], pytz.timezone('US/Pacific')).strftime('%Y-%m-%d %H:%M:%S')

   #turn list of lists into a dataframe and label columns
   weather_pred = pd.DataFrame.from_records(all_hours, columns = ['time', 'temp', 'pressure', 'wind', 'humidity'])

   #remove current weather because that information comes from ozone monitor source
   weather_pred = weather_pred.iloc[1:]

   #convert to datetime, set as index, and set base to current time for resampling in 8 hour intervals
   weather_pred['time'] = pd.to_datetime(weather_pred['time'])
   weather_pred.set_index('time', inplace = True)
   date_base = weather_pred.index[0]
   hour_base = int(date_base.strftime('%H'))
   weather_pred_8hr = weather_pred.resample('8H', base = hour_base).max()

   #change units to match model input
   weather_pred_8hr['wind'] = weather_pred_8hr['wind'] * 1.94384
   weather_pred_8hr['humidity'] = weather_pred_8hr['humidity'] * 100
   #rearrange column order to make it easier to format input arry
   weather_pred_8hr= weather_pred_8hr[['temp', 'pressure', 'humidity', 'wind']]
   print(weather_pred_8hr)

########### CHANGE PATH ###########
   #get dataframe with all weather and ozone data from the last 24 hours
   past_df_8hr = pd.read_csv('/home/ubuntu/application/flaskexample/past_df_8hr.csv')
   #past_df_8hr = pd.read_csv('~/Documents/Insight/past_df_8hr.csv')
   past_df_8hr.set_index('Time', inplace = True)
   print(past_df_8hr)


   #add in month, day, hour for current time
   #there has to be a better way - FIX LATER
   my_date = datetime.now(pytz.timezone('US/Pacific'))
   data = [{'Aug': 0, 'Dec': 0, 'Feb':0,'Jan': 0, 'July': 0, 'June':0, 'March':0,
         'May': 0, 'Nov': 0, 'Oct':0,'Sept': 0, '8': 0, '16':0,
        'Mon': 0, 'Sat': 0, 'Sun':0, 'Thur': 0, 'Tue': 0, 'Wed':0}]
   date_info = pd.DataFrame(data)

   #Month fill in
   if my_date.strftime('%B') == 'September':
       date_info['Sept'] = 1
   elif my_date.strftime('%B') == 'October':
       date_info['Oct'] = 1
   elif my_date.strftime('%B') == 'November':
       date_info['Nov'] = 1
   elif my_date.strftime('%B') == 'December':
       date_info['Dec'] = 1
   elif my_date.strftime('%B') == 'January':
       date_info['Jan'] = 1
   elif my_date.strftime('%B') == 'February':
       date_info['Feb'] = 1
   elif my_date.strftime('%B') == 'March':
       date_info['March'] = 1
   elif my_date.strftime('%B') == 'May':
       date_info['May'] = 1
   elif my_date.strftime('%B') == 'June':
       date_info['June'] = 1
   elif my_date.strftime('%B') == 'July':
       date_info['July'] = 1
   elif my_date.strftime('%B') == 'August':
       date_info['Aug'] = 1

   #Hour range fill in
   if (int(my_date.strftime('%H')) >= 16) & (int(my_date.strftime('%H')) <= 23):
       date_info['16'] = 1
   elif (int(my_date.strftime('%H')) >= 8) & (int(my_date.strftime('%H')) <= 15):
       date_info['8'] = 1
    
   #Day of week fill in
   if my_date.strftime('%A') == 'Monday':
       date_info['Mon'] = 1
   elif my_date.strftime('%A') == 'Tuesday':
       date_info['Tue'] = 1
   elif my_date.strftime('%A') == 'Thursday':
       date_info['Thur'] = 1
   elif my_date.strftime('%A') == 'Wednesday':
       date_info['Wed'] = 1
   elif my_date.strftime('%A') == 'Saturday':
       date_info['Sat'] = 1
   elif my_date.strftime('%A') == 'Sunday':
       date_info['Sun'] = 1
   date_info = date_info[['Aug','Dec','Feb','Jan','July','June','March',
   'May','Nov','Oct','Sept','8','16','Mon','Sat','Sun','Thur','Tue','Wed']]
   
   print(my_date)
   print(date_info)

   ordered_input = np.array([past_df_8hr.iloc[3].values, past_df_8hr.iloc[2].values, past_df_8hr.iloc[1].values, past_df_8hr.iloc[0].values, 
                           weather_pred_8hr.iloc[0].values, weather_pred_8hr.iloc[1].values, weather_pred_8hr.iloc[2].values, 
                           weather_pred_8hr.iloc[3].values, weather_pred_8hr.iloc[4].values, weather_pred_8hr.iloc[5].values,
                           date_info.iloc[0].values])
   #arrays within array, concatenate to format appropriately for predictions from model fit
   ordered_input = np.concatenate(ordered_input).ravel()
   print(ordered_input)

   #read in data for model and format
   df = pd.read_csv('/home/ubuntu/application/flaskexample/uncertainty.csv')
   #df = pd.read_csv('~/Documents/Insight/uncertainty.csv')
   #df = pd.read_csv('uncertainty.csv')
   df.rename(columns = {"Unnamed: 0": "Date"}, inplace = True) 
   df = df.set_index('Date')
   df.index = pd.to_datetime(df.index)
   df.dropna(inplace = True) 
   df = pd.get_dummies(df, columns = ['Month', 'Hour', 'Day'], drop_first = True)
   reg = LinearRegression()
   X = df.drop(['Ozone+8', 'Ozone+16', 'Ozone+24', 'Ozone+32', 'Ozone+40', 'Ozone+48'], axis = 1)
   #normalize in a way that I can re-use normalization conditions for future prediction values
   scaler = preprocessing.StandardScaler().fit(X)
   X_norm = scaler.transform(X)
   y = df[['Ozone+8', 'Ozone+16', 'Ozone+24', 'Ozone+32', 'Ozone+40', 'Ozone+48']]
   X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size = 0.3, random_state = 21)
   reg.fit(X_train, y_train)  
   
   #use my ordered input of live data with the model fit to predict ozone
   ordered_input_scaled = scaler.transform(ordered_input.reshape(1, -1))
   pred_app = reg.predict(ordered_input_scaled)
   print(pred_app)

   output = pd.DataFrame(data=pred_app[:,:])  
   output.columns = ['+8 hours', '+16 hours', '+24 hours', '+32 hours', '+40 hours', '+48 hours']
   output = output.transpose()
   output = output.reset_index()
   output['risk'] = "low"
   output.columns = ['time', 'level', 'risk']
   output['level'] = round(output['level'],3)
 
      
   if request.method == 'POST':
      #get data from drop down
      sensitivity = dict(request.form)['sensitivity']
      intensity = dict(request.form)['intensity']
      duration = dict(request.form)['duration']
      
      if (sensitivity == "low") & (intensity == "low") & (duration == "short"):
         output.risk[output['level'] <0.06] = 'low'
         output.risk[(output['level'] >= 0.06) & (output['level'] < 0.075)] = 'medium'
         output.risk[(output['level'] >= 0.075) & (output['level'] < 0.09)] = 'high'
         output.risk[(output['level'] >= 0.09)] = 'very high'
	 
      if (sensitivity == "low") & (intensity == "low") & (duration == "medium"):
         output.risk[output['level'] <0.055] = 'low'
         output.risk[(output['level'] >= 0.055) & (output['level'] < 0.07)] = 'medium'
         output.risk[(output['level'] >= 0.07) & (output['level'] < 0.085)] = 'high'
         output.risk[(output['level'] >= 0.085)] = 'very high'
	 
      if (sensitivity == "low") & (intensity == "low") & (duration == "long"):
         output.risk[output['level'] <0.05] = 'low'
         output.risk[(output['level'] >= 0.05) & (output['level'] < 0.065)] = 'medium'
         output.risk[(output['level'] >= 0.065) & (output['level'] < 0.08)] = 'high'
         output.risk[(output['level'] >= 0.08)] = 'very high'
	 
      if (sensitivity == "low") & (intensity == "mod") & (duration == "short"):
         output.risk[output['level'] <0.055] = 'low'
         output.risk[(output['level'] >= 0.055) & (output['level'] < 0.07)] = 'medium'
         output.risk[(output['level'] >= 0.07) & (output['level'] < 0.085)] = 'high'
         output.risk[(output['level'] >= 0.085)] = 'very high'
	 
      if (sensitivity == "low") & (intensity == "mod") & (duration == "medium"):
         output.risk[output['level'] <0.05] = 'low'
         output.risk[(output['level'] >= 0.05) & (output['level'] < 0.065)] = 'medium'
         output.risk[(output['level'] >= 0.065) & (output['level'] < 0.08)] = 'high'
         output.risk[(output['level'] >= 0.08)] = 'very high'
	 
      if (sensitivity == "low") & (intensity == "mod") & (duration == "long"):
         output.risk[output['level'] <0.045] = 'low'
         output.risk[(output['level'] >= 0.045) & (output['level'] < 0.06)] = 'medium'
         output.risk[(output['level'] >= 0.06) & (output['level'] < 0.075)] = 'high'
         output.risk[(output['level'] >= 0.075)] = 'very high'    
	 
      if (sensitivity == "low") & (intensity == "high") & (duration == "short"):
         output.risk[output['level'] <0.050] = 'low'
         output.risk[(output['level'] >= 0.05) & (output['level'] < 0.065)] = 'medium'
         output.risk[(output['level'] >= 0.065) & (output['level'] < 0.080)] = 'high'
         output.risk[(output['level'] >= 0.080)] = 'very high'
	 
      if (sensitivity == "low") & (intensity == "high") & (duration == "medium"):
         output.risk[output['level'] <0.05] = 'low'
         output.risk[(output['level'] >= 0.05) & (output['level'] < 0.060)] = 'medium'
         output.risk[(output['level'] >= 0.060) & (output['level'] < 0.08)] = 'high'
         output.risk[(output['level'] >= 0.08)] = 'very high'
	 
      if (sensitivity == "low") & (intensity == "high") & (duration == "long"):
         output.risk[output['level'] <0.040] = 'low'
         output.risk[(output['level'] >= 0.040) & (output['level'] < 0.06)] = 'medium'
         output.risk[(output['level'] >= 0.06) & (output['level'] < 0.075)] = 'high'
         output.risk[(output['level'] >= 0.075)] = 'very high' 

	 
	 
		######################################################
		######  All combos with sensitivity high ##############
		######################################################

      if (sensitivity == "high") & (intensity == "low") & (duration == "short"):
         output.risk[output['level'] <0.055] = 'low'
         output.risk[(output['level'] >= 0.055) & (output['level'] < 0.065)] = 'medium'
         output.risk[(output['level'] >= 0.065) & (output['level'] < 0.75)] = 'high'
         output.risk[(output['level'] >= 0.075)] = 'very high'
	 
      if (sensitivity == "high") & (intensity == "low") & (duration == "medium"):
         output.risk[output['level'] <0.05] = 'low'
         output.risk[(output['level'] >= 0.05) & (output['level'] < 0.065)] = 'medium'
         output.risk[(output['level'] >= 0.065) & (output['level'] < 0.07)] = 'high'
         output.risk[(output['level'] >= 0.070)] = 'very high'
	 
      if (sensitivity == "high") & (intensity == "low") & (duration == "long"):
         output.risk[output['level'] <0.045] = 'low'
         output.risk[(output['level'] >= 0.045) & (output['level'] < 0.06)] = 'medium'
         output.risk[(output['level'] >= 0.06) & (output['level'] < 0.07)] = 'high'
         output.risk[(output['level'] >= 0.07)] = 'very high'
	 
      if (sensitivity == "high") & (intensity == "mod") & (duration == "short"):
         output.risk[output['level'] <0.05] = 'low'
         output.risk[(output['level'] >= 0.05) & (output['level'] < 0.065)] = 'medium'
         output.risk[(output['level'] >= 0.065) & (output['level'] < 0.07)] = 'high'
         output.risk[(output['level'] >= 0.070)] = 'very high'
	 
      if (sensitivity == "high") & (intensity == "mod") & (duration == "medium"):
         output.risk[output['level'] <0.045] = 'low'
         output.risk[(output['level'] >= 0.045) & (output['level'] < 0.06)] = 'medium'
         output.risk[(output['level'] >= 0.06) & (output['level'] < 0.07)] = 'high'
         output.risk[(output['level'] >= 0.07)] = 'very high'
	 
      if (sensitivity == "high") & (intensity == "mod") & (duration == "long"):
         output.risk[output['level'] <0.040] = 'low'
         output.risk[(output['level'] >= 0.04) & (output['level'] < 0.055)] = 'medium'
         output.risk[(output['level'] >= 0.055) & (output['level'] < 0.07)] = 'high'
         output.risk[(output['level'] >= 0.070)] = 'very high'    
	 
	 
      if (sensitivity == "high") & (intensity == "high") & (duration == "short"):
         output.risk[output['level'] <0.045] = 'low'
         output.risk[(output['level'] >= 0.045) & (output['level'] < 0.06)] = 'medium'
         output.risk[(output['level'] >= 0.06) & (output['level'] < 0.07)] = 'high'
         output.risk[(output['level'] >= 0.07)] = 'very high'
	 
      if (sensitivity == "high") & (intensity == "high") & (duration == "medium"):
         output.risk[output['level'] <0.040] = 'low'
         output.risk[(output['level'] >= 0.04) & (output['level'] < 0.055)] = 'medium'
         output.risk[(output['level'] >= 0.055) & (output['level'] < 0.07)] = 'high'
         output.risk[(output['level'] >= 0.070)] = 'very high'      
	 
      if (sensitivity == "high") & (intensity == "high") & (duration == "long"):
         output.risk[output['level'] <0.035] = 'low'
         output.risk[(output['level'] >= 0.035) & (output['level'] < 0.05)] = 'medium'
         output.risk[(output['level'] >= 0.05) & (output['level'] < 0.065)] = 'high'
         output.risk[(output['level'] >= 0.065)] = 'very high' 
      
      
      ozone = []
      for i in range(len(output)):
         my_dict = {output.columns[0]:output.iloc[i][0], output.columns[1]:output.iloc[i][1], output.columns[2]:output.iloc[i][2]}
         ozone.append(my_dict)      
 
      
      
   return render_template("model_input.html", ozone = ozone)
