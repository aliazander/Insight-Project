# Insight-Project
No-zone: predicting ozone levels 48 hours in advance


www.no-zone.xyz provides real time ozone levels predictions for Los Angeles, California.

Description for each final:
1. ozone_combined_cleaned_LA.ipynb - downloads all necessary data, cleans it up, and exports new csv
2. EDA_FeatureEngineering.ipynb - creates new features necessary for fitting predictive model
3. Add_uncertainty.ipynb - add uncertainty to weather data for the future time points to mimic realistic circumstances
4. Model_testing.ipynb - try several different types of models to determine which is best to use in the final web app
5. Model_final.ipynb - one final model (linear regression) with output plots and deeper dive into results
6. Real_time_clean_noAPIkey.ipynb - bring together all the information needed to real time estimates 
7. FLASK_2 folder - contains all files that were used to transform the above code into a webapp

Ozone is a dangerous pollutant at high levels, but even at moderate levels it can be harmful for vulnerable populations. 
Currently, the Environmental Protection Agency only shows the ozone levels for the current time and the next day, 
which is not enough advance notice for people to plan their schedules around.

No-zone is a personalized app that takes in user input for sensitivity to pollution and the intensity and 
duration of an activity a user wishes to perform. It outputs the predicted ozone levels for the next 48 hours 
in 8 hour segments and the user's personal risk level for their desired activity. This allows users to plan their 
outdoor activities to decrease their exposure to ozone and improve respiratory health.

This project was done during my time as a Health Data Science Fellow at Insight Data Science. Insight Data Science is
a program that helps PhDs transition from academia to careers into data science.
