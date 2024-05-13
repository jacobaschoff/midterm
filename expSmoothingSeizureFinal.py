# %%
# Importing libraries

import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels .tsa.api import ExponentialSmoothing, SimpleExpSmoothing
import time


# %%
# Reading in and cleaning data

# Office of Field Operations data 2019 - 2023
OFOTwentyThree = pd.read_csv("https://raw.githubusercontent.com/jacobaschoff/midterm/main/nationwide-drugs-fy20-fy23%20(3).csv")
nineteen = pd.read_csv("https://raw.githubusercontent.com/jacobaschoff/midterm/main/nationwide-drugs-fy19-fy22.csv")
OFOTwnetyNineteen = nineteen[nineteen['FY']==2019]

# Air and Marine Operations data 2019 - 2023
AMOTwentyThree = pd.read_csv("https://raw.githubusercontent.com/jacobaschoff/midterm/main/amo-drug-seizures-fy20-fy23.csv")
AMOnineteen = pd.read_csv("https://raw.githubusercontent.com/jacobaschoff/midterm/main/amo-drug-seizures-fy19-fy22.csv")
AMOTwentyNineteen = AMOnineteen[AMOnineteen['FY']==2019]

# Combining both sets of data

fentanyl = pd.concat([OFOTwnetyNineteen, AMOTwentyNineteen, OFOTwentyThree, AMOTwentyThree])


# Month ordered in able to organize the final dataset chronologically in order to display graphs as a timeline.

month_order = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}
fentanyl['Month (abbv)'] = fentanyl['Month (abbv)'].map(month_order)


fentanyl = fentanyl.sort_values(by=['FY', 'Month (abbv)'])
fentanyl = fentanyl.reset_index(drop=True)

# %%
# Breaking out data from total to only show Fentanyl seizures

fentanyl_df = fentanyl[fentanyl['Drug Type'] == 'Fentanyl']

# Group by year and sum quantity

monthly_totals = fentanyl_df.groupby(['FY', 'Month (abbv)'])['Sum Qty (lbs)'].sum()

# Transfer data into array for using in model

data = monthly_totals.values

# Finding average of 2023 values and 2019 values

avgT3 = np.sum(data[-12:])/12
avgTn9 = np.sum(data[:12])/12

print(f"{round(avgTn9)} is the average pounds per month seized in 2019")
print(f"{round(avgT3)} is the average pounds per month seized in 2023")

# %%
# Basic alpha predictions as seen in lecture videos

alpha020 = SimpleExpSmoothing(data).fit(
                                        smoothing_level=0.2,
                                        optimized=False)

alpha050 = SimpleExpSmoothing(data).fit(
                                        smoothing_level=0.5,
                                        optimized=False)

alpha080 = SimpleExpSmoothing(data).fit(
                                        smoothing_level=0.8,
                                        optimized=False)

level2 = alpha020.forecast(1)
level5 = alpha050.forecast(1)
level8 = alpha080.forecast(1)

print(level2, level5, level8)

# %%
# Plotting non-optimized alpha prediction values

levels = pd.DataFrame([data, 
[float(level2) for i in range(61)], 
[float(level5) for i in range(61)], 
[float(level8) for i in range(61)]]).T

levels.columns = ['fentanyl_seizures', 'alpha020', 'alpha050', 'alpha080']

levels.reset_index(drop=True, inplace=True)

# Plot using Plotly
fig = px.line(levels, y=['fentanyl_seizures', 'alpha020', 'alpha050', 'alpha080'])

# Show the plot
fig.show()

# %%
# Exponential smoothing model

trend = ExponentialSmoothing(data, trend='add').fit()
forecast = trend.forecast(steps=60)
values = np.concatenate([data, forecast])
yesnoforecast = [0] * 60 + [1] * 60

# Create DataFrame for plotting

trends = pd.DataFrame({'seizures': values, 'forecast': yesnoforecast})
                   
# Plotting
fig = px.line(trends, y='seizures', color='forecast', title='Time Series Forecasting for Total Seizures in the United States')
fig.show() 

# %%
# Finding yearly averages for all forecasted years

avgT4 = round(np.sum(forecast[:12])/12)
avgT5 = round(np.sum(forecast[12:24])/12)
avgT6 = round(np.sum(forecast[24:36])/12)
avgT7 = round(np.sum(forecast[36:48])/12)
avgT8 = round(np.sum(forecast[-12:])/12)
print(avgT4, avgT5, avgT6, avgT7, avgT8)

# %%
# Exponential smoothing model with seasonality

# seasonal_periods were chosen based on visualization starting in 2022

trend = ExponentialSmoothing(data, trend='add', seasonal='mul', seasonal_periods=8).fit()
forecast = trend.forecast(steps=60)
values = np.concatenate([data, forecast])
yesnoforecast = [0] * 60 + [1] * 60

# Create DataFrame for plotting

trends = pd.DataFrame({'seizures': values, 'forecast': yesnoforecast})
                   
# Plotting
fig = px.line(trends, y='seizures', color='forecast', title='Time Series Forecasting for Total Seizures in the United States')
fig.show() 


