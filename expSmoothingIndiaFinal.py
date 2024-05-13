# %%
# Import libraries

import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels .tsa.api import ExponentialSmoothing, SimpleExpSmoothing

# %%
#Importing necessary data which has been obtained via UN source. Pulled all import and export data available for selected country.

india = pd.read_csv('https://raw.githubusercontent.com/jacobaschoff/midterm/main/New%20India%20Dataset%20(Commodity%20Coded).csv') #Reading in CSV file.

# %%
# Cleaning data for model and plotting

fentanylData = india[india['Comm. Code'] == 293333] # Commodity code for alfentanil and carfentanil
exportData = fentanylData[fentanylData['Flow'] == 'Export'] # Data tracking exports from India
importData = fentanylData[fentanylData['Flow'] == 'Import'] # Data tracking imports to India

# %%
# Plotting exports value (USD) by year 2002 through 2022

figIndiaExValue = px.line(exportData, x='Year', y='Trade (USD)')
figIndiaExValue

# %%
# Organizing data to be broken out into an array

df = exportData.groupby(['Year'])['Trade (USD)'].sum()
data = df.values
data

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
[float(level2) for i in range(20)], 
[float(level5) for i in range(20)], 
[float(level8) for i in range(20)]]).T

levels.columns = ['fentanyl_seizures', 'alpha020', 'alpha050', 'alpha080']

levels.reset_index(drop=True, inplace=True)

fig = px.line(levels, y=['fentanyl_seizures', 'alpha020', 'alpha050', 'alpha080'])

fig.show()


# %%
# Exponential smoothing model

trend = ExponentialSmoothing(data, trend='add', damped_trend=True).fit()
forecast = trend.forecast(20)
values = np.concatenate([data, forecast])
yesnoforecast = [0] * 20 + [1] * 20

# DataFrame for plotting

trends = pd.DataFrame({'exports': values, 'forecasted': yesnoforecast})

# Plotting
px.line(trends, y='exports', color='forecasted')

# %%
np.sum(df[:13])
df


