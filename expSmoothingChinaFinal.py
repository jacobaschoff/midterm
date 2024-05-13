# %%
# Importing libraries

import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels .tsa.api import ExponentialSmoothing, SimpleExpSmoothing

# %%
# Importing necessary data which has been obtained via UN source. Pulled all import and export data available for selected country.

china = pd.read_csv('https://raw.githubusercontent.com/jacobaschoff/midterm/main/New%20China%20Data%20(Commodity%20Coded).csv') #Reading in CSV file.
china

# %%
# Cleaning data for model and plotting

fentanylData = china[china['Comm. Code'] == 293333] # Commodity code for alfentanil and carfentanil
exportData = fentanylData[fentanylData['Flow'] == 'Export'] # Data tracking exports from China
importData = fentanylData[fentanylData['Flow'] == 'Import'] # Data tracking imports to China

# %%
# Plotting exports value (USD) by year 2002 through 2022

figChinaExValue = px.line(exportData, x='Year', y='Trade (USD)')
figChinaExValue

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
[float(level2) for i in range(21)], 
[float(level5) for i in range(21)], 
[float(level8) for i in range(21)]]).T

levels.columns = ['fentanyl_seizures', 'alpha020', 'alpha050', 'alpha080']

levels.reset_index(drop=True, inplace=True)

fig = px.line(levels, y=['fentanyl_seizures', 'alpha020', 'alpha050', 'alpha080'])

fig.show()


# %%
# Exponential smoothing model

trend = ExponentialSmoothing(data, trend='add', damped_trend=True).fit()
forecast = trend.forecast(21) # forecasting values
values = np.concatenate([data, forecast])
yesnoforecast = [0] * 21 + [1] * 21 # identifying forecasted values for plotting

#Data frame for plotting

trends = pd.DataFrame({'exports': values, 'forecasted': yesnoforecast})

# Plotting
px.line(trends, y='exports', color='forecasted', title='Total USD Value of Fentanyl Exports from China between 2002 and 2022')

# %%
# https://www.cbp.gov/newsroom/local-media-release/philadelphia-cbp-seizes-nearly-17-million-fentanyl-shipped-china
# article suggests $34k per kg
lb_price = 34000/2.20462
twentyfour = forecast[1]/lb_price
twentyfour


