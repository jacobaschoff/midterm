#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECON 8310 Final Project
Spring 2024
Pygam model for the analysis of US Border Patrol seizure data
Jacob Aschoff, Ryan Holz, Brett Thompson
"""

# Install pygam
#pip install pygam

# Import statements
from pygam import LinearGAM, s, f
import pandas as pd
import patsy as pt
import numpy as np
import plotly.offline as py
from plotly.subplots import make_subplots
import plotly.graph_objs as go


# Prep the dataset
data = pd.read_csv("https://raw.githubusercontent.com/jacobaschoff/midterm/main/nationwide-drugs-fy20-fy23.csv")

# Create new reformatted dataset
refdata = pd.DataFrame(0, index=np.arange(len(data)), columns=['FY','Month (abbv)','Component','Region','Land Filter','Area of Responsibility','Other', 'Ketamine', 'Khat', 'Cocaine','Ecstasy', 'Fentanyl', 'Heroin', 'Lsd', 'Marijuana','Methamphetamine'])

# Adding column data for the year, month, component, region, land filter, and area of resposibility to the reformatted dataframe
for i in range(len(data)):
    refdata['FY'][i] = data['FY'][i]
    refdata['Month (abbv)'][i] = data['Month (abbv)'][i]
    refdata['Component'][i] = data['Component'][i]
    refdata['Region'][i] = data['Region'][i]
    refdata['Land Filter'][i] = data['Land Filter'][i]
    refdata['Area of Responsibility'][i] = data['Area of Responsibility'][i];
    
# Transferring the weights of the substances from the original dataframe to their corresponding locations in the new dataframe
for i in range(len(refdata)):
    if data['Drug Type'][i] == 'Other Drugs**':
        refdata['Other'][i] = data['Sum Qty (lbs)'][i];
    elif data['Drug Type'][i] == 'Ketamine':
        refdata['Ketamine'][i] = data['Sum Qty (lbs)'][i];
    elif data['Drug Type'][i] == 'Khat (Catha Edulis)':
        refdata['Khat'][i] = data['Sum Qty (lbs)'][i];
    elif data['Drug Type'][i] == 'Cocaine':
        refdata['Cocaine'][i] = data['Sum Qty (lbs)'][i];
    elif data['Drug Type'][i] == 'Ecstasy':
        refdata['Ecstasy'][i] = data['Sum Qty (lbs)'][i];
    elif data['Drug Type'][i] == 'Fentanyl':
        refdata['Fentanyl'][i] = data['Sum Qty (lbs)'][i];
    elif data['Drug Type'][i] == 'Heroin':
        refdata['Heroin'][i] = data['Sum Qty (lbs)'][i];
    elif data['Drug Type'][i] == 'Lsd':
        refdata['Lsd'][i] = data['Sum Qty (lbs)'][i];
    elif data['Drug Type'][i] == 'Marijuana':
        refdata['Marijuana'][i] = data['Sum Qty (lbs)'][i];
    elif data['Drug Type'][i] == 'Methamphetamine':
        refdata['Methamphetamine'][i] = data['Sum Qty (lbs)'][i];
        
# Combining rows for substances seized by the same office in the same year and month
refdata = refdata.groupby(['FY','Month (abbv)','Component','Region','Land Filter','Area of Responsibility'], as_index = False).sum()

# Dropping the rows that do not include any fentanyl seizure data and resetting the index
refdata = refdata[refdata.Fentanyl != 0].reset_index()

# Generate x and y matrices
eqn = """Fentanyl ~ -1 + Other + Ketamine + Khat + Cocaine + Ecstasy + Heroin + Lsd + Marijuana + Methamphetamine"""
y,x = pt.dmatrices(eqn, data=refdata)

# Initialize and fit the model
gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8))
gam = gam.gridsearch(np.asarray(x), y)

# Specify plot titles and shape
titles = ['Other', 'Ketamine', 'Khat', 'Cocaine', 'Ecstacy', 'Heroin', 'Lsd', 'Marijuana', 'Methamphetamine']

fig = make_subplots(rows=3, cols=3, subplot_titles=titles)
fig['layout'].update(height=800, width=1200, title='pyGAM', showlegend=False)

# Adding traces and confidence intervals
for i, title in enumerate(titles):
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, width=.95)
    trace = go.Scatter(x=XX[:,i], y=pdep, mode='lines', name='Effect')
    ci1 = go.Scatter(x = XX[:,i], y=confi[:,0], line=dict(dash='dash', color='grey'), name='95% CI')
    ci2 = go.Scatter(x = XX[:,i], y=confi[:,1], line=dict(dash='dash', color='grey'), name='95% CI')
    
    
    if i<3:
        fig.append_trace(trace, 1, i+1)
        fig.append_trace(ci1, 1, i+1)
        fig.append_trace(ci2, 1, i+1)
    elif i<6:
        fig.append_trace(trace, 2, i-2)
        fig.append_trace(ci1, 2, i-2)
        fig.append_trace(ci2, 2, i-2)
    else:
        fig.append_trace(trace, 3, i-5)
        fig.append_trace(ci1, 3, i-5)
        fig.append_trace(ci2, 3, i-5)
    
py.plot(fig)









