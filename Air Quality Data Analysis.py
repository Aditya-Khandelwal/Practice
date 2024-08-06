#!/usr/bin/env python
# coding: utf-8

# # Analyzing Air Quality Data: Trends and Patterns with Python

# In[61]:


import pandas as pd

df = pd.read_csv("air_quality.csv", sep=';', decimal=',', na_values=-200)

print(df.head())


# In[62]:


df


# In[63]:


print(df.columns)


# In[64]:


df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
df.set_index('datetime', inplace=True)

selected_columns = ['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'NOx(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH', 'CO(GT)', 'NMHC(GT)', 'NOx(GT)', 'NO2(GT)']
df = df[selected_columns]

df.dropna(inplace=True)

df.sort_index(inplace=True)

print(df.head())


# In[65]:


import matplotlib.pyplot as plt

df1 = df.resample('D').mean()

df2 = df1.rolling(window=7).mean()

plt.figure(figsize=(12, 6))
plt.plot(df1['NO2(GT)'], label='Daily Levels (NO2(GT))')
plt.plot(df2['NO2(GT)'], label='7-Day Moving Average', color='orange')
plt.title('Daily Levels and 7-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('NO2(GT))')
plt.legend()
plt.grid()
plt.show()


# In[66]:


from statsmodels.tsa.seasonal import seasonal_decompose

df1 = df1.asfreq('D')

# Interpolate missing values
df3 = df1.interpolate()

# Perform seasonal decomposition on one of the columns (e.g., 'NO2(GT))') with a weekly period
decomposition_result = seasonal_decompose(df3['NO2(GT)'], model='multiplicative', period=7)

# Plot the decomposed components
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# Observed
ax1.plot(decomposition_result.observed, label='Observed')
ax1.legend(loc='lower left')
ax1.set_ylabel('NO2(GT)')
ax1.grid()
# Trend
ax2.plot(decomposition_result.trend, label='Trend', color='green')
ax2.legend(loc='lower left')
ax2.set_ylabel('NO2(GT)')
ax2.grid()

# Seasonal
ax3.plot(decomposition_result.seasonal, label='Seasonal', color='orange')
ax3.legend(loc='lower left')
ax3.set_ylabel('NO2(GT)')
ax3.grid()

# Residual
ax4.plot(decomposition_result.resid, label='Residual', color='red')
ax4.legend(loc='lower left')
ax4.set_ylabel('NO2(GT))')
ax4.set_xlabel('Date')
ax4.grid()

plt.tight_layout()
plt.show()


# In[67]:


columns_to_plot = ['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)']

for column in columns_to_plot:
    plt.figure(figsize=(12, 6))
    plt.plot(df1[column], label=f'Daily Levels ({column})')
    plt.plot(df2[column], label='7-Day Moving Average', color='orange')
    plt.title(f'Daily Levels and 7-Day Moving Average for {column}')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.legend()
    plt.show()

