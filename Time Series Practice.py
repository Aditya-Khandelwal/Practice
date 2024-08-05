#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('time_series.csv', parse_dates=['date'], index_col='date')

print(df.head())

print(df.info())

print(df.describe())


# In[4]:


import pandas as pd

df = {
    'date': ['2021-01-01', '2021-01-02', '2021-01-03'],
    'sales': [100, 105, 102]
}
df = pd.DataFrame(df)

df['date'] = pd.to_datetime(df['date'])

print(df)


# In[5]:


# Set 'date' column as index
df.set_index('date', inplace=True)

print(df)


# In[6]:


# Sample data with datetime index
data = {
    'date': ['2021-01-01 10:00', '2021-01-02 11:00', '2021-01-03 12:00'],
    'sales': [100, 105, 102]
}
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Localize to a specific time zone (UTC)
df = df.tz_localize('UTC')

# Convert to another time zone (US/Eastern)
df = df.tz_convert('US/Eastern')

print(df)


# In[7]:


# Generate a range of dates with daily frequency
date_range = pd.date_range(start='2021-01-01', end='2021-01-10', freq='D')
print(date_range)

# Generate a range of dates with custom frequency (every 2 days)
custom_date_range = pd.date_range(start='2021-01-01', periods=5, freq='2D')
print(custom_date_range)


# # Summary

# Mastering date/time indexing is essential for managing and analyzing time series data. Here’s a recap of the steps involved:
# 
# 1.Introduction to Date/Time Indexing: Understand the importance and benefits of using date/time indexing in time series analysis.
# 
# 2.Converting Columns to DateTime: Use pd.to_datetime() to convert columns to datetime format.
# 
# 3.Setting Date/Time as Index: Use set_index() to set the date/time column as the index.
# 
# 4.Handling Multiple Time Zones: Use tz_localize() and tz_convert() to handle different time zones.
# 
# 5.Generating Custom Date/Time Frequencies: Use pd.date_range() to create custom date ranges with specified frequencies.

# In[8]:


import pandas as pd

data = pd.date_range(start='2023-01-01', periods=365, freq='D')
sales = np.random.randint(50, 200, size=365)
df = pd.DataFrame({'date': data, 'sales': sales})

print(df.head())
print(df.tail())


# In[9]:


# Convert 'date' to datetime and set it as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Downsample to monthly frequency and aggregate using mean
monthly_sales = df.resample('M').mean()

print(monthly_sales)


# In[10]:


# Upsample to daily frequency and interpolate missing values
daily_sales = monthly_sales.resample('D').interpolate(method='linear')

print(daily_sales)


# In[11]:


hourly_sales_ffill = df.resample('H').ffill()
print("Upsampled Data (Hourly Intervals with Forward Fill):")
print(hourly_sales_ffill.head(10))


# In[17]:


hourly_sales_bfill = df.resample('H').bfill()
print("Upsampled Data (Hourly Intervals with Backward Fill):")
print(hourly_sales_bfill.head(10))


# In[12]:


# Downsample to a custom frequency (every 10 days) and sum the data
ten_day_sales = df.resample('10D').sum()

print(ten_day_sales)


# # Summary

# Resampling is an essential technique in time series analysis that allows you to modify the frequency of your data for better analysis and insights. Here’s a recap:
# 
# 1.What is Resampling?: The process of changing the frequency of time series data.
# 
# 2.Downsampling: Reducing the frequency by aggregating data over a specified period.
# 
# 3.Upsampling: Increasing the frequency by interpolating or filling in missing values.
# 
# 4.Custom Resampling Intervals: Using specific periods to resample data according to your analysis requirements.

# In[ ]:





# In[13]:


import pandas as pd
import numpy as np

data = pd.date_range(start='2023-01-01', periods=365, freq='D')
sales = np.random.randint(50, 200, size=365)
df = pd.DataFrame({'date': data, 'sales': sales})
df.set_index('date', inplace=True)

# Calculate rolling mean with a 7-day window
df['rolling_mean'] = df['sales'].rolling(window=7).mean()

print(df.head(10))

print(df.tail(10))


# In[14]:


# Calculate rolling standard deviation with a 7-day window
df['rolling_std'] = df['sales'].rolling(window=7).std()

print(df.head(10))


# In[15]:


# Define a custom function to calculate the range within each window
def range_function(series):
    return series.max() - series.min()

# Apply the custom function with a 7-day window
df['rolling_range'] = df['sales'].rolling(window=7).apply(range_function)

print(df.head(10))


# In[16]:


import matplotlib.pyplot as plt

# Plot original sales data and rolling statistics
plt.figure(figsize=(20, 10))
plt.plot(df.index, df['sales'], label='Original Sales', color='blue', alpha=0.5)
plt.plot(df.index, df['rolling_mean'], label='Rolling Mean (7 days)', color='orange')
plt.plot(df.index, df['rolling_std'], label='Rolling Std Dev (7 days)', color='green')
plt.title('Rolling Statistics of Daily Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


# # Summary

# Rolling windows are a valuable tool in time series analysis, allowing you to compute various statistics over a moving subset of data. Here’s a recap:
# 
# 1.Introduction to Rolling Windows: Apply functions to data subsets within a moving window to analyze trends and patterns.
# 
# 2.Calculating Rolling Mean: Smooth data by computing the average over a rolling window.
# 
# 3.Calculating Rolling Standard Deviation: Measure variability within each rolling window to assess consistency.
# 
# 4.Applying Custom Functions with Rolling Windows: Use custom functions for specialized calculations within rolling windows.
# 
# 5.Visualizing Rolling Statistics: Create visual representations of rolling statistics to better understand data trends and patterns.
# 

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate example data: Monthly retail sales
data = pd.date_range(start='2023-01-01', periods=24, freq='M')
sales = [200, 220, 210, 240, 300, 320, 280, 310, 350, 360, 370, 400] * 2
df = pd.DataFrame({'date': data, 'sales': sales})
df.set_index('date', inplace=True)

# Introduce missing values
df.loc['2023-06-30', 'sales'] = np.nan

# Handle missing values using interpolation
df['sales'] = df['sales'].interpolate()

# Perform seasonal decomposition
decomposition = seasonal_decompose(df['sales'], model='additive')

# Extract components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot components
plt.figure(figsize=(14, 10))
plt.subplot(4, 1, 1)
plt.plot(df['sales'], label='Original Series')
plt.title('Original Time Series')
plt.legend(loc='best')

plt.subplot(4, 1, 2)
plt.plot(trend, label='Trend')
plt.title('Trend Component')
plt.legend(loc='best')

plt.subplot(4, 1, 3)
plt.plot(seasonal, label='Seasonality')
plt.title('Seasonal Component')
plt.legend(loc='best')

plt.subplot(4, 1, 4)
plt.plot(residual, label='Residuals')
plt.title('Residual Component')
plt.legend(loc='best')

plt.tight_layout()
plt.show()# Introduce missing values
df.loc['2023-06-30', 'sales'] = np.nan

# Handle missing values using interpolation
df['sales'] = df['sales'].interpolate()


# # Summary

# Seasonal decomposition is a crucial technique for understanding time series data. Here’s a quick recap:
# 
# 1.What is Seasonal Decomposition: Decompose a time series into trend, seasonal, and residual components to analyze underlying patterns.
# 
# 2.Additive vs. Multiplicative Models: Choose the model based on whether seasonal effects are constant or proportional to the trend.
# 
# 3.Performing Seasonal Decomposition: Use seasonal_decompose to break down the time series and visualize the components.
# 
# 4.Interpreting Decomposition Results: Analyze trend, seasonal, and residual components to gain insights into the data.
# Handling Missing Values: Address missing data through interpolation or imputation to maintain accurate decomposition

# In[22]:


# Performing seasonal decomposition
decomposition = seasonal_decompose(df['sales'], model='additive')
decomposition.plot()
plt.show()


# In[25]:


from statsmodels.tsa.seasonal import seasonal_decompose

# Perform seasonal decomposition (additive model)
decomposition = seasonal_decompose(df['sales'], model='additive')
seasonally_adjusted = df['sales'] - decomposition.seasonal

# Plot the seasonally adjusted series
plt.figure(figsize=(10, 5))
plt.plot(seasonally_adjusted, label='Seasonally Adjusted Sales')
plt.title('Seasonally Adjusted Time Series')
plt.legend(loc='best')
plt.show()


# In[34]:


from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(df['sales'], order=(5, 1, 0))
model_fit = model.fit()

# Forecast future values
forecast = model_fit.forecast(steps=18)
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['sales'], label='Historical Sales')
plt.plot(pd.date_range(start=df.index[-1], periods=19, freq='M')[1:], forecast, label='Forecast', color='red')
plt.title('Sales Forecast')
plt.legend(loc='best')
plt.show()


# In[35]:


from sklearn.metrics import mean_squared_error

# Split data into training and test sets
train_size = int(len(df) * 0.8)
train, test = df['sales'][:train_size], df['sales'][train_size:]

# Fit model on training data and make predictions
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)

# Calculate and print error metrics
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse:.2f}')


# # Summary

# Implementing best practices in time series analysis ensures robust and accurate results. Here’s a recap:
# 
# 1.	Understanding Your Data: Gain insights into data structure and characteristics.
# 
# 2.	Handling Missing Values: Use interpolation, imputation, or exclusion to manage missing data.
# 
# 3.	Ensuring Proper Date/Time Indexing: Convert columns to datetime and set as index for correct analysis.
# 
# 4.	Using Resampling Effectively: Aggregate or interpolate data to different frequencies for better analysis.
# 
# 5.	Applying Rolling Windows Correctly: Use rolling windows to calculate statistics and smooth data.
# 
# 6.	Decomposing Time Series for Insights: Break down time series into components to understand patterns.
# 
# 7.	Visualizing Data at Every Step: Use plots to visualize and interpret time series components.
# 
# 8.	Performing Seasonal Adjustment: Remove seasonal effects to focus on underlying trends.
# 
# 9.	Using Appropriate Forecasting Models: Choose models based on data characteristics for accurate predictions.
# 
# 10.	Validating Your Models: Evaluate model performance using cross-validation and error metrics.
# 
# 

# In[ ]:




