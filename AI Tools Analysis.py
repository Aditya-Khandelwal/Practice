#!/usr/bin/env python
# coding: utf-8

# # AI Tools Analysis (2023)

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


# In[8]:


df=pd.read_csv("ai_tools_usage_2023.csv")
df


# In[9]:


missing_values = df.isnull().sum()
df


# In[10]:


df['Month'] = pd.to_datetime(df['Month'].astype(str) + '-2023', format='%m-%Y')
df


# In[35]:


monthly_usage = df.groupby('Month')['Usage Count'].sum().sort_index()
monthly_usage = monthly_usage.resample('M').sum()

plt.figure(figsize=(12, 6))
plt.plot(monthly_usage, marker='o')
plt.title('Monthly AI Tool Usage Count in 2023')
plt.xlabel('Month')
plt.ylabel('Usage Count')
plt.grid()
plt.show()


# In[36]:


rolling_usage = monthly_usage.rolling(window=3).mean()

plt.figure(figsize=(12, 8))
plt.plot(monthly_usage, label='Original', marker='o')
plt.plot(rolling_usage, label='3-Month Moving Average', marker='o', linestyle='--')
plt.title('AI Tool Usage with 3-Month Moving Average')
plt.xlabel('Month')
plt.ylabel('Usage Count')
plt.legend()
plt.grid()
plt.show()


# In[37]:


decomposition = seasonal_decompose(monthly_usage, model='additive', period=6)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

decomposition.trend.plot(ax=ax1)
ax1.set_ylabel('Trend')
ax1.set_xlabel('Month')
decomposition.seasonal.plot(ax=ax2)
ax2.set_ylabel('Seasonal')
ax2.set_xlabel('Month')
decomposition.resid.plot(ax=ax3)
ax3.set_ylabel('Residual')
ax3.set_xlabel('Month')
plt.tight_layout()
plt.show()


# In[38]:


forecast = monthly_usage.shift(1).bfill()

plt.figure(figsize=(10,6))
plt.plot(monthly_usage, label='Actual', marker='o')
plt.plot(forecast, label='Forecast', marker='o', linestyle='--')
plt.title('AI Tool Usage Forecast')
plt.xlabel('Month')
plt.ylabel('Usage Count')
plt.legend()
plt.grid()
plt.show()


# In[48]:


category_usage = df.groupby(['Month', 'Category'])['Usage Count'].sum().unstack()

plt.figure(figsize=(12, 6))
plt.plot(category_usage,marker='o')
plt.title('AI Tool Usage by Category in 2023')
plt.xlabel('Month')
plt.ylabel('Usage Count')
plt.legend(df.Category,title='Category', loc='best')
plt.grid()
plt.show()


# In[ ]:




