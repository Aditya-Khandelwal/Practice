#!/usr/bin/env python
# coding: utf-8

# # Project 2: Customer Segmentation
# Explanation: Use SQL to retrieve customer data and employ Pandas for segmentation and clustering analysis to identify customer groups.

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


# In[3]:


engine = create_engine('sqlite:///purchase.db')


# In[5]:


customers_df = pd.read_sql('SELECT * FROM customers', con = engine)
print(customers_df.head)


# In[6]:


products_df = pd.read_sql('SELECT * FROM products', con = engine)
print(products_df.head)


# In[7]:


purchases_df = pd.read_sql('SELECT * FROM purchases', con = engine)
print(purchases_df.head)


# In[9]:


# Merge the DataFrames for extractiong insights 
customer_purchase_df = pd.merge(purchases_df, customers_df, on = 'customer_id', how = 'inner')
merged_df = pd.merge(customer_purchase_df, products_df, on = 'product_id', how = 'inner')
merged_df


# In[10]:


# Calculate total sales per Customer 
merged_df['total_purchase'] = merged_df["quantity"] * merged_df["price"]

total_sales_df = merged_df.groupby('customer_id')['total_purchase'].sum().reset_index()
total_sales_df.columns = ['customer_id', 'total_sales']


# In[11]:


# Calculate purchase frequency
purchase_frequency_df = merged_df.groupby('customer_id')['purchase_id'].count().reset_index()
purchase_frequency_df.columns = ['customer_id', 'purchase_count']


# In[12]:


# Calculate average purchase value 
avg_purchase_value_df = merged_df.groupby('customer_id')['total_purchase'].mean().reset_index()
avg_purchase_value_df.columns = ['customer_id', 'avg_purchase_value']


# In[15]:


# Analyse
top_customers = total_sales_df.sort_values(by = 'total_sales', ascending = False)


# In[16]:


customer_analysis_df = pd.merge(total_sales_df, purchase_frequency_df, on = 'customer_id')
customer_analysis_df = pd.merge(customer_analysis_df, avg_purchase_value_df, on = 'customer_id')


# In[17]:


# Visualize the data
top_customers.head(10).plot(x = 'customer_id', y = 'total_sales', kind = 'bar')
plt.title('Top 10 Customers by Total Sales')
plt.xlabel('Customer ID')
plt.ylabel('Total Sales')
plt.show()


# In[22]:


plt.scatter(customer_analysis_df['total_sales'], customer_analysis_df['avg_purchase_value'])
plt.title('Total Sales vs. Average Purchase Value')
plt.xlabel('Total Sales')
plt.ylabel('Average Purchase Value')
plt.show()


# In[ ]:




