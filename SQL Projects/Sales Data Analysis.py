#!/usr/bin/env python
# coding: utf-8

# # Project 1: Sales Data Analysis
# Explanation:
# 
# In this project, we’ll work with a sales database to extract transaction data and perform an in-depth analysis of sales trends using Pandas. The objective is to gain insights into sales performance over time, identify top-performing products, and visualize the results for better decision-making.

# In[2]:


get_ipython().system('pip install faker')


# # Using the Faker Library to Generate Sample Data
# 
# Faker is a Python library that allows you to generate fake data, including names, addresses, and transactions. We can use it to create a sample sales database.

# In[7]:


from sqlalchemy import create_engine
import pandas as pd
from faker import Faker
import random

fake = Faker()

data = []
for _ in range(1000):
    transaction = {
        'transaction_id': fake.uuid4(),
        'product_id': random.randint(1, 50),
        'quantity': random.randint(1, 10),
        'price': round(random.uniform(5, 100), 2),
        'transaction_date': fake.date_between(start_date='-1y', end_date='today')
    }
    data.append(transaction)

df = pd.DataFrame(data)

engine = create_engine('sqlite:///sales_data.db')
df.to_sql('transactions', con=engine, if_exists='replace', index=False)

print("Sample sales data created successfully!")


# # Querying Transaction Data
# 
# Next, we’ll write and execute SQL queries to extract relevant transaction data from the database.

# In[8]:


query = """
SELECT transaction_id, product_id, quantity, price, transaction_date
FROM transactions
WHERE transaction_date BETWEEN '2023-01-01' AND '2023-12-31'
"""
df_transactions = pd.read_sql(query, con=engine.connect())


# In[9]:


df_transactions


# # Performing Sales Trend Analysis
# 
# With the transaction data in a DataFrame, we can perform various analyses to uncover sales trends.

# In[11]:


df_transactions['total_sales'] = df_transactions['quantity'] * df_transactions['price']
sales_summary = df_transactions.groupby('product_id')['total_sales'].sum().reset_index()

top_products = sales_summary.sort_values(by='total_sales', ascending=False).head(10)

print(top_products)


# # Visualizing Sales Trends
# 
# Finally, we’ll create visualizations to present your findings.

# In[14]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(top_products['product_id'], top_products['total_sales'],color="red")
plt.xlabel('Product ID')
plt.ylabel('Total Sales')
plt.title('Top 10 Best-Selling Products in 2023')
plt.show()


# In[ ]:




