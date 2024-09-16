#!/usr/bin/env python
# coding: utf-8

# # Project : Sales Data Analysis with Postgre
# Explanation:
# 
# In this project, we’ll work with a sales database to extract transaction data and perform an in-depth analysis of sales trends using Pandas. The objective is to gain insights into sales performance over time, identify top-performing products, and visualize the results for better decision-making.

# In[8]:


import psycopg2
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from faker import Faker
import random
from sqlalchemy import create_engine


# In[20]:


fake = Faker()

data = []
for _ in range(1000):
    transaction = {
        'order_id': fake.uuid4(),
        'product_id': random.randint(1, 50),
        'quantity': random.randint(1, 10),
        'price': round(random.uniform(5, 100), 2),
        'order_date': fake.date_between(start_date='-1y', end_date='today')
    }
    data.append(transaction)

df = pd.DataFrame(data)

engine = create_engine('sqlite:///sales_data.db')
df.to_sql('transactions', con=engine, if_exists='replace', index=False)

print("Sample sales data created successfully!")


# In[21]:


df


# In[22]:


conn = psycopg2.connect(
    dbname="mydatabase",
    user="postgres",
    password="0000",
    host="localhost",
    port="5432"
)

# Create a cursor object
cursor = conn.cursor()
print("connection succesful.")


# In[24]:


query = """
SELECT order_id, product_id, quantity, price, order_date
FROM transactions
WHERE order_date BETWEEN '2023-01-01' AND '2023-12-31'
"""
df_transactions = pd.read_sql(query, con=engine.connect())
df_transactions


# In[25]:


df_transactions['total_sales'] = df_transactions['quantity'] * df_transactions['price']
sales_summary = df_transactions.groupby('product_id')['total_sales'].sum().reset_index()

top_products = sales_summary.sort_values(by='total_sales', ascending=False).head(10)

print(top_products)


# In[28]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(top_products['product_id'], top_products['total_sales'],color="red")
plt.xlabel('Product ID')
plt.ylabel('Total Sales')
plt.title('Top 10 Best-Selling Products in 2023')
plt.show()


# In[26]:


df['order_date'] = pd.to_datetime(df['order_date'])
monthly_sales = df.groupby(df['order_date'].dt.to_period('M')).size()

print(monthly_sales)


# In[27]:


plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='line', marker='o', color='b')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Number of Orders')
plt.grid(True)
plt.show()


# In[ ]:




