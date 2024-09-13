#!/usr/bin/env python
# coding: utf-8

# # Project 1 : Personal Expense Tracker 
# Objective:
# Build an application to track personal expenses, allowing users to add, view, update, and delete expense records.
# 
# Setup and Requirements:
# 
# Database: Postgre SQL
# 
# 
# Technology: Python

# In[7]:


get_ipython().system('pip install psycopg2')


# In[8]:


import psycopg2


# In[9]:


conn = psycopg2.connect(host = "localhost", user = "postgres", password = "0000", dbname = 'mydatabase')


# In[10]:


cursor = conn.cursor()
conn.commit()


# In[11]:


cursor.execute('''
    CREATE TABLE IF NOT EXISTS expenses (
        id SERIAL PRIMARY KEY,  -- Use SERIAL for auto-incrementing integers
        date VARCHAR(50) NOT NULL,
        category VARCHAR(50) NOT NULL,
        amount DECIMAL(20, 2) NOT NULL,  -- Specify precision for DECIMAL
        description VARCHAR(50)
    )
''')


# In[12]:


def add_expenses(date, category, amount, description):
    cursor.execute('''
                   INSERT INTO expenses(date, category, amount, description )
                   VALUES(%s,%s,%s,%s)
                   ''', (date, category, amount, description))
    conn.commit()


# In[13]:


add_expenses('2024-09-10', 'Groceries', 50.75, 'Weekly grocery shopping')
add_expenses('2024-09-11', 'Transport', 15.00, 'Taxi fare')
add_expenses('2024-09-12', 'Utilities', 100.00, 'Electricity bill')


# In[14]:


def view_expenses():
    cursor.execute('SELECT * FROM expenses')
    rows = cursor.fetchall()
    for row in rows:
        print(row) 


# In[15]:


view_expenses()


# In[19]:


def update_expenses(expense_id, date=None, category=None, amount=None, description=None):
    query = "UPDATE expenses SET "
    params = []
    
    if date:
        query += "date = %s, "
        params.append(date)
    
    if category:
        query += "category = %s, "
        params.append(category)
        
    if amount:
        query += "amount = %s, "
        params.append(amount)
        
    if description:
        query += "description = %s, "
        params.append(description)
        
    # Remove the trailing comma and space, and add WHERE clause
    query = query.rstrip(', ') + " WHERE id = %s"
    params.append(expense_id)
    
    # Execute the query
    cursor.execute(query, tuple(params))
    conn.commit()


# In[20]:


update_expenses(1, date='2024-09-15', category='Dining')
update_expenses(2, amount=75.50, description='Grocery shopping for the week')
update_expenses(3, date='2024-09-12', category='Transport', amount="25.00", description='Bus fare')


# In[21]:


view_expenses()


# In[22]:


def delete_expenses(expense_id):
    cursor.execute('DELETE FROM expenses WHERE id = %s', (expense_id,))
    conn.commit()


# In[25]:


delete_expenses(3)


# In[26]:


view_expenses()


# In[ ]:




