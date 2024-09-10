#!/usr/bin/env python
# coding: utf-8

# # Project 1: Personal Expense Tracker
# 
# Objective:
# Build an application to track personal expenses, allowing users to add, view, update, and delete expense records.
# 
# Setup and Requirements:
# 
# Database: SQLite
# 
# Technology: Python

# In[2]:


import sqlite3

conn = sqlite3.connect('expense_tracker.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS expenses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        category TEXT NOT NULL,
        amount REAL NOT NULL,
        description TEXT
    )
''')
conn.commit()


# In[3]:


def add_expense(date, category, amount, description):
    cursor.execute('''
        INSERT INTO expenses (date, category, amount, description)
        VALUES (?, ?, ?, ?)
    ''', (date, category, amount, description))
    conn.commit()


# In[6]:


add_expense('2024-09-07',"Travel",200,"Monthly Travel Expenses")
add_expense('2024-09-08',"Bills",300,"Monthly Electricity Bill")
add_expense('2024-09-09',"Recharge",109,"Monthly Mobile Recharge")
add_expense('2024-09-10',"Food",500,"Monthly Food Expenses")


# In[7]:


def view_expenses():
    cursor.execute('SELECT * FROM expenses')
    rows = cursor.fetchall()
    for row in rows:
        print(row)


# In[8]:


view_expenses()


# In[14]:


def update_expense(expense_id, date=None, category=None, amount=None, description=None):
    query = 'UPDATE expenses SET'
    params = []
    if date:
        query += ' date = ?,'
        params.append(date)
    if category:
        query += ' category = ?,'
        params.append(category)
    if amount:
        query += ' amount = ?,'
        params.append(amount)
    if description:
        query += ' description = ?,'
        params.append(description)
    query = query.rstrip(',') + ' WHERE id = ?'
    params.append(expense_id)
    cursor.execute(query, params)
    conn.commit()

def delete_expense(expense_id):
    cursor.execute('DELETE FROM expenses WHERE id = ?', (expense_id,))
    conn.commit()


# In[15]:


update_expense(2, date='2024-09-06', amount=550.00)
update_expense(4, category='Repair', description='Laptop Repair')


# In[16]:


view_expenses()


# In[18]:


delete_expense(1)


# In[19]:


view_expenses()


# In[ ]:




