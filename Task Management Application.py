#!/usr/bin/env python
# coding: utf-8

# # Project 3: Task Management Application
# Objective:
# Create a task management system where users can add, update, view, and delete tasks.
# 
# Setup and Requirements:
# 
# Database: SQLite
# 
# 
# Technology: Python

# In[1]:


import sqlite3
conn = sqlite3.connect('tasks.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_name TEXT NOT NULL,
        due_date TEXT,
        status TEXT
    )
''')
conn.commit()


# In[11]:


def add_task(task_name, due_date, status):
    cursor.execute('''
        INSERT INTO tasks (task_name, due_date, status)
        VALUES (?, ?, ?)
    ''', (task_name, due_date, status))
    conn.commit()


# In[12]:


add_task("Project 1",'2024-09-15',"Half work done")
add_task("Project 2",'2024-09-12',"Almost done")
add_task("Project 3",'2024-09-11',"Completed")


# In[13]:


def view_tasks():
    cursor.execute('SELECT * FROM tasks')
    rows = cursor.fetchall()
    for row in rows:
        print(row)


# In[14]:


view_tasks()


# In[15]:


def update_task(task_id, task_name=None, due_date=None, status=None):
    query = 'UPDATE tasks SET'
    params = []
    if task_name:
        query += ' task_name = ?,'
        params.append(task_name)
    if due_date:
        query += ' due_date = ?,'
        params.append(due_date)
    if status:
        query += ' status = ?,'
        params.append(status)
    query = query.rstrip(',') + ' WHERE id = ?'
    params.append(task_id)
    cursor.execute(query, params)
    conn.commit()

def delete_task(task_id):
    cursor.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
    conn.commit()


# In[16]:


update_task(2,"Project 2",'2024-09-12',"completed")


# In[17]:


view_tasks()


# In[20]:


delete_task(17)


# In[21]:


view_tasks()


# In[ ]:




