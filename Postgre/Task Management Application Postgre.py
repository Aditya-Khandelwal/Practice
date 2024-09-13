#!/usr/bin/env python
# coding: utf-8

# # Project 3: Task Management Application
# Objective:
# Create a task management system where users can add, update, view, and delete tasks.
# 
# Setup and Requirements:
# 
# Database: Postgre SQL
# 
# 
# Technology: Python

# In[1]:


import psycopg2


# In[2]:


conn = psycopg2.connect(host = "localhost", user = "postgres", password = "0000", dbname = 'mydatabase')


# In[3]:


cursor = conn.cursor()
conn.commit()


# In[4]:


cursor.execute('''
               CREATE TABLE IF NOT EXISTS tasks(
                   id SERIAL PRIMARY KEY ,
                   task_name VARCHAR(50) NOT NULL,
                   due_date VARCHAR(50),
                   status VARCHAR(50)
               )
               ''')
conn.commit()


# In[5]:


def add_task(task_name,due_date,status):
    cursor.execute('''
                   INSERT INTO tasks(task_name,due_date,status)
                   VALUES(%s,%s,%s)
                   ''',(task_name,due_date,status))
    conn.commit()


# In[6]:


add_task("Project 1",'2024-09-15',"Half work done")
add_task("Project 2",'2024-09-12',"Almost done")
add_task("Project 3",'2024-09-11',"Completed")


# In[7]:


def view_tasks():
    cursor.execute('SELECT*FROM tasks')
    rows=cursor.fetchall()
    for row in rows:
        print(row)


# In[8]:


view_tasks()


# In[9]:


def update_task(task_id, task_name=None, due_date=None, status=None):
    query = 'UPDATE tasks SET'
    params = []
    
    if task_name:
        query += ' task_name = %s,'
        params.append(task_name)
        
    if due_date:
        query += ' due_date = %s,'
        params.append(due_date)
        
    if status:
        query += ' status = %s,'
        params.append(status)
        
    # Remove the trailing comma and add the WHERE clause
    query = query.rstrip(',') + ' WHERE id = %s'
    params.append(task_id)
    
    cursor.execute(query, params)
    conn.commit()


# In[10]:


update_task(2,"Project 2",'2024-09-12',"completed")


# In[11]:


view_tasks()


# In[12]:


def delete_task(task_id):
    cursor.execute('DELETE FROM tasks WHERE id=%s',(task_id,))
    conn.commit()


# In[13]:


delete_task(3)


# In[14]:


view_tasks()

