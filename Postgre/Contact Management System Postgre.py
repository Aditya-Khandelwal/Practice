#!/usr/bin/env python
# coding: utf-8

# # Project 4: Contact Management System
# Objective:
# Develop a system to manage contacts, allowing users to add, view, update, and delete contact information.
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
               CREATE TABLE IF NOT EXISTS contacts(
                   id SERIAL PRIMARY KEY ,
                   first_name VARCHAR(50) NOT NULL,
                   last_name VARCHAR(50) NOT NULL,
                   phone_number VARCHAR(50),
                   email VARCHAR(50))
                   '''
               )
conn.commit()


# In[5]:


def add_contact(first_name,last_name,phone_number,email):
    cursor.execute('''
                   INSERT INTO contacts(first_name,last_name,phone_number,email)
                   VALUES(%s,%s,%s,%s)
                   ''',(first_name,last_name,phone_number,email))
    conn.commit()


# In[6]:


add_contact("Aditya","Khandelwal",'9828060440',"aditya@gmail.com")
add_contact("Harsh","Soni",'7852075858',"harsh@gmail.com")
add_contact("Rishabh","Raj",'1234567890',"rishabh@gmail.com")


# In[7]:


def view_contacts():
    cursor.execute('SELECT*FROM contacts')
    rows=cursor.fetchall()
    if rows:
        for row in rows:
            print(f"ID: {row[0]}, First Name: {row[1]}, Last Name: {row[2]}, Phone: {row[3]}, Email: {row[4]}")
    else:
        print('No contact found.')


# In[8]:


view_contacts()


# In[9]:


def update_contact(contact_id,first_name=None,last_name=None,phone_number=None,email=None):
    query='UPDATE contacts SET'
    params=[]

    if first_name:
        query+= ' first_name = %s,'
        params.append(first_name)
    if last_name:
        query+= ' last_name = %s,'
        params.append(last_name)
    if phone_number:
        query+=' phone_number = %s,'
        params.append(phone_number)
    if email:
        query+= ' email = %s,'
        params.append(email)
    query=query.rstrip(',')  + ' WHERE id = %s'
    params.append(contact_id)
    
    cursor.execute(query,params)
    conn.commit()


# In[10]:


update_contact(3,"Rishabh","Raj",'9876543210',"rishabh.updated@gmail.com")


# In[11]:


view_contacts()


# In[12]:


def delete_contact(contact_id):
    cursor.execute('DELETE FROM contacts WHERE id = %s', (contact_id,))
    conn.commit()


# In[13]:


delete_contact(3)


# In[14]:


view_contacts()

