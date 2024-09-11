#!/usr/bin/env python
# coding: utf-8

# # Project 4: Contact Management System
# Objective:
# Develop a system to manage contacts, allowing users to add, view, update, and delete contact information.
# 
# Setup and Requirements:
# 
# Database: SQLite
# 
# 
# Technology: Python

# In[1]:


import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('contacts.db')
cursor = conn.cursor()

# Create table for contacts
cursor.execute('''
    CREATE TABLE IF NOT EXISTS contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        phone_number TEXT,
        email TEXT
    )
''')
conn.commit()


# In[2]:


def add_contact(first_name, last_name, phone_number, email):
    cursor.execute('''
        INSERT INTO contacts (first_name, last_name, phone_number, email)
        VALUES (?, ?, ?, ?)
    ''', (first_name, last_name, phone_number, email))
    conn.commit()


# In[3]:


add_contact("Aditya","Khandelwal",'9828060440',"aditya@gmail.com")
add_contact("Harsh","Soni",'7852075858',"harsh@gmail.com")
add_contact("Rishabh","Raj",'1234567890',"rishabh@gmail.com")


# In[4]:


def view_contacts():
    cursor.execute('SELECT * FROM contacts')
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            print(f"ID: {row[0]}, First Name: {row[1]}, Last Name: {row[2]}, Phone: {row[3]}, Email: {row[4]}")
    else:
        print("No contacts found.")


# In[5]:


view_contacts()


# In[6]:


def update_contact(contact_id, first_name=None, last_name=None, phone_number=None, email=None):
    query = 'UPDATE contacts SET'
    params = []

    if first_name:
        query += ' first_name = ?,'
        params.append(first_name)
    if last_name:
        query += ' last_name = ?,'
        params.append(last_name)
    if phone_number:
        query += ' phone_number = ?,'
        params.append(phone_number)
    if email:
        query += ' email = ?,'
        params.append(email)

    query = query.rstrip(',') + ' WHERE id = ?'
    params.append(contact_id)

    cursor.execute(query, params)
    conn.commit()


# In[7]:


update_contact(3,"Rishabh","Raj",'9876543210',"rishabh.updated@gmail.com")


# In[8]:


view_contacts()


# In[9]:


def delete_contact(contact_id):
    cursor.execute('DELETE FROM contacts WHERE id = ?', (contact_id,))
    conn.commit()


# In[10]:


delete_contact(3)


# In[11]:


view_contacts()


# In[12]:


# Close the connection to the database
conn.close()


# In[ ]:




