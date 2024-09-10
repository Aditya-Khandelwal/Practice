#!/usr/bin/env python
# coding: utf-8

# # Project 2: Library Management System
# Objective:
# Develop a system to manage books in a library, including adding new books, updating information, viewing book details, and removing books.
# 
# Setup and Requirements:
# 
# Database: SQLite
# 
# 
# Technology: Python

# In[1]:


import sqlite3
conn = sqlite3.connect('library.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        author TEXT NOT NULL,
        published_date TEXT,
        isbn TEXT UNIQUE
    )
''')
conn.commit()


# In[2]:


def add_book(title, author, published_date, isbn):
    cursor.execute('''
        INSERT INTO books (title, author, published_date, isbn)
        VALUES (?, ?, ?, ?)
    ''', (title, author, published_date, isbn))
    conn.commit()


# In[3]:


add_book("Bhagawad Gita","Maharishi Ved Vyas Ji","2nd Century BCE",9388883802)
add_book("Ramayan","Maharishi Valmiki Ji","5th Century BCE",8192540804)
add_book("Bible","Moses","100 CE",9531101132)


# In[4]:


def view_books():
    cursor.execute('SELECT * FROM books')
    rows = cursor.fetchall()
    for row in rows:
        print(row)


# In[5]:


view_books()


# In[6]:


def update_book(book_id, title=None, author=None, published_date=None, isbn=None):
    query = 'UPDATE books SET'
    params = []
    if title:
        query += ' title = ?,'
        params.append(title)
    if author:
        query += ' author = ?,'
        params.append(author)
    if published_date:
        query += ' published_date = ?,'
        params.append(published_date)
    if isbn:
        query += ' isbn = ?,'
        params.append(isbn)
    query = query.rstrip(',') + ' WHERE id = ?'
    params.append(book_id)
    cursor.execute(query, params)
    conn.commit()

def delete_book(book_id):
    cursor.execute('DELETE FROM books WHERE id = ?', (book_id,))
    conn.commit()


# In[7]:


delete_book(3)


# In[8]:


view_books()


# In[ ]:




