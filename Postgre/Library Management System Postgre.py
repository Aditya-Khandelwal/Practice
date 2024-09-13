#!/usr/bin/env python
# coding: utf-8

# # Project 2: Library Management System
# Objective:
# Develop a system to manage books in a library, including adding new books, updating information, viewing book details, and removing books.
# 
# Setup and Requirements:
# 
# Database: Postgre SQL
# 
# 
# Technology: Python

# In[1]:


import psycopg2


# In[4]:


conn = psycopg2.connect(host = "localhost", user = "postgres", password = "0000", dbname = 'mydatabase')


# In[5]:


cursor = conn.cursor()
conn.commit()


# In[6]:


cursor.execute('''
            CREATE TABLE IF NOT EXISTS books (
        id SERIAL PRIMARY KEY ,
        title VARCHAR(50) NOT NULL,
        author VARCHAR(50) NOT NULL,
        published_date VARCHAR(50),
        isbn VARCHAR(50) UNIQUE
    )
''')


# In[7]:


def add_book(title, author, published_date, isbn):
    cursor.execute('''
                   INSERT INTO books(title, author, published_date, isbn)
                   VALUES (%s, %s, %s, %s)
                   ''',(title, author, published_date, isbn))
    conn.commit()


# In[8]:


add_book("Bhagawad Gita","Maharishi Ved Vyas Ji","2nd Century BCE",9388883802)
add_book("Ramayan","Maharishi Valmiki Ji","5th Century BCE",8192540804)
add_book("Bible","Moses","100 CE",9531101132)


# In[9]:


def view_books():
    cursor.execute('SELECT * FROM books')
    rows = cursor.fetchall()
    for row in rows:
        print(row) 


# In[10]:


view_books()


# In[12]:


def update_book(book_id, title=None, author=None, published_date=None, isbn=None):
    query = 'UPDATE books SET'
    params = []
    if title:
        query += ' title = %s,'
        params.append(title)
    if author:
        query += ' author = %s,'
        params.append(author)
    if published_date:
        query += ' published_date = %s,'
        params.append(published_date)
    if isbn:
        query += ' isbn = %s,'
        params.append(isbn)
    query = query.rstrip(',') + ' WHERE id = %s'
    params.append(book_id)
    cursor.execute(query, params)
    conn.commit()
    
def delete_book(book_id):
    cursor.execute('DELETE FROM books WHERE id = %s', (book_id,))
    conn.commit()


# In[13]:


delete_book(3)


# In[14]:


view_books()

