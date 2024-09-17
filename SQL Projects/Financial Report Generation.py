#!/usr/bin/env python
# coding: utf-8

# # Project 3: Financial Report Generation
# Explanation: Combine data from multiple financial tables, perform analysis with Pandas, and generate comprehensive financial reports.

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
from faker import Faker
from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import random


# In[17]:


engine = create_engine('sqlite:///financial_data.db')


# In[18]:


fake = Faker()
Faker.seed(42)

# Set up SQLAlchemy
engine = create_engine('sqlite:///financial_data.db')
Base = declarative_base()

# Define the income_statement table
class IncomeStatement(Base):
    __tablename__ = 'income_statement'

    id = Column(Integer, primary_key=True)
    ticker = Column(String)
    date = Column(Date)
    revenue = Column(Float)
    net_income = Column(Float)
    current_assets = Column(Float)

# Define the balance_sheet table
class BalanceSheet(Base):
    __tablename__ = 'balance_sheet'

    id = Column(Integer, primary_key=True)
    ticker = Column(String)
    date = Column(Date)
    current_assets = Column(Float)
    current_liabilities = Column(Float)
    total_liabilities = Column(Float)
    shareholders_equity = Column(Float)

# Define the cash_flow table
class CashFlow(Base):
    __tablename__ = 'cash_flow'

    id = Column(Integer, primary_key=True)
    ticker = Column(String)
    date = Column(Date)
    operating_cash_flow = Column(Float)
    investing_cash_flow = Column(Float)
    financing_cash_flow = Column(Float)

# Create the database tables
Base.metadata.create_all(engine)

# Generate fake data and insert into the database
Session = sessionmaker(bind=engine)
session = Session()

tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
dates = [fake.date_this_year() for _ in range(10)]

for ticker in tickers:
    for date in dates:
        # Generate random financial data
        income_statement_entry = IncomeStatement(
            ticker=ticker,
            date=date,
            revenue=random.uniform(50000, 200000),
            net_income=random.uniform(10000, 50000),
            current_assets=random.uniform(20000, 100000)
        )
        session.add(income_statement_entry)

        balance_sheet_entry = BalanceSheet(
            ticker=ticker,
            date=date,
            current_assets=random.uniform(20000, 100000),
            current_liabilities=random.uniform(10000, 50000),
            total_liabilities=random.uniform(30000, 150000),
            shareholders_equity=random.uniform(20000, 80000)
        )
        session.add(balance_sheet_entry)

        cash_flow_entry = CashFlow(
            ticker=ticker,
            date=date,
            operating_cash_flow=random.uniform(5000, 30000),
            investing_cash_flow=random.uniform(-20000, -5000),
            financing_cash_flow=random.uniform(-10000, -1000)
        )
        session.add(cash_flow_entry)

# Commit and close the session
session.commit()
session.close()


# In[19]:


income_statement = pd.read_sql('SELECT * FROM income_statement', engine)
balance_sheet = pd.read_sql('SELECT * FROM balance_sheet', engine)
cash_flow = pd.read_sql('SELECT * FROM cash_flow', engine)


# In[20]:


income_statement.head()


# In[21]:


balance_sheet.head()


# In[22]:


cash_flow.head()


# In[23]:


financial_data = income_statement.merge(balance_sheet, on=['ticker', 'date'], suffixes=('_income', '_balance'))
financial_data = financial_data.merge(cash_flow, on=['ticker', 'date'])


# In[24]:


financial_data.head()


# In[26]:


financial_data['current_ratio'] = financial_data['current_assets_balance'] / financial_data['current_liabilities']
financial_data['debt_to_equity'] = financial_data['total_liabilities'] / financial_data['shareholders_equity']


# In[27]:


report = financial_data.groupby('ticker').agg({
    'current_ratio': 'mean',
    'debt_to_equity': 'mean',
    'net_income': 'sum'
}).reset_index()

report.to_csv('financial_report.csv', index=False)


# In[ ]:




