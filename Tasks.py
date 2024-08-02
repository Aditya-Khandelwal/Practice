#!/usr/bin/env python
# coding: utf-8

# # Matplotlib

# # Task 1

# In[2]:


#Using historical stock price data for multiple companies, create an interactive plot to analyze trends and patterns over time. 
#Include the following features:
# Line plots for stock prices of three different companies over a year.
# Highlight significant events or anomalies with annotations.
# Add a moving average line to smooth out short-term fluctuations and highlight longer-term trends.

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("GOOGL.csv")
df1=pd.read_csv("AAPL.csv")
df2=pd.read_csv("NVDA.csv")


# In[3]:


df.describe()


# In[4]:


df1.describe()


# In[5]:


df2.describe()


# In[6]:


df['Date'] = pd.to_datetime(df['Date'])
df['month_year'] = df['Date'].dt.to_period('M')
monthly_data = df.groupby('month_year')['Close'].max().reset_index()
monthly_data['month_year'] = monthly_data['month_year'].dt.to_timestamp()


# In[7]:


df1['Date'] = pd.to_datetime(df1['Date'])
df1['month_year'] = df1['Date'].dt.to_period('M')
monthly_data1 = df1.groupby('month_year')['Close'].max().reset_index()
monthly_data1['month_year'] = monthly_data1['month_year'].dt.to_timestamp()


# In[8]:


df2['Date'] = pd.to_datetime(df2['Date'])
df2['month_year'] = df2['Date'].dt.to_period('M')
monthly_data2 = df2.groupby('month_year')['Close'].max().reset_index()
monthly_data2['month_year'] = monthly_data2['month_year'].dt.to_timestamp()


# In[9]:


monthly_data.tail()


# In[10]:


monthly_data2.tail()


# In[11]:


monthly_data1.tail()


# In[12]:


plt.plot(monthly_data['month_year'], monthly_data['Close'], marker='o')
plt.plot(monthly_data1['month_year'], monthly_data1['Close'], marker='o')
plt.plot(monthly_data2['month_year'], monthly_data2['Close'], marker='*')
plt.grid()
plt.show()


# # Task 2

# In[13]:


#Visualize the results of a customer segmentation analysis using K-Means clustering. 
#Use a dataset of customer purchase behavior, and create scatter plots to display clusters.
# Include:
# Scatter plots of customers segmented into different clusters based on purchase frequency and amount.
# Distinguish clusters with different colors and markers.
# Annotate centroids of each cluster.


import pandas as pd
file_path = 'customer_segmentation_data.csv'
data = pd.read_csv(file_path)
features = data[['purchase_frequency', 'last_purchase_amount']]
from sklearn.cluster import KMeans
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(features)
data['Cluster'] = kmeans.labels_

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
markers = ['o', 's', 'D']
for i in range(n_clusters):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['purchase_frequency'], cluster_data['last_purchase_amount'],
                label=f'Cluster {i}', marker=markers[i])

centroids = kmeans.cluster_centers_
for i, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[1], s=100, c='black', marker='X')
    plt.annotate(f'Centroid {i}', (centroid[0], centroid[1]), textcoords='offset points', xytext=(0,10), ha='center')

plt.xlabel('Purchase Frequency')
plt.ylabel('Purchase Amount')
plt.title('Customer Segmentation using K-Means Clustering')
plt.legend()
plt.show()


# # Task 3

# In[14]:


# Create a dashboard to visualize sales performance across different regions and product categories.
# Include:
# Bar charts showing total sales per region.
# Stacked bar charts to break down sales by product categories.
# A line plot for monthly sales trends over the past year.

df=pd.read_csv("salesdata.csv",parse_dates=['order_date'])
df.columns


# In[15]:


df.columns = [col.strip() for col in df.columns]

total_sales_per_region = df.groupby('region')['sales'].sum()

sales_by_category_region = df.groupby(['region', 'category'])['sales'].sum().unstack()

df['order_date'] = pd.to_datetime(df['order_date'])
monthly_sales_trends = df.resample('M', on='order_date')['sales'].sum()

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18))

axes[0].bar(total_sales_per_region.index, total_sales_per_region.values, color='skyblue')
axes[0].set_title('Total sales per region')
axes[0].set_xlabel('region')
axes[0].set_ylabel('Total sales')

sales_by_category_region.plot(kind='bar', stacked=True, ax=axes[1])
axes[1].set_title('sales by Product Categories in Each region')
axes[1].set_xlabel('region')
axes[1].set_ylabel('Total sales')
axes[1].legend(title='category')

axes[2].plot(monthly_sales_trends.index, monthly_sales_trends.values, marker='o', linestyle='-', color='green')
axes[2].set_title('Monthly sales Trends')
axes[2].set_xlabel('Month')
axes[2].set_ylabel('Total sales')
axes[2].grid(True)

plt.tight_layout()

plt.show()


# # Task 4

# In[16]:


# Analyze and visualize the distribution of real estate prices in different neighborhoods of a city.
# Use a dataset of real estate listings and create:
# A histogram to show the price distribution.
# A box plot to compare price distributions across different neighborhoods.
# Scatter plots to show the relationship between property size and price.

df=pd.read_csv("AmesHousing.csv")
df.head()


# In[17]:


sc=df[['GrLivArea','SalePrice','Neighborhood']]


# In[18]:


sc.head()


# In[19]:


plt.figure(figsize=(10,6))
plt.hist(sc.SalePrice,edgecolor="k")
plt.grid()


# In[20]:


import seaborn as sns
plt.figure(figsize=(20, 10))
sns.boxplot(y=sc.SalePrice,x=sc.Neighborhood)
plt.xticks(rotation=90)


# In[21]:


plt.figure(figsize=(14,10))
plt.scatter(sc.GrLivArea,sc.SalePrice,color="r")
plt.show()


# # Seaborn

# # Task 1

# In[22]:


# Project 1: Customer Segmentation Analysis
# Dataset

# Customer data including purchase history, demographics, and behavior metrics. Download a suitable dataset from Kaggle, such as the Mall Customer Segmentation Data.

# Visualization Goals

# Identify distinct customer segments.
# Visualize purchasing patterns across segments.

import pandas as pd

df = pd.read_csv('mall_segmentation.csv')

df.head()


# In[23]:


from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)


# In[24]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1')
plt.title('Customer Segments based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()


# In[25]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Cluster', y='Spending Score (1-100)', palette='Set1')
plt.title('Spending Score Distribution across Customer Segments')
plt.xlabel('Customer Segment')
plt.ylabel('Spending Score (1-100)')
plt.show()


# # Task 2

# In[38]:


# Project 2: Sales Performance Dashboard
# Dataset

# Sales data including regions, sales reps, products, and monthly sales figures. Download a suitable dataset from Kaggle, such as the Superstore Dataset.

# Visualization Goals

# Compare sales performance across regions.
# Track sales trends over time.

import pandas as pd
df=pd.read_csv("superstore.csv")
df.head()


# In[39]:


df['Order Date'] = pd.to_datetime(df['Order Date'])
df['YearMonth'] = df['Order Date'].dt.to_period('M')
df.head()


# In[40]:


import seaborn as sns
import matplotlib.pyplot as plt

region_sales = df.groupby('Region')['Sales'].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=region_sales, x='Region', y='Sales', palette='viridis')
plt.title('Sales Performance Across Regions')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.show()


# In[46]:


df['Order Date'] = pd.to_datetime(df['Order Date'])

df['YearMonth'] = df['Order Date'].dt.to_period('M')
monthly_sales = df.groupby('YearMonth')['Sales'].sum().reset_index()

monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)

plt.figure(figsize=(14, 6))
sns.lineplot(x='YearMonth', y='Sales', data=monthly_sales, marker='o')
plt.title('Sales Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# # Task 3

# In[30]:


# Health Metrics Analysis
# Dataset

# Health metrics data including patient records, treatments, outcomes, and demographics. Download a suitable dataset from Kaggle, such as the Heart Disease Dataset.

# Visualization Goals

# Analyze the distribution of various health metrics.
# Identify correlations between different health parameters.
# Visualize patient outcomes across different demographic groups.

import pandas as pd
 
df=pd.read_csv("heartdisease.csv")
df.head()


# In[31]:


sns.set_style("whitegrid")

columns_to_plot = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

axes = axes.flatten()

for i, column in enumerate(columns_to_plot):
    sns.histplot(df[column], kde=True, ax=axes[i], color='blue', bins=30)
    axes[i].set_title(f'Distribution of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')

fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()


# In[32]:


plt.figure(figsize=(12, 10))

sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'shrink': .5})

plt.title('Correlation Matrix of Health Metrics', fontsize=16)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()


# In[33]:


sns.set_style("whitegrid")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

sns.countplot(x='sex', hue='target', data=df, ax=axes[0, 0], palette='Set1')
axes[0, 0].set_title('Heart Disease Distribution by Sex')
axes[0, 0].set_xlabel('Sex (0 = Female, 1 = Male)')
axes[0, 0].set_ylabel('Count')

age_bins = [29, 40, 50, 60, 70, 80]
df['age_group'] = pd.cut(df['age'], bins=age_bins)
sns.countplot(x='age_group', hue='target', data=df, ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Heart Disease Distribution by Age Group')
axes[0, 1].set_xlabel('Age Group')
axes[0, 1].set_ylabel('Count')

sns.countplot(x='cp', hue='target', data=df, ax=axes[1, 0], palette='Set3')
axes[1, 0].set_title('Heart Disease Distribution by Chest Pain Type')
axes[1, 0].set_xlabel('Chest Pain Type')
axes[1, 0].set_ylabel('Count')

sns.countplot(x='fbs', hue='target', data=df, ax=axes[1, 1], palette='Set1')
axes[1, 1].set_title('Heart Disease Distribution by Fasting Blood Sugar')
axes[1, 1].set_xlabel('Fasting Blood Sugar (0 = <120 mg/dl, 1 = >120 mg/dl)')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()


# # Task 4

# In[34]:


# Project 4: Data Science Job Salaries Analysis
# Dataset

# Data on job salaries for data science positions across different years. A suitable dataset is the Data Science Job Salaries Dataset from Kaggle.

# Visualization Goals

# Analyze trends in data science salaries over different years.
# Compare salary distributions across various job titles.
# Visualize the impact of experience and location on salaries.

import pandas as pd
df = pd.read_csv("data_science_salaries.csv")
df.head()


# In[35]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))
sns.lineplot(x='year', y='salary_in_usd', data=df, estimator='mean', ci=None)
plt.title('Average Data Science Salaries Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Salary in USD')
plt.show()


# In[36]:


plt.figure(figsize=(15, 10))
sns.boxplot(y='salary_in_usd', x='job_title', data=df)
plt.title('Salary Distribution Across Job Titles')
plt.xlabel('Job Title')
plt.xticks(rotation=90)
plt.ylabel('Salary in USD')
plt.tight_layout()
plt.show()


# In[37]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='experience_level', y='salary_in_usd', data=df)
plt.title('Impact of Experience Level on Salaries')
plt.xlabel('Experience Level')
plt.ylabel('Salary in USD')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
top_locations = df['company_location'].value_counts().head(10).index
sns.boxplot(y='salary_in_usd',x='company_location', data=df[df['company_location'].isin(top_locations)])
plt.title('Impact of Company Location on Salaries (Top 10 Locations)')
plt.xlabel('Salary in USD')
plt.ylabel('Company Location')
plt.tight_layout()
plt.show()


# In[ ]:




