#!/usr/bin/env python
# coding: utf-8

# In[43]:


import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

plt.plot(x, y, color='green', linestyle='--', marker='o')

plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Customized Line Plot')
plt.legend(['Data Line'])

plt.show()


# In[44]:


fig, axs = plt.subplots(2, 2)


x = [1, 2, 3, 4, 5]
y1 = [2, 3, 5, 7, 11]
y2 = [1, 4, 6, 8, 10]
y3 = [5, 6, 7, 8, 9]
y4 = [7, 8, 9, 10, 11]

axs[0, 0].plot(x, y1)
axs[0, 0].set_title('Plot 1')

axs[0, 1].plot(x, y2, 'tab:orange')
axs[0, 1].set_title('Plot 2')

axs[1, 0].plot(x, y3, 'tab:green')
axs[1, 0].set_title('Plot 3')

axs[1, 1].plot(x, y4, 'tab:red')
axs[1, 1].set_title('Plot 4')

for ax in axs.flat:
    ax.set(xlabel='X-axis', ylabel='Y-axis')

plt.tight_layout()

plt.show()
plt.style.use('seaborn-darkgrid')


# In[45]:


import numpy as np

data = np.random.randn(1000)

plt.hist(data, bins=30, alpha=1, color='black')

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram ')

plt.show()


# In[46]:


x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

plt.scatter(x, y, color='red')

plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Scatter Plot Example')

plt.show()


# In[47]:


import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

sns.boxplot(x='day', y='total_bill', data=tips)

plt.show()


# In[2]:


import plotly.express as px

df = px.data.iris()

fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species')

fig.show()


# In[49]:


import seaborn as sns
import matplotlib.pyplot as plt
data = sns.load_dataset('iris')
x=data.sepal_width
y=data.sepal_length
plt.plot(x,color="red",linestyle="--",marker="o")
plt.plot(y)
plt.show()


# In[50]:


import seaborn as sns
import matplotlib.pyplot as plt
data = sns.load_dataset('iris')
fig,axs=plt.subplots(2,2)
x=data.sepal_width
y=data.sepal_length
a=data.petal_width
b=data.petal_length
axs[0,0].plot(x,color="red")
axs[0,0].set_title("sepal width")
axs[0,1].plot(y,color="green")
axs[0,1].set_title("sepal length")
axs[1,0].plot(a,color="blue")
axs[1,0].set_title("petal width")
axs[1,1].plot(b,color="black")
axs[1,1].set_title("petal length")
plt.tight_layout()
plt.show()


# In[51]:


plt.hist(data.sepal_length)
plt.show()


# In[52]:


z=[0.3,0.8]
plt.scatter(data.sepal_length,data.petal_length,c="red",alpha=z)
plt.show()


# # Exercise 1

# In[53]:


#Create a simple line plot showing the relationship between two sets of data points: x = [0, 1, 2, 3, 4] and y = [0, 2, 4, 6, 8].
#Add appropriate labels for the x-axis and y-axis and a title for the plot.

import matplotlib.pyplot as plt
x = [0, 1, 2, 3, 4]
y = [0, 2, 4, 6, 8]
plt.plot(x,y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Line Plot")
plt.show()


# # Exercise 2

# In[54]:


#Create a bar chart representing the number of students in different classes. 
#Use the classes [‘Class A’, ‘Class B’, ‘Class C’] and the corresponding number of students [30, 25, 40]. 
#Customize the colors and add grid lines.

import matplotlib.pyplot as plt
Classes=["Class A","Class B","Class C"]
Students=[30,25,40]
plt.bar(Classes,Students,color=["red","blue","black"])
plt.grid()
plt.show()


# # Exercise 3

# In[63]:


#Generate random data following a normal distribution and create a histogram to visualize the distribution. 
#Use 1000 data points and 30 bins. 
#Add a density plot overlay to show the data’s distribution curve.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
data=np.random.randn(1000)
plt.hist(data,bins=30,density=True,color="red",alpha=0.7)
sns.kdeplot(data,color="black",linewidth=2)
plt.grid()
plt.show()


# # Exercise 4

# In[56]:


#Create a figure with four subplots (2×2 grid)showing different types of plots:line plot, scatter plot, histogram, and bar chart. 
#Use random data for the plots.

import numpy as np
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,2)
name=['A','B','C','D','E']
data=np.random.uniform(1,100,5)
data1=np.random.randn(5)
data2=np.random.randn(5)
axs[0,0].scatter(data2,data1,color='red')
axs[0,0].set_title("Scatterplot")
axs[0,1].bar(name,data,color='g')
axs[0,1].set_title("Bar Plot")
axs[1,0].hist(data,bins=5,color='b')
axs[1,0].set_title("Histogram")
axs[1,1].plot(data,color='black')
axs[1,1].set_title("Line Plot")
plt.tight_layout()
plt.show()


# # Exercise 5

# In[57]:


#Create a scatter plot showing the relationship between two variables.
#Generate 50 random data points for x and y, and add a regression line to indicate the trend.

x=np.random.rand(50)
y=np.random.rand(50)
sns.regplot(x=x,y=y,color="red")
plt.show()


#  # Exercise 6

# In[58]:


#Create a pie chart showing the market share of different companies. 
#Use the companies [‘Company A’, ‘Company B’, ‘Company C’, ‘Company D’] and their respective market shares [35, 25, 25, 15]. 
#Display the percentage for each segment.

import matplotlib.pyplot as plt
comp=['Company A','Company B','Company C','Company D']
shares=[35,25,25,15]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
plt.pie(shares,autopct='%1.1f%%',labels=comp,colors=colors)
plt.legend(loc="upper left")
plt.show()


# # Exercise 7

# In[59]:


#Generate random data for four different groups and create a box plot to visualize the data distribution. 
#Use 100 data points for each group and ensure the plot includes quartiles and potential outliers.

import pandas as pd
data = {
    'Group A': np.random.normal(loc=0, scale=1, size=100),
    'Group B': np.random.normal(loc=1, scale=2, size=100),
    'Group C': np.random.normal(loc=2, scale=1.5, size=100),
    'Group D': np.random.normal(loc=-1, scale=1.2, size=100)
}

df = pd.DataFrame(data)

df_melted = df.melt(var_name='Group', value_name='Value')

plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Value', data=df_melted)
plt.show()


# # Exercise 8

# In[60]:


#Create a line plot with multiple lines to compare different datasets. 
#Use the data: x = [1, 2, 3, 4, 5], y1 = [1, 4, 9, 16, 25], y2 = [1, 2, 3, 4, 5].
#Customize the line styles, colors, and markers. Add a legend to distinguish between the datasets.

x = [1, 2, 3, 4, 5]
y1 = [1, 4, 9, 16, 25]
y2 = [1, 2, 3, 4, 5]
plt.plot(x,y1,linestyle='--',marker='o',color="red")
plt.plot(x,y2,linestyle='-',marker='*',color="blue")
plt.legend(["x,y1","x,y2"])
plt.show()


# # Exercise 9

# In[61]:


#Create a scatter plot and annotate specific points of interest. 
#Use the data: x = [1, 2, 3, 4, 5], y = [2, 3, 5, 7, 11]. Annotate the point (3, 5) with the label ‘Important Point’.

x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
xy=[3,5]
plt.scatter(x,y)
plt.annotate('Important Point', (3, 5))


# # Exercise 10

# In[62]:


#Combine a line plot and a bar chart in a single figure to compare monthly sales data. 
#Use the data: months = [‘Jan’, ‘Feb’, ‘Mar’, ‘Apr’, ‘May’],sales = [100, 150, 200, 250, 300], costs = [80, 130, 180, 230, 280].
#Plot the sales as a line plot and the costs as a bar chart.

import matplotlib.pyplot as plt
months = ["Jan", "Feb", "Mar", "Apr", "May"]
sales = [100, 150, 200, 250, 300]
costs = [80, 130, 180, 230, 280]
colors=["red","blue","orange","green","purple"]
plt.plot(months,sales,color="black")
plt.bar(months,costs,color=colors)
plt.show()


# In[10]:


# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()

# Load an example dataset
tips = sns.load_dataset("tips")

# Create a visualization
sns.relplot(
    data=tips,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="smoker", size="size",
)


# In[12]:


import matplotlib.pyplot as plt
# Load the tips dataset
tips = sns.load_dataset('tips')

# Create a FacetGrid
g = sns.FacetGrid(tips, col="time", row="smoker")
g.map(sns.histplot, "total_bill")
plt.show()


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Generate example data
np.random.seed(0)
data = np.random.rand(10, 12)
data_frame = pd.DataFrame(data, columns=[f'Col{i}' for i in range(1, 13)], 
                          index=[f'Row{i}' for i in range(1, 11)])

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data_frame, annot=True, fmt=".2f", cmap='viridis', linewidths=.5)
plt.title('Seaborn Heatmap Example')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()


# In[48]:


# Scatter plot with customization
iris = sns.load_dataset('iris')
sns.boxplot(data=tips, x='day', y='total_bill', palette='Set2')
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species')
plt.show()


# In[64]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Data Science job salaries dataset
salary_data = pd.read_csv('data_science_salaries.csv')

# Convert the 'year' column to datetime for better handling
salary_data['year'] = pd.to_datetime(salary_data['year'], format='%Y')

# Line plot for salaries over the years
plt.figure(figsize=(14, 10))
sns.lineplot(data=salary_data, x='year', y='salary', hue='job_title', palette='Set1')
plt.title('Data Science Job Salaries Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Salary (USD)')
plt.grid(True)
plt.show()

# Box plot for salary distribution by job title
plt.figure(figsize=(10, 10))
sns.boxplot(data=salary_data, x='salary', y='job_title', palette='Set2')
plt.title('Salary Distribution by Job Title')
plt.xlabel('Job Title')
plt.ylabel('Salary (USD)')
plt.xticks(rotation=90)
plt.show()

# FacetGrid for salary trends by experience level
salary_data['experience_level'] = pd.Categorical(salary_data['experience_level'], 
                                                  categories=['Junior', 'Mid-level', 'Senior', 'Lead'], 
                                                  ordered=True)
g = sns.FacetGrid(salary_data, col="experience_level", hue="job_title", height=4, aspect=1)
g.map(sns.lineplot, "year", "salary")
g.add_legend()
plt.title('Salary Trends by Experience Level')
plt.xlabel('Year')
plt.ylabel('Salary (USD)')
plt.show()


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[20]:


tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
flights = sns.load_dataset('flights')
diamonds = sns.load_dataset('diamonds')


# In[21]:


sns.violinplot(data=tips, x='day', y='total_bill', hue='sex', split=True)
plt.title('Violin Plot of Total Bill by Day and Sex')
plt.show()


# In[23]:


pivot_table = flights.pivot(index='month', columns='year', values='passengers')
sns.heatmap(pivot_table, annot=True, fmt="d", cmap='YlGnBu')
plt.title('Heatmap of Passengers')
plt.show()


# In[25]:


sns.pairplot(iris, hue='species')
plt.title('Pair Plot of Iris Dataset')
plt.show()


# In[26]:


sns.jointplot(data=tips, x='total_bill', y='tip', kind='reg')
plt.title('Joint Plot of Total Bill and Tip')
plt.show()


# In[27]:


sns.relplot(data=tips, x='total_bill', y='tip', hue='day', style='time')
plt.title('Relational Plot of Total Bill and Tip')
plt.show()


# In[28]:


g = sns.FacetGrid(tips, col='sex', hue='smoker')
g.map(sns.scatterplot, 'total_bill', 'tip')
g.add_legend()
plt.show()


# In[29]:


sns.regplot(data=tips, x='total_bill', y='tip')
plt.title('Regression Plot of Total Bill and Tip')
plt.show()


# In[30]:


sns.residplot(data=tips, x='total_bill', y='tip')
plt.title('Residual Plot of Total Bill and Tip')
plt.show()


# In[31]:


sns.kdeplot(data=diamonds, x='price', hue='cut', fill=True)
plt.title('KDE Plot of Diamond Prices by Cut')
plt.show()


# In[34]:


sns.stripplot(data=tips, x='day', y='total_bill', jitter=True)
plt.title('Strip Plot of Total Bill by Day')
plt.show()


# In[35]:


sns.swarmplot(data=tips, x='day', y='total_bill', hue='sex')
plt.title('Swarm Plot of Total Bill by Day and Sex')
plt.show()


# In[36]:


sns.boxenplot(data=diamonds, x='cut', y='price')
plt.title('Boxen Plot of Diamond Prices by Cut')
plt.show()


# In[37]:


sns.catplot(data=tips, x='day', y='total_bill', kind='box', hue='sex')
plt.title('Cat Plot of Total Bill by Day and Sex')
plt.show()


# In[38]:


sns.pointplot(data=tips, x='day', y='total_bill', hue='sex')
plt.title('Point Plot of Total Bill by Day and Sex')
plt.show()

