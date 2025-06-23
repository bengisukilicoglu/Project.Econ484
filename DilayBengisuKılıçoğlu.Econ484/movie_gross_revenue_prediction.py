#!/usr/bin/env python
# coding: utf-8

# # Movie Gross Revenue Prediction Project

# The goal of this project is to predict the gross revenue of a movie from the IMDB Top 1000 Movies list.
# First, I will clean the data to address inconsistencies such as null values and changing data types.
# Next, I will perform EDA to identify possible relationships between gross revenue and other factors.
# Then, I will create a machine learning model to try to predict the gross revenue.
# 
# Dataset: https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows

# In[1]:


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re


# ## Data Cleaning

# In[2]:


# Reading in data
data = pd.read_csv('dataset/imdb_top_1000.csv')
data.head()


# In[3]:


# Dropping columns that won't be used for analysis
data.drop(['Poster_Link','Series_Title','Overview','Director','Star1','Star2','Star3','Star4'], axis=1, inplace=True)

# Renaming relevant columns
data.rename(columns={'Released_Year':'Release Year', 
             'Certificate':'Age Rating', 
             'IMDB_Rating':'IMDB Rating',
             'Meta_score':'Metascore',
             'No_of_Votes':'Votes',
             'Gross':'Gross Revenue'}, inplace=True)

data.head()


# In[4]:


# Checking columns for nulls
data.isna().sum()


# In[5]:


# Removing rows where Gross Revenue is null
data = data[data['Gross Revenue'].isna() == False]

# Standardizing the Age Rating column to U, UA, and A
data['Age Rating'] = data['Age Rating'].map({'U':'U','G':'U','PG':'U','GP':'U','TV-PG':'U',
                                             'UA':'UA','PG-13':'UA','U/A':'UA','Passed':'UA','Approved':'UA',
                                             'A':'A','R':'A'})

# Removing rows where Age Rating is null
data = data[data['Age Rating'].isna() == False]

data['Age Rating'].value_counts()


# In[6]:


# Creating new column to indicate whether a Metascore exists
data['Metascore Exists'] = data['Metascore'].notnull()
data.drop('Metascore',axis=1,inplace=True)
data.head()


# In[7]:


# Checking datatypes of each column
data.dtypes


# In[8]:


# Filtering Release Year values for right format and changing type to int
year_format = r'\d\d\d\d'
data = data[data['Release Year'].str.match(year_format)]
data['Release Year'] = data['Release Year'].astype(int)

# Changing Runtime type to int 
data['Runtime'] = data['Runtime'].str[:-4].astype(int)

# Changing Gross Revenue type to int and changing units to millions
data['Gross Revenue'] = data['Gross Revenue'].str.replace(',','').astype(int)
data['Gross Revenue'] = data['Gross Revenue']*(10**-6)

# Creating new column to count the number of Genres
data['Genres'] = data['Genre'].apply(lambda x: len(x.split(', ')))

# Creating new column for Primary Genre which will be the first genre listed, then dropping Genre column
data['Primary Genre'] = data['Genre'].str.split(', ').str[0]
data.drop('Genre', axis=1, inplace=True)

data.head()


# In[9]:


# Confirming data is clean
print(data.isna().sum())
print(data.dtypes)


# ## Exploratory Data Analysis

# In[10]:


# Scatterplot
# IMDB Rating vs Gross Revenue
# No distinct trend

fig = px.scatter(data, x='IMDB Rating', y='Gross Revenue', trendline='ols')
fig.update_layout(title='IMDB Rating vs Gross Revenue')
fig.show()


# In[11]:


# Scatterplot
# Runtime vs Gross Revenue
# No distinct trend

fig = px.scatter(data, x='Runtime', y='Gross Revenue', trendline='ols')
fig.update_layout(title='Runtime vs Gross Revenue')
fig.show()


# In[12]:


# Scatterplot
# Votes vs Gross Revenue
# Positive trend (more votes, higher revenue)

fig = px.scatter(data, x='Votes', y='Gross Revenue', trendline='ols')
fig.update_layout(title='Votes vs Gross Revenue')
fig.show()


# In[13]:


# Bar plot
# Average Gross Revenue per Age Rating
# UA movies gross the most on average

fig = px.bar(data[['Age Rating','Gross Revenue']].groupby('Age Rating').mean().reset_index(), x='Age Rating', y='Gross Revenue')
fig.update_layout(title='Age Rating vs Gross Revenue', xaxis={'categoryorder':'total descending'})
fig.show()


# In[14]:


# Pie chart
# Distribution of Age Rating
# Fairly even distribution

fig = px.pie(data['Age Rating'], names='Age Rating')
fig.update_layout(title='Distribution of Age Ratings')
fig.show()


# In[15]:


# Bar plot
# Average Gross Revenue per Primary Genre
# Family movies gross the most on average

fig = px.bar(data[['Primary Genre','Gross Revenue']].groupby('Primary Genre').mean().reset_index(), x='Primary Genre', y='Gross Revenue')
fig.update_layout(title='Primary Genre vs Gross Revenue', xaxis={'categoryorder':'total descending'})
fig.show()


# In[16]:


# Pie chart
# Distribution of Primary Genre
# Drama, Action, and Comedy make up most of the top 1000 movies
# Family has a very small percentage even though it has the highest average gross

fig = px.pie(data['Primary Genre'], names='Primary Genre')
fig.update_layout(title='Distribution of Primary Genre')
fig.show()


# In[17]:


# Bar plot
# Average Gross Revenue depending on number of genres
# Movies with more listed genres gross higher on average

fig = px.bar(data[['Genres','Gross Revenue']].groupby('Genres').mean().reset_index(), x='Genres', y='Gross Revenue')
fig.update_layout(title='Genres vs Gross Revenue', xaxis={'categoryorder':'total descending'})
fig.show()


# In[18]:


# Bar plot
# Average Gross Revenue depending on whether Metascore exists or not
# Movies with a Metascore gross higher on average

fig = px.bar(data[['Metascore Exists','Gross Revenue']].groupby('Metascore Exists').mean().reset_index(), x='Metascore Exists', y='Gross Revenue')
fig.update_layout(title='Metascore Exists vs Gross Revenue', xaxis={'categoryorder':'total descending'})
fig.show()


# In[19]:


# Trendline
# Year vs Gross Revenue
# Increasing over time

fig = px.line(data[['Release Year','Gross Revenue']].groupby('Release Year').mean().reset_index(), x='Release Year', y='Gross Revenue')
fig.update_layout(title='Average Gross Revenue Over Time')
fig.show()


# ## Prediction Model
# 
# Features that seem to have most trends: Release Year, Age Rating, Primary Genre, Votes, Metascore Exists
# 
# Features that will also be considered: Runtime, IMDB Rating

# In[20]:


data.head()


# ### Data Transformation

# In[21]:


prediction_data = data

# Converting Age Rating into a numerical value
prediction_data['Age Rating'] = prediction_data['Age Rating'].map({'U':0, 'UA':1, 'A':2})

# One-hot encoding categorical variables for Genre
prediction_data = pd.get_dummies(prediction_data, columns=['Primary Genre'])

prediction_data.head()


# ### Random Forest Model

# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[23]:


# Splitting dataset into features (X) and target (y)
X = prediction_data.drop('Gross Revenue', axis=1)
y = prediction_data['Gross Revenue']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# In[24]:


# Creating and training random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=101)
rf_model.fit(X_train, y_train)


# In[25]:


# Making predictions and getting metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)

print('MSE: ', round(mse, 2))
print('MAE: ', round(mae, 2))
print('R^2: ', round(r2, 2))
print('Training score: ', format(train_score, '.2%'))
print('Testing score: ', format(test_score, '.2%'))


# ## Conclusion & Results

# I was able to create a random forest regression model that predicts the Gross Revenue of a movie given it's:
# - Release Year
# - Age Rating
# - Primary Genre
# - Runtime
# - IMDB Rating
# - Number of votes on IMDB
# - Whether it received a Metascore or not
# 
# The model produced the following metrics:
# - Mean squared error: 2995.04 (in millions)
# - Mean absolute error: 36.28 (in millions)
# - R-squared: 0.7
