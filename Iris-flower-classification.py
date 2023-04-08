#!/usr/bin/env python
# coding: utf-8

# # OASIS INFOBYTE Data Science Internship task 1
# 

# # Iris Flower classification

# Steps to build a ML model:
# 
# 1. import dataset
# 2. visualizing the dataset
# 3. Data preparation
# 4. Training the algorithm
# 5. Making Perdiction.
# 6. Model Evolution

# # 1. Importing Libraries

# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error


# In[2]:


df=pd.read_csv('Iris.csv')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[7]:


df.shape


# In[10]:


df.isnull().sum()


# In[11]:


df.dtypes


# In[12]:


data=df.groupby('Species')


# In[13]:


data.head()


# In[14]:


df['Species'].unique()


# In[15]:


df.info()


# # 2. Visualizing the dataset

# In[16]:


sns.boxplot(df['SepalLengthCm'])


# In[17]:


sns.boxplot(df['SepalWidthCm'])


# In[18]:


sns.boxplot(df['PetalLengthCm'])


# In[19]:


sns.boxplot(df['PetalWidthCm'])


# In[20]:


sns.heatmap(df.corr())


# #  3. Data Preparation
# 

# In[21]:


df.drop('Id',axis=1,inplace=True)


# In[22]:


sp={'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}


# In[23]:


df.Species=[sp[i] for i in df.Species]


# In[24]:


df


# In[25]:


X=df.iloc[:,0:4]


# In[26]:


X


# In[27]:


y=df.iloc[:,4]


# In[28]:


y


# In[29]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# #  4. Training Model

# In[30]:


model=LinearRegression()


# In[32]:


model.fit(X_train,y_train)


# In[33]:


model.score(X_train,y_train)


# In[34]:


model.coef_


# In[35]:


model.intercept_


# #  5. Making Prediction

# In[36]:


pred=model.predict(X_test)


# #  6. Model Evaluation

# In[39]:


print("Mean absolute error: %.2f" % mean_absolute_error(pred,y_test))


# In[41]:


print("Mean squared error: %.2f" % np.mean((pred - y_test) ** 2))


# In[ ]:




