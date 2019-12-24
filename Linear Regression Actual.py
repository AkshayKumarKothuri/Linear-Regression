#!/usr/bin/env python
# coding: utf-8

# # Building Machine Learning Model

# # Part1-Data Preprocessing

# Step1- Importing Libraries for Preprocessing

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 
# Step2- Import Dataset

# In[2]:


dataset =pd.read_csv('Salary_Data.csv')


# In[3]:


type(dataset)


# In[4]:


dataset


# In[5]:


dataset.head()


# In[6]:


dataset.head(5)


# Step3- Split Independent and Dpendent Variables

# In[7]:


x= dataset.iloc[:,:1]


# In[8]:


x


# In[9]:


type(x)


# In[10]:


x= dataset.iloc[:,:-1].values #convert from dataframe to numpy array


# In[11]:


x


# In[12]:


x.ndim #mandatory to be in 2 dimesion for Linear Regression


# In[13]:


type(x)


# In[14]:


y= dataset.iloc[:,1:]


# In[15]:


y


# In[16]:


y= dataset.iloc[:,1:].values


# In[17]:


y

If you have null values in the dataset 
# Step6- Split Test and Train Data

# In[18]:


from sklearn.model_selection import train_test_split                #previously cros_validation was used in sklearn
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[19]:


x_train


# In[20]:


x_test

tinyurl.com/sample-linear
# In[21]:


from sklearn.linear_model import LinearRegression


# In[22]:


lr=LinearRegression()


# In[23]:


lr.fit(x_train,y_train)


# In[24]:


y_predict=lr.predict(x_test)


# In[25]:


y_predict


# In[26]:


y_test


# In[27]:


lr.predict(np.array([[5]]))


# In[28]:


from sklearn.metrics import r2_score
r2= r2_score(y_predict,y_test)
r2


# In[29]:


#visualization of train data
plt.scatter(x_train,y_train,color = 'green')
plt.plot(x_train,lr.predict(x_train),color = 'Red')
plt.xlabel("Exper.")
plt.ylabel("Salary")
plt.title("Salary vs Experience(train)")
plt.show()


# In[30]:


#visualization of train data
plt.scatter(x_test,y_test,color = 'green')
plt.plot(x_test,lr.predict(x_test),color = 'Red')
plt.xlabel("Exper.")
plt.ylabel("Salary")
plt.title("Salary vs Experience(train)")
plt.show()

