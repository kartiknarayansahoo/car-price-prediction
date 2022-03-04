#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing the dataset

# In[2]:


data = pd.read_csv('data.csv')
data.head()


# In[23]:


x = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values


# # Correlation between different variables

# In[4]:


plt.figure(figsize=(8,8))
sns.heatmap(data.corr(),annot=True,cmap='mako',linewidths=.5)


# # Encoding categorical variables

# In[5]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[6]:


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[2,3,4,5])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
# we have encoded the categorical variables


# In[7]:


print(x[:2])
print(y[:2])


# In[8]:


len(x)


# # Splitting the dataset into training and test set

# In[9]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=0)


# # Training the random forest regression model on the training set

# In[10]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(x_train, y_train)


# # Predicting the test set values

# In[11]:


predictions = regressor.predict(x_test)


# In[12]:


np.set_printoptions(precision= 2)
np.concatenate((y_test.reshape(len(y_test),1), predictions.reshape(len(predictions),1)),1)


# # Evaluating the model performance

# In[13]:


from sklearn.metrics import r2_score
r2_score(y_test, predictions)


# # Encoding categorical variables

# In[14]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[15]:


ct_fuel = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[2,3,4,5])], remainder='passthrough')
x = np.array(ct_fuel.fit_transform(x))
# we have encoded the categorical variables


# In[16]:


print(x[:2])
print(y[:2])


# In[17]:


len(x)


# # Splitting the dataset into training and test set

# In[18]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=0)


# # Training the random forest regression model on the training set

# In[19]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(x_train, y_train)


# # Predicting the test set values

# In[20]:


predictions = regressor.predict(x_test)


# In[21]:


np.set_printoptions(precision= 2)
np.concatenate((y_test.reshape(len(y_test),1), predictions.reshape(len(predictions),1)),1)


# # Evaluating the model performance

# In[22]:


from sklearn.metrics import r2_score
r2_score(y_test, predictions)


# In[ ]:




