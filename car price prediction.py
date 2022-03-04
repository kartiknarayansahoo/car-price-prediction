#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# # Importing the dataset

# In[2]:


dataset = pd.read_csv('Car details v3.csv')
dataset.head()


# In[3]:


dataset.shape


# # Cleaning the dataset

# ### Deleting missing values

# In[4]:


# counting missing values
dataset.isnull().sum()


# In[5]:


# deleting the rows with missing values
dataset = dataset.dropna()


# In[6]:


dataset.isnull().sum()


# In[7]:


dataset.shape
dataset.isnull().sum()


# In[8]:


import re


# ### Creating new torque_rpm column without only numeric values

# In[11]:


torque_list = dataset["torque"].to_list()
torque_rpm = []
def extraction_torque(x):
    for i in x:
        res = i.replace(",","")
        temp = [int(s) for s in re.findall(r'\d+', res)]
        torque_rpm.append(max(temp))
        
extraction_torque(torque_list)
print(torque_rpm[:2])


# In[12]:


dataset['torque_rpm'] = torque_rpm
dataset.head(2)


# ### extracting mileage column

# In[13]:


# extracting mileage
mil_list = dataset['mileage'].to_list()
mil_kmpl = []
def extraction_mil(x):
  for item in x:
    temp = []
    try:
      for s in item.split(" "):
        temp.append(float(s))
    except:
      pass
    mil_kmpl.append(max(temp))

extraction_mil(mil_list)
print(mil_list[:2])
print(mil_kmpl[:2])


# In[14]:


dataset['mil_kmpl'] = mil_kmpl
dataset.head()


# In[15]:


engine_list = dataset['engine'].to_list()
engine_cc = []
def extraction_engine(x):
    for item in x:
        temp = []
        try:
            for s in item.split(" "):
                temp.append(float(s))
        except:
            pass
        engine_cc.append(max(temp))

extraction_engine(engine_list)
print(engine_list[:2])
print(engine_cc[:2])


# In[16]:


dataset['engine_cc'] = engine_cc
dataset.head(2)


# In[17]:


# for max power 
max_power_list = dataset['max_power'].to_list()
max_power_bhp = []

def extraction_maxpower(x):
    for item in x:
        temp = []
        try:
            for s in item.split(" "):
                temp.append(float(s))
        except:
            pass
        max_power_bhp.append(max(temp))
        
extraction_maxpower(max_power_list)
print(max_power_list[:2])
print(max_power_bhp[:2])


# In[18]:


dataset["max_power_bhp"] = max_power_bhp
dataset.head()


# In[19]:


# so now let us create a new set dataframe with the text columns deleted
data_new = dataset.drop(["mileage","engine","max_power","torque"], axis = 1)
data_new.head()


# In[20]:


data_new.describe()


# In[42]:


plt.figure(figsize=(8,8))
sns.heatmap(data_new.corr(),annot=True,cmap='mako',linewidths=.5)


# # Splitting the dataset

# ### input and output

# In[22]:


data_new.head()


# In[23]:


df = data_new.pop('selling_price')
data_new['selling_price'] = df


# In[24]:


x = data_new.iloc[:,1:-1].values
y = data_new.iloc[:,-1].values


# In[25]:


print(data_new['fuel'].value_counts())
print(data_new['seller_type'].value_counts())
print(data_new['transmission'].value_counts())
print(data_new['owner'].value_counts())
print(data_new['name'].value_counts())


# In[26]:


data_new.head()


# # Encoding categorical variables

# In[27]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[28]:


ct_fuel = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[2,3,4,5])], remainder='passthrough')
x = np.array(ct_fuel.fit_transform(x))
# we have encoded the categorical variables


# In[29]:


print(x[:2])
print(y[:2])


# In[30]:


len(x)


# # Splitting the dataset into training and test set

# In[31]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=0)


# # Training the random forest regression model on the training set

# In[32]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(x_train, y_train)


# # Predicting the test set values

# In[33]:


predictions = regressor.predict(x_test)


# In[34]:


np.set_printoptions(precision= 2)
np.concatenate((y_test.reshape(len(y_test),1), predictions.reshape(len(predictions),1)),1)


# # Evaluating the model performance

# In[35]:


from sklearn.metrics import r2_score
r2_score(y_test, predictions)


# In[ ]:




