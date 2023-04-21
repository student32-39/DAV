#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# # Load Dataset

# In[2]:


train_data = pd.read_csv ('D://Datasets//simple_linear_regression_train.csv')


# In[3]:


display (train_data)


# In[4]:


test_data = pd.read_csv ('D://Datasets//simple_linear_regression_test.csv')


# In[5]:


display (test_data)


# # EDA

# In[6]:


train_data.info ()


# In[7]:


test_data.info ()


# In[8]:


train_data.describe ()


# In[9]:


test_data.describe ()


# In[10]:


train_data.shape 


# In[11]:


test_data.shape


# In[12]:


train_data.ndim, test_data.ndim


# In[13]:


train_data.dtypes


# In[14]:


test_data.dtypes


# In[15]:


print (train_data.isna ().sum ())
print (test_data.isna ().sum ())


# In[16]:


print (train_data.duplicated ().sum ())
print (test_data.duplicated ().sum ())


# In[17]:


train_mean, train_median, train_mode = train_data.mean (), train_data.median (), train_data.mode ()
print (train_mean, '\n', train_median, '\n', train_mode)
test_mean, test_median, test_mode = test_data.mean (), test_data.median (), test_data.mode ()
print (test_mean, '\n', test_median, '\n', test_mode)


# In[18]:


train_data ['y'].fillna (train_data ['y'].mean (), inplace=True)


# In[19]:


train_data.isna ().sum ()


# ## Data Split

# In[20]:


x_train, y_train, x_test, y_test = train_data['x'], train_data['y'], test_data ['x'], test_data ['y']


# In[21]:


x_train.shape , x_test.shape, y_train.shape , y_test.shape


# In[22]:


x = pd.concat ([x_train, x_test], axis=0)
x


# In[23]:


y = pd.concat ([y_train, y_test], axis=0)
y


# In[24]:


x.shape, y.shape 


# ## Data Visualization

# In[25]:


rcParams ['figure.figsize'] = 10, 7


# In[26]:


plt.title ('Train Data - X Vs. Y')
plt.xlabel ('X')
plt.ylabel ('Y')
plt.scatter (x_train, y_train)
plt.show ()


# In[27]:


plt.title ('Test Data - X Vs. Y')
plt.xlabel ('X')
plt.ylabel ('Y')
plt.scatter (x_test, y_test)
plt.show ()


# In[28]:


plt.title ('X Vs. Y')
plt.xlabel ('X')
plt.ylabel ('Y')
plt.scatter (x, y)
plt.grid ()
plt.show ()


# In[29]:


train_data.iloc [40:61, 1]


# In[30]:


train_data [train_data ['x'] > 3500]


# In[31]:


train_data.iloc [200:220, :]


# In[32]:


train_data.drop ([213], axis=0, inplace=True)


# In[33]:


train_data.iloc [200:220, :]


# In[34]:


new_x_train, new_y_train = train_data ['x'], train_data ['y']
new_x_train.shape, new_y_train.shape


# In[35]:


plt.title ('Train Data - X Vs. Y')
plt.xlabel ('X')
plt.ylabel ('Y')
plt.scatter (new_x_train, new_y_train)
plt.show ()


# In[36]:


plt.title ('Train Data - X Vs. Y')
plt.bar (new_x_train, new_y_train, label='Y')
plt.legend ()
plt.show ()


# In[37]:


plt.title ('Test Data - X Vs. Y')
plt.bar (x_test, y_test, label='Y')
plt.legend ()
plt.show ()


# In[38]:


plt.hist (new_y_train)
plt.show ()


# In[39]:


x = pd.concat ([pd.DataFrame ({'x': new_x_train}), pd.DataFrame ({'x': x_test})], axis=0)
x.shape


# In[40]:


x


# In[41]:


y = pd.concat ([pd.DataFrame ({'y': new_y_train}), pd.DataFrame ({'y': y_test})], axis=0)
y.shape


# In[42]:


y


# In[43]:


df = pd.concat ([x, y], axis=1)
df.shape 


# In[44]:


df


# In[45]:


plt.title ('Whole corrected Dataset')
plt.scatter (df.x, df.y)
plt.show ()


# In[46]:


corr = df.corr ()
corr


# In[47]:


rcParams ['figure.figsize'] = 4,2
plt.title ('Correlation between X & Y')
sns.heatmap (corr.iloc [0:1, :], annot=True)
plt.show ()


# ### Regression Line

# In[48]:


m, c = np.polyfit (new_x_train, new_y_train, 1)
m, c


# In[49]:


rcParams ['figure.figsize'] = 10, 7
plt.title ('Regression Line')
plt.xlabel ('X')
plt.ylabel ('y')
plt.plot (new_x_train, (m*new_x_train + c), label='RegL')
plt.grid ()
plt.legend ()
plt.show ()


# ### Model Building, Testing, Evaluating

# In[50]:


model = LinearRegression ()
model.fit (new_x_train.values.reshape (-1, 1), new_y_train)
r2_score (y_test, model.predict (x_test.values.reshape (-1, 1)))


# In[51]:


mean_squared_error(y_test, model.predict (x_test.values.reshape (-1, 1)), squared=False)


# # Prediction

# In[52]:


model.predict ([[92]])


# In[53]:


model.predict ([[121]])


# #### It shows the direct proportionality of Data.
