#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("C:/Users/muriu/Downloads/IT Salary Survey EU  2020.csv")


# In[10]:


df.head()


# In[11]:


df.shape


# In[31]:


df.info()


# In[32]:


df.describe()
df.isnull().sum()
df.duplicated().sum()


# In[22]:


df.describe()


# In[15]:


df.isnull().sum()


# In[17]:


df.duplicated().sum()


# In[33]:


df.describe()


# In[23]:


import seaborn as sns


# In[24]:


corr = df.corr()
sns.heatmap(corr, annot=True, cmap= 'coolwarm')


# In[27]:


import pandas as pd
import numpy as np
df = pd.DataFrame({'A : np.random.normal(0.1, 10,100)'})


# In[28]:


print(df)


# In[29]:


df = pd.read_csv("C:/Users/muriu/Downloads/IT Salary Survey EU  2020.csv")
print(df.skew())


# In[30]:


for column in df.columns:
    plt.figure()
    sns.histplot(df[column], kde=True)
    plt.title(f"{column} Skewness: {df[column].skew()}")
    plt.show()


# In[ ]:




