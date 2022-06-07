#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_excel (r'C:\Users\Stefan\Desktop\UAX\IRONHACKK\Proyecto Final\Solución Regresión IH\data\regression_data.xls')
df


# In[2]:


df.describe()


# In[3]:


sns.displot(df['price']);


# In[4]:


df.hist(bins=50, figsize=(15,15)) #shows us all the graphs
plt.show()


# In[5]:


pd.plotting.scatter_matrix(df, alpha=0.2)
plt.show()


# In[6]:


plt.figure(figsize=(15,15)) #considering we have latitude and longitude values we can learn what are the most common areas where
sns.jointplot(x=df.lat.values, y=df.long.values, height=15)#properties are bought
plt.ylabel('Longitude')
plt.xlabel('Latitude')
plt.show()


# In[7]:


df['bedrooms'].value_counts().plot(kind='bar') #we'll plot the number of bedrooms counted to see the most common ones
plt.title('Bedrooms number')
plt.xlabel('Bedrooms')
plt.ylabel('Counted')


# In[8]:


plt.scatter(df.price,df.sqft_living)
plt.title('Value of the property depending to living area')
#this way we render a graph that shows what's the estimated price depending on living area


# In[9]:


plt.scatter(df.price,df.lat)
plt.title('Value of the property depending on latitude location') #we obtain data according to lat location
plt.xlabel('Price')
plt.ylabel('Latitude')


# In[10]:


plt.scatter(df.price,df.long)
plt.title('Value of the property depending on longitude location') #we obtain data according to longitude location
plt.xlabel('Price')
plt.ylabel('Longitude')


# In[11]:


plt.scatter(df.price,df.bedrooms)
plt.title('Price according to ammount of bedrooms')
plt.xlabel('Price')
plt.ylabel('Bedrooms')
plt.show()


# In[12]:


plt.scatter(df.price,df.sqft_basement)
plt.title('Price related to sqft of basement')
plt.xlabel('Price')
plt.ylabel('Basement Area')


# In[13]:


#since we have both sqft of living area and basement area, it's ideal to combine both of them in the same plot
plt.scatter(df['price'],(df['sqft_living']+df['sqft_basement']))
plt.xlabel('Price')
plt.ylabel('Total Area')


# In[14]:


plt.scatter(df.price,df.waterfront) #Waterfront-Price
plt.xlabel('Price')
plt.ylabel('Waterfront?')


# In[15]:


plt.scatter(df.price,df.bedrooms) #Bedrooms-Price
plt.xlabel('Price')
plt.ylabel('Bedrooms')


# In[16]:


plt.scatter(df.price,df.floors) #Floors-Price
plt.xlabel('Price')
plt.ylabel('Floors')


# In[17]:


plt.scatter(df.price,df.condition) #Condition-Price
plt.xlabel('Price')
plt.ylabel('Condition')


# In[18]:


plt.scatter(df.price,df.grade) #Grade-Price
plt.xlabel('Price')
plt.ylabel('Grade')


# In[19]:


plt.figure(figsize=(5,5)) #Zipcode-Price
sns.jointplot(x=df.price, y=df.zipcode, height=5)
plt.ylabel('Zip-Code')
plt.xlabel('Price')
plt.show()


# In[20]:


df.floors.value_counts().plot(kind='bar')


# In[21]:


corrmat = df.corr() #we check correlation
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=.8, square= True);


# In[22]:


k=20
cols=corrmat.nlargest(k, 'price')['price'].index
f,ax = plt.subplots(figsize=(14,10))
sns.heatmap(df[cols].corr(), vmax=.8, square=True);


# In[23]:


#we can appreciate absolutely everything can affect values on the property price


# In[24]:


#########################################################################


# In[25]:


train = df.drop(['id', 'price'],axis=1)
train


# In[26]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


# In[27]:


reg = LinearRegression()


# In[28]:


labels = df['price']
dates = [1 if values == 2014 else 0 for values in df.date ]
df['date'] = dates
train1 = df.drop(['id', 'price'],axis=1)


# In[29]:


x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =1)


# In[30]:


reg.fit(x_train,y_train)


# In[31]:


reg.score(x_test,y_test)


# In[32]:


clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')


# In[33]:


clf.fit(x_train, y_train)


# In[34]:


clf.score(x_test,y_test) #our score is right above 85% so it should be valid


# In[35]:


t_sc = np.zeros((400),dtype=np.float64)
y_pred = reg.predict(x_test)
for i,y_pred in enumerate(clf.staged_predict(x_test)):
    t_sc[i]=clf.loss_(y_test,y_pred)


# In[36]:


testsc = np.arange((400))+1


# In[37]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(testsc,clf.train_score_,'b-',label= 'Set dev train')
plt.plot(testsc,t_sc,'r-',label = 'set dev test')


# In[38]:


pca = PCA()
pca.fit_transform(scale(train1))


# In[39]:


#all the properties over 650k check all the conditions basically which is very diverse, but it can be confirmed
#with all the plots, that, the properties that are within a long of -122.2 and lat of 47.6 have way bigger price ranges
#seems like zip codes around 98040 and 98000 are quite expensive, too, probably they have advantage of location
#the better the grading the more expensive properties tend to get, but, on the other hand, houses with a condition 3 tend 
#to be more expensive on avg than anything else

