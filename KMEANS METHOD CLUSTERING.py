#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('C:\\Mis archivos\\Business Analytics\\Proyecto final Netflix\\netflix_titles.csv')


# In[3]:


#creating an overall class of the problem
class Data_Understanding:
    #initialzing the class
    def __init__(self,columns,tail,shape):
        self.columns = columns
        self.tail = tail
        self.shape = shape


results1 = Data_Understanding(data.columns, data.tail, data.shape)
#viewing the columns present
print(results1.columns)
#viewing the last 5 records
print(results1.tail)
#visualizing the shape of the data
print(results1.shape)


# In[4]:


#visualizing the data
import seaborn as sns
import matplotlib.pyplot as plt
class Data_visualization:
    def __init__(self,data):
        self.data = data
        
    #creating a heatmap that is going to show correlation between the variables    
    def Correlation(self):
        
        return sns.heatmap(self.data.corr(),annot=True,cmap = 'coolwarm')
        
heatmap = Data_visualization(data)
print(heatmap.Correlation())


# In[5]:


#cleaning data by checking null values
class Clean(Data_visualization):
    def __init__(self,data):
        super().__init__(data)
            
missing_values = Clean(data.isna().sum())

print(missing_values.data)# there are no null values


# In[6]:


data


# In[7]:


#checking the data types
data.dtypes


# In[8]:


#visualizing the target variable
data['type'].value_counts().plot(kind='pie',radius=2)


# In[9]:


#converting the object data types from non numerical to numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['show_id'] = le.fit_transform(data['show_id'])
data['type'] = le.fit_transform(data['type'])
data['title'] = le.fit_transform(data['title'])
data['show_id'] = le.fit_transform(data['show_id'])
data['director'] = le.fit_transform(data['director'])
data['cast'] = le.fit_transform(data['cast'])
data['date_added'] = le.fit_transform(data['date_added'])
data['country'] = le.fit_transform(data['country'])
data['rating'] = le.fit_transform(data['rating'])
data['duration'] = le.fit_transform(data['duration'])
data['listed_in'] = le.fit_transform(data['listed_in'])
data['description'] = le.fit_transform(data['description'])
data.dtypes


# In[10]:


#dropping the 'date_added' and 'description' columns

data = data.drop(columns=['date_added','description'])
data
        


# In[11]:


#creating x and y data matrices
x = data.drop(columns='type')
y = data['type']


# In[12]:



#MODELLING
#importing the random forest classifier
from sklearn.cluster import KMeans 

Kmeans_model = KMeans(n_clusters=2)

#splitting the data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=50,test_size=0.3)


# In[13]:


y_test


# In[14]:



Kmeans_model.fit(x_train,y_train)


# In[15]:


y_pred = Kmeans_model.predict(x_test)
y_pred


# In[16]:


#THE ELBOW METHOD


# In[17]:


# WCSS (within cluster sum of squares)


# In[18]:


## WCSS is the sum of squared distance between each point and the centroid in a cluster. When we plot the WCSS with the K value, the plot looks like an Elbow. As the number of clusters increases, the WCSS value will start to decrease. WCSS value is largest when K = 1.


# In[19]:


wcss = Kmeans_model.inertia_
wcss


# In[20]:


WCSS=[]
for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(x_test)
    wcss_iteration = kmeans.inertia_
    WCSS.append(wcss_iteration)
    
WCSS


# it has 6 clusters

# In[21]:

print('clustering')
number_clusters = range(1,7)
plt.figure(figsize=(10,7))
plt.plot(number_clusters,WCSS,color='red')
plt.scatter(number_clusters,WCSS,color='blue')
plt.xlabel('number clusters')
plt.ylabel('within cluster sum of squares')
plt.title('THE ELBOW METHOD')
plt.show()


# In[ ]:




