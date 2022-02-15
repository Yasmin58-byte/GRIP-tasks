#!/usr/bin/env python
# coding: utf-8

# # unsupervised learning using kmeans
# 

# ## Author : Yasmin Soliman
# 

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
df = pd.DataFrame(iris.data,columns = iris.feature_names)
print(df.head())
df.describe()


# In[14]:


from sklearn.cluster import KMeans
ks = range(1,10)
n = df.iloc[:, [0, 1, 2,3]].values

inertias = []
for k in ks:
    #make instance of kmeans
    model = KMeans(n_clusters = k)
    model.fit(n)
    inertias.append(model.inertia_)
    
plt.plot(ks,inertias)
plt.xlabel('Number of clusters')
plt.ylabel('Inertias')
plt.show()


# ## from here we notice that the optimum number of clusters is 3

# In[16]:


model = KMeans(n_clusters = 3)
labels= model.fit_predict(n)


# In[21]:


xs = n[:,0]
ys = n[:,1]
plt.scatter(xs,ys,c = labels,s = 50)
centroids = model.cluster_centers_

centroid_x= centroids[:,0]
centroid_y= centroids[:,1]
plt.scatter(centroid_x,centroid_y,s = 50,marker='D')


# In[ ]:




