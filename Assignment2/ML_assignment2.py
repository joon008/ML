#!/usr/bin/env python
# coding: utf-8

# # DBSCAN

# It is a density-based clustering algorithm.
# Clusters are dense regions in the data space, separated by regions of the lower density of points. The DBSCAN algorithm is based on this intuitive notion of “clusters” and “noise”. The key idea is that for each point of a cluster, the neighborhood of a given radius has to contain at least a minimum number of points.

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# In[2]:


blob_df=pd.read_csv(r"C:\Users\Hp\Documents\ML_lab\cluster_blobs.csv")


# In[3]:


sns.pairplot(blob_df)


# In[4]:


cir_df=pd.read_csv(r"C:\Users\Hp\Documents\ML_lab\cluster_circles.csv")


# In[5]:


#PairPlot of the dataframe.
sns.pairplot(cir_df)


# In[6]:


moon_df=pd.read_csv(r"C:\Users\Hp\Documents\ML_lab\cluster_moons.csv")


# In[7]:


sns.pairplot(moon_df)


# In[8]:


df1=blob_df.values


# In[9]:


df2=cir_df.values


# In[10]:


df3=moon_df.values


# # eps: 
# The distance that specifies the neighborhoods. Two points are considered to be neighbors if the distance between them are less than or equal to eps.
# 
# if the eps value chosen is too small, a large part of the data will not be clustered. It will be considered outliers because don’t satisfy the number of points to create a dense region. On the other hand, if the value that was chosen is too high, clusters will merge and the majority of objects will be in the same cluster. The eps should be chosen based on the distance of the dataset (we can use a k-distance graph to find it), but in general small eps values are preferable.

# # minPoints:
# Minimum number of data points to define a cluster.
# 
# As a general rule, a minimum minPoints can be derived from a number of dimensions (D) in the data set, as minPoints ≥ D + 1. Larger values are usually better for data sets with noise and will form more significant clusters. The minimum value for the minPoints must be 3, but the larger the data set, the larger the minPoints value that should be chosen.

# # Dataset--> Blob.csv

# We just need to define eps and minPts values using eps and min_samples parameters.
# 
# We do not have to specify the number of clusters for DBSCAN which is a great advantage of DBSCAN over k-means clustering.

# In[11]:



# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
df1, labels_true = make_blobs(n_samples=1500, centers=centers, cluster_std=0.3,
                            random_state=0)

df1 = StandardScaler().fit_transform(df1)

# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(df1)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(df1, labels))

# Plot result
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = df1[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = df1[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


# Homogeneity :
# 
# Homogeneity measures how much the sample in a cluster are similar.
# means all of the observations with the same class label are in the same cluster. 
# 

# Completeness:
# 
# Completeness although measures how much similar samples are put together by the clustering algorithm.
# means all members of the same class are in the same cluster.

# V_measure :
# 
# a measure of the goodness of our clustering algorithm we can consider the harmonic average between homogeneity and completeness and obtain the V-measure 

# Kmeans for blob dataset

# In[12]:


kmeans =  KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(df1)
plt.scatter(df1[:,0], df1[:,1], c=kmeans_labels)
plt.show()


# In[13]:


metrics.silhouette_score(df1, kmeans_labels)


# silhouette_score for Kmeans is more closer to 1 which is means KMeans clusters are more well seperated in comparison to DBSCAN.
# 
# Both the cluster has given appropriate visualization but DBSCAN was able to locate the outliers in dataset.

# # Dataset--> Circle.csv

# In[14]:



df2, y = make_circles(n_samples=1500, factor=0.3, noise=0.1)
df2 = StandardScaler().fit_transform(df2)
y_pred = DBSCAN(eps=0.3, min_samples=10).fit_predict(df2)

plt.scatter(df2[:,0], df2[:,1], c=y_pred)
print('Number of clusters: {}'.format(len(set(y_pred[np.where(y_pred != -1)]))))
print('Homogeneity: {}'.format(metrics.homogeneity_score(y, y_pred)))
print('Completeness: {}'.format(metrics.completeness_score(y, y_pred)))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(df2, labels))


# Kmeans for circle dataset

# In[15]:


kmeans =  KMeans(n_clusters=2, random_state=0)
kmeans_labels = kmeans.fit_predict(df2)
plt.scatter(df2[:,0], df2[:,1], c=kmeans_labels)
plt.show()


# In[16]:


metrics.silhouette_score(df2, kmeans_labels)


# silhouette_score for Kmeans is more closer to 1 which is means KMeans clusters are more well seperated in comparison to DBSCAN
# 
# As we can see the DBSCAN is much more accurate. It is able to capture complex relationships between features. Further more, the algorithms was able to spot outliers

# # Dataset--> Moon.csv

# In[17]:


df3, y = make_moons(n_samples=1500,  noise=0.05)
df3 = StandardScaler().fit_transform(df3)
y_pred = DBSCAN(eps=0.3, min_samples=10).fit_predict(df3)

plt.scatter(df3[:,0], df3[:,1], c=y_pred)
print('Number of clusters: {}'.format(len(set(y_pred[np.where(y_pred != -1)]))))
print('Homogeneity: {}'.format(metrics.homogeneity_score(y, y_pred)))
print('Completeness: {}'.format(metrics.completeness_score(y, y_pred)))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(df3, labels))


# Kmeans for moon dataset

# In[18]:


kmeans =  KMeans(n_clusters=2, random_state=0)
kmeans_labels = kmeans.fit_predict(df3)
plt.scatter(df3[:,0], df3[:,1], c=kmeans_labels)
plt.show()


# In[19]:


metrics.silhouette_score(df3, kmeans_labels)


# silhouette_score for Kmeans is more closer to 1 which is means KMeans clusters are more well seperated in comparison to DBSCAN.
# 
# DBSCAN was able to form cluster which is more appropriate than cluster formed by Kmeans.

# # conclusion :
# 
# DBSCAN figure outs the number of clusters. DBSCAN works by determining whether the minimum number of points are close enough to one another to be considered part of a single cluster. DBSCAN is very sensitive to scale since epsilon is a fixed value for the maximum distance between two points. 
# In some cases, determining an appropriate distance of neighborhood (eps) is not easy and it requires domain knowledge.

# In[ ]:




