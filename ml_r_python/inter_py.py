# -*- coding: utf-8 -*-
"""
Machine Learning in Python
"""


import pandas as pd
import numpy as np
import os

###set working directory
#use this call to see the current working directory
os.getcwd()
os.chdir("/home/eli/Documents/Columbia/CDSS/cdss-tutorials/ml_r_python")

#####################SUPERVISED LEARNING########################################
#############Classification
#Let's load in the data
#load data
data = pd.read_csv("uspsdata.txt", sep = "\t", header=None)
#get as numpy array
data = data.values
labels = pd.read_csv("uspscl.txt", sep = "\t", header = None)
#get as numpy array
labels = labels.values.reshape(labels.shape[0])

##Split data into test and training data
#choose 10% of data randomly for testing
from sklearn import cross_validation
XTrain,XTest,yTrain,yTest = cross_validation.train_test_split(data,labels,
                                                              test_size = .1)

###Q: What is this data?
##A: Images!
#Use matplotlib imshow to see data
import matplotlib.pyplot as plt
#make a function to show image
def plotImg(v):
    #reshape the vector v
    v = v.reshape(16,16)
    plt.imshow(v, cmap = "winter")


######SVM
###Linear SVM
#Use sklearn's SVM
from sklearn import grid_search
from sklearn.svm import SVC
##Linear SVM has 1 parameter: cost. We need to find a good value for this
#tune the SVM
parameters = {"kernel":["linear"],'C':[10**i for i in range(-10,0)]}#create svm object
#use 5-fold cross validation
search = grid_search.GridSearchCV(SVC(),param_grid = parameters,refit=True,
                                  cv = 5)
search.fit(XTrain,yTrain)
#plot the validation error
errs = [1- score[1] for score in search.grid_scores_]
cs = parameters["C"]
plt.plot(cs,errs)
plt.semilogx()
plt.xlabel("Cost")
plt.ylabel("Validation Error")
plt.title("Linear SVM 5-Fold Cross Validation Error")
#what were the best parameters?
print(search.best_params_)
#what's our training error?
print(1 - search.score(XTest,yTest))
#Pretty good!

###Kernalized SVM
#Kernel Explanation video: https://www.youtube.com/watch?v=3liCbRZPrZA
#RBF Kernel SVM has two parameters: cost, and gamma. Let's do the same thing
params = {"kernel":["rbf"], "C":[10**i for i in np.arange(-1,2,.5)],
              "gamma":[10**i for i in np.arange(-10,0,1)]}
              
search = grid_search.GridSearchCV(SVC(),param_grid = params, refit = True,
                                  cv = 5)
search.fit(XTrain,yTrain)
#plot errors
errs = [1 - score[1] for score in search.grid_scores_]

errs = np.array(errs).reshape(len(params["C"]), len(params["gamma"]))
plt.imshow(errs,cmap="winter")
plt.xlabel('Gamma')
plt.ylabel('C')
plt.title("RBF SVM 5-Fold Cross Validation Error")
plt.colorbar()
plt.xticks(np.arange(len(params["gamma"])), params["gamma"], rotation=45)
plt.yticks(np.arange(len(params["C"])),params["C"])

#what were the best parameters?
print(search.best_params_)
#what's our training error?
print(1 - search.score(XTest,yTest))
#That's really good!

###Randomf FOrest
from sklearn import ensemble
#add trees one by one and see the error
rf = ensemble.RandomForestClassifier(warm_start = True, oob_score = True)
errs = []
#try 100 trees
for i in range(1,100):
    #change number of trees
    rf.set_params(n_estimators=i)
    rf.fit(XTrain,yTrain)
    errs.append(1 - rf.oob_score_)
#plot oob error to see how the number of trees affects the error
plt.plot([i for i in range(1,100)],errs)
plt.xlabel("Number of Trees")
plt.ylabel("OOB Error")
plt.title("OOB Error for Random Forest")  

#it seems to stabalize at around 65-70 trees, let's fit again
rf = ensemble.RandomForestClassifier(n_estimators = 70)
rf.fit(XTrain,yTrain)
print(1 - rf.score(XTest,yTest))
#Good accuracy!

#NOTE: Don't expect this kind of accuracy on a messy, complicated dataset

#####################UNSUPERVISED LEARNING#####################################
##############Dimensionality Reduction
######PCA
##Prepare the data
#extract the sixes
sixes = data[labels == 1]
#take a look at a six
plotImg(sixes[0,:])
from sklearn import decomposition
pca = decomposition.PCA(n_components = 16*16)
pca.fit(sixes)
#what does the average six look like?
plotImg(pca.mean_)
#let's look at the first few eigen-sixes
plotImg(pca.components_[0,:])
plotImg(pca.components_[1,:])
plotImg(pca.components_[2,:])
#how much variance is explained?
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of componenets")
plt.ylabel("Cumulative Variance Explained")
plt.title("Variance Explained in PCA")
#Can also reduce dimension of whole dataset
pca = decomposition.PCA(n_componenets = 2)
pca.fit(data)
#reduce to 2 dimensions
reduced = pca.transform(data)
#plot in 2 dimensions
plt.scatter(reduced[:,0],reduced[:,1], c = labels,cmap="jet")

##########Clustering
#####K-Means
##Cool visualization of K-means: http://www.naftaliharris.com/blog/visualizing-k-means-clustering/
####Let's generate some 2-d data
#normal mean (0,0) identity covariance
x1 = np.random.multivariate_normal([0,0],np.eye(2),150)
#normal mean (1,2) correlation = .5
x2 = np.random.multivariate_normal([3,3],np.array([[1,-.5],[-.5,1]]),50)
x3 = np.random.multivariate_normal([-5,1],np.array([[1,.8],[.8,1]]),200)
#combine matrices
X = np.vstack((x1,x2,x3))
color = np.concatenate([np.repeat(1,150),np.repeat(2,50),np.repeat(3,200)])
#plot data
plt.scatter(X[:,0],X[:,1],c = color, cmap = "Set1")

#do kmeans with 3 clusters
from sklearn import cluster
kmeans = cluster.KMeans(n_clusters = 3,init = 'random')
kmeans.fit(X)
#get cluster labels
clusters = kmeans.predict(X)
#plot
plt.scatter(X[:,0],X[:,1],c = clusters, cmap = "Set1")
#What about k=5?
#There are many ways for k-means to converge
def plot_kmeans(X):
    kmeans = cluster.KMeans(n_clusters = 5,init = 'random')
    kmeans.fit(X)
    #get cluster labels
    clusters = kmeans.predict(X)
    #plot
    plt.scatter(X[:,0],X[:,1],c = clusters, cmap = "Set1")   
    
