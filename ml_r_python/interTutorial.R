#####CDSS Intermediate R + Machine Learning Tutorial 4/22/15
#####Eli Ben-Michael

###set working directory
#use this call to see the current working directory
getwd()
#change working directory to where titanic test and train data are
setwd("~/Documents/Columbia/CDSS/Inter R")
#alternatively, go to Session > Set working Directory > Choose Directory...

#####################SUPERVISED LEARNING########################################
#############Classification
#Let's load in the data
#load data
data <- read.table("uspsdata.txt", sep = "\t", header=FALSE)
#load labels
labels <- read.table("uspscl.txt", sep = "\t", header=FALSE)
#convert labels to logical
labels <- labels == 1

##split into test and training data
#choose 10% of the data randomly for testing
ind <- sample(nrow(data),floor(nrow(data)/10))
#make training and test sets
train <- data[-ind,]
train.lab <- labels[-ind,]
test <- data[ind,]
test.lab <- labels[ind,]



###Q: What is this data?
##A: Images!
#helper rotation function
rot <- function(m) t(m)[,nrow(m):1]
#Make a function to show the image
plotImg <- function(x) {
  #convert from row of data frame to numeric vector
  x <- as.numeric(x)
  #convert to matrix
  mat <- matrix(x,16,16,byrow = TRUE)
  #use image function with grey scale
  image(rot(mat),col = grey((0:256) / 256))
}   

######SVM
#install package
install.packages("e1071")
#load library
library("e1071")

###Linear SVM
##Linear SVM has 1 parameter: cost. We need to find a good value for this
#tune the SVM
costs <- 2^(seq(-16,-6,1))

linear <- tune("svm",train,train.lab,ranges = list(cost = costs),kernel = "linear",
                  type = "C-classification",tune.control = tune.control(
                    sampling = "cross",cross = 5))
plot(linear, main="Linear SVM 5-Fold CV Error")
summary(linear)
#Which value of cost was the best?
cost.linear <- linear$best.model$cost
#train an SVM on all of the training data with this cost
linear.final <- svm(train,train.lab,cost = cost.linear,kernel="linear",
                    type = "C")
#Check our training error
prediction <- predict(linear.final,test)
misClass.linear <- sum(prediction != test.lab) / length(test.lab)
misClass.linear
#That's good test error! 

###RBF Kernel SVM
#Kernel Explanation video: https://www.youtube.com/watch?v=3liCbRZPrZA
##RBF Kernel SVM has two parameters: cost, and gamma. Let's do the same thing
##to find good values for these
costs <- 2^(seq(-1,2,.5))
gammas <- 2^(seq(-20,-13,1))
rbf <- tune("svm",train,train.lab,ranges = list(cost = costs,gamma = gammas),
              kernel = "radial", type = "C-classification",
              tune.control = tune.control(sampling = "cross",cross = 5))
plot(rbf,main = "Linear SVM 5-Fold CV Error")
summary(rbf)
#What cost,gamma pair was the best?
cost.rbf <- rbf$best.model$cost
gamma.rbf <- rbf$best.model$gamma
#The best coast,gamma pair is below
#cost.rbf <- 4 
#gamma.rbf <- 0.0001220703

#train and SVM with all of the training data
rbf.final <- svm(train,train.lab,cost = cost.rbf,gamma = gamma.rbf,
                  kernel="radial",type = "C")
#Check our training error
prediction <- predict(rbf.final,test)
misClass.rbf <- sum(prediction != test.lab) / length(test.lab)
misClass.rbf



######Random Forest
#install package
install.packages('randomForest')
#load library
library('randomForest')

##fit a random forest
#use as.factor() to force randomForest to do classification
randForest <- randomForest(train,as.factor(train.lab),type = "C")
#take a look at how the number of trees affects the error
plot(randForest)

##predict our training labels and see our error rate
prediction <- predict(randForest,test)
misClass.randForest <- sum(prediction != test.lab) / length(test.lab)
misClass.randForest


#########################UNSUPERVISED LEARNING################################
##############Dimensionality Reduction
######PCA
##Prepare the data
#Extract the 6s
sixes <- data[labels,]
#look at a six
plotImg(sixes[1,])

##Do PCA
pca <- prcomp(sixes,retx=TRUE)
#Plot the eigenvalues of the covariance matrix
plot(pca,type="l")
#Look at the summary
summary(pca)
##Some plots
#What does the average 6 look like?
plotImg(pca$center)
#Look at the first few "eigen-sixes"
plotImg(pca$rotation[,1])
plotImg(pca$rotation[,2])
plotImg(pca$rotation[,3])

##########Clustering
#####K-Means
##Cool visualization of K-means: http://www.naftaliharris.com/blog/visualizing-k-means-clustering/
####Let's generate some 2-d data
library(MASS)
###generate dataset with 3 multivariate random normal distributions

dat <- rbind(mvrnorm(150,c(0,0),diag(2)), #mean 0, identity covariance
             mvrnorm(50,c(3,3),matrix(c(1,-.5,-.5,1),ncol = 2,byrow = T)),
             mvrnorm(200,c(-5,1),matrix(c(1,.8,.8,1),ncol = 2,byrow = T)))
color.true <- c(rep(1,150),rep(2,50),rep(3,200))
##plot data
plot(dat,ylab='',xlab='')#col=color.true)
##run k means and see how data is clustered
clusters <- kmeans(dat,3)
#plot data with clusters
plot(dat,ylab='',xlab='',col=clusters$cluster)
#notice how we get different clusters with each restart
#there are multiple ways for k-means to converge
#try k means with more clusters
clusters <- kmeans(dat,5)
#plot data with clusters
plot(dat,ylab='',xlab='',col=clusters$cluster)
