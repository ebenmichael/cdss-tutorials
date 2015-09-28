#####CDSS R Tutorial
##Adapted from http://trevorstephens.com/post/72916401642/titanic-getting-started-with-r
##and https://statsguys.wordpress.com/2014/01/03/first-post/


###set working directory
#use this call to see the current working directory
getwd()
#change working directory to where titanic test and train data are
setwd("~/Documents/Columbia/CDSS/R Tutorial")
#alternatively, go to Session > Set working Directory > Choose Directory...


###Some basic data types
#numbers, '<-' is assignment operator
num <- 3
#vectors, use 'c()'
vec <- c(1,2,3,4)
#matrices, use 'matrix()' 
mat <- matrix(c(1,2,3,4,5,6,7,8,9),3,3)

###Functions
myFun <- function(x) {
  return(sin(x) + .07*x^2 - cos(x))
}

###Basic Plotting
##seq function
?seq
vec <- seq(from=-10, to = 10, by = .1)
##most basic plot
plot(vec,myFun(vec))
lines(vec,myFun(vec))

###Load in test and train data
#read.csv takes a file name as input and returns a data frame version of the spreadsheet
test <- read.csv("test.csv")
train <- read.csv("train.csv")
#take a look at the data frames
View(test)
View(train)
#Important: test and train data are different!
#Take a look at structure of train
str(train)

###How many people died?
#train$Survived isolates the column in the data frame called 'Survive'
#table cross tabulates
table(train$Survived)
#to get a proportion
prop.table(table(train$Survived))
#Most people died! Make that our predicition
testLength <- dim(test)[1]
test$Survived <- rep(0, testLength)
#data.frame() creates a new data frame
output1 <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
#write this new data frame to csv
write.csv(output1,file = 'prediction1.csv', row.names = FALSE)
#Submit to Kaggle and see that out prediction wasn't so good...


###Sex Model
#Maybe looking at the sex of those who survived can give us a better prediction?
#use prop.table and table again
table(train$Survived,train$Sex)
#we want column proportions
props <- prop.table(table(train$Survived,train$Sex),2)
#let's visualize this using the built in barplot function
barplot(props, xlab = "Sex", ylab = "Proportion", main = "Male and Female survival")
# so most women survived and most men died! Let's use that to make a new prediction
#set all values in test$Survived to 0
test$Survived <- 0
#bracket operator creates a subset of the columns and assigns 1 to them
#subset is defined to be the rows were test$Sex is female
#read this as "Where the sex of the passenger is female, set survived to 1
test$Survived[test$Sex == 'female'] <- 1

#create a new data frame and save to a csv
output2 <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(output2,file = 'prediction2.csv', row.names = FALSE)
#Getting better!

###Sex and age
#Let's take a look at age
#use R's built in hist command to make a histogram
hist(train$Age,breaks = 20, main = 'Histogram of Age', xlab = 'Age', ylab = 'Count')
#As of now we've only been using categorical values, so let's keep doing that
#How do we convert age to a categorical value? Child or not child
#create new child column in train and test data
train$Child <- 0
train$Child[train$Age < 18] <- 1
test$Child <- 0
test$Child[test$Age < 18] <- 1
#use aggregate function
? aggregate
aggregate(Survived ~ Child + Sex, data=train, FUN=function(x) {sum(x)/length(x)})
#there isn't much to help us here, male children were still less likely to survive than female children


###Sex fare and class
#what about the price of each ticket?
#use the hist function again
hist(train$Fare, breaks = 30)
#Convert Fare into a categorical variable: use quartiles
summary(train$Fare)
#create new FareQuartile column in train and test data
train$FareQuartile <- 0
train$FareQuartile[train$Fare <= 7.91] <- 'q1'
train$FareQuartile[7.91 < train$Fare & train$Fare <= 14.45] <- 'q2'
train$FareQuartile[14.45 < train$Fare & train$Fare <= 31] <- 'q3'
train$FareQuartile[31 < train$Fare] <- 'q4'

test$FareQuartile <- 0
test$FareQuartile[test$Fare <= 7.91] <- 'q1'
test$FareQuartile[7.91 < test$Fare & test$Fare <= 14.45] <- 'q2'
test$FareQuartile[14.45 < test$Fare & test$Fare <= 31] <- 'q3'
test$FareQuartile[31 < test$Fare] <- 'q4'
#use aggregate function again
aggregate(Survived ~ FareQuartile + Sex, data=train, FUN=function(x) {sum(x)/length(x)})
#still nothing too interesting

#try adding in class
aggregate(Survived ~ FareQuartile + Sex + Pclass, data=train, FUN=function(x) {sum(x)/length(x)})
#most women in third class who were in the upper 2 quartiles of Fare did not survive
#let's use this in our next prediction
test$Survived <- 0
test$Survived[test$Sex == 'female'] <- 1
test$Survived[test$Sex == 'female' & test$Pclass == 3 & test$FareQuartile != 'q1'] <- 0

#create a new data frame and save to a csv
output3 <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(output3,file = 'prediction3.csv', row.names = FALSE)
#Incremental change is good!

###Decision Trees
##library() loads a package
library(rpart)
##We need to install some new packages to print these trees nicely
#install.packages() instals a package. You can also do this in R studio
#by going to the packages tab and selecting install
install.packages('rattle')
install.packages('rpart.plot')
install.packages('RColorBrewer')
library(rattle)
library(rpart.plot)
library(RColorBrewer)

#Decision tree on Sex
fitS <- rpart(Survived ~ Sex, data=train,method="class")
#use the new package we just installed and loaded to plot this tree nicely
fancyRpartPlot(fitS)

#Fit a decision tree on the data using the original variables using rpart
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, method="class")
#plot decision tree
fancyRpartPlot(fit)
#use decision tree to predict the test data
prediction <- predict(fit,test,type = "class")
#save this prediction to a new dataframe, write to csv and submit
output4 <- data.frame(PassengerId = test$PassengerId, Survived = prediction)
write.csv(output4,file = 'prediction4.csv', row.names = FALSE)
#we can change some parameters if we want, check out rpart.control
? rpart.control


###Logistic Regression
##totally different approach
#First need to clean the data, get rid of NA values in age
#Simple way: if the age is NA, assign the average age
#use mean function, and only look at data that isn't NA
avgAge <- mean(train$Age[!is.na(train$Age)])
#set every NA age value to the average age
train$Age[is.na(train$Age)] <- avgAge
#do the same for test data
avgAgeTest <- mean(test$Age[!is.na(test$Age)])
test$Age[is.na(test$Age)] <- avgAgeTest


#now run logistic regression
logit <- glm(Survived ~ Pclass + Sex + Age, family = binomial, data = train)
#predict
survived <- predict.glm(logit, test,type = "response")
#logistic regression tells you the probability of having survived given sex, age, etc.
#need to round
predictionLogit <- round(survived)
#save this prediction to a new dataframe, write to csv and submit
output5 <- data.frame(PassengerId = test$PassengerId, Survived = predictionLogit)
write.csv(output5,file = 'prediction5.csv', row.names = FALSE)

