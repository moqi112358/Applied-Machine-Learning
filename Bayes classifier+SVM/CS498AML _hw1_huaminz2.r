###3.1
#The UC Irvine machine learning data repository hosts a famous collection of data on whether a #patient has diabetes (the Pima Indians dataset), originally owned by the National Institute of #Diabetes and Digestive and Kidney Diseases and donated by Vincent Sigillito. This can be found #at http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes. This data has a set of #attributes of patients, and a categorical variable telling whether the patient is diabetic or #not. For several attributes in this data set, a value of 0 may indicate a missing value of the #variable.

##(a)
#Build a simple naive Bayes classifier to classify this data set. You should hold out 20% of the #data for evaluation, and use the other 80% for training. You should use a normal distribution to #model each of the class-conditional distributions. You should write this classifier yourself #(it’s quite straight- forward), but you may find the function createDataPartition in the R #package caret helpful to get the random partition.

setwd("C:/Users/98302/Desktop")
set.seed(3)
library(klaR)
library(caret)
#load data file
read.csv("pima-indians-diabetes.data",header=F)->data
head(data)

input <- data[,-9]#extract the input data
output <- data[,9]#extract the output data
accuracy_train<-array(dim=10)
accuracy_test<-array(dim=10)
sensitivity<-array(dim=10)
specificity<-array(dim=10)
#run 10 times
for(i in 1:10){
  #get the random partition tag
  createDataPartition(output, p=.8, list=FALSE) ->tag
  #80% is training data
  #Set up train and test splits
  input[tag,] -> train_input
  input[-tag,] -> test_input
  output[tag] -> train_output
  output[-tag] -> test_output
  #Set up positive and negative training splits
  train_output == 0 -> negative_tag
  train_output == 1 -> positive_tag
  train_input[negative_tag,] -> negative_train_input
  train_input[positive_tag,] -> positive_train_input
  #calculate log(P(+)) and log(P(-))
  log_p_positive <- log(sum(positive_tag)/length(train_output))
  log_p_negative <- log(sum(negative_tag)/length(train_output))
  #compute the raw model params of +/- examples
  #According to the training data, we compute the model params, positive_mean, 
  #positive_sd, negative_mean, negative_sd, which is the mean and standard deviation 
  #of each feature given +/- examples. These params will be used to compute the 
  #posterior probability
  positive_mean <- sapply(positive_train_input,mean,na.rm=T)
  positive_sd <- sapply(positive_train_input,sd,na.rm=T)
  negative_mean <- sapply(negative_train_input,mean,na.rm=T)
  negative_sd <- sapply(negative_train_input,sd,na.rm=T)
  #The classifier created from the training set using a Gaussian distribution assumption.
  #Assume all attributes in training data are normal distribution
  #(Actually we know some of them may not be)
  #Transform to Standard normal distribution so that we can calculate 
  #the probability easily.
  positive_tr_input <- t((t(train_input)-positive_mean) / positive_sd)
  negative_tr_input <- t((t(train_input)-negative_mean ) / negative_sd)
  #calculate the log(P(training example|+)) and log(P(training example|-))
  positive_tr_logP <- rowSums(apply(positive_tr_input,1:2,function(x)
                      log(dnorm(x))),na.rm = T) + log_p_positive
  negative_tr_logP <- rowSums(apply(negative_tr_input,1:2,function(x) 
                      log(dnorm(x))),na.rm = T) + log_p_negative
  #compare log(P(training example|+)) and log(P(training example|-))
  positive_tr_logP > negative_tr_logP -> tr_p_tag
  #calculate the accuracy of training data
  accuracy_train[i] <- sum(tr_p_tag == train_output) / length(train_output)
  #predict the label of test data
  positive_te_input <- t((t(test_input)-positive_mean) / positive_sd)
  negative_te_input <- t((t(test_input)-negative_mean ) / negative_sd)
  positive_te_logP <- rowSums(apply(positive_te_input,1:2,function(x)                               
                      log(dnorm(x))),na.rm = T) + log_p_positive
  negative_te_logP <- rowSums(apply(negative_te_input,1:2,function(x)                                
                      log(dnorm(x))),na.rm = T) + log_p_negative
  positive_te_logP > negative_te_logP -> te_p_tag
  #calculate the accuracy,sensitivity and specificity of test data
  result<-confusionMatrix(data=as.numeric(te_p_tag),test_output)
  accuracy_test[i] <- result$overall["Accuracy"]
  sensitivity[i] <- result$byClass["Sensitivity"]
  specificity[i] <- result$byClass["Specificity"]
}

#We run 10 times and the following is the accuracy, sensitivity, 
#specificity of the naive Bayes classifier

accuracy_train
accuracy_test
sensitivity
specificity
mean(accuracy_test)

#The mean accuracy of the test data is 0.7522876 The sensitivity is about 65-80% 
#and Specificity is about 65-80%.

##(b)
#Now adjust your code so that, for attribute 3 (Diastolic blood pressure), attribute 4 (Triceps #skin fold thickness), attribute 6 (Body mass index), and attribute 8 (Age), it regards a value #of 0 as a missing value when estimating the class-conditional distributions, and the posterior. #R uses a special number NA to flag a missing value. Most functions handle this number in #special, but sensible, ways; but you’ll need to do a bit of looking at manuals to check. Does #this affect the accuracy of your classifier?

rm(list=ls())
setwd("C:/Users/98302/Desktop")
set.seed(3)
library(klaR)
library(caret)
#load data file
read.csv("pima-indians-diabetes.data",header=F)->data
input <- data[,-9]#extract the input data
output <- data[,9]#extract the output data
#change value=0 to NA
for (j in c(3,5,6,8)){
  input[,j] == 0 -> flag
  input[flag,j] <- NA
}
accuracy_train<-array(dim=10)
accuracy_test<-array(dim=10)
sensitivity<-array(dim=10)
specificity<-array(dim=10)
#run 10 times
for(i in 1:10){
  #get the random partition tag
  #80% is training data
  createDataPartition(output, p=.8, list=FALSE) ->tag
  #Set up train and test splits
  input[tag,] -> train_input
  input[-tag,] -> test_input
  output[tag] -> train_output
  output[-tag] -> test_output
  #Set up positive and negative training splits
  train_output == 0 -> negative_tag
  train_output == 1 -> positive_tag
  train_input[negative_tag,] -> negative_train_input
  train_input[positive_tag,] -> positive_train_input
  #calculate log(P(+)) and log(P(-))
  log_p_positive <- log(sum(positive_tag)/length(train_output))
  log_p_negative <- log(sum(negative_tag)/length(train_output))
  #compute the raw model params of +/- examples
  #According to the training data, we compute the model params, positive_mean, 
  #positive_sd, negative_mean, negative_sd, which is the mean and standard deviation 
  #of each feature given +/- examples. These params will be used to compute the 
  #posterior probability
  positive_mean <- sapply(positive_train_input,mean,na.rm=T)
  positive_sd <- sapply(positive_train_input,sd,na.rm=T)
  negative_mean <- sapply(negative_train_input,mean,na.rm=T)
  negative_sd <- sapply(negative_train_input,sd,na.rm=T)
  #The classifier created from the training set using a Gaussian distribution assumption.
  #Assume all attributes in training data are normal distribution
  #(Actually we know some of them may not be)
  #Transform to Standard normal distribution so that we can calculate 
  #the probability easily.
  positive_tr_input <- t((t(train_input)-positive_mean) / positive_sd)
  negative_tr_input <- t((t(train_input)-negative_mean ) / negative_sd)
  #calculate the log(P(training example|+)) and log(P(training example|-))
  positive_tr_logP <- rowSums(apply(positive_tr_input,1:2,function(x)                               log(dnorm(x))), na.rm = T) + log_p_positive
  negative_tr_logP <- rowSums(apply(negative_tr_input,1:2,function(x)                               log(dnorm(x))), na.rm = T) + log_p_negative
  #compare log(P(training example|+)) and log(P(training example|-))
  positive_tr_logP > negative_tr_logP -> tr_p_tag
  #calculate the accuracy of training data
  accuracy_train[i] <- sum(tr_p_tag == train_output) / length(train_output)
  #predict the label of test data
  positive_te_input <- t((t(test_input)-positive_mean) / positive_sd)
  negative_te_input <- t((t(test_input)-negative_mean ) / negative_sd)
  positive_te_logP <- rowSums(apply(positive_te_input,1:2,function(x)                               log(dnorm(x))),na.rm = T) + log_p_positive
  negative_te_logP <- rowSums(apply(negative_te_input,1:2,function(x)                               log(dnorm(x))),na.rm = T) + log_p_negative
  positive_te_logP > negative_te_logP -> te_p_tag
  #calculate the accuracy,sensitivity and specificity of test data
  result<-confusionMatrix(data=as.numeric(te_p_tag),test_output)
  accuracy_test[i] <- result$overall["Accuracy"]
  sensitivity[i] <- result$byClass["Sensitivity"]
  specificity[i] <- result$byClass["Specificity"]
}

#Compared with (a), we regard a missing value as NA. We run 10 times and the following is the #accuracy, sensitivity, specificity of the naive Bayes classifier

accuracy_train
accuracy_test
sensitivity
specificity
mean(accuracy_test)

#The mean accuracy of the test data is 0.7176471 which is a little less the #result in (a) The #sensitivity is about 70-80% and Specificity is about 60-75%.

##(c)
#Now use the caret and klaR packages to build a naive bayes classifier for this data, assuming #that no attribute has a missing value. The caret package does cross-validation (look at train) #and can be used to hold out data. The klaR package can estimate class-conditional densities #using a density estimation procedure that I will describe much later in the course. Use the #cross-validation mechanisms in caret to estimate the accuracy of your classifier. I have not #been able to persuade the combination of caret and klaR to handle missing values the way I’d #like them to, but that may be ignorance (look at the na.action argument).

rm(list=ls())
setwd("C:/Users/98302/Desktop")
set.seed(3)
library(klaR)
library(caret)
#load data file
read.csv("pima-indians-diabetes.data",header=F)->data
input <- data[,-9]#extract the input data
output <- as.factor(data[,9])#extract the output data
#Set up train and test splits
#80% is training data
createDataPartition(output, p=.8, list=FALSE) ->tag
input[tag,] -> train_input
output[tag] -> train_output
#train the data with a naive bayes classifier
#use 10-fold cross-validation
#non-preProcessing
model<-train(train_input, train_output, 'nb', trControl=trainControl(method='cv', number=10))
model
#predict the label of test data
predict_result<-predict(model,newdata=input[-tag,])
#calculate the accuracy, sensitivity and specificity of test data
confusionMatrix(data=predict_result, output[-tag])

#Using the 10-fold cross-validation mechanisms, the accuracy of the naive bayes classifier was #estimated as 77.8%. The sensitivity and Specificity is 0.8700 and 0.6038.

##(d)
#Now install SVMLight, which you can find at http://svmlight.joachims.org, via the interface in #klaR (look for svmlight in the manual) to train and evaluate an SVM to classify this data. You #don’t need to understand much about SVM’s to do this — we’ll do that in following exercises. You #should hold out 20% of the data for evaluation, and use the other 80% for training. You should #NOT substitute NA values for zeros for attributes 3, 4, 6, and 8.

rm(list=ls())
setwd("C:/Users/98302/Desktop")
set.seed(3)
library(klaR)
library(caret)
#load data file
read.csv("pima-indians-diabetes.data",header=F)->data
input <- data[,-9]#extract the input data
output <- as.factor(data[,9])#extract the output data
#Set up train and test splits
#80% is training data
createDataPartition(output, p=.8, list=FALSE) ->tag
#train the data with svm
svm_model <- svmlight(input[tag,], output[tag], pathsvm='C:/Users/98302/Desktop/svm_light_windows64')
#predict the label
result <- predict(svm_model, input[-tag,])
predict_label <- result$class
#calculate the accuracy
accuracy <- sum(predict_label == output[-tag])/ length(output[-tag])
accuracy
#calculate sensitivity and specificity 
test_output <- output[-tag]
result<-confusionMatrix(data=as.numeric(predict_label),as.numeric(test_output))
sensitivity <- result$byClass["Sensitivity"]
specificity <- result$byClass["Specificity"]
sensitivity
specificity

#We hold out 20% of the data for testing, and use the other 80% for training. Using the SVM #model, the accuracy of the naive bayes classifier was estimated as 78.4%. The sensitivity and #specificity is 0.91 and 0.5471698.

###3.3
#The UC Irvine machine learning data repository hosts a collection of data on heart disease. The #data was collected and supplied by Andras Janosi, M.D., of the Hungarian Institute of #Cardiology, Budapest; William Steinbrunn, M.D., of the University Hospital, Zurich, Switzerland; #Matthias Pfisterer, M.D., of the University Hospital, Basel, Switzerland; and Robert Detrano, #M.D., Ph.D., of the V.A. Medical Center, Long Beach and Cleveland Clinic Foundation. You can #find this data at https://archive.ics.uci.edu/ml/datasets/Heart+Disease. Use the processed #Cleveland dataset, where there are a total of 303 instances with 14 attributes each. The #irrelevant attributes described in the text have been removed in these. The 14’th attribute is #the disease diagnosis. There are records with missing attributes, and you should drop these.

##(a)
#Take the disease attribute, and quantize this into two classes, num = 0 and num > 0. Build and #evaluate a naive bayes classifier that predicts the class from all other attributes Estimate #accuracy by cross-validation. You should use at least 10 folds, excluding 15% of the data at #random to serve as test data, and average the accuracy over those folds. Report the mean and #standard deviation of the accuracy over the folds.

rm(list=ls())
setwd("C:/Users/98302/Desktop")
set.seed(3)
library(klaR)
library(caret)
accuracy_test<-array(dim=10)
sensitivity<-array(dim=10)
specificity<-array(dim=10)
#load data file
read.csv("processed.cleveland.data",header=F)->data
#remove records with missing attributes
rowSums(data == "?") == 0 -> remove
data <- data[remove,]
#Quantize the disease attribute into two classes,num = 0 and num > 0
data[,14] > 0 -> flag
data[flag,14] <- 1
input <- data[,-14]#extract the input data
output <- as.factor(data[,14])#extract the output data
#Set up train and test splits
#85% is training data
for(i in 1:10){
  createDataPartition(output, p=.85, list=FALSE) ->tag
  input[tag,] -> train_input
  output[tag] -> train_output
  #train the data with a naive bayes classifier
  #use 10-fold cross-validation
  #non-preProcessing
  model<-train(train_input, train_output, 'nb', trControl=trainControl(method='cv', number=10))
  model
  #predict the label of test data
  predict_result<-predict(model,newdata=input[-tag,])
  #calculate the accuracy, sensitivity and specificity of test data
  result <- confusionMatrix(data=predict_result, output[-tag])
  accuracy_test[i] <- result$overall["Accuracy"]
  sensitivity[i] <- result$byClass["Sensitivity"]
  specificity[i] <- result$byClass["Specificity"]
}
accuracy_test
sensitivity
specificity
mean(accuracy_test)
sd(accuracy_test)

#Using the 10-fold cross-validation mechanisms, the mean of accuracy of the naive bayes #classifier was estimated as 82.5%. The standard deviation of accuracy was 0.0643. The #sensitivity is about 70-90% and Specificity is about 70-90%.

##(b).
#Now revise your classifier to predict each of the possible values of the disease attribute (0-4 #as I recall). Estimate accuracy by cross-validation. You should use at least 10 folds, excluding #15% of the data at random to serve as test data, and average the accuracy over those folds. #Report the mean and standard deviation of the accuracy over the folds.

rm(list=ls())
setwd("C:/Users/98302/Desktop")
set.seed(3)
library(klaR)
library(caret)
accuracy_test<-array(dim=10)
#load data file
read.csv("processed.cleveland.data",header=F)->data
#remove records with missing attributes
rowSums(data == "?") == 0 -> remove
data <- data[remove,]
input <- data[,-14]#extract the input data
output <- as.factor(data[,14])#extract the output data
#Set up train and test splits
#85% is training data
for(i in 1:10){
  createDataPartition(output, p=.85, list=FALSE) ->tag
  input[tag,] -> train_input
  output[tag] -> train_output
  #train the data with a naive bayes classifier
  #use 10-fold cross-validation
  #non-preProcessing
  model<-train(train_input, train_output, 'nb', trControl=trainControl(method='cv', number=10))
  model
  #predict the label of test data
  predict_result<-predict(model,newdata=input[-tag,])
  #calculate the accuracy of test data
  result <- confusionMatrix(data=predict_result, output[-tag])
  accuracy_test[i] <- result$overall["Accuracy"]
}
accuracy_test
mean(accuracy_test)
sd(accuracy_test)

#Using the 10-fold cross-validation mechanisms, the mean of accuracy of the naive bayes #classifier was estimated as 59.1%. The standard deviation of accuracy was 0.0333. 