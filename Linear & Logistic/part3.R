#install.packages("glmnet", repos = "http://cran.us.r-project.org")
#install.packages("SubLasso", repos = "http://cran.us.r-project.org")
#install.packages("caret")
#install.packages("ROCR")
#install.packages("AUC")
#install.packages("pROC")
library(pROC)
library(AUC)
library(caret)
library(ROCR)
library(glmnet)

set.seed(50602)

#df <- read.table(file.choose() ,sep="\t", header= F, fill = TRUE)
df <- read.table("//ad.uillinois.edu/engr/instructional/rwang67/Desktop/CS 498 Homework/tissues.txt" ,sep="\t", header= F, fill = TRUE)

# a positive sign to a normal tissue, and a negative sign to a tumor tissue.
result <- as.vector(df$V1)
for (i in 1:length(result))
{
  if(result[i] > 0)
  {
    result[i] = 1
  }
  else 
  {
    result[i] = 0
  }
}
#data <- (matrix(scan(file.choose()), nrow = 2000, byrow = T))   # row = 2000 col = 62
data <- (matrix(scan("//ad.uillinois.edu/engr/instructional/rwang67/Desktop/CS 498 Homework/matrix.txt"), nrow = 2000, byrow = T))

data <- t(data)       # row = 62  col = 2000   each row is a sample with 2000 genes
#f1 <- SubLasso(data, result, nfold = 5)

accuracy = 0

cv_partition <- createDataPartition(y = result, p = 0.7, list=FALSE) 
train_result <- result[cv_partition]
test_result <- result[-cv_partition]
train_data <- data[cv_partition, ]
test_data <- data[-cv_partition, ]

# aplha = 1 lasso with binomial distribution
#cvfit  <- cv.glmnet(train_data, train_result, alpha=1, family="binomial", type.measure = "class")
#lot(cvfit)

glmmod <- cv.glmnet(train_data, train_result, alpha=1, family="binomial", type.measure = "deviance", nfold = 5)
#glmmodauc <- cv.glmnet(train_data, train_result, alpha=1, family="binomial", type.measure = "auc", nfold = 5)

best_lambda = 0
best_accuracy = 0
for(i in 1:length(glmmod$lambda))
{
  prediction <- predict(glmmod, train_data, s = glmmod$lambda[i], type = "class")    # Give the misclassification error
  prediction <- as.numeric(prediction)
  accuracy <- sum(prediction == train_result)/length(prediction)
  if(best_accuracy < accuracy)
  {
    best_accuracy = accuracy
    best_lambda = glmmod$lambda[i]
  }
}
best_lambda
best_accuracy

coefficient <- coef(glmmod, s = best_lambda)
sprintf("The best model uses total of %d genes", sum(coefficient != 0))

plot(glmmod)
max(glmmod$cvm)  # deviance value
summary(glmmod$cvm)

#=======================================
#plot(glmmodauc)
#max(glmmodauc$cvm)
#summary(glmmodauc$cvm)
#=======================================

#prediction <- predict(glmmod, test_data, s = "lambda.min", type = "class")

# calculate probabilities for TPR/FPR for predictions
prediction <- predict(glmmod, test_data, s = best_lambda, type = "class")
prediction <- as.numeric(prediction)
testaccuracy <- sum(prediction == test_result)/length(prediction)

sprintf("The accuracy is: %.3f", testaccuracy)
roc_result <- roc(test_result, prediction)
sprintf("The Area under the curve is: %.3f", roc_result$auc)
plot(roc_result)


#pred <- prediction(prediction, test_result)
#perf <- performance(pred,"tpr","fpr")
#performance(pred,"auc")
#plot(perf,colorize=FALSE, col="red")
#lines(c(0,1),c(0,1),col = "gray", lty = 4 )
#=======================================
#accuracy <- accuracy /10

