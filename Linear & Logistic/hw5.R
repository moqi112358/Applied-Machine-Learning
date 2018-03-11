#1.1
library(glmnet)
setwd("C:/Users/98302/Desktop/hw5")
read.csv("default_plus_chromatic_features_1059_tracks.txt",header=F)->data
latitude = as.matrix(data[,dim(data)[2]-1])
longitude = as.matrix(data[,dim(data)[2]])
data = as.matrix(data[,-c(dim(data)[2]-1,dim(data)[2])])
latitude_lm = lm(latitude~data)
mat<-matrix(1:4,2,2) 
layout(mat) 
layout.show(4)
plot(latitude_lm)
plot(latitude_lm$fitted.values,latitude)
latitude_r2 = summary(latitude_lm)$adj.r.squared
latitude_r2

longitude_lm = lm(longitude~data)
layout(mat) 
layout.show(4)
plot(longitude_lm)
longitude_r2 = summary(longitude_lm)$adj.r.squared
longitude_r2
#1.2
summary(latitude)
summary(longitude)
latitude_orginal <- latitude
longitude_orginal <- longitude

#negative = which(latitude<0)
#latitude[negative] = latitude[negative] + 90
#negative = which(longitude<0)
#longitude[negative] = longitude[negative] + 180
#latitude_new <- latitude
#longitude_new <- longitude

latitude_new <- latitude + 90
longitude_new <- longitude + 180

#boxcox for new
layout(1)
la_boxcox_new = boxcox(lm(latitude_new~data),lambda = seq(-5, 7, length = 100))
la_lambda_new = la_boxcox_new$x[which.max(la_boxcox_new$y)]
latitude_new_trans=(latitude_new^la_lambda_new-1)/la_lambda_new
latitude_new_trans.lm = lm(latitude_new_trans~data)
summary(latitude_new_trans.lm)$adj.r.squared
layout(mat) 
layout.show(4)
plot(latitude_new_trans.lm)
latitude_new_trans.mse = mean((latitude_new_trans.lm$fitted.values - latitude_new_trans)^2)
latitude_new_trans.mse

latitude_new_without_trans.lm = lm(latitude_new~data)
latitude_new_without_trans.r2 = summary(latitude_new_without_trans.lm)$adj.r.squared
latitude_new_without_trans.r2
plot(latitude_new_without_trans.lm)
latitude_new_without_trans.pred = (latitude_new_without_trans.lm$fitted.values^la_lambda_new-1)/la_lambda_new
latitude_new_without_trans.mse = mean((latitude_new_without_trans.pred - latitude_new_trans)^2)
latitude_new_without_trans.mse

layout(1)
lo_boxcox_new = boxcox(lm(longitude_new~data))
lo_lambda_new = lo_boxcox_new$x[which.max(lo_boxcox_new$y)]
longitude_new_trans=(longitude_new^lo_lambda_new-1)/lo_lambda_new
longitude_new_trans.lm = lm(longitude_new_trans~data)
summary(longitude_new_trans.lm)$adj.r.squared
layout(mat) 
layout.show(4)
plot(longitude_new_trans.lm)
longitude_new_trans.mse = mean((longitude_new_trans.lm$fitted.values - longitude_new_trans)^2)
longitude_new_trans.mse

longitude_new_without_trans.lm = lm(longitude_new~data)
longitude_new_without_trans.r2 = summary(longitude_new_without_trans.lm)$adj.r.squared
longitude_new_without_trans.r2
plot(longitude_new_without_trans.lm)
longitude_new_without_trans.pred = (longitude_new_without_trans.lm$fitted.values^lo_lambda_new-1)/lo_lambda_new
longitude_new_without_trans.mse = mean((longitude_new_without_trans.pred - longitude_new_trans)^2)
longitude_new_without_trans.mse


#1.3a ridge
#latitude
layout(1)
latitude_ridge_trans.lm = cv.glmnet(x=data,y=latitude_new_trans,alpha=0,nfold = 10,family = "gaussian")
plot(latitude_ridge_trans.lm)
plot(latitude_ridge_trans.lm$cvm)
latitude_ridge_trans.pred <- predict(latitude_ridge_trans.lm, s = 
                                       latitude_ridge_trans.lm$lambda.min, newx = data)
#latitude_ridge_trans.pred.original = (latitude_ridge.pred * la_lambda_new + 1)^(1/la_lambda_new)
latitude_ridge_trans.R2 = var(latitude_ridge_trans.pred)/var(latitude_new_trans)
latitude_ridge_trans.R2
latitude_ridge_trans.mse = mean((latitude_ridge_trans.pred - latitude_new_trans)^2)
latitude_ridge_trans.mse


latitude_ridge_without_trans.lm = cv.glmnet(x=data,y=latitude_new,alpha=0,nfold = 10,family = "gaussian")
plot(latitude_ridge_without_trans.lm )
plot(latitude_ridge_without_trans.lm$cvm)
latitude_ridge_without_trans.pred <- predict(latitude_ridge_without_trans.lm,
                                             s = latitude_ridge_without_trans.lm$lambda.min, newx = data)
#latitude_ridge.pred.original = (latitude_ridge.pred * la_lambda_new + 1)^(1/la_lambda_new)
latitude_ridge_without_trans.pred <- (latitude_ridge_without_trans.pred^la_lambda_new-1)/la_lambda_new
latitude_ridge_without_trans.R2 = var(latitude_ridge_without_trans.pred)/var(latitude_new_trans)
latitude_ridge_without_trans.R2
latitude_ridge_without_trans.mse = mean((latitude_ridge_without_trans.pred - latitude_new_trans)^2)
latitude_ridge_without_trans.mse

#longitude
longitude_ridge_trans.lm = cv.glmnet(x=data,y=longitude_new_trans,alpha=0,nfold = 10,family = "gaussian")
plot(longitude_ridge_trans.lm)
plot(longitude_ridge_trans.lm$cvm)
longitude_ridge_trans.pred <- predict(longitude_ridge_trans.lm, s = 
                                        longitude_ridge_trans.lm$lambda.min, newx = data)
longitude_ridge_trans.R2 = var(longitude_ridge_trans.pred)/var(longitude_new_trans)
longitude_ridge_trans.R2
longitude_ridge_trans.mse = mean((longitude_ridge_trans.pred - longitude_new_trans)^2)
longitude_ridge_trans.mse


longitude_ridge_without_trans.lm = cv.glmnet(x=data,y=longitude_new,alpha=0,nfold = 10,family = "gaussian")
plot(longitude_ridge_without_trans.lm )
plot(longitude_ridge_without_trans.lm$cvm)
longitude_ridge_without_trans.pred <- predict(longitude_ridge_without_trans.lm,
                                             s = longitude_ridge_without_trans.lm$lambda.min, newx = data)
longitude_ridge_without_trans.pred <- (longitude_ridge_without_trans.pred^lo_lambda_new-1)/lo_lambda_new
longitude_ridge_without_trans.R2 = var(longitude_ridge_without_trans.pred)/var(longitude_new_trans)
longitude_ridge_without_trans.R2
longitude_ridge_without_trans.mse = mean((longitude_ridge_without_trans.pred - longitude_new_trans)^2)
longitude_ridge_without_trans.mse



#1.3b lasso
#latitude
layout(1)
latitude_lasso_trans.lm = cv.glmnet(x=data,y=latitude_new_trans,alpha=1,nfold = 10,family = "gaussian")
plot(latitude_lasso_trans.lm)
plot(latitude_lasso_trans.lm$cvm)
latitude_lasso_trans.pred <- predict(latitude_lasso_trans.lm, s = 
                                       latitude_lasso_trans.lm$lambda.min, newx = data)
#latitude_lasso_trans.pred.original = (latitude_lasso.pred * la_lambda_new + 1)^(1/la_lambda_new)
latitude_lasso_trans.R2 = var(latitude_lasso_trans.pred)/var(latitude_new_trans)
latitude_lasso_trans.R2
latitude_lasso_trans.mse = mean((latitude_lasso_trans.pred - latitude_new_trans)^2)
latitude_lasso_trans.mse


latitude_lasso_without_trans.lm = cv.glmnet(x=data,y=latitude_new,alpha=1,nfold = 10,family = "gaussian")
plot(latitude_lasso_without_trans.lm )
plot(latitude_lasso_without_trans.lm$cvm)
latitude_lasso_without_trans.pred <- predict(latitude_lasso_without_trans.lm,
                                             s = latitude_lasso_without_trans.lm$lambda.min, newx = data)
#latitude_lasso.pred.original = (latitude_lasso.pred * la_lambda_new + 1)^(1/la_lambda_new)
latitude_lasso_without_trans.pred <- (latitude_lasso_without_trans.pred^la_lambda_new-1)/la_lambda_new
latitude_lasso_without_trans.R2 = var(latitude_lasso_without_trans.pred)/var(latitude_new_trans)
latitude_lasso_without_trans.R2
latitude_lasso_without_trans.mse = mean((latitude_lasso_without_trans.pred - latitude_new_trans)^2)
latitude_lasso_without_trans.mse

#longitude
longitude_lasso_trans.lm = cv.glmnet(x=data,y=longitude_new_trans,alpha=1,nfold = 10,family = "gaussian")
plot(longitude_lasso_trans.lm)
plot(longitude_lasso_trans.lm$cvm)
longitude_lasso_trans.pred <- predict(longitude_lasso_trans.lm, s = 
                                        longitude_lasso_trans.lm$lambda.min, newx = data)
longitude_lasso_trans.R2 = var(longitude_lasso_trans.pred)/var(longitude_new_trans)
longitude_lasso_trans.R2
longitude_lasso_trans.mse = mean((longitude_lasso_trans.pred - longitude_new_trans)^2)
longitude_lasso_trans.mse


longitude_lasso_without_trans.lm = cv.glmnet(x=data,y=longitude_new,alpha=1,nfold = 10,family = "gaussian")
plot(longitude_lasso_without_trans.lm )
plot(longitude_lasso_without_trans.lm$cvm)
longitude_lasso_without_trans.pred <- predict(longitude_lasso_without_trans.lm,
                                              s = longitude_lasso_without_trans.lm$lambda.min, newx = data)
longitude_lasso_without_trans.pred <- (longitude_lasso_without_trans.pred^lo_lambda_new-1)/lo_lambda_new
longitude_lasso_without_trans.R2 = var(longitude_lasso_without_trans.pred)/var(longitude_new_trans)
longitude_lasso_without_trans.R2
longitude_lasso_without_trans.mse = mean((longitude_lasso_without_trans.pred - longitude_new_trans)^2)
longitude_lasso_without_trans.mse

#2
read.table("default_of_credit_card_clients.txt",header=T)->data
data[1,]
dim(data)
x = as.matrix(data[,2:24])
y = data[25]
y <- as.factor(as.matrix(y))
alpha = c(0,0.25,0.5,0.75,1)
acc = NULL
e = 2.718282
for (i in 1:length(alpha)){
  lm = cv.glmnet(x, y, family = "binomial", alpha = alpha[i], type.measure = "class",nfold = 10)
  plot(lm,main=paste("alpha = ",alpha[i]))
  lm.pred <- predict(lm,s = lm$lambda.min, newx = x)
  p = (e^lm.pred)/(1+e^lm.pred)
  p[p>0.5] <- 1
  p[p<=0.5] <- 0
  lm.acc = sum(p == y)/length(y)
  acc[i] = lm.acc
}