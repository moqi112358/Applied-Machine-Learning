#library(reshape)

setwd("/home/huaminz2/cs498/hw7")
load_image <- function(path){
  val = list()
  fd = file(path,'rb')
  readBin(fd,'integer',n=1,size=4,endian='big')
  val$n = readBin(fd,'integer',n=1,size=4,endian='big')
  nrow = readBin(fd,'integer',n=1,size=4,endian='big')
  ncol = readBin(fd,'integer',n=1,size=4,endian='big')
  x = readBin(fd,'integer',n=val$n*nrow*ncol,size=1,signed=F)
  val$x = matrix(x, ncol=nrow*ncol, byrow=T)
  close(fd)
  val$x <- (val$x)[1:500, ]/255
  val$x[val$x < 0.5] <- -1.0
  val$x[val$x >= 0.5] <- 1.0
  return(val$x)
}

load_label <- function(path){
  fd = file(path,'rb')
  readBin(fd,'integer',n=1,size=4,endian='big')
  n = readBin(fd,'integer',n=1,size=4,endian='big')
  y = readBin(fd,'integer',n=n,size=1,signed=F)
  close(fd)
  return(y[1:500])
}

plot_img <- function(current){
  if(class(current) == "matrix"){
    mtx <- apply(current, 2, rev)
    image(1:28, 1:28, t(mtx))
  }else{
    mtx <- matrix(unlist(current), ncol = 28, byrow = T)
    mtx <- apply(mtx, 2, rev)
    image(1:28, 1:28, t(mtx))   
  }
}

img_data <- load_image("train-images.idx3-ubyte")
img_label <- load_label("train-labels.idx1-ubyte")

roc_helper <- function(real, output){
  tn <- sum(real[output == -1] == -1)
  fp <- sum(real[output == 1] == -1)
  fn <- sum(real[output == -1] == 1)
  tp <- sum(real[output == 1] == 1)
  return(c(tp, tn, fp, fn))
}

mean_field <- function(hh,pi){
  hx <- 2       #theta_{ij}=2 for the H_i, X_j terms
  tp <- 0
  tn <- 0
  fp <- 0
  fn <- 0
  
  stop_val <- 0.0000001
  best_acc <- 0
  worst_acc <- 1000
  correct_pixel <- 0
  best_real <- NULL
  best_noise <- NULL
  best_recon <- NULL
  worst_real <- NULL
  worst_noise <- NULL
  worst_recon <- NULL
  for(i in seq(500)){
    old_pi <- matrix(pi, 28, 28)
    new_pi <- matrix(pi, 28, 28)
    real_img <- matrix(unlist(img_data[i, ]), ncol = 28, byrow = T)
    sample_pixel <- sample(1:length(real_img), 16, replace = F)#random sample 2%
    noise_img <- real_img
    noise_img[sample_pixel] <- -noise_img[sample_pixel]#flip the bits
    #plot_img(noise_img)
    stop = 0
    while(stop < 1000){
      for(row in seq(28)){
        for(col in seq(28)){
          total <- 0
          if(col != 1)#left one
            total <- total + hh*(2*old_pi[row, col-1]-1) + hx*(noise_img[row, col-1]) 
          if(col != 28)#right one
            total <- total + hh*(2*old_pi[row, col+1]-1) + hx*(noise_img[row, col+1])
          if(row != 1)#upper one
            total <- total + hh*(2*old_pi[row-1, col]-1) + hx*(noise_img[row-1, col])
          if(row != 28)#lower one
            total <- total + hh*(2*old_pi[row+1, col]-1) + hx*(noise_img[row+1, col])
          new_pi[row, col] <- exp(total)/(exp(-total) + exp(total))
        }
      }
      stop = stop + 1
      if(norm(new_pi - old_pi, type = "F") < stop_val){
        break
      }
      old_pi <- new_pi
    }
    output_img <- new_pi
    output_img[output_img < 0.5] <- -1
    output_img[output_img >= 0.5] <- 1
    
    roc_result <- roc_helper(real_img, output_img)   # return(c(tp, tn, fp, fn))
    tp = tp + roc_result[1]
    tn = tn + roc_result[2]
    fp = fp + roc_result[3]
    fn = fn + roc_result[4]
    
    correct <- sum(output_img == real_img)
    correct_pixel = correct_pixel + correct
    if(correct < worst_acc){
      worst_acc = correct
      worst_real = real_img
      worst_noise = noise_img
      worst_recon = output_img
    }
    if(correct > best_acc){
      best_acc = correct
      best_real = real_img
      best_noise = noise_img
      best_recon = output_img
    }
  }# end of for loop
  TPR <- tp/(tp +fn)
  TNR <- tn/(tn + fp)
  return(list(c(TPR, TNR),best_real,best_noise,best_recon,worst_real,worst_noise,worst_recon,correct_pixel))
}

# reconstruction
set.seed(0)
result = mean_field(0.2,0.5)
TPR = result[[1]][1]
TNR = result[[1]][2]
c(TPR,TNR)
plot_img(result[[2]])
plot_img(result[[3]])
plot_img(result[[4]])
plot_img(result[[5]])
plot_img(result[[6]])
plot_img(result[[7]])
overall_acc <- result[[8]]/(500 * 784)
overall_acc
save(result,file = "result.Rdata")
#ROC
c <- seq(-1,1,0.1)
TPR_list <- NULL
TNR_list <- NULL
acc <- NULL
for(i in seq(length(c))){
  data <- mean_field(c[i],0.5)
  TPR_list <- c(TPR_list, data[[1]][1])
  TNR_list <- c(TNR_list, data[[1]][2])
  overall_acc <- data[[8]]/(500 * 784)
  acc <- c(acc,overall_acc)
}
save(TPR_list,TNR_list,acc,file = "ROC.Rdata")




