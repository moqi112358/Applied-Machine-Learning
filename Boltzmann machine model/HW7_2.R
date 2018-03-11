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

roc_helper <- function(real, output){
  tn <- sum(real[output == -1] == -1)
  fp <- sum(real[output == 1] == -1)
  fn <- sum(real[output == -1] == 1)
  tp <- sum(real[output == 1] == 1)
  return(c(tp, tn, fp, fn))
}

mean_field <- function(img_data,hh,method){
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
    real_img <- matrix(unlist(img_data[i, ]), ncol = 28, byrow = T)
    sample_pixel <- sample(1:length(real_img), 16, replace = F)#random sample 2%
    noise_img <- real_img
    noise_img[sample_pixel] <- -noise_img[sample_pixel]#flip the bits
    if(method == 0){
      old_pi <- matrix(0.5, 28, 28)
      new_pi <- matrix(0.5, 28, 28)
    }else if(method == 1){
      pi = noise_img
      pi[pi == -1] = 0
      old_pi <- pi
      new_pi <- pi
    }else if(method == 2){
      image = noise_img
      pi = matrix(0, 28, 28)
      count =  matrix(0, 28, 28)
      for(row in seq(28)){
        for(col in seq(28)){
          if(col != 1){#left one
            pi[row,col] = pi[row,col] + (image[row, col-1] == 1)
            count[row,col] = count[row,col] + 1
          }
          if(col != 28){#right one
            pi[row,col] = pi[row,col] + (image[row, col+1] == 1)
            count[row,col] = count[row,col] + 1
          }
          if(row != 1){#upper one
            pi[row,col] = pi[row,col] + (image[row-1, col] == 1)
            count[row,col] = count[row,col] + 1
          }
          if(row != 28){#lower one
            pi[row,col] = pi[row,col] + (image[row+1, col] == 1)
            count[row,col] = count[row,col] + 1
          }
        }
      }
      old_pi <- pi/count
      new_pi <- pi/count
    }else if(method == 3){
      old_pi = matrix(sample(0:1000,28*28)/1000, ncol = 28, byrow = T)
      new_pi = old_pi
    }
    
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

mean_field_diagonal <- function(img_data,hh,method){
  hx <- 2       #theta_{ij}=2 for the H_i, X_j terms
  tp <- 0
  tn <- 0
  fp <- 0
  fn <- 0
  stop_val <- 0.0000001
  correct_pixel <- 0
  for(i in seq(500)){
    real_img <- matrix(unlist(img_data[i, ]), ncol = 28, byrow = T)
    sample_pixel <- sample(1:length(real_img), 16, replace = F)#random sample 2%
    noise_img <- real_img
    noise_img[sample_pixel] <- -noise_img[sample_pixel]#flip the bits
    if(method == 0){
      old_pi <- matrix(0.5, 28, 28)
      new_pi <- matrix(0.5, 28, 28)
    }else if(method == 1){
      pi = noise_img
      pi[pi == -1] = 0
      old_pi <- pi
      new_pi <- pi
    }else if(method == 2){
      image = noise_img
      pi = matrix(0, 28, 28)
      count =  matrix(0, 28, 28)
      for(row in seq(28)){
        for(col in seq(28)){
          if(col != 1){#left one
            pi[row,col] = pi[row,col] + (image[row, col-1] == 1)
            count[row,col] = count[row,col] + 1
          }
          if(col != 28){#right one
            pi[row,col] = pi[row,col] + (image[row, col+1] == 1)
            count[row,col] = count[row,col] + 1
          }
          if(row != 1){#upper one
            pi[row,col] = pi[row,col] + (image[row-1, col] == 1)
            count[row,col] = count[row,col] + 1
          }
          if(row != 28){#lower one
            pi[row,col] = pi[row,col] + (image[row+1, col] == 1)
            count[row,col] = count[row,col] + 1
          }
        }
      }
      old_pi <- pi/count
      new_pi <- pi/count
    }else if(method == 3){
      old_pi = matrix(sample(0:1000,28*28)/1000, ncol = 28, byrow = T)
      new_pi = old_pi
    }
    
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
          if(col != 1 && row != 1)
            total <- total + hh*(2*old_pi[row-1, col-1]-1) + hx*(noise_img[row-1, col-1]) 
          if(col != 28 && row != 1)
            total <- total + hh*(2*old_pi[row-1, col+1]-1) + hx*(noise_img[row-1, col+1])
          if(col != 1 && row != 28)
            total <- total + hh*(2*old_pi[row+1, col-1]-1) + hx*(noise_img[row+1, col-1])
          if(col != 28 && row != 28)
            total <- total + hh*(2*old_pi[row+1, col+1]-1) + hx*(noise_img[row+1, col+1])
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
  }# end of for loop
  TPR <- tp/(tp +fn)
  TNR <- tn/(tn + fp)
  return(list(c(TPR, TNR),correct_pixel))
}
img_data <- load_image("train-images.idx3-ubyte")
img_label <- load_label("train-labels.idx1-ubyte")
method = c(0,1,2,3)
pi_TPR = NULL
pi_TNR = NULL
pi_acc = NULL
set.seed(0)
for(i in seq(length(c))){
  data <- mean_field(img_data,0.2,method = method[i])
  pi_TPR <- c(pi_TPR, data[[1]][1])
  pi_TNR <- c(pi_TNR, data[[1]][2])
  overall_acc <- data[[8]]/(500 * 784)
  pi_acc <- c(pi_acc,overall_acc)
}
save(pi_TPR,pi_TNR,pi_acc,file = "pi.Rdata")


