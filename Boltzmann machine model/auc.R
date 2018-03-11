mean_field_roc <- function(img_data,hh,method){
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
  real = NULL
  pred = NULL
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
    
    pred = c(pred,as.vector(new_pi))
    real = c(real,as.vector(real_img))
    
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
  acc <- correct_pixel/(500*28*28)
  return(list(c(TPR, TNR),best_real,best_noise,best_recon,worst_real,worst_noise,worst_recon,acc,pred,real))
}
img_data <- load_image("train-images.idx3-ubyte")
img_label <- load_label("train-labels.idx1-ubyte")
result = mean_field_roc(img_data,0.2,method = 0)
pred = result[[9]]
real = result[[10]]
roc = prediction(pred,real)
auc = performance(roc,"tpr","fpr")
plot(auc)

