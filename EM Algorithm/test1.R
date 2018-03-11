#install.packages("jpeg")
#install.packages("matrixStats")
library(jpeg)
library(matrixStats)
options(digits=10)
setwd("C:/Users/98302/Desktop/EM/HW_6_part_2")
fish <- readJPEG("fish.jpg")
flower <- readJPEG("flower.jpg")
sky <- readJPEG("sky.jpg")


timestart<-Sys.time()
pic_cluster <- function(picture, clusters, seed){
  timestart<-Sys.time()
  img_row <- dim(picture)[1]
  img_col <- dim(picture)[2]
  vector_length <- img_row * img_col
  mtx <- matrix(0, 3, vector_length)   # xi   3 * (307200) #pixels
  w_ij <- matrix(0, vector_length, clusters)
  segment_mean <- matrix(0, 3, clusters)   # segment mean matrix 3 * number of segments
  segment_weight <- matrix(0, 1, clusters)  # weight of each segment 1 * number of segments 
  stop_criteria <- .1
  smooth_val <- 0.000001
  Q_value <- NULL
  
  for(i in 1:img_row){
    for(j in 1:img_col){
      mtx[,((i-1)*img_col)+j] <- picture[i, j, ]
    }
  }
  #================kmean, setting init==================
  image_dim <- dim(picture)
  imgRGB <- data.frame(
    x = rep(1:image_dim[2], each = image_dim[1]),
    y = rep(image_dim[1]:1, image_dim[2]),
    R = as.vector(picture[,,1]),
    G = as.vector(picture[,,2]),
    B = as.vector(picture[,,3])
  )
  set.seed(seed)
  kmean <- kmeans(imgRGB[, c("R", "G", "B")], centers = clusters, iter.max = 500)
  for(j in 1:clusters){
    segment_mean[, j] <- kmean$centers[j, ]
    segment_weight[j] <- kmean$size[j]/vector_length   # initial probability of each segment
  }
  
  #================EM==================
  while(TRUE){
    #================E Step==================
    current_q <- 0
    inner <- matrix(0, vector_length, clusters)
    for(j in 1:clusters){
      very_inner <- t(mtx - t(segment_mean)[j, ])
      #inner[, j] <- (-1/2) * rowSums(very_inner ^ 2)
      inner[, j] <- 100 * -(1/2) * rowSums(very_inner ^ 2) + log(segment_weight[j])
    }
    #calculate w_ij s
	top <- exp(inner)
    bottom <- rowSums(top)
    w_ij <- top/bottom     # vector_length * clusters
	 if(sum(is.nan(w_ij)) > 0){
	    w_ij <- matrix(0,vector_length,clusters) 
	    for(i in 1:vector_length){
	      w_ij[i,] = exp(inner[i,]-logSumExp(inner[i,]))
	    }
	 }
    
    current_q <- sum(inner * w_ij)  # inner product ----> Q value
    #print(current_q)
    Q_value <- c(Q_value, current_q)
    #================M Step==================
    for(j in 1:clusters){
      top <- colSums(t(mtx) * w_ij[,j]) + smooth_val
      segment_mean[, j] <- top/(sum(w_ij[, j]) + smooth_val)
      segment_weight[j] <- (sum(w_ij[, j]))/vector_length  
    }
    
    #stopping rule
    
    if(length(Q_value) > 1){
      #if(abs((Q_value[length(Q_value)] - Q_value[length(Q_value)-1])) < stop_criteria){
      if(abs(Q_value[length(Q_value)] - Q_value[length(Q_value)-1]) < stop_criteria){
        break
      }
    }
  }

  
  final <- array(0, c(img_row, img_col, 3))
  for(i in 1:img_row){
    for(j in 1:img_col){
      index <- (i-1)*img_col + j
      points <- mtx[, index]
      meanseg <- which(w_ij[index,] == max(w_ij[index,]))
      final[i, j, ] <- segment_mean[, meanseg]
    }
  }
  timeend<-Sys.time()
  runningtime<-timeend-timestart

  write(runningtime,file = "time",append = T)
  write(Q_value,file = "time",append = T)
  return(final)
}

writeJPEG(pic_cluster(fish, 10, 0), "fish_segmented10.jpg",quality = 1)
writeJPEG(pic_cluster(fish, 20, 0), "fish_segmented20.jpg",quality = 1)
writeJPEG(pic_cluster(fish, 50, 0), "fish_segmented50.jpg",quality = 1)

writeJPEG(pic_cluster(flower, 10, 0), "flower_segmented10.jpg",quality = 1)
writeJPEG(pic_cluster(flower, 20, 0), "flower_segmented20.jpg",quality = 1)
writeJPEG(pic_cluster(flower, 50, 0), "flower_segmented50.jpg",quality = 1)

writeJPEG(pic_cluster(sky, 10, 0), "sky_segmented10.jpg",quality = 1)
writeJPEG(pic_cluster(sky, 20, 0), "sky_segmented20.jpg",quality = 1)
writeJPEG(pic_cluster(sky, 50, 0), "sky_segmented50.jpg",quality = 1)

writeJPEG(pic_cluster(sky, 20, 0), "sky_segmented20_1.jpg",quality = 1)
writeJPEG(pic_cluster(sky, 20, 1), "sky_segmented20_2.jpg",quality = 1)
writeJPEG(pic_cluster(sky, 20, 2), "sky_segmented20_3.jpg",quality = 1)
writeJPEG(pic_cluster(sky, 20, 3), "sky_segmented20_4.jpg",quality = 1)
writeJPEG(pic_cluster(sky, 20, 4), "sky_segmented20_5.jpg",quality = 1)

timeend<-Sys.time()
runningtime<-timeend-timestart
write(runningtime,file = "time",append = T)       

