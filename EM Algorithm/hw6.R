library(matrixStats)

##Problem 1
#read in files
set.seed(0)
setwd("C:/Users/98302/Desktop/EM")
vocab <- read.csv("vocab.nips.txt",header = F)
documents <- read.csv("docword.nips.txt", sep = " ", skip = 3, header = FALSE)

#Constants
document_num <- 1500 
vocab_num <- length(vocab[,1])
word_num <- dim(documents)[1]
topic_num <- 30
smoothing_constant <- .00025
stop_criteria <- .001

#Setup

colnames(documents) <- c("document", "word_id", "count")

#create documents
document_vocab <- matrix(0,document_num,vocab_num)
for(i in 1:word_num){
  document_vocab[documents[i,1], documents[i,2]] = documents[i,3]
}

#set initial values for topic_weight and topic_vocab
topic_weight<- matrix(1/topic_num,1,topic_num)
topic_vocab <- matrix(0,topic_num,vocab_num)
#set probs to be random values that sum to 1 for each topic 

set.seed(0)
for(i in 1:topic_num){
  x <- runif(vocab_num)
  topic_vocab[i,] <- x / sum(x)
}

Q_value <- NULL

while(TRUE){
  #E Step - calculate the expected value of log liklihood:
  inner <- document_vocab %*% t(log(topic_vocab)) #[1500*30] sums of features multiplied by probs for each doc and cluster
  w_ij <- matrix(0,document_num,topic_num) #inner + probablity of each cluster 
  #add logs of the topic_weight
  for(i in 1:topic_num){
    inner[,i] <- inner[,i] + log(topic_weight[i])
  }
  #calculate w_ij s
  for(i in 1:document_num){
    w_ij[i,] = exp(inner[i,]-logSumExp(inner[i,]))
  }
  #calculate Q_value
  Q_value <- c(Q_value, sum(inner * w_ij))
  #stopping rule
  if(length(Q_value) > 1){
    if(abs((Q_value[length(Q_value)] - Q_value[length(Q_value)-1])) < stop_criteria){
      break
    }
  }
  #M Step - update
  for(j in 1:topic_num){
    #Update with additive smoothing
    dem <- 0
    for(z in 1:1500){
        dem = dem + sum(document_vocab[z, ]) * w_ij[z, j]
    }
    topic_vocab[j, ] <- (colSums(document_vocab * w_ij[, j])+ smoothing_constant)/(dem + smoothing_constant)
    topic_weight[j] <- sum(w_ij[,j]) / document_num
    #numerator <- colSums(document_vocab * w_ij[,j]) + smoothing_constant
    #denominator <- sum(rowSums(document_vocab) * w_ij[,j]) + (smoothing_constant * vocab_num)
    #topic_vocab[j,] <-numerator/denominator
    #update pis
    #topic_weight[j] <- sum(w_ij[,j]) / document_num
  }
}

#plot
plot(as.vector(topic_weight), type='l', ylab = "probability", xlab="topic", main = "Probability a Topic is Selected")

colnames(topic_vocab) <- as.character(vocab[,1])
#10 highest occuring words per topic
table = NULL
topic = NULL
for(i in 1:topic_num){
  word <- rownames(as.data.frame(sort(topic_vocab[i,], decreasing = TRUE)[1:10]))
  table = rbind(table,word)
  topic = c(topic,paste("Topic", i , ":",seq=""))
}
rownames(table) <- topic
table
