j = 1
# setwd(paste("/Users/chenqiyang/Downloads/UIUC_PhD/UIUC-PhD/Fall-2022/CS598PSL/CS598PSL_Project3/split_", j, sep=""))

setwd(paste("/Users/chenqiyang/Downloads/UIUC_PhD/UIUC-PhD/Fall-2022/CS598PSL/CS598PSL_Project3/",sep=""))
getwd()
train = read.table("alldata.tsv", stringsAsFactors = FALSE, header = TRUE)

library(text2vec)
library(tidyverse)
library(text2vec)
library(glmnet)
library(ggrepel)
library(slam)
train$review = gsub('<.*?>', ' ', train$review)
stop_words = c("i", "me", "my", "myself", 
               "we", "our", "ours", "ourselves", 
               "you", "your", "yours", 
               "their", "they", "his", "her", 
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were", 
               "him", "himself", "has", "have", 
               "it", "its", "the", "us")
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
tmp.vocab = create_vocabulary(it_train, 
                              stopwords = stop_words, 
                              ngram = c(1L,4L))
tmp.vocab = prune_vocabulary(tmp.vocab, term_count_min = 10,
                             doc_proportion_max = 0.5,
                             doc_proportion_min = 0.001)
dtm_train  = create_dtm(it_train, vocab_vectorizer(tmp.vocab))
tmpfit = glmnet(x = dtm_train, 
                y = train$sentiment, 
                alpha = 1,
                family='binomial')
tmpfit$df
myvocab = colnames(dtm_train)[which(tmpfit$beta[, 44] != 0)]
txt <- c(myvocab)
writeLines(txt, "myvocab_R.txt")

v.size = dim(dtm_train)[2]
ytrain = train$sentiment

summ = matrix(0, nrow=v.size, ncol=4)
summ[,1] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==1, ]), mean)
summ[,2] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==1, ]), var)
summ[,3] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==0, ]), mean)
summ[,4] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==0, ]), var)

n1 = sum(ytrain); 
n = length(ytrain)
n0 = n - n1

myp = (summ[,1] - summ[,3])/
  sqrt(summ[,2]/n1 + summ[,4]/n0)

words = colnames(dtm_train)
id = order(abs(myp), decreasing=TRUE)[1:2000]
pos.list = words[id[myp[id]>0]]
neg.list = words[id[myp[id]<0]]

id1 = which(summ[, 2] == 0) # same as: which(summ[id0, 1] != 0)
id0 = which(summ[, 4] == 0) #same as: which(summ[id1, 3] != 0)
words[id1]
words[id0]

words[id0[! (id0 %in% id)]]
words[id1[! (id1 %in% id)]]
# words_train  = create_dtm(words, vocab_vectorizer(tmp.vocab))
words_train = dtm_train[,id]

Myvocab_triedTwo = glmnet(x = words_train, 
                y = ytrain, 
                alpha = 1,
                family='binomial')
Myvocab_triedTwo$df
myvocab2 = colnames(dtm_train)[which(Myvocab_triedTwo$beta[, 44] != 0)]
txt2 <- c(myvocab2)
writeLines(txt2, "myvocab_R2.txt")
