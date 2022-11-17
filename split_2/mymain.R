#####################################
# Load libraries
# Load your vocabulary and training data
#####################################
library(text2vec)
library(glmnet)
library(pROC)
library(slam)
library(dplyr)

myvocab <- scan(file = "myvocab.txt", what = character())
train = read.table("train.tsv",
                   stringsAsFactors = FALSE,
                   header = TRUE)
train$review = gsub('<.*?>', ' ', train$review)

vectorizer = vocab_vectorizer(create_vocabulary(myvocab,
                                                ngram = c(1L, 2L)))

#####################################
# Train a binary classification model
#####################################
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)

dtm_train = create_dtm(it_train, vectorizer)

set.seed(1852)
fit <- cv.glmnet(y = train$sentiment, x = dtm_train, family = 'binomial', alpha = 0)

#####################################
# Load test data, and 
# Compute prediction
#####################################
test <- read.table("test.tsv", stringsAsFactors = FALSE,
                   header = TRUE)

it_test = itoken(test$review,
                 preprocessor = tolower,
                 tokenizer = word_tokenizer)

dtm_test = create_dtm(it_test, vectorizer)

output = matrix(0,length(test$id), 2)

output[,1] <- test$id
output[,2] <- predict(fit, dtm_test, type="response", s = fit$lambda.1se)

output <- as.data.frame(output)
colnames(output) <- c("id","prob")

#####################################
# Store your prediction for test data in a data frame
# "output": col 1 is test$id
#           col 2 is the predicted probs
#####################################
write.table(output, file = "mysubmission.txt", 
            row.names = FALSE, sep='\t')