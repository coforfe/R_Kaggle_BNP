#------------------------------------------------------------
# BNP - Paribas - Claims Projections - 2016-02-15
#------------------------------------------------------------

# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 
setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP")

#Library loading
library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(xgboost)
library(stringr)

#Train and Test Load.
train <- read_csv("train.csv") 
y     <- train[, 'target']
train <- train[, -2]
test  <- read_csv("test.csv") 
train[is.na(train)] <- -1
test[is.na(test)]   <- -1

#Reduce significative variables
load("information_value.RData")
head(information_value, 14)
variables_top_14 <- information_value$Variable[1:14]

train02 <- train[, variables_top_14]
test02  <- test[, variables_top_14] 


#Character columns to numeric
col_char <- 0
j <- 0
for (i in 1:ncol(train02)) {
  cltmp <- class(train02[, i])
  if (cltmp == "character") {
    j <- j + 1
    col_char[j] <- i
    train02[,i] <- as.numeric( as.factor(train02[,i]) )
  } else next
}

col_char <- 0
j <- 0
for (i in 1:ncol(test02)) {
  cltmp <- class(test02[, i])
  if (cltmp == "character") {
    j <- j + 1
    col_char[j] <- i
    test02[,i] <- as.numeric( as.factor(test02[,i]) )
  } else next
}


#Define xgboost main function
dotest <- function(y, train, test, param0, iter) {
      n         <- nrow(train)
      xgtrain   <- xgb.DMatrix(as.matrix(train), label = y)
      xgval     <- xgb.DMatrix(as.matrix(test))
      watchlist <- list('train02' = xgtrain)
      model <- xgb.train(nrounds       = iter, 
                         params        = param0,
                         data          = xgtrain,
                         watchlist     = watchlist,
                         print.every.n = 100,
                         nthread       = 4 
      )
      p <- predict(model, xgval)
      rm(model)
      gc()
      p
}

# general , non specific params - just guessing
param012 <- list("objective"        = "binary:logistic",
                 "eval_metric"      = "logloss",
                 "eta"              = 0.002,
                 "subsample"        = 0.8,
                 "colsample_bytree" = 0.9,
                 "min_child_weight" = 1,
                 "max_depth"        = 15
)

# total analysis
pred012  <- read_csv("sample_submission.csv")
ensemble <- rep(0, nrow(test02))

# change to 1:5 to get result
ex_a <- Sys.time();  
for (i in 1:5) {
      set.seed(6879)
      p <- dotest(y, train02, test02, param012, 1300) 
      # change to 1300 or 1200, test02 by trial and error, have to add to local check which suggests 900, 
      # but have another 20% train02ing data to concider which gives longer optimal train02ing time
      ensemble <- ensemble + p
      (head(ensemble))
}
ex_b <- Sys.time(); 
ex_t <- ex_b - ex_a

pred012$PredictedProb <- ensemble/i


dat_tim <- str_replace_all(Sys.time()," |:","_")
file_tmp <- paste("Res_xxxxx_XGB_",dat_tim,"_.csv", sep = "")

write.csv(pred012, file_tmp, row.names = F, quote = F)
