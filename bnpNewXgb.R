#------------------------------------------------------------
# BNP - Paribas - Claims Projections - 2016-02-15
#------------------------------------------------------------

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

#Library loading
library(data.table)
library(caret)
library(stringr)
library(lubridate)
library(readr)
library(xgboost)

datIn <- read_csv("train.csv")
datIn <- datIn[, c(2,1, 3:ncol(datIn))]
datIn[is.na(datIn)] <- -1
#datIn$target <- as.factor(datIn$target)

datTest <- read_csv("test.csv")
datTest[is.na(datTest)] <- -1

col_char <- 0
j <- 0
for (i in 2:ncol(datIn)) {
  cltmp <- class(datIn[, i])
  if (cltmp == "character") {
    j <- j + 1
    col_char[j] <- i
    datIn[,i] <- as.numeric( as.factor(datIn[,i]) )
  } else next
}

col_char <- 0
j <- 0
for (i in 1:ncol(datTest)) {
  cltmp <- class(datTest[, i])
  if (cltmp == "character") {
    j <- j + 1
    col_char[j] <- i
    datTest[,i] <- as.numeric( as.factor(datTest[,i]) )
  } else next
}

y <- datIn[, 'target']

#-----------------------------------
# To remove correlated columns
datNum <- datIn[, -c(1,2)]
# without "-1"
for (i in 1:ncol(datNum)) {
  ctmp <- datNum[,i]
  val_lo <- ctmp == -1
  datNum[val_lo ,i] <- NA 
}
cor_col <- cor(datNum, use = 'pairwise.complete.obs')
fin_cor <- findCorrelation(cor_col, cutoff = .98, names = TRUE); fin_cor
nam_col <- names(datIn)
nam_def <- setdiff(nam_col, fin_cor)
datIn <- datIn[, nam_def]
nam_col_t <- names(datTest)
nam_def_t <- setdiff(nam_col_t, fin_cor)
datTest <- datTest[, nam_def_t]


#---------------------------------
#---------------------- XGB
#---------------------------------
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
                     nthread       = 5 
  )
  p <- predict(model, xgval)
  #rm(model)
  #gc()
  p
}

# general , non specific params - just guessing
paramXgb <- list("objective"        = "binary:logistic",
                 "eval_metric"      = "logloss",
                 "eta"              = 0.01,
                 "subsample"        = 0.8,
                 "colsample_bytree" = 0.8,
                 "min_child_weight" = 1,
                 "max_depth"        = 10
)

#Run Analysis
predXgb  <- read_csv("sample_submission.csv")
ensemble <- rep(0, nrow(datTest))

#Iterations to build an ensemble
ex_a <- Sys.time();  
n_iter <- 501
for (i in 1:1) {
  print(i)
   ex_ina <- Sys.time();  
  v_rnd <- round(abs(rnorm(1)*1e5),0)
  set.seed( v_rnd )
  
  p <- dotest(y, datIn[, 3:ncol(datIn)], datTest[, 2:ncol(datTest)], paramXgb, n_iter) 
  ensemble <- ensemble + p
  
  print(head(ensemble))
   ex_inb <- Sys.time(); ex_int <- ex_inb - ex_ina; 
   print(ex_int)
}
ex_b <- Sys.time(); 
ex_t <- ex_b - ex_a; ex_t

#--------------------------------------------------------
#-------------- PREDICTION WEIGHTED
#--------------------------------------------------------
predXgb$PredictedProb <- ensemble/i

#--------------------------------------------------------
#-------------- FILE UPLOAD
#--------------------------------------------------------
dat_tim <- str_replace_all(Sys.time()," |:","_")
file_tmp <- paste("Res_xxxxx_XGB_",dat_tim,"_niter_", n_iter,"_iter_",i,"_.csv", sep = "")

write.csv(predXgb, file_tmp, row.names = F, quote = F)


#---------------------------------------------------

# Without correlated columns
# Just one iterarion
# nrounds = 501.
# Time difference of 19.03439 mins
# [0]	train02-logloss:0.689159
# [100]	train02-logloss:0.491383
# [200]	train02-logloss:0.432660
# [300]	train02-logloss:0.405251
# [400]	train02-logloss:0.388226
# [500]	train02-logloss:0.375805

# Error I hadn't change datTest...
# Without correlated columns
# Just one iterarion
# nrounds = 1500.
# general , non specific params - just guessing
# paramXgb <- list("objective"        = "binary:logistic",
#                  "eval_metric"      = "logloss",
#                  "eta"              = 0.01,
#                  "subsample"        = 0.8,
#                  "colsample_bytree" = 0.8,
#                  "min_child_weight" = 1,
#                  "max_depth"        = 10
# [0]	train02-logloss:0.689197
# [100]	train02-logloss:0.491513
# [200]	train02-logloss:0.432969
# [300]	train02-logloss:0.405096
# [400]	train02-logloss:0.387688
# [500]	train02-logloss:0.375318
# [600]	train02-logloss:0.364260
# [700]	train02-logloss:0.354271
# [800]	train02-logloss:0.345690
# [900]	train02-logloss:0.337360
# [1000]	train02-logloss:0.329321
# [1100]	train02-logloss:0.321251
# [1200]	train02-logloss:0.313613
# [1300]	train02-logloss:0.306431
# [1400]	train02-logloss:0.299130
# [1500]	train02-logloss:0.292704
# Time difference of 51.72743 mins  


# paramXgb <- list("objective"        = "binary:logistic",
#                  "eval_metric"      = "logloss",
#                  "eta"              = 0.0081,
#                  "subsample"        = 0.80,
#                  "colsample_bytree" = 0.80,
#                  "min_child_weight" = 1,
#                  "max_depth"        = 15
# )
# 0]	train02-logloss:0.689778
# [100]	train02-logloss:0.459668
# [200]	train02-logloss:0.358014
# [300]	train02-logloss:0.300945
# [400]	train02-logloss:0.266116
# [500]	train02-logloss:0.244326
# [600]	train02-logloss:0.228641
# [700]	train02-logloss:0.217103
# [800]	train02-logloss:0.205261
# [900]	train02-logloss:0.194878
# [1000]	train02-logloss:0.185525
# [1100]	train02-logloss:0.175980
# [1200]	train02-logloss:0.167702
# [1300]	train02-logloss:0.159748
# [1400]	train02-logloss:0.152351
# (0.46642)




# paramXgb <- list("objective"        = "binary:logistic",
#                  "eval_metric"      = "logloss",
#                  "eta"              = 0.01,
#                  "subsample"        = 0.8,
#                  "colsample_bytree" = 0.8,
#                  "min_child_weight" = 1,
#                  "max_depth"        = 10
# )
# 0]	train02-logloss:0.688765
# [100]	train02-logloss:0.454200
# [200]	train02-logloss:0.369278
# [300]	train02-logloss:0.325294
# [400]	train02-logloss:0.299946
# [500]	train02-logloss:0.282869
# [600]	train02-logloss:0.268133
# [700]	train02-logloss:0.255154
# [800]	train02-logloss:0.243457
# [900]	train02-logloss:0.233304
# [1000]	train02-logloss:0.222038
# [1100]	train02-logloss:0.212373
# [1200]	train02-logloss:0.202796
# [1300]	train02-logloss:0.193615
# [1400]	train02-logloss:0.185561
# (0.46352)




# [100]	train02-logloss:0.481817
# [1000]	train02-logloss:0.472305
# [2000]	train02-logloss:0.467750
# [3000]	train02-logloss:0.464303
# [4000]	train02-logloss:0.461479
# [5000]	train02-logloss:0.458933
# [6000]	train02-logloss:0.456638
# [7000]	train02-logloss:0.454563
# [8000]	train02-logloss:0.452656
# [9000]	train02-logloss:0.450848
# [10000]	train02-logloss:0.449124
# [11000]	train02-logloss:0.447471
# [12000]	train02-logloss:0.445917
# [13000]	train02-logloss:0.444486
# [14000]	train02-logloss:0.443093
# [15000]	train02-logloss:0.441737
# [16000]	train02-logloss:0.440469
# [17000]	train02-logloss:0.439205
# [18000]	train02-logloss:0.437989
# [19000]	train02-logloss:0.436829
# [20000]	train02-logloss:0.435697
# [21000]	train02-logloss:0.434520
# [22000]	train02-logloss:0.433455
# [23000]	train02-logloss:0.432396
# [24000]	train02-logloss:0.431404
# [25000]	train02-logloss:0.430373
# [26000]	train02-logloss:0.429434
# [27000]	train02-logloss:0.428474
# [28000]	train02-logloss:0.427545
# [29000]	train02-logloss:0.426632
# [30000]	train02-logloss:0.425735 (1.18_hrs)



# Results
# paramXgb <- list("objective"        = "binary:logistic",
#                  "eval_metric"      = "logloss",
#                  "eta"              = 0.3,
#                  "subsample"        = 0.8,
#                  "colsample_bytree" = 0.6,
#                  "min_child_weight" = 1,
#                  "max_depth"        = 1
# )
# [1900]	train02-logloss:0.468105

