#------------------------------------------------------------
# BNP - Paribas - Claims Projections - 2016-02-15
#------------------------------------------------------------

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

# load("datIn1000.RData")
# setwd("~/Downloads")


#Library loading
library(data.table)
library(caret)
library(stringr)
library(lubridate)
library(readr)

#Data loading
# datIn <- fread("train.csv")
# datIn <- as.data.frame(datIn)
datIn <- read_csv("train.csv")
datOr <- read_csv("train.csv")

# Second columns "target" is the column to predict 1 ok / 0 fail
# Gotal to predict probability.
# Reorder datIn to have target the first variable
datIn <- datIn[, c(2,1, 3:ncol(datIn))]
#datIn$target <- ifelse(datIn$target == "0", "no", "yes")
#datIn$target <- as.factor(datIn$target)

## Info about dataset:
# 114321 x 133
# ID is not repeated (meaningful for model?)
# Many NAs around 44% per column. (Also in test.csv)
#na_col <- apply(as.matrix(datIn), 2, function(x) round(sum(is.na(x))/length(x),2) )

# No NAs -> 62561 x 133
#datIn_nona <- datIn[complete.cases(datIn), ]
# But problems later for prediction.
# For xgboost and randomForest change missing for value 10000

# If I transform based on chage as a matrix, everything get transformed as factor
# transform with a loop.
# numeric -> replace NA with 1000
# character -> replace "missing values" with majority of column (mode).

# #Feature engineering - NAs per row (%)
# na_row <- function(x) {
#   vtmp <- sum(is.na(x)) / length(x)
#   return(vtmp)
# }
# feperna <- apply(datIn, 1, na_row )
# datIn$feperna <- feperna

datIn[is.na(datIn)] <- -999

# val <- -999
# for( i in 3:ncol(datIn) ) {
#   vtmp <- datIn[,i]
#   if(class(vtmp) =="numeric") {
#     vtmp <- ifelse(is.na(vtmp), val, vtmp)
#     datIn[,i] <- vtmp
#   }  
#   if(class(vtmp) =="character") {
#     tbl_tmp <- table(vtmp)
#     val_s <- names(tbl_tmp[tbl_tmp==max(tbl_tmp)])
#     vtmp <- ifelse(vtmp=="", val_s, vtmp)
#     datIn[,i] <- as.factor(vtmp)
#   }   
# }
# rm(tbl_tmp, vtmp, val_s)

# datIn$target <- ifelse(datIn$target == "0", "no", "yes")
datIn$target <- as.factor(datIn$target)

# There are some columns with many factors
# Count number of levels and remove those with more than 30
# lev_df <- data.frame(nu_col=0, nu_lev=0)
# cont <- 0
# for(i in 1:ncol(datIn)) {
#   vtmp <- datIn[,i]
#   if(class(vtmp)=="factor") {
#     cont <- cont + 1
#     n_lev <- length(levels(vtmp))
#     lev_df[cont, 1] <- i
#     lev_df[cont, 2] <- n_lev
#   }
# }
# rm(cont, n_lev, i)
# nu_fac <- 30
# idx_tmp <- which(lev_df$nu_lev> nu_fac, arr.ind=TRUE)
# col_del <- lev_df[idx_tmp,1]
# datIn <- datIn[, -col_del]
# 
# rm(idx_tmp, val, col_del)

#Transform factors in numeric 
col_char <- 0
j <- 0
for( i in 2:ncol(datIn)) {
  cltmp <- class(datIn[, i])
  if(cltmp == "character") {
    j <- j + 1
    col_char[j] <- i
    datIn[,i] <- as.numeric( as.factor(datIn[,i]) )
  } else next
}

# To remove correlated columns
#col_num <- setdiff(3:ncol(datIn), col_char)
# datNum <- datIn[, -c(1,2,col_char)]
# cor_col <- cor(datNum)
# fin_cor <- findCorrelation(cor_col, cutoff = .95, names=TRUE)
# nam_col <- names(datIn)
# nam_def <- setdiff(nam_col, fin_cor)
# datIn <- datIn[, nam_def]


#save(datIn, file="datIn1000.RData")
#load("datIn1000.RData")

#--------------------------------------------------------
#-------------- READY TO MODEL
#--------------------------------------------------------
#Change set's size just to get something.
sizMod <- 1 * nrow(datIn)
datSamp <- datIn[sample(1:nrow(datIn), sizMod) , ]
#rm(datIn);gc()

inTrain <- createDataPartition(datSamp$target, p = 0.70 , list = FALSE)
trainDat <- datSamp[ inTrain, ]
testDat <- datSamp[ -inTrain, ]

library(doMC)
numCor <- parallel::detectCores() - 2; numCor
#numCor <- 2
registerDoMC(cores = numCor)


#---------------------------------
#---------------------- XGB (new version CARET)
#---------------------------------
setwd("~/Downloads")

set.seed(6879)
a <- Sys.time();a
# bootControl <- trainControl(number = 5,
#                             summaryFunction = twoClassSummary,
#                             classProbs = TRUE,
#                             verboseIter = FALSE)

bootControl <- trainControl(number=10)

xgbGrid <- expand.grid(
  eta = 0.3,
  max_depth = 1,
  nrounds = 50,
  gamma = 0,
  colsample_bytree = 0.6,
  min_child_weight = 1
)

#nrounds, max_depth, eta, gamma, colsample_bytree, min_child_weight
#eta (0,1) - default: 0.3
#max_depth (1-Inf) - default: 6
#gamma (0-Inf) - default: 0
#min_child_weight (0-Inf) - default: 1
#colsample_bytree (0-1) - default:1

modFitxgb <-  train(
  target ~ .,
  data = trainDat,
  trControl = bootControl,
  #tuneGrid = xgbGrid,
  #preProc = 'BoxCox',
  tuneLength = 3,
  metric = "Accuracy",
  method = "xgbTree",
  verbose = 1,
  objective = "binary:logistic",
  eval_metric = "logloss",
  num_class = 2
  #early.stop.round = 10
)

modFitxgb

predxgb <- predict( modFitxgb, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatxgb <- confusionMatrix(testDat$target, predxgb); conMatxgb
conMatxgbdf <- as.data.frame(conMatxgb$overall); xgbAcc <- conMatxgbdf[1,1]; xgbAcc <- as.character(round(xgbAcc*100,2))
b <- Sys.time();b; b-a

if( nrow(xgbGrid) < 2  )  { resampleHist(modFitxgb) } else
{ plot(modFitxgb, as.table=T) }
# resampleHist(modFitxgb) 
plot(modFitxgb, as.table=T) 

# #Variable Importance
# Impxgb <- varImp( modFitxgb, scale=F)
# plot(Impxgb, top=20)

#Best iteration
modBest <- modFitxgb$bestTune; modBest
modBestc <- paste(modBest[1],modBest[2],modBest[3], sep="_")
#Execution time:
modFitxgb$times$final[3]
#Samples
samp <- dim(modFitxgb$resample)[1]
numvars <- ncol(trainDat)
xgbAcc <- round(100*max(modFitxgb$results$Accuracy),2); xgbAcc

#Save trainDat, testDat and Model objects.
format(object.size(modFitxgb), units = "Gb")

save(
  trainDat, testDat, modFitxgb,
  file=paste("XGB_",numvars,"vars_n",samp,"_grid",modBestc,"_",xgbAcc,"__.RData", sep="")
)

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

#---------------------------------------------------

results <- function() {
  
# # Results - Reduced set no correlation
# Time difference of 6.800389 mins
# bootControl <- trainControl(number=5)
# tuneLength = 3,
#       nrounds max_depth eta gamma colsample_bytree min_child_weight
# 1      50         1 0.3     0              0.6                1
  
  
#------------------  
 # # Results - 14-15-16-17... - No Feature - No Remove columns. 
# Tested several options with sample 0.70 y 0.90.
# Wtih all of them, finally the xgbGrid is reduced to this very simple values..
# xgbGrid <- expand.grid(
#   eta = 0.01,
#   max_depth = 1,
#   nrounds = 1,
#   gamma = 0,
#   colsample_bytree = 0.01,
#   min_child_weight = 0
# )



# # Results - 13 - No Feature - No Remove columns. NOT remove columns
# bootControl <- trainControl(number=10)
# eta = seq(0.001, 1, length.out = 10),
# max_depth = seq(1, 20, length.out = 10),
# Time difference of 3.912359 hours
# nrounds max_depth   eta gamma colsample_bytree min_child_weight
# 41      50         1 0.445     0             0.35             0.55
# xgbAcc [1] 76.12

# # Results - 12 - No Feature - No Remove columns. NOT remove columns
# bootControl <- trainControl(number=10)
# Time difference of 3.512593 mins
# nrounds max_depth  eta gamma colsample_bytree min_child_weight
# 1      50         5 0.05     0              0.5              0.7
# xgbAcc [1] 76.06

# # Results - 11 - No Feature - No Remove columns. NOT remove columns
# grid 81
# Time difference of 1.153407 hours
# nrounds max_depth  eta gamma colsample_bytree min_child_weight
# 45      50         5 0.05     0              0.5              0.7
# bootControl <- trainControl(number=5)
# xgbAcc [1] 76.12

# # Results - 18 - No Feature - No Remove columns. NOT remove columns
# It improves...!!
# bootControl <- trainControl(number=5)
# xgbAcc [1] 76.12

# # Results - 17 - With Feature Engineering. NOT remove columns
# It does not improve...
# bootControl <- trainControl(number=5)
# xgbAcc [1] 76.08

# # Results - 16 - With Feature Engineering.
# It does not improve...
# bootControl <- trainControl(number=5)
# xgbAcc [1] 76.08

# # Results - 16 - With Feature Engineering.
# bootControl <- trainControl(number=50)
# xgbAcc [1] 76.14


# # Results - 15
# Time difference of 1.373918 hours
# bootControl <- trainControl(number=500)
# nrounds max_depth  eta gamma colsample_bytree min_child_weight
# 1      50         5 0.05     0              0.3              0.5
# xgbAcc [1] 76.13

# # Results - 14
# Time difference of 5.420202 mins
# nrounds max_depth  eta gamma colsample_bytree min_child_weight
# 1      50         5 0.05     0              0.3              0.5
# xgbAcc [1] 76.24

# # Results - 13
# Time difference of 5.88768 mins
# nrounds max_depth  eta gamma colsample_bytree min_child_weight
# 15      50         5 0.05     0              0.3              0.5
# xgbAcc [1] 76.24

# # Results - 12
# Time difference of 6.083278 mins
# nrounds max_depth  eta gamma colsample_bytree min_child_weight
# 1      50         1 0.03     0             0.01                1
# xgbAcc [1] 76.24

# Results - 11
# Time difference of 8.180648 mins
# nrounds max_depth  eta gamma colsample_bytree min_child_weight
# 1      50         1 0.01     0              0.3                1
# xgbAcc [1] 76.24

# Results - 10
# Time difference of 2.525026 mins
# nrounds max_depth  eta gamma colsample_bytree min_child_weight
# 1      50         1 0.05     0              0.6                1
# xgbAcc [1] 76.23


# Results - 9
# Time difference of 1.919867 hours
# bootControl <- trainControl(number=3)
# tuneLength = 6,
# nrounds max_depth eta gamma colsample_bytree min_child_weight
# 1      50         1 0.3     0              0.6                1
# xgbAcc [1] 76.23


# Results - 8
# bootControl <- trainControl(number=50)
# Time difference of 5.220811 mins
# nrounds max_depth eta gamma colsample_bytree min_child_weight
# 1      50         1 0.3     0              0.6                1
# Accuracy   Kappa          Accuracy SD  Kappa SD    
# 0.7611266  -1.904001e-06  0.002301814  1.649993e-05

# Results - 7
# Time difference of 2.551433 mins
# bootControl <- trainControl(number=25)
# metric = "Accuracy",
# Accuracy   Kappa          Accuracy SD  Kappa SD    
# 0.7606751  -3.757486e-06  0.002008261  2.954248e-05
# nrounds max_depth eta gamma colsample_bytree min_child_weight
# 1      50         1 0.3     0              0.6                1


# Results - 6
# tuneLength = 5,
# bootControl <- trainControl(number=5)
#       nrounds max_depth eta gamma colsample_bytree min_child_weight
# 1      50         1 0.3     0              0.6                1
# > xgbAcc [1] 0.7616033

# # Results - 5: 
# # With XGBLinear no improvement at all...
# Time difference of 22.02989 mins
# xgbGrid <- expand.grid(
#   nrounds = 200,
#   lambda = 0.4,
#   alpha = 0.25
# )
# ROC        Sens        Spec       ROC SD        Sens SD      Spec SD    
# 0.5003403  0.02311747  0.9766594  0.0009062016  0.001222147  0.001391988

# Results - 4:
# Even with basic results and leaving new parameters with
# default values... No gains..
# Time difference of 39.78067 mins
# xgbGrid <- expand.grid(
#   eta = 0.12,
#   max_depth = 10,
#   nrounds = 200,
#   gamma = 0,
#   colsample_bytree = 1,
#   min_child_weight = 1
# )
# ROC        Sens        Spec      ROC SD       Sens SD       Spec SD     
# 0.5015339  0.01109749  0.988743  0.002221733  0.0004452026  0.0009693951
# 


# #Results - 2 - 3
# With changes in eta, max_depth, gamma... ROC does not improve. (??).
# It is strange but it is happening...

# #Results - 1
# bootControl <- trainControl(number = 5,
# xgbGrid <- expand.grid(
#   eta = 0.13,
#   max_depth = 20,
#   nrounds = 50,
#   gamma = 0.501,
#   colsample_bytree = 0.201,
#   min_child_weight = 0.01
# )
# Time difference of 7.077075 mins
# ROC        Sens       Spec       ROC SD        Sens SD     Spec SD     
# 0.5015582  0.0177682  0.9821317  0.0001539689  0.00118234  0.0008852989


# #Results - 3 - With Feature Engineering - datIn + datIn_70s + datIn_150s
# Time difference of 4.658915 hours
# bootControl <- trainControl(number=5, verboseIter=TRUE)
# nrounds max_depth  eta gamma colsample_bytree min_child_weight
# 1     300        20 0.13 0.501            0.201             0.01
# Accuracy : 0.9915
# 95% CI : (0.9905, 0.9924)

}

#------------------------------------------------------------
# TEST
#------------------------------------------------------------

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

#load("datIn1000.RData")
#setwd("~/Downloads")

#Library loading
library(data.table)
library(caret)
library(stringr)
library(lubridate)
library(readr)

#Data loading
# datTestori <- fread("test.csv")
# datTestori <- as.data.frame(datTestori)
# datTestpre <- fread("test.csv")
# datTestpre <- as.data.frame(datTestpre)

datTestori <- read_csv("test.csv")
datTestpre <- read_csv("test.csv")

# If I transform based on chage as a matrix, everything get transformed as factor
# transform with a loop.
# numeric -> replace NA with 1000
# character -> replace "missing values" with majority of column (mode).
# val <- 1000
# for( i in 1:ncol(datTestpre) ) {
#   vtmp <- datTestpre[,i]
#   if(class(vtmp) =="numeric") {
#     vtmp <- ifelse(is.na(vtmp), val, vtmp)
#     datTestpre[,i] <- vtmp
#   }  
#   if(class(vtmp) =="character") {
#     tbl_tmp <- table(vtmp)
#     val_s <- names(tbl_tmp[tbl_tmp==max(tbl_tmp)])
#     vtmp <- ifelse(vtmp=="", val_s, vtmp)
#     datTestpre[,i] <- as.factor(vtmp)
#   }   
# }
# rm(tbl_tmp, vtmp, val_s)

datTestpre[is.na(datTestpre)] <- -999
datTestori[is.na(datTestori)] <- -999

# There are some columns with many factors
# Count number of levels and remove those with more than 30
# lev_df <- data.frame(nu_col=0, nu_lev=0)
# cont <- 0
# for(i in 1:ncol(datTestpre)) {
#   vtmp <- datTestpre[,i]
#   if(class(vtmp)=="factor") {
#     cont <- cont + 1
#     n_lev <- length(levels(vtmp))
#     lev_df[cont, 1] <- i
#     lev_df[cont, 2] <- n_lev
#   }
# }
# rm(cont, n_lev, i)
# nu_fac <- 30
# idx_tmp <- which(lev_df$nu_lev> nu_fac, arr.ind=TRUE)
# col_del <- lev_df[idx_tmp,1]
# datTestpre <- datTestpre[, -col_del]
# 
# rm(idx_tmp, val, col_del)

# #Transform factors in numeric 
# for( i in 1:ncol(datTestpre)) {
#   cltmp <- class(datTestpre[, i])
#   if(cltmp == "factor") {
#     datTestpre[,i] <- as.numeric( datTestpre[,i] )
#   } else next
# }
# rm(lev_df, i, cltmp, nu_fac, vtmp)

#Transform factors in numeric 
col_char <- 0
j <- 0
for( i in 2:ncol(datTestpre)) {
  cltmp <- class(datTestpre[, i])
  if(cltmp == "character") {
    j <- j + 1
    col_char[j] <- i
    datTestpre[,i] <- as.numeric( as.factor(datTestpre[,i]) )
  } else next
}

# To remove correlated columns
# nam_col <- names(datTestpre)
# nam_def <- setdiff(nam_col, fin_cor)
# datTestpre <- datTestpre[, nam_def]



#save(datTestpre, file="datTestpre.RData")

#------------------------------------------------------------
# PREDICT
#------------------------------------------------------------
  modFit <- modFitxgb 
  in_err <- xgbAcc
  modtype <-modFit$method
  samptmp <- modFit$resample; samp <- length(unique(samptmp$Resample))
  numvars <- length(modFit$coefnames)
  timval <- str_replace_all(Sys.time(), " |:", "_")
  
  #Is it "prob" of 'yes' or 'no'....?
  pred_BNP <- predict(modFit, newdata = datTestpre, type = "prob")
  toSubmit <- data.frame(ID = datTestpre$ID, PredictedProb = pred_BNP[1:114393,2])
  
  write.table(toSubmit, file=paste("Res_xxxx_", modtype,"_",numvars,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep=""),sep=","
              , row.names=FALSE,col.names=TRUE, quote=FALSE)
  
