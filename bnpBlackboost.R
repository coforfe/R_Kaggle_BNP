
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

#Data loading
datIn <- fread("train.csv")
datIn <- as.data.frame(datIn)

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

val <- 1000
for( i in 3:ncol(datIn) ) {
  vtmp <- datIn[,i]
  if(class(vtmp) =="numeric") {
    vtmp <- ifelse(is.na(vtmp), val, vtmp)
    datIn[,i] <- vtmp
  }  
  if(class(vtmp) =="character") {
    tbl_tmp <- table(vtmp)
    val_s <- names(tbl_tmp[tbl_tmp==max(tbl_tmp)])
    vtmp <- ifelse(vtmp=="", val_s, vtmp)
    datIn[,i] <- as.factor(vtmp)
  }   
}
rm(tbl_tmp, vtmp, val_s)

datIn$target <- ifelse(datIn$target == "0", "no", "yes")
datIn$target <- as.factor(datIn$target)

# There are some columns with many factors
# Count number of levels and remove those with more than 30
lev_df <- data.frame(nu_col=0, nu_lev=0)
cont <- 0
for(i in 1:ncol(datIn)) {
  vtmp <- datIn[,i]
  if(class(vtmp)=="factor") {
    cont <- cont + 1
    n_lev <- length(levels(vtmp))
    lev_df[cont, 1] <- i
    lev_df[cont, 2] <- n_lev
  }
}
rm(cont, n_lev, i)
nu_fac <- 30
idx_tmp <- which(lev_df$nu_lev> nu_fac, arr.ind=TRUE)
col_del <- lev_df[idx_tmp,1]
datIn <- datIn[, -col_del]

rm(idx_tmp, val, col_del)

#Transform factors in numeric 
for( i in 2:ncol(datIn)) {
  cltmp <- class(datIn[, i])
  if(cltmp == "factor") {
    datIn[,i] <- as.numeric( datIn[,i] )
  } else next
}

# Feature Engineering - kurtosis each column vs.v50..
# but per row??.....

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
#- blackboost CART - Boosted Tree
#---------------------------------
setwd("~/Downloads")

a <- Sys.time();a
set.seed(20792)

bootControl <- trainControl(
  number=10, verboseIter=TRUE, 
  summaryFunction = twoClassSummary
  ,classProbs = TRUE
) 


bootControl <- trainControl(number=5)

blackboostGrid <- expand.grid(
  mstop = 50,
  maxdepth = seq(3,10, 1)
)

modFitblackboost <- train(
  target ~ .
  ,data = trainDat
  ,method = "blackboost"
  ,trControl = bootControl
  ,tuneGrid = blackboostGrid
  #,tuneLength = 3
  #,metric =  "ROC"
  ,metric =  "Accuracy"
)

modFitblackboost

predblackboost <- predict( modFitblackboost, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatblackboost <- confusionMatrix(testDat$target, predblackboost); conMatblackboost 
conMatblackboostdf <- as.data.frame(conMatblackboost$overall);
blackboostAcc <- conMatblackboostdf[1,1]; 
blackboostAcc <- as.character(round(blackboostAcc*100,2))
blackboostAcc
b <- Sys.time();b; b-a   

if( nrow(blackboostGrid) < 2  )  { resampleHist(modFitblackboost) } else  
{ plot(modFitblackboost, as.table=T) }
plot(modFitblackboost, as.table=T)

#Best iteration
modBest <- modFitblackboost$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitblackboost$times$final[3]
#Samples
samp <- dim(modFitblackboost$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
# Impblackboost <- varImp( modFitblackboost, scale=F)
# plot(Impblackboost, top=(2:ncol(trainDat)))


#Save trainDat, testDat and Model objects.
object.size(modFitblackboost)

# save(
#   trainDat, testDat, modFitblackboost,
#   file=paste("blackboost_",numvars,"vars_n",samp,"_grid_",blackboostAcc,"_.RData", sep="")
# )

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

#-------------------------------------------

# Results 3 - Mac 
# Time difference of 58.19683 mins
# bootControl <- trainControl(number=5)
# mstop maxdepth
# 7    50        9
# Accuracy : 0.7827         
# 95% CI : (0.7783, 0.787)


# Results 2 - Mac 
# Time difference of 21.48914 mins
# mstop maxdepth
# 1    50        9
# Accuracy : 0.7827         
# 95% CI : (0.7783, 0.787)


# # # # Results 2 - Mac 
# Time difference of 17.11868 mins
# blackboostGrid <- expand.grid(
#   mstop = 50,
#   maxdepth = 9
# )
# Accuracy : 0.7806         
# 95% CI : (0.7762, 0.785)

# # # # Results 1 - Mac 
# Time difference of ??????? 
# ,tuneLength = 10
# ROC was used to select the optimal model using  the largest value.
# The final values used for the model were mstop = 50 and maxdepth = 9. 
# Accuracy : 0.778           
# 95% CI : (0.7735, 0.7824)


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

#Data loading
datTestori <- fread("test.csv")
datTestori <- as.data.frame(datTestori)
datTestpre <- fread("test.csv")
datTestpre <- as.data.frame(datTestpre)

# If I transform based on chage as a matrix, everything get transformed as factor
# transform with a loop.
# numeric -> replace NA with 1000
# character -> replace "missing values" with majority of column (mode).
val <- 1000
for( i in 1:ncol(datTestpre) ) {
  vtmp <- datTestpre[,i]
  if(class(vtmp) =="numeric") {
    vtmp <- ifelse(is.na(vtmp), val, vtmp)
    datTestpre[,i] <- vtmp
  }  
  if(class(vtmp) =="character") {
    tbl_tmp <- table(vtmp)
    val_s <- names(tbl_tmp[tbl_tmp==max(tbl_tmp)])
    vtmp <- ifelse(vtmp=="", val_s, vtmp)
    datTestpre[,i] <- as.factor(vtmp)
  }   
}
rm(tbl_tmp, vtmp, val_s)

# There are some columns with many factors
# Count number of levels and remove those with more than 30
lev_df <- data.frame(nu_col=0, nu_lev=0)
cont <- 0
for(i in 1:ncol(datTestpre)) {
  vtmp <- datTestpre[,i]
  if(class(vtmp)=="factor") {
    cont <- cont + 1
    n_lev <- length(levels(vtmp))
    lev_df[cont, 1] <- i
    lev_df[cont, 2] <- n_lev
  }
}
rm(cont, n_lev, i)
nu_fac <- 30
idx_tmp <- which(lev_df$nu_lev> nu_fac, arr.ind=TRUE)
col_del <- lev_df[idx_tmp,1]
datTestpre <- datTestpre[, -col_del]

rm(idx_tmp, val, col_del)

#Transform factors in numeric 
for( i in 1:ncol(datTestpre)) {
  cltmp <- class(datTestpre[, i])
  if(cltmp == "factor") {
    datTestpre[,i] <- as.numeric( datTestpre[,i] )
  } else next
}
rm(lev_df, i, cltmp, nu_fac, vtmp)

#save(datTestpre, file="datTestpre.RData")

#------------------------------------------------------------
# PREDICT
#------------------------------------------------------------
modFit <- modFitblackboost 
in_err <- blackboostAcc
modtype <- modFit$method
samptmp <- modFit$resample; samp <- length(unique(samptmp$Resample))
numvars <- length(modFit$coefnames)
timval <- str_replace_all(Sys.time(), " |:", "_")

#Is it "prob" of 'yes' or 'no'....?
pred_BNP <- predict(modFit, newdata = datTestpre, type = "prob")
toSubmit <- data.frame(ID = datTestpre$ID, PredictedProb = pred_BNP$yes)

write.table(toSubmit, file=paste("Res_xxxx_", modtype,"_",numvars,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep=""),sep=","
            , row.names=FALSE,col.names=TRUE, quote=FALSE)


