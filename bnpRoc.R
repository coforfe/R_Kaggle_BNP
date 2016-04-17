
#------------------------------------------------------------
# BNP - Paribas - Claims Projections - 2016-02-15
#------------------------------------------------------------

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

load("datIn1000.RData")
setwd("~/Downloads")


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
#- ROC Based classifier
#---------------------------------
setwd("~/Downloads")

a <- Sys.time();a
set.seed(20792)

bootControl <- trainControl(
  number=5, verboseIter=TRUE,
  summaryFunction = twoClassSummary
  ,classProbs = TRUE
)

bootControl <- trainControl(number=10)

# rocGrid <- expand.grid(
#   xgenes =
# )

modFitroc <- train(
  target ~ .
  ,data = trainDat
  ,method = "rocc"
  ,trControl = bootControl
  #,tuneGrid = rocGrid
  ,tuneLength = 10
  ,metric =  "Accuracy"
)

modFitroc

predroc <- predict( modFitroc, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatroc <- confusionMatrix(testDat$target, predroc); conMatroc 
conMatrocdf <- as.data.frame(conMatroc$overall); 
rocAcc <- conMatrocdf[1,1]; 
rocAcc <- as.character(round(rocAcc*100,2))
rocAcc
b <- Sys.time();b; b-a   

if( nrow(rocGrid) < 2  )  { resampleHist(modFitroc) } else  
{ plot(modFitroc, as.table=T) }
plot(modFitroc, as.table=T) 

#Best iteration
modBest <- modFitroc$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitroc$times$final[3]
#Samples
samp <- dim(modFitroc$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
# Improc <- varImp( modFitroc, scale=F)
# plot(Improc, top=(2:ncol(trainDat)))


#Save trainDat, testDat and Model objects.
format(object.size(modFitroc), units = "Gb")

save(
  trainDat, testDat, modFitroc,
  file=paste("roc_",numvars,"vars_n",samp,"_grid_",rocAcc,"_.RData", sep="")
)

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

#-------------------------------------------

# Results 2
# Time difference of 15.15049 mins
# bootControl <- trainControl(number=10)
# ,tuneLength = 10
# xgenes
# 1      2
# Accuracy : 0.7618          
# 95% CI : (0.7573, 0.7663)


# Results 1 
# Time difference of 13.00901 mins
# ,tuneLength = 10
# Accuracy was used to select the optimal model using  the largest value.
# The final value used for the model was xgenes = 100. 
# Accuracy : 0.7611          
# 95% CI : (0.7565, 0.7656)



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

modFit <- modFitroc
in_err <- rocAcc
modtype <-modFit$method
samptmp <- modFit$resample; samp <- length(unique(samptmp$Resample))
numvars <- length(modFit$coefnames)
timval <- str_replace_all(Sys.time(), " |:", "_")

#Is it "prob" of 'yes' or 'no'....?
pred_BNP <- predict(modFit, newdata = datTestpre, type = "raw")
toSubmit <- data.frame(ID = datTestpre$ID, PredictedProb = pred_BNP$yes)

write.table(toSubmit, file=paste("Res_xxxx_", modtype,"_",numvars,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep=""),sep=","
            , row.names=FALSE,col.names=TRUE, quote=FALSE)

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

modFit <- modFitgbm
in_err <- gbmAcc
modtype <-modFit$method
samptmp <- modFit$resample; samp <- length(unique(samptmp$Resample))
numvars <- length(modFit$coefnames)
timval <- str_replace_all(Sys.time(), " |:", "_")

#Is it "prob" of 'yes' or 'no'....?
pred_BNP <- predict(modFit, newdata = datTestpre, type = "prob")
toSubmit <- data.frame(ID = datTestpre$ID, PredictedProb = pred_BNP$yes)

write.table(toSubmit, file=paste("Res_xxxx_", modtype,"_",numvars,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep=""),sep=","
            , row.names=FALSE,col.names=TRUE, quote=FALSE)

# Result...
# It only provides "class" not "prob"...