
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
#---------------------- NNET - MODEL AVERAGED
#---------------------------------
setwd("~/Downloads")

a <- Sys.time();a
set.seed(20792)

bootControl <- trainControl(
  number=3, verboseIter=TRUE, 
  summaryFunction = twoClassSummary
  ,classProbs = TRUE
) 

bootControl <- trainControl(number=10)

nnetGrid <- expand.grid(
  size = 7,
  decay = 0.2
  #bag = FALSE
)

modFitnnet <- train(
  target ~ .
  ,data = trainDat
  ,method = "pcaNNet"
  ,trControl = bootControl
  ,tuneGrid = nnetGrid
  #,tuneLength = 15
  #,metric =  "ROC"
  ,metric =  "Accuracy"
  ,preProc = c('center', 'scale')
  ,entropy = TRUE
)

modFitnnet

prednnet <- predict( modFitnnet, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatnnet <- confusionMatrix(testDat$target, prednnet); conMatnnet 
conMatnnetdf <- as.data.frame(conMatnnet$overall); 
nnetAcc <- conMatnnetdf[1,1]; 
nnetAcc <- as.character(round(nnetAcc*100,2))
nnetAcc
b <- Sys.time();b; b-a   

if( nrow(nnetGrid) < 2  )  { resampleHist(modFitnnet) } else  
{ plot(modFitnnet, as.table=T) }

#Best iteration
modBest <- modFitnnet$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitnnet$times$final[3]
#Samples
samp <- dim(modFitnnet$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
# Impnnet <- varImp( modFitnnet, scale=F)
# plot(Impnnet, top=(2:ncol(trainDat)))


#Save trainDat, testDat and Model objects.
format(object.size(modFitnnet), units = "Gb")

save(
  trainDat, testDat, modFitnnet,
  file=paste("pcannet_",numvars,"vars_n",samp,"_",nnetAcc,"_.RData", sep="")
)

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")


#-------------------------------------------

results <- function() {
# # # # Results 6 - Mac 
# Time difference of 22.57427 mins
# ROC        Sens        Spec      ROC SD       Sens SD     Spec SD    
# 0.6922917  0.07156548  0.978466  0.007403278  0.02120645  0.005921239
# Accuracy : 0.7643          
# 95% CI : (0.7598, 0.7688)
# Time difference of 22.57427 mins

# # # Results 5 - Mac 
# Time difference of 6.816169 hours
# bootControl <- trainControl( number=3, verboseIter=TRUE, 
# ROC was used to select the optimal model using  the largest value.
# The final values used for the model were size = 23 and decay = 0.1. 
# ,method = "pcaNNet"
# ,tuneLength = 15
# Accuracy : 0.7614          
# 95% CI : (0.7568, 0.7659)

# # Results 5 - Mac 
# bootControl <- trainControl( number=3, verboseIter=TRUE, 
# ,method = "pcaNNet"
# ,tuneLength = 5
# Accuracy : 0.7628          
# 95% CI : (0.7583, 0.7673)


# Results 4 - Mac 
# Time difference of 16.8823 mins
# nnetGrid <- expand.grid(
#   size = 7,
#   decay = 0.5,
#   bag = TRUE
# )
# ROC        Sens        Spec       ROC SD       Sens SD      Spec SD    
# 0.7262798  0.06518336  0.9913844  0.001885331  0.009196912  0.001819861
# Accuracy : 0.7679          
# 95% CI : (0.7634, 0.7723)


# Results 3 - Mac 
# Time difference of 16.57544 mins
# nnetGrid <- expand.grid(
#   size = 7,
#   decay = 0.5,
#   bag = FALSE
# )
# ROC        Sens        Spec      ROC SD       Sens SD     Spec SD    
# 0.7272529  0.06856398  0.988411  0.003412229  0.02472516  0.005078946
# Accuracy : 0.7706          
# 95% CI : (0.7662, 0.7751)

  
# Results 2 - Mac 
# Time difference of 21.87191 mins
# bootControl <- trainControl(
#   number=3, verboseIter=TRUE, 
# nnetGrid <- expand.grid(
#   size = 7,
#   decay = 0.2,
#   bag = FALSE
# )
# ROC        Sens       Spec       ROC SD       Sens SD     Spec SD    
# 0.7330545  0.1049715  0.9835557  0.001575801  0.02465711  0.0044595
# Accuracy : 0.772           
# 95% CI : (0.7675, 0.7764)


# Results 1 - Mac 
# Time difference of 4.024185 hours
# ,tuneLength = 5
# Tuning parameter 'bag' was held constant at a value of FALSE
# ROC was used to select the optimal model using  the largest value.
# The final values used for the model were size = 7, decay = 0.1 and bag = FALSE. 
# size decay   bag
#   7   0.1 FALSE 
# size  decay  ROC        Sens        Spec       ROC SD       Sens SD      Spec SD   
# 7     1e-01  0.7335668  0.09783656  0.9848445  0.001615119  0.029838115  0.005787503
# Accuracy : 0.7733          
# 95% CI : (0.7688, 0.7777)
# NAs for "Hidden Units" when equal to "9".

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
setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")
#load("datTestpre.RData")

  modFit <- modFitnnet 
  in_err <- nnetAcc
  modtype <-modFit$method
  samptmp <- modFit$resample; samp <- length(unique(samptmp$Resample))
  numvars <- length(modFit$coefnames)
  timval <- str_replace_all(Sys.time(), " |:", "_")
  
  #Is it "prob" of 'yes' or 'no'....?
  pred_BNP <- predict(modFit, newdata = datTestpre, type = "prob")
  toSubmit <- data.frame(ID = datTestpre$ID, PredictedProb = pred_BNP$yes)
  
  write.table(toSubmit, file=paste("Res_xxxx_", modtype,"_",numvars,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep=""),sep=","
              , row.names=FALSE,col.names=TRUE, quote=FALSE)
  
