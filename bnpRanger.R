
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
#---------------------- RANGER (randomForest)
#---------------------------------
setwd("~/Downloads")

a <- Sys.time();a
set.seed(5789)
#set.seed(457856+rnorm(1)*10000)
#bootControl <- trainControl(method='boot',number=50, verboseIter=TRUE)
#bootControl <- trainControl(method='oob',number=25, verboseIter=TRUE)
#bootControl <- trainControl( number=5, verboseIter=TRUE )

bootControl <- trainControl(number = 50,
                             summaryFunction = twoClassSummary,
                             classProbs = TRUE,
                             verboseIter = FALSE)

#rfGrid <- expand.grid(mtry=seq(9,12,1))
rfGrid <- expand.grid(mtry = 10)

modFitrf <-  train(
  target ~ .,
  data = trainDat,
  trControl = bootControl,
  tuneGrid = rfGrid,
  metric = "ROC",
  method = "ranger",
  num.trees = 500,
  importance = 'impurity',
  respect.unordered.factors = TRUE,
  verbose = TRUE,
  classification = TRUE
)

modFitrf

predrf <- predict( modFitrf, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatrf <- confusionMatrix(testDat$target, predrf); conMatrf
conMatrfdf <- as.data.frame(conMatrf$overall); rfAcc <- conMatrfdf[1,1]; rfAcc <- as.character(round(rfAcc*100,2))
b <- Sys.time();b; b-a

if( nrow(rfGrid) < 2  )  { resampleHist(modFitrf) } else
{ plot(modFitrf, as.table=T) }

#Best iteration
modBest <- modFitrf$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitrf$times$final[3]
#Samples
samp <- dim(modFitrf$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
Imprf <- varImp( modFitrf, scale=F)
plot(Imprf, top=20)

#Save trainDat, testDat and Model objects.
save(
  trainDat, testDat, modFitrf,
  file=paste("ranger_",numvars,"vars_rf_n",samp,"_grid",modBestc,"_",rfAcc,"__.RData", sep="")
)

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

#-----------------------

results <- function() {
# Results 4 - columns with many levels removed (>30)
# bootControl <- trainControl(number = 50,
# rfGrid <- expand.grid(mtry = 10)
# num.trees = 500,
# Time difference of 3.317524 hours
# Accuracy : 0.7792          
# 95% CI : (0.7748, 0.7836)


# Results 3 - columns with many levels removed (>30)
# All columns are NUMERIC (factors converted to numeric)
# bootControl <- trainControl(number = 5,
# rfGrid <- expand.grid(mtry = 10)
# num.trees = 100,
# Time difference of 6.679732 mins
# Accuracy : 0.7788          
# 95% CI : (0.7744, 0.7832)

# Results 2 - columns with many levels removed (>30)
# bootControl <- trainControl(number = 5,
# rfGrid <- expand.grid(mtry = 10)
# num.trees = 100,
# Time difference of 5.699979 mins
# Accuracy : 0.7779          
# 95% CI : (0.7735, 0.7823)

# Results 1 - columns with many levels removed (>30)
# Some columns are character.
# bootControl <- trainControl(number = 5,
# rfGrid <- expand.grid(mtry=seq(9,12,1))
# The final value used for the model was mtry = 10. 
# num.trees = 25,
# Time difference of 5.36344 mins
# Accuracy : 0.7751          
# 95% CI : (0.7706, 0.7795)

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
load("ranger_129vars_rf_n50_grid10_77.92__.RData")
  modFit <- modFitrf 
  in_err <- rfAcc
  in_err <- 77.92
  modtype <-modFit$method
  samptmp <- modFit$resample; samp <- length(unique(samptmp$Resample))
  numvars <- length(modFit$coefnames)
  timval <- str_replace_all(Sys.time(), " |:", "_")
  
  #Is it "prob" of 'yes' or 'no'....?
  pred_BNP <- predict(modFit, newdata = datTestpre, type = "prob")
  toSubmit <- data.frame(ID = datTestpre$ID, PredictedProb = pred_BNP$yes)
  
  write.table(toSubmit, file=paste("Res_xxxx_", modtype,"_",numvars,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep=""),sep=","
              , row.names=FALSE,col.names=TRUE, quote=FALSE)
  

  
#--------------------- OTHER MODELS ----------------
  
#---------------------------------
#---------------------- randomGLM
#---------------------------------
a <- Sys.time();a
set.seed(5789)
#set.seed(457856+rnorm(1)*10000)
#bootControl <- trainControl(method='boot',number=50, verboseIter=TRUE)
#bootControl <- trainControl(method='oob',number=25, verboseIter=TRUE)
#bootControl <- trainControl( number=5, verboseIter=TRUE )

bootControl <- trainControl(number = 3,
                            summaryFunction = twoClassSummary,
                             classProbs = TRUE,
                            verboseIter = FALSE)

#rfGrid <- expand.grid(mtry=seq(43,47,1))
glmranGrid <- expand.grid(maxInteractionOrder = 1)
#rfGrid <- expand.grid(maxdepth = 8)

modFitglmran <-  train(
  target ~ .,
  data = trainDat,
  trControl = bootControl,
  #tuneGrid = glmranGrid,
  tuneLength = 3,
  metric = "ROC",
  method = "randomGLM",
  classify = TRUE,
  replace = TRUE,
  verbose = 1
)

modFitglmran

predglmran <- predict( modFitglmran, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatglmran <- confusionMatrix(testDat$target, predglmran); conMatglmran
conMatglmrandf <- as.data.frame(conMatglmran$overall); glmranAcc <- conMatglmrandf[1,1]; glmranAcc <- as.character(round(glmranAcc*100,2))
b <- Sys.time();b; b-a

if( nrow(glmranGrid) < 2  )  { resampleHist(modFitglmran) } else
{ plot(modFitglmran, as.table=T) }

#Best iteration
modBest <- modFitglmran$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitglmran$times$final[3]
#Samples
samp <- dim(modFitglmran$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
Impglmran <- varImp( modFitglmran, scale=F)
plot(Impglmran, top=20)

#Save trainDat, testDat and Model objects.
save(
  trainDat, testDat, modFitglmran,
  file=paste("ranglmran_",numvars,"vars_n",samp,"_grid",modBestc,"_",glmranAcc,"__.RData", sep="")
)


#---------------------------------
#---------------------- RFERNS
#---------------------------------
a <- Sys.time();a
set.seed(457856+rnorm(1)*10000) 
bootControl <- trainControl( number=5, verboseIter=TRUE) 

#rferGrid <- expand.grid( .depth = seq(9,16,1)) 
rferGrid <- expand.grid( .depth = 15) 

modFitrfer <-  train(
  target ~ .,
  data = trainDat,
  trControl = bootControl,
  metric = "Accuracy",
  tuneLength = 100,
  method = "rFerns"
)
modFitrfer

predrfer <- predict( modFitrfer, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatrfer <- confusionMatrix(testDat$target, predrfer); conMatrfer 
conMatrferdf <- as.data.frame(conMatrfer$overall); rferAcc <- conMatrferdf[1,1]; rferAcc <- as.character(round(rferAcc*100,2))
b <- Sys.time();b; b-a   

if( nrow(rferGrid) < 2  )  { resampleHist(modFitrfer) } else  
{ plot(modFitrfer, as.table=T) }

#Variable Importance
# Imprfer <- varImp( modFitrfer, scale=F)
# plot(Imprfer, top=20)

#Best iteration
modBest <- modFitrfer$bestTune; modBest
modBestc <- paste(modBest[1],modBest[2],modBest[3], sep="_")
#Execution time:
modFitrfer$times$final[3]
#Samples
samp <- dim(modFitrfer$resample)[1]


#Save trainDat, testDat and Model objects.
save(
  trainDat, testDat, modFitgbm,
  file=paste("rFerns_",numvars,"vars_n",samp,"_grid",modBestc,"_",rferAcc,"__.RData", sep="")
)

#-------------------------------------------
# Results 1 - NO GOOD...
# Time difference of 3.481633 mins
# tuneLength = 100,
# Accuracy was used to select the optimal model using  the largest value.
# The final value used for the model was depth = 6. 
# Accuracy : 0.5971          
# 95% CI : (0.5919, 0.6023)



#---------------------------------
#---------------------- ADABOOST
#---------------------------------
a <- Sys.time();a
set.seed(457856+rnorm(1)*10000) 
#bootControl <- trainControl( number=5, verboseIter=TRUE) 
bootControl <- trainControl(number = 5,
                            summaryFunction = twoClassSummary,
                            classProbs = TRUE,
                            verboseIter = FALSE)



modFitadab <-  train(
  target ~ .,
  data = trainDat,
  trControl = bootControl,
  metric = "ROC",
  tuneLength = 10,
  method = "AdaBag"
)
modFitadab

predadab <- predict( modFitadab, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatadab <- confusionMatrix(testDat$target, predadab); conMatadab 
conMatadabdf <- as.data.frame(conMatadab$overall); adabAcc <- conMatadabdf[1,1]; adabAcc <- as.character(round(adabAcc*100,2))
b <- Sys.time();b; b-a   

if( nrow(adabGrid) < 2  )  { resampleHist(modFitadab) } else  
{ plot(modFitadab, as.table=T) }

#Variable Importance
# Impadab <- varImp( modFitadab, scale=F)
# plot(Impadab, top=20)

#Best iteration
modBest <- modFitadab$bestTune; modBest
modBestc <- paste(modBest[1],modBest[2],modBest[3], sep="_")
#Execution time:
modFitadab$times$final[3]
#Samples
samp <- dim(modFitadab$resample)[1]


#Save trainDat, testDat and Model objects.
save(
  trainDat, testDat, modFitgbm,
  file=paste("adabns_",numvars,"vars_n",samp,"_grid",modBestc,"_",adabAcc,"__.RData", sep="")
)

#-------------------------------------------
# Results 1 - NO GOOD...
# Time difference of 3.481633 mins
# tuneLength = 100,
# Accuracy was used to select the optimal model using  the largest value.
# The final value used for the model was depth = 6. 
# Accuracy : 0.5971          
# 95% CI : (0.5919, 0.6023)



#---------------------------------------------
# ------ NODEHARVEST  - Tree-Based Ensembles
#---------------------------------------------
setwd("~/Downloads")

a <- Sys.time();a
set.seed(20792)

bootControl <- trainControl(
  number=5, verboseIter=TRUE,
  summaryFunction = twoClassSummary
  ,classProbs = TRUE
)

# bootControl <- trainControl(
#   number=5, verboseIter=TRUE
# ) 

# nodeHarGrid <- expand.grid(
#   xgenes =
# )

modFitnodeHar <- train(
  target ~ .
  ,data = trainDat
  ,method = "nodeHarvest"
  ,trControl = bootControl
  #,tuneGrid = nodeHarGrid
  ,tuneLength = 10
  ,metric =  "ROC"
)

modFitnodeHar

prednodeHar <- predict( modFitnodeHar, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatnodeHar <- confusionMatrix(testDat$target, prednodeHar); conMatnodeHar 
conMatnodeHardf <- as.data.frame(conMatnodeHar$overall); nodeHarAcc <- conMatnodeHardf[1,1]; nodeHarAcc <- as.character(round(nodeHarAcc*100,2))
b <- Sys.time();b; b-a   

if( nrow(nodeHarGrid) < 2  )  { resampleHist(modFitnodeHar) } else  
{ plot(modFitnodeHar, as.table=T) }

#Best iteration
modBest <- modFitnodeHar$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitnodeHar$times$final[3]
#Samples
samp <- dim(modFitnodeHar$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
# ImpnodeHar <- varImp( modFitnodeHar, scale=F)
# plot(ImpnodeHar, top=(2:ncol(trainDat)))


#Save trainDat, testDat and Model objects.
save(
  trainDat, testDat, modFitnodeHar,
  file=paste("nodeHar_",numvars,"vars_n",samp,"_grid_",nodeHarAcc,"_.RData", sep="")
)

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

#-------------------------------------------
# Results 1 
# It takes a lot of time... aborted...



#---------------------------------------------
# ------ svmRadial  - SVM - Radial Basis Function Kernel
#---------------------------------------------
setwd("~/Downloads")

a <- Sys.time();a
set.seed(20792)

bootControl <- trainControl(
  number=5, verboseIter=TRUE,
  summaryFunction = twoClassSummary
  ,classProbs = TRUE
)

bootControl <- trainControl(number=5)

# svmRadGrid <- expand.grid(
#   Sigma = ,
#   Cost = 
# )

modFitsvmRad <- train(
  target ~ .
  ,data = trainDat
  ,method = "svmRadial"
  ,trControl = bootControl
  #,tuneGrid = svmRadGrid
  ,tuneLength = 2
  ,metric =  "Accuracy"
)

modFitsvmRad

predsvmRad <- predict( modFitsvmRad, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatsvmRad <- confusionMatrix(testDat$target, predsvmRad); conMatsvmRad 
conMatsvmRaddf <- as.data.frame(conMatsvmRad$overall);
svmRadAcc <- conMatsvmRaddf[1,1]; 
svmRadAcc <- as.character(round(svmRadAcc*100,2))
svmRadAcc
b <- Sys.time();b; b-a   

if( nrow(svmRadGrid) < 2  )  { resampleHist(modFitsvmRad) } else  
{ plot(modFitsvmRad, as.table = TRUE ) }
plot(modFitsvmRad, as.table = TRUE) 

#Best iteration
modBest <- modFitsvmRad$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitsvmRad$times$final[3]
#Samples
samp <- dim(modFitsvmRad$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
# ImpsvmRad <- varImp( modFitsvmRad, scale=F)
# plot(ImpsvmRad, top=(2:ncol(trainDat)))


#Save trainDat, testDat and Model objects.
object.size(modFitsvmRad)

save(
  trainDat, testDat, modFitsvmRad,
  file=paste("svmRad_",numvars,"vars_n",samp,"_grid_",svmRadAcc,"_.RData", sep="")
)

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

#-------------------------------------------
# Results 1 
