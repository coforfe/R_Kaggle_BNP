#------------------------------------------------------------
# BNP - Paribas - Claims Projections - 2016-02-15
#------------------------------------------------------------

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

# load("datIn1000.RData")
# setwd("~/Downloads")


#Library loading
library(lubridate)
library(stringr)
library(data.table)
library(caret)

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
#---------------------- GBM
#---------------------------------
setwd("~/Downloads")

a <- Sys.time();a
set.seed(5789)
#set.seed(457856+rnorm(1)*10000)
#bootControl <- trainControl(method='boot',number=50, verboseIter=TRUE)
#bootControl <- trainControl(method='oob',number=25, verboseIter=TRUE)
#bootControl <- trainControl( number=5, verboseIter=TRUE )

bootControl <- trainControl(number = 5,
                            summaryFunction = twoClassSummary,
                            classProbs = TRUE,
                            verboseIter = FALSE)


bootControl <- trainControl(number = 5)
                            
gbmGrid <- expand.grid(
  interaction.depth = 3,
  n.trees = 100,
  shrinkage = 0.155,
  n.minobsinnode = 15
)


modFitgbm <-  train(
  target ~ .,
  data = trainDat,
  trControl = bootControl,
  tuneGrid = gbmGrid,
  #tuneLength = 3,
  metric = "Accuracy",
  #metric = "ROC",
  method = "gbm",
  verbose = TRUE
)

modFitgbm

predgbm <- predict( modFitgbm, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatgbm <- confusionMatrix(testDat$target, predgbm); conMatgbm
conMatgbmdf <- as.data.frame(conMatgbm$overall); 
gbmAcc <- conMatgbmdf[1,1]; 
gbmAcc <- as.character(round(gbmAcc*100,2))
gbmAcc
b <- Sys.time();b; b-a

if( nrow(gbmGrid) < 2  )  { resampleHist(modFitgbm) } else
{ plot(modFitgbm, as.table=T) }
plot(modFitgbm, as.table=T) 

ggplot(modFitgbm) +
  geom_smooth(se = FALSE, span = .8, method = loess) +
  theme(legend.position = "top")

#Best iteration
modBest <- modFitgbm$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitgbm$times$final[3]
#Samples
samp <- dim(modFitgbm$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
# Impgbm <- varImp( modFitgbm, scale=F)
# plot(Impgbm, top=20)

#Save trainDat, testDat and Model objects.
format(object.size(modFitgbm), units = "Gb")

save(
  trainDat, testDat, modFitgbm,
  #file=paste("gbm_",numvars,"vars_n",samp,"_grid",modBestc,"_",gbmAcc,"__.RData", sep="")
  file=paste("gbm_",numvars,"vars_n",samp,"_grid_1_",gbmAcc,"__.RData", sep="")
)

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")


#-------------------------------

results <- function() {
  
# Results 8 - Mac
 #  Time difference of 38.1634 mins
 #  bootControl <- trainControl(number = 5)
 #  shrinkage = seq(0.01, 0.3, length.out = 3),
 #  n.minobsinnode = seq(5,15,length.out = 3)
 # n.trees interaction.depth shrinkage n.minobsinnode
 #  100                 3     0.155             15  
 #  Accuracy : 0.7821          
 #  95% CI : (0.7777, 0.7864)
  
# Results 7 - Mac
# Time difference of 13.23194 mins
# tuneLength = 3,
# bootControl <- trainControl(number = 5)
# n.trees interaction.depth shrinkage n.minobsinnode
#   100                 3       0.1             10
#   Accuracy : 0.782           
#   95% CI : (0.7776, 0.7863)  
  
# Results 6 - Mac
# Time difference of 20.9931 mins 
# bootControl <- trainControl(number = 5,
# gbmGrid <- expand.grid(
#     interaction.depth = 10,
#     n.trees = 150,
#     shrinkage = 0.06,
#     n.minobsinnode = 12
# )
# n.trees interaction.depth shrinkage n.minobsinnode
# 1     150                10      0.06             12
# Accuracy : 0.7818          
# 95% CI : (0.7773, 0.7861)  
  
# Results 1 - Mac
# Time difference of 1.5 hours
# gbmGrid <- expand.grid(
#   interaction.depth = 10,
#   n.trees = 150,
#   shrinkage = seq(0.04, 0.06, 0.01),
#   n.minobsinnode = 12
# )
# Tuning parameter 'n.trees' was held constant at a value of 150
# Tuning parameter 'interaction.depth' was held constant at a value of 10
# Tuning parameter 'n.minobsinnode' was held constant at a value of 12
 # ROC was used to select the optimal model using  the largest value.
# The final values used for the model were n.trees = 150, interaction.depth = 10, shrinkage = 0.06
# and n.minobsinnode = 12. 
# Accuracy : 0.7802          
# 95% CI : (0.7758, 0.7846)  
  
# Results 5 - Mac
# Time difference of 57.92468 mins
# gbmGrid <- expand.grid(
#   interaction.depth = 10,
#   n.trees = 150,
#   shrinkage = seq(0.04, 0.06, 0.01),
#   n.minobsinnode = 12
# )
# Tuning parameter 'n.trees' was held constant at a value of 150
# Tuning parameter 'interaction.depth' was held constant at a value of 10
# Tuning parameter 'n.minobsinnode' was held constant at a value of 12
# ROC was used to select the optimal model using  the largest value.
# The final values used for the model were n.trees = 150, interaction.depth = 10, shrinkage = 0.06
# and n.minobsinnode = 12. 
# Accuracy : 0.7802          
# 95% CI : (0.7758, 0.7846)

# Results 4 - Mac
# Time difference of 22.23347 mins
# gbmGrid <- expand.grid(
#   interaction.depth = 10,
#   n.trees = 150,
#   shrinkage = 0.08,
#   n.minobsinnode = 12
# )
# ROC        Sens      Spec       ROC SD      Sens SD      Spec SD    
# 0.7427892  0.190869  0.9662981  0.00156953  0.005830568  0.002433789
# Tuning parameter 'n.trees' was held constant at a value of 150
# Tuning parameter held constant at a value of 0.08
# Tuning parameter 'n.minobsinnode' was held constant at a value
# of 12
# Accuracy : 0.7794         
# 95% CI : (0.775, 0.7838)

# Results 4 - Mac
# Time difference of 22.63166 mins
# Tuning parameter 'n.trees' was held constant at a value of 150
# Tuning # parameter 'shrinkage' was held constant at a value of 0.01
# Tuning # parameter 'n.minobsinnode' was held constant at a value of 12
# gbmGrid <- expand.grid(
#   interaction.depth = 10,
#   n.trees = 150,
#   shrinkage = 0.01,
#   n.minobsinnode = 12
# )
# Accuracy : 0.7776         
# 95% CI : (0.7732, 0.782)


# Results 3 - Mac - THE BEST...
# Time difference of 22.83691 mins
# gbmGrid <- expand.grid(
#   interaction.depth = 10,
#   n.trees = 150,
#   shrinkage = 0.05,
#   n.minobsinnode = 12
# )
# ROC        Sens       Spec       ROC SD       Sens SD      Spec SD    
# 0.7441522  0.1758769  0.9711634  0.001338755  0.002406988  0.001599055
# Accuracy : 0.7824         
# 95% CI : (0.778, 0.7867)



# Results 3 - Mac
# Time difference of 4.657586 hours
# gbmGrid <- expand.grid(
#   interaction.depth = 10,
#   n.trees = 150,
#   shrinkage = seq(0.1, 0.4, 0.1),
#   n.minobsinnode = seq(10,14,1)
# )
# Tuning parameter 'n.trees' was held constant at a value of 150
# Tuning # parameter 'interaction.depth' was held constant at a value of 10
# ROC was used to select the optimal model using  the largest value.
# The final values used for the model were n.trees = 150, interaction.depth = 10, shrinkage
# = 0.1 and n.minobsinnode = 12.
# Accuracy : 0.7822          
# 95% CI : (0.7778, 0.7866)



# Results 3 - Instance type c4.4xlarge
# Time difference of 1.177033 hours
# bootControl <- trainControl(number = 5,
# tuneLength = 10,
# Tuning parameter 'shrinkage' was held constant at a value of 0.1
# Tuning parameter 'n.minobsinnode' was
# held constant at a value of 10
# ROC was used to select the optimal model using  the largest value.
# The final values used for the model were n.trees = 150, interaction.depth = 10, shrinkage = 0.1
# and n.minobsinnode = 10. 
# Accuracy : 0.7813          
# 95% CI : (0.7769, 0.7857)


# Results 2 - Instance type c4.4xlarge
# Time difference of 14.71116 mins
# Tuning parameter 'shrinkage' was held constant at a value of 0.1
# Tuning
# parameter 'n.minobsinnode' was held constant at a value of 10
# ROC was used to select the optimal model using  the largest value.
# The final values used for the model were n.trees = 250, interaction.depth =
#   5, shrinkage = 0.1 and n.minobsinnode = 10. 
# Accuracy : 0.7794         
# 95% CI : (0.775, 0.7838)

# Results 1
# Time difference of 1.855072 hours
# bootControl <- trainControl(number = 5,
# n.trees interaction.depth shrinkage n.minobsinnode
# 1     800                11      0.08             17
# Accuracy : 0.7802          
# 95% CI : (0.7758, 0.7846)
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
  
