
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
library(imputeR)

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

#NAs in 'numeric' columns are 'NA'
#But there are missing values in 'character' columns. These replaced with 'NA'
#Once everything is numeric with NA they will be managed by 'WaverR'.
val <- NA
for( i in 3:ncol(datIn) ) {
  vtmp <- datIn[,i]
  if(class(vtmp) =="character") {
    tbl_tmp <- table(vtmp)
    val_s <- names(tbl_tmp[tbl_tmp==max(tbl_tmp)])
    vtmp <- ifelse(vtmp=="", val , vtmp)
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

rm(idx_tmp, col_del, val)

#Transform 'factor' into 'numeric', NA will remain as NA.
cont <- 0
idx_fac <- 0
for( i in 2:ncol(datIn)) {
  cltmp <- class(datIn[, i])
  if(cltmp == "factor") {
    datIn[,i] <- as.numeric( datIn[,i] )
    cont <- cont + 1
    idx_fac[cont] <- i #which were factors needed for imputation
  } else next
}

rm(lev_df, cltmp, i, nu_fac, vtmp)

# waverr saves file "ReconstructedData.txt" in working directory
# with 10 repetitions it's very slow... (6 hours) and still many "NAs"...!
# with "imputeR"..
# Impute as Regression
cols_in <- 1:ncol(datIn)
cols_reg <- setdiff(cols_in, c(1,2,idx_fac)) 
cols_cla <- idx_fac
datIn_reg <- impute(as.matrix(datIn[,cols_reg]), lmFun = "lassoR", verbose = TRUE, maxiter=10)$imp

for(i in 1:length(cols_cla)) {
  datIn[, cols_cla[i]] <- as.factor(datIn[, cols_cla[i]])
}
datIn_cla <- impute(as.matrix(datIn[,cols_cla]), cFun = "rpartC", verbose = TRUE, maxiter=10)$imp
datIn_nona <- cbind.data.frame(datIn[,c(1,2)], as.data.frame(datIn_reg), as.data.frame(datIn_cla))

# save(datIn_nona, file="datInnoNA.RData")
# load("datInnoNA.RData")

datIn <- datIn_nona

#--------------------------------------------------------
#--------------------------------------------------------
#-------------- READY TO MODEL
#--------------------------------------------------------
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

bootControl <- trainControl(number = 5,
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

# Results 1 - No NAs - No big differences in Accuracy
# ROC        Sens       Spec       ROC SD      Sens SD      Spec SD    
# 0.7155159  0.1521468  0.9660248  0.00308978  0.002942251  0.002277474
# inTrain <- createDataPartition(datSamp$target, p = 0.70 , list = FALSE)
# bootControl <- trainControl(number = 5,
# num.trees = 500,
# Time difference of 35.6616 mins
# Accuracy : 0.7769          
# 95% CI : (0.7724, 0.7813)






#---------------------------------
#---------------------- XGB (new version CARET)
#---------------------------------
set.seed(387)
a <- Sys.time();a
bootControl <- trainControl(number = 1,
                            summaryFunction = twoClassSummary,
                            classProbs = TRUE,
                            verboseIter = FALSE)

# xgbGrid <- expand.grid(
#   eta = 0.12,
#   max_depth = 10,
#   nrounds = 100,
#   gamma = seq(0.3, 1, length.out=5),
#   colsample_bytree = 1,
#   min_child_weight = 1
# )
 # xgbGrid <- expand.grid(
 #   eta = 0.12,
 #   max_depth = 10,
 #   nrounds = 200,
 #   gamma = 0,
 #   colsample_bytree = 1,
 #   min_child_weight = 1
 # )

# xgbGrid <- expand.grid(
#   nrounds = 200,
#   lambda = 0.4,
#   alpha = 0.25
# )

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
  tuneLength = 4,
  metric = "ROC",
  method = "xgbTree",
  verbose = 1,
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

#Save trainDat, testDat and Model objects.
save(
  trainDat, testDat, modFitxgb,
  file=paste("XGB_",numvars,"vars_n",samp,"_grid",modBestc,"_",xgbAcc,"__.RData", sep="")
)

#---------------------------------------------------

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
#---------------------- GLMBOOST
#---------------------------------
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

glmGrid <- expand.grid(
                         #mstop = seq(10000,20000,10000),
                         mstop = 20000,
                         prune = "no"
                       )

modFitglm <-  train(
  target ~ .,
  data = trainDat,
  trControl = bootControl,
  tuneGrid = glmGrid,
  #tuneLength = 30,
  metric = "ROC",
  method = "glmboost"
  #center = TRUE
)

modFitglm

predglm <- predict( modFitglm, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatglm <- confusionMatrix(testDat$target, predglm); conMatglm
conMatglmdf <- as.data.frame(conMatglm$overall); glmAcc <- conMatglmdf[1,1]; glmAcc <- as.character(round(glmAcc*100,2))
b <- Sys.time();b; b-a

if( nrow(glmGrid) < 2  )  { resampleHist(modFitglm) } else
{ plot(modFitglm, as.table=T) }

#Best iteration
modBest <- modFitglm$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitglm$times$final[3]
#Samples
samp <- dim(modFitglm$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
Impglm <- varImp( modFitglm, scale=F)
plot(Impglm, top=20)

#Save trainDat, testDat and Model objects.
save(
  trainDat, testDat, modFitglm,
  file=paste("glmboost_",numvars,"vars_n",samp,"_grid_",glmAcc,"__.RData", sep="")
)
#-------------------------------

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

#---------------------------------
#---------------------- GBM
#---------------------------------
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


gbmGrid <- expand.grid(
  interaction.depth = 10,
  n.trees = 150,
  shrinkage = seq(0.04, 0.06, 0.01),
  n.minobsinnode = 12
)



modFitgbm <-  train(
  target ~ .,
  data = trainDat,
  trControl = bootControl,
  tuneGrid = gbmGrid,
  metric = "ROC",
  method = "gbm",
  verbose = TRUE
)

modFitgbm

predgbm <- predict( modFitgbm, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatgbm <- confusionMatrix(testDat$target, predgbm); conMatgbm
conMatgbmdf <- as.data.frame(conMatgbm$overall); gbmAcc <- conMatgbmdf[1,1]; gbmAcc <- as.character(round(gbmAcc*100,2))
b <- Sys.time();b; b-a

if( nrow(gbmGrid) < 2  )  { resampleHist(modFitgbm) } else
{ plot(modFitgbm, as.table=T) }

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
Impgbm <- varImp( modFitgbm, scale=F)
plot(Impgbm, top=20)

#Save trainDat, testDat and Model objects.
save(
  trainDat, testDat, modFitgbm,
  #file=paste("gbm_",numvars,"vars_n",samp,"_grid",modBestc,"_",gbmAcc,"__.RData", sep="")
  file=paste("gbm_",numvars,"vars_n",samp,"_grid_1_",gbmAcc,"__.RData", sep="")
)

#-------------------------------

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




#---------------------------------
#---------------------- NNET - MODEL AVERAGED
#---------------------------------
a <- Sys.time();a
set.seed(20792)

bootControl <- trainControl(
  number=15, verboseIter=TRUE, 
  summaryFunction = twoClassSummary
  ,classProbs = TRUE
) 

nnetGrid <- expand.grid(
  size = 23,
  decay = 0.1
  #bag = FALSE
)

modFitnnet <- train(
  target ~ .
  ,data = trainDat
  ,method = "pcaNNet"
  ,trControl = bootControl
  ,tuneGrid = nnetGrid
  #,tuneLength = 15
  ,metric =  "ROC"
  ,preProc = c('center', 'scale')
  ,entropy = TRUE
)

modFitnnet

prednnet <- predict( modFitnnet, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatnnet <- confusionMatrix(testDat$target, prednnet); conMatnnet 
conMatnnetdf <- as.data.frame(conMatnnet$overall); nnetAcc <- conMatnnetdf[1,1]; nnetAcc <- as.character(round(nnetAcc*100,2))
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
Impnnet <- varImp( modFitnnet, scale=F)
plot(Impnnet, top=(2:ncol(trainDat)))


#Save trainDat, testDat and Model objects.
save(
  trainDat, testDat, modFitnnet,
  file=paste("pcannet_",numvars,"vars_n",samp,"_grid_.RData", sep="")
)

#-------------------------------------------

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





#---------------------------------
#- blackboostGED CART
#---------------------------------
a <- Sys.time();a
set.seed(20792)

bootControl <- trainControl(
  number=10, verboseIter=TRUE, 
  summaryFunction = twoClassSummary
  ,classProbs = TRUE
) 

blackboostGrid <- expand.grid(
   mstop = 50,
   maxdepth = 9
)

modFitblackboost <- train(
  target ~ .
  ,data = trainDat
  ,method = "blackboost"
  ,trControl = bootControl
  ,tuneGrid = blackboostGrid
  #,tuneLength = 10
  ,metric =  "ROC"
)

modFitblackboost

predblackboost <- predict( modFitblackboost, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatblackboost <- confusionMatrix(testDat$target, predblackboost); conMatblackboost 
conMatblackboostdf <- as.data.frame(conMatblackboost$overall); blackboostAcc <- conMatblackboostdf[1,1]; blackboostAcc <- as.character(round(blackboostAcc*100,2))
b <- Sys.time();b; b-a   

if( nrow(blackboostGrid) < 2  )  { resampleHist(modFitblackboost) } else  
{ plot(modFitblackboost, as.table=T) }

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
save(
  trainDat, testDat, modFitblackboost,
  file=paste("blackboost_",numvars,"vars_n",samp,"_grid_",blackboostAcc,"_.RData", sep="")
)

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

#-------------------------------------------

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
conMatrocdf <- as.data.frame(conMatroc$overall); rocAcc <- conMatrocdf[1,1]; rocAcc <- as.character(round(rocAcc*100,2))
b <- Sys.time();b; b-a   

if( nrow(rocGrid) < 2  )  { resampleHist(modFitroc) } else  
{ plot(modFitroc, as.table=T) }

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
save(
  trainDat, testDat, modFitroc,
  file=paste("roc_",numvars,"vars_n",samp,"_grid_",rocAcc,"_.RData", sep="")
)

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

#-------------------------------------------
# Results 1 
# Time difference of 13.00901 mins
# ,tuneLength = 10
# Accuracy was used to select the optimal model using  the largest value.
# The final value used for the model was xgenes = 100. 
# Accuracy : 0.7611          
# 95% CI : (0.7565, 0.7656)



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
  ,tuneLength = 10
  ,metric =  "ROC"
)

modFitsvmRad

predsvmRad <- predict( modFitsvmRad, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatsvmRad <- confusionMatrix(testDat$target, predsvmRad); conMatsvmRad 
conMatsvmRaddf <- as.data.frame(conMatsvmRad$overall); svmRadAcc <- conMatsvmRaddf[1,1]; svmRadAcc <- as.character(round(svmRadAcc*100,2))
b <- Sys.time();b; b-a   

if( nrow(svmRadGrid) < 2  )  { resampleHist(modFitsvmRad) } else  
{ plot(modFitsvmRad, as.table=T) }

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
save(
  trainDat, testDat, modFitsvmRad,
  file=paste("svmRad_",numvars,"vars_n",samp,"_grid_",svmRadAcc,"_.RData", sep="")
)

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

#-------------------------------------------
# Results 1 
