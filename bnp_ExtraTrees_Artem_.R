#------------------------------------------------------------
# BNP - Paribas - Claims Projections - 2016-02-15
#------------------------------------------------------------

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP")

#Library loading
library(ggplot2)
library(readr)
library(xgboost)
library(stringr)
library(caret)
library(extraTrees)

cat("Read the train and test data\n")
train <- as.data.frame(read_csv("train.csv"))
test  <- as.data.frame(read_csv("test.csv"))

cat("Recode NAs to -999\n")
train[is.na(train)] <- -999
test[is.na(test)]   <- -999

nams_ini <- names(train)

cat("Remove highly correlated features\n")
re_move <- c('v8','v23','v25','v31','v36','v37',
            'v46','v51','v53','v54','v63','v73','v75','v79',
            'v81','v82','v89','v92','v95','v105','v107','v108',
            'v109','v110','v116','v117','v118','v119','v123',
            'v124','v128')

nams_end <- setdiff(nams_ini, re_move)
nams_end <- nams_end[ 3:length(nams_end)]

tra_new <- train[, nams_end ]
tes_new <-  test[, nams_end ]

cat("Replace categorical variables with integers\n")
for (i in 1:length(nams_end)) {
  if (class(tra_new[, i]) == "character") {
    lev_tmp <- unique( c( tra_new[, i], tes_new[, i]) )
    tra_new[, i] <- as.integer( factor( tra_new[, i], levels = lev_tmp) )
    tes_new[, i] <- as.integer( factor( tes_new[, i], levels = lev_tmp) )
  } else next
}

# Add "target" column
tra_new <- cbind.data.frame( target = train[ , 2], tra_new)

rm(i, lev_tmp)

datIn <- tra_new
datIn$target <- as.factor(ifelse(datIn$target == 1, 'yes', 'no'))
datTestpre <- tes_new

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
##numCor <- 2
#registerDoMC(cores = numCor)

#---------------------------------
#---------------------- EXTRA-TREES (¡error with caret!!??)
#---------------------------------
a <- Sys.time();a
set.seed(5789)

bootControl <- trainControl(number = 1,
                            summaryFunction = twoClassSummary,
                            classProbs = TRUE,
                            verboseIter = FALSE)

bootControl <- trainControl(number = 1)

#rfGrid <- expand.grid(mtry=seq(9,12,1))
rfGrid <- expand.grid(
                      mtry = 10,
                      numRandomCuts = 3
                     )

trainDat_mat <- as.matrix(trainDat)
x_mat <- as.matrix(trainDat[, 2:ncol(trainDat)])
y_fac <- trainDat[, 1]

x_df <- trainDat[ , 2:ncol(trainDat)]
y_fac <- trainDat[, 1]

modFitrf <-  train(
  target ~ .,
  # x = x_df,
  # y = y_fac,
  data = trainDat_mat,
  #data = trainDat,
  trControl = bootControl,
  #tuneGrid = rfGrid,
  #metric = "ROC",
  #metric = "Accuracy",
  method = "extraTrees"
)

modFitrf

predrf <- predict( modFitrf, newdata=testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatrf <- confusionMatrix(testDat$target, predrf); conMatrf
conMatrfdf <- as.data.frame(conMatrf$overall);
rfAcc <- conMatrfdf[1,1];
rfAcc <- as.character(round(rfAcc*100,2))
b <- Sys.time();b; b - a

if (nrow(rfGrid) < 2  )  { resampleHist(modFitrf) } else
{ plot(modFitrf, as.table = T) }

#Best iteration
modBest <- modFitrf$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitrf$times$final[3]
#Samples
samp <- dim(modFitrf$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
Imprf <- varImp( modFitrf, scale = F)
plot(Imprf, top = 20)

#---------------------------------------------
########------------ EXTRATREES DIRECTLY
#---------------------------------------------
# load("~/extraTrees_Dos_model.RData")
options( java.parameters = "-Xmx6g" )

library(extraTrees)
x_mat <- as.matrix(tra_new[, 2:ncol(tra_new)])
y_fac <- as.factor(tra_new[, 1])
test_mat <- as.matrix(tes_new[,1:ncol(tes_new)])

modFitexT <- extraTrees(
                        x = x_mat,
                        y = y_fac,
                        mtry = 10,
                        numRandomCuts = 3
                        #numThreads = 3
                       )

# ## use 4G memory
# prepareForSave(modFitexT)
# save(modFitexT, file = "extraTrees_model.RData")
# save(tra_new, tes_new, x_mat, y_fac, test_mat, file = "extraTrees_Dos_model.RData")
# #load("extraTrees_model.RData")

predexT <- predict( modFitexT, test_mat, probability = TRUE )

#ConfusionMatrix
conMatrf <- confusionMatrix(testDat$Class, predrf); conMatrf

# ExtraTreesClassifier(
#                       n_estimators=1000,max_features= 50,
#                       criterion= 'entropy',min_samples_split= 4,
#                       max_depth= 35, min_samples_leaf= 2, n_jobs = -1
#                    )


file_aws <- 'predicted_extraTrees_noExt_rcut_5_evCuts.csv'
pred1 <- read.csv(file = file_aws)
submission <- data.frame(ID = test$ID, PredictedProb = pred1$X1)

cat("Create submission file\n")
submission <- submission[order(submission$ID),]

library(stringr)
dat_tim  <- str_replace_all(Sys.time()," |:","_")
file_tmp <- paste("Res_xxxxx_extraTrees_no_extended_noNA_", dat_tim,"_.csv", sep = "")
write.csv(submission,file_tmp,row.names = F)
