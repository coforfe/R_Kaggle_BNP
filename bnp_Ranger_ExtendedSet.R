#------------------------------------------------------------
# BNP - Paribas - Claims Projections - 2016-02-15
#------------------------------------------------------------
setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP")

#Library loading
library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(xgboost)
library(stringr)
library(caret)


cat("Read the train and test data\n")
train <- read_csv("train.csv")
test  <- read_csv("test.csv")

cat("Recode NAs to -997\n")
train[is.na(train)]   <- -997
test[is.na(test)]   <- -997

cat("Get feature names\n")
feature.names <- names(train)[c(3:ncol(train))]

cat("Remove highly correlated features\n")
highCorrRemovals <- c("v8","v23","v25","v36","v37","v46",
                      "v51","v53","v54","v63","v73","v81",
                      "v82","v89","v92","v95","v105","v107",
                      "v108","v109","v116","v117","v118",
                      "v119","v123","v124","v128")
feature.names <- feature.names[!(feature.names %in% highCorrRemovals)]

cha_col <- 0
cont <- 0
cat("Replace categorical variables with integers\n")
for (f in feature.names) {
  if (class(train[[f]]) == "character") {
    cont <- cont + 1
    cha_col[cont] <- f
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels = levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels = levels))
  }
}

tra <- train[,feature.names]
tra <- cbind.data.frame( train[, 1:2], tra)
tes <- test[,feature.names]

#Feature Engineering....
#High important variables - 10 most improtant - Numeric.
#See end of program about how to calculate them.
high_imp_var <- c(
  "v49", "v39", "v11", "v20", "v21", 
  "v33", "v13", "v9",  "v65"
)
# Create combinations of two and calculate ratios
# They will be new columns in the
#comb_high <- as.data.frame(t(combn(high_imp_var, 2)))
comb_high_3 <- as.data.frame(t(combn(high_imp_var[1:5], 2)))

#----- Train--------
# With just the three best one high importance
ratio_df <- 0
for (j in 1:nrow(comb_high_3)) {
  print(j)
  nam_a  <- as.character(comb_high_3$V1[j])
  nam_b  <- as.character(comb_high_3$V2[j])
  col_a  <- tra[,nam_a]
  col_b  <- tra[,nam_b]
  col_a[col_a == -997] <- NA
  col_b[col_b == -997] <- NA
  col_ab <- col_a / col_b  
  col_ab[is.na(col_ab)] <- -997
  ratio_df <- cbind.data.frame( ratio_df, col_ab) 
}
ratio_df <- ratio_df[, 2:ncol(ratio_df)]
names(ratio_df) <- paste(comb_high_3$V1,comb_high_3$V2, sep = "_")

#Add these new "ratio" columns to "tra".
tra_ex <- cbind.data.frame(tra, ratio_df)

#----- Test----------
# With just the three best one high importance
ratio_df <- 0
for (j in 1:nrow(comb_high_3)) {
  print(j)
  nam_a  <- as.character(comb_high_3$V1[j])
  nam_b  <- as.character(comb_high_3$V2[j])
  col_a  <- tes[,nam_a]
  col_b  <- tes[,nam_b]
  col_a[col_a == -997] <- NA
  col_b[col_b == -997] <- NA
  col_ab <- col_a / col_b  
  col_ab[is.na(col_ab)] <- -997
  ratio_df <- cbind.data.frame( ratio_df, col_ab) 
}
ratio_df <- ratio_df[, 2:ncol(ratio_df)]
names(ratio_df) <- paste(comb_high_3$V1,comb_high_3$V2, sep = "_")

#Add these new "ratio" columns to "tes".
tes_ex <- cbind.data.frame(tes, ratio_df)

datIn <- tra_ex; datIn <- datIn[, 2:ncol(datIn)]
datIn$target <- as.factor(datIn$target)
datIn$target <- as.factor(ifelse(datIn$target == 1, 'yes', 'no'))
datTestpre <- tes_ex

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
#setwd("~/Downloads")

a <- Sys.time();a
set.seed(5789)

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

#Save trainDat, testDat and Model objects.
save(
  trainDat, testDat, modFitrf,
  file = paste("ranger_",numvars,"vars_rf_n",samp,"_grid",modBestc,"_",rfAcc,"__.RData", sep="")
)

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")


#------------------------------------------------------------
# TEST
#------------------------------------------------------------

#------------------------------------------------------------
# PREDICT
#------------------------------------------------------------
modFit <- modFitrf 
in_err <- rfAcc
#in_err <- 77.92
modtype <- modFit$method
samptmp <- modFit$resample; samp <- length(unique(samptmp$Resample))
numvars <- length(modFit$coefnames)
timval <- str_replace_all(Sys.time(), " |:", "_")

#Is it "prob" of 'yes' or 'no'....?
pred_BNP <- predict(modFit, newdata = datTestpre, type = "prob")
toSubmit <- data.frame(ID = test$ID, PredictedProb = pred_BNP$yes)

write.table(toSubmit, file = paste("Res_xxxx_datArtem_", modtype,"_",numvars,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep=""),sep=","
            , row.names = FALSE,col.names = TRUE, quote = FALSE)
