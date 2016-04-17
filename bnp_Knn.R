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

datIn <- read_csv("train.csv")
datIn <- datIn[, c(2,1, 3:ncol(datIn))]
datIn[is.na(datIn)] <- -999
datIn$target <- as.factor(datIn$target)

datTest <- read_csv("test.csv")
datTest[is.na(datTest)] <- -999

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
#----------------------  KNN
#---------------------------------
setwd("~/Downloads")

a <- Sys.time();a
set.seed(5789)

bootControl <- trainControl(number = 5,
                            summaryFunction = twoClassSummary,
                            classProbs = TRUE,
                            verboseIter = FALSE)


bootControl <- trainControl(number = 5)

knnGrid <- expand.grid(
  k = 2
)


modFitknn <-  train(
  target ~ .,
  data = trainDat,
  trControl = bootControl,
  tuneGrid = knnGrid,
  #tuneLength = 3,
  metric = "Accuracy",
  #metric = "ROC",
  method = "knn",
  preProc = c('scale', 'center')
)

modFitknn

predknn <- predict( modFitknn, newdata = testDat[,2:ncol(testDat)], type = "prob" )
#ConfusionMatrix
conMatknn <- confusionMatrix(testDat$target, predknn); conMatknn
conMatknndf <- as.data.frame(conMatknn$overall); 
knnAcc <- conMatknndf[1,1]; 
knnAcc <- as.character(round(knnAcc*100,2))
knnAcc
b <- Sys.time();b; b - a

if( nrow(knnGrid) < 2  )  { resampleHist(modFitknn) } else
{plot(modFitknn, as.table = T) }
plot(modFitknn, as.table = T) 

ggplot(modFitknn) +
  geom_smooth(se = FALSE, span = .8, method = loess) +
  theme(legend.position = "top")

#Best iteration
modBest <- modFitknn$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitknn$times$final[3]
#Samples
samp <- dim(modFitknn$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
# Impknn <- varImp( modFitknn, scale=F)
# plot(Impknn, top=20)

#Save trainDat, testDat and Model objects.
format(object.size(modFitknn), units = "Gb")

save(
  trainDat, testDat, modFitknn,
  #file=paste("knn_",numvars,"vars_n",samp,"_grid",modBestc,"_",knnAcc,"__.RData", sep="")
  file=paste("knn_",numvars,"vars_n",samp,"_grid_1_",knnAcc,"__.RData", sep = "")
)

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

