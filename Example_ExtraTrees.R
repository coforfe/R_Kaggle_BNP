
#----- EXTRATREES
library(caret)
data(iris)

irisbig <- rbind(iris, iris)
for(i in 1:400){
  irisbig <- rbind(irisbig, iris)
}

irisbig$Species <- as.factor(as.numeric(irisbig$Species))

inTrain <- createDataPartition(irisbig$Species, p = 0.70 , list = FALSE)
trainDat <- irisbig[ inTrain, ]
testDat <- irisbig[ -inTrain, ]




datIn <- twoClassSim(1000)

inTrain <- createDataPartition(datIn$Class, p = 0.70 , list = FALSE)
trainDat <- datIn[ inTrain, ]
testDat  <- datIn[ -inTrain, ]

set.seed(6879)

bootControl <- trainControl(number=5)

rfGrid <- expand.grid(
  mtry = c(2,3,4,5, 6, 7),
  numRandomCuts = seq(1,3, 1)
)

trainDat_mat <- as.matrix(trainDat)
x_mat <- as.matrix( trainDat[, 1:(ncol(trainDat)-1)] )
y_fac <- trainDat$Class
testDat_mat <- as.matrix( testDat[, 1:(ncol(testDat)-1)] )

# x_df <- trainDat[ , 2:ncol(trainDat)]
# y_fac <- trainDat[, 1]

modFitexT <-  train(
  #target ~ .,
  x = x_mat,
  y = y_fac,
  #data = trainDat_mat,
  #data = trainDat,
  trControl = bootControl,
  tuneGrid = rfGrid,
  metric = "Kappa",
  #metric = "Accuracy",
  method = "extraTrees",
  maximize = FALSE
  
)

modFitexT

predexT <- predict( modFitexT, newdata=testDat_mat , type = "prob")

#ConfusionMatrix
conMatexT <- confusionMatrix(testDat$Class, predexT); conMatexT

library(Metrics)
act_val <- ifelse(testDat$Class == "Class1", 1, 0)
pre_val <- ifelse(predexT == "Class1" , 1 , 0)
logLoss(act_val, pre_val)



