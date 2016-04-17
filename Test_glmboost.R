library(mlbench)
library(mboost)
library(caret)

data(Sonar)

set.seed(25)
trainIndex = createDataPartition(Sonar$Class, p = 0.9, list = FALSE)
training = Sonar[ trainIndex,]
testing  = Sonar[-trainIndex,]

### set training parameters
fitControl = trainControl(method = "cv",
                          number = 3,
                          ## Estimate class probabilities
                          classProbs = TRUE,
                          verboseIter = TRUE,
                          ## Evaluate a two-class performances  
                          ## (ROC, sensitivity, specificity) using the following function 
                          summaryFunction = twoClassSummary)

### train the models

# Use the expand.grid to specify the search space   
glmBoostGrid1 = expand.grid(mstop = c(50, 100, 150),
                            prune = c("no"))

set.seed(4242)
glmBoostFit1 = train(Class ~ ., 
                     data = training,
                     method = "glmboost",
                     trControl = fitControl,
                     tuneGrid = glmBoostGrid1,
                     metric = "ROC")

print(glmBoostFit1)





set.seed(4242)

fitControl = trainControl(#method = "cv",
                          #number = 3,
                          ## Estimate class probabilities
                          classProbs = TRUE,
                          verboseIter = TRUE,
                          ## Evaluate a two-class performances  
                          ## (ROC, sensitivity, specificity) using the following function 
                          summaryFunction = twoClassSummary)

glmBoostGrid2 = expand.grid(mstop = c(50, 100, 150),
                            prune = c("no"))

glmBoostFit2 = train(Class ~ ., 
                     data = training,
                     method = "glmboost",
                     trControl = fitControl,
                     #tuneGrid = glmBoostGrid2,
                     tuneLength  = 5,
                     metric = "ROC")


print(glmBoostFit2)


