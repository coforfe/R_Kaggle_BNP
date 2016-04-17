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

#library(doMC)
#numCor <- parallel::detectCores() - 2; numCor
##numCor <- 2
#registerDoMC(cores = numCor)

#---------------------------------------------
########------------ EXTRATREES DIRECTLY
#---------------------------------------------
library(extraTrees)
tra_newre <- tra_new[  sample(1:nrow(tra_new), 0.8*nrow(tra_new)), ]
x_mat <- as.matrix(tra_new[, 2:ncol(tra_new)])
y_fac <- as.factor(tra_new[, 1])
test_mat <- as.matrix(tes_new[,1:ncol(tes_new)])

setJavaMemory(6000)
modFitexT <- extraTrees(
                        x = x_mat,
                        y = y_fac,
                        mtry = 10,
                        numRandomCuts = 3
                        #numThreads = 3
                       )

## use 4G memory
prepareForSave(modFitexT)
save(modFitexT, file = "extraTrees_model.RData")
save(tra_new, tes_new, x_mat, y_fac, test_mat, file = "extraTrees_Dos_model.RData")
load("extraTrees_model.RData")
load("extraTrees_Dos_model.RData")

predexT <- predict( modFitexT, test_mat, probability = TRUE )
