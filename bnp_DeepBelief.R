#------------------------------------------------------------
# BNP - Paribas - Claims Projections - 2016-02-15
#------------------------------------------------------------

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

#Library loading
library(ggplot2) 
library(readr) 
library(xgboost)
library(stringr)
library(caret)
library(Matrix)


cat("Read the train and test data\n")
train <- as.data.frame(read_csv("train.csv"))
test  <- as.data.frame(read_csv("test.csv"))
te_id <- test$ID

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

train_y <- train[, 2]

rm(i, lev_tmp)

#---------------------------------
#---------------------- DEEPLEARNING RCCPDL
#---------------------------------
library(RSNNS)
library(RcppDL)
library(Metrics)

dat_spl <- splitForTrainingAndTest(tra_new, train_y, ratio=0.15)
x_train <- normalizeData(dat_spl$inputsTrain, type = "0_1")
y_train <- dat_spl$targetsTrain
temp <- ifelse(y_train == 0, 1, 0)
y_train <- cbind(y_train ,temp)

hidden <- c(12,10)
fit <- Rdbn(x_train , y_train , hidden)
pretrain(fit)
finetune(fit)

x_test <- normalizeData(dat_spl$inputsTest, type = "0_1")
pred_ict <- predict(fit, x_test )






