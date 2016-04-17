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
library(h2o)

datIn <- read_csv("train.csv")
datIn <- datIn[, c(2,1, 3:ncol(datIn))]
datIn[is.na(datIn)] <- -999
datIn$target <- as.factor(datIn$target)

datTest <- read_csv("test.csv")
datTest[is.na(datTest)] <- -999

col_char <- 0
j <- 0
for( i in 2:ncol(datIn)) {
  cltmp <- class(datIn[, i])
  if(cltmp == "character") {
    j <- j + 1
    col_char[j] <- i
    datIn[,i] <- as.numeric( as.factor(datIn[,i]) )
  } else next
}

col_char <- 0
j <- 0
for( i in 1:ncol(datTest)) {
  cltmp <- class(datTest[, i])
  if(cltmp == "character") {
    j <- j + 1
    col_char[j] <- i
    datTest[,i] <- as.numeric( as.factor(datTest[,i]) )
  } else next
}

load("information_value.RData")
var_god <- information_value$Variable[1:14]
datIn_new <- cbind.data.frame(datIn[,1:2], datIn[, var_god ])
datTest_new <- cbind.data.frame(datTest[,1], datTest[, var_god ])
names(datTest_new)[1] <- names(datTest)[1]

#--------------------------------------------------------
#-------------- READY TO MODEL
#--------------------------------------------------------

library(h2o)
h2o.init(nthreads=-1,max_mem_size = '8G')

### load both files in using H2O's parallel import
#train<-h2o.uploadFile("train.csv",destination_frame = "train.hex")
#test<-h2o.uploadFile("test.csv",destination_frame = "test.hex")
# train <- as.h2o(datIn, destination_frame = 'train.hex')
# test<- as.h2o(datTest,destination_frame = "test.hex")
train <- as.h2o(datIn_new, destination_frame = 'train.hex')
test<- as.h2o(datTest_new,destination_frame = "test.hex")

train$target<-as.factor(train$target)

splits<-h2o.splitFrame(train,0.9,destination_frames = c("trainSplit","validSplit"),seed=111111111)

#---------------------------------
#---------------------- RandomForest
#---------------------------------
a <- Sys.time();a

rfGrid <- expand.grid(
  mx_dp = 28,
  sa_ra = seq(0.794, 0.797, length.out = 10),
  ma_tr = 11
)

res_df <- data.frame(rfAcc=0, mx_dp=0, sa_ra=0, ma_tr=0, ex_t=0)

for( i in 1:nrow(rfGrid)) {
print(i)
  
ex_a <- Sys.time();  
  
rf<-h2o.randomForest(
  x = 3:133,
  y=1,
  training_frame = splits[[1]],
  validation_frame = splits[[2]],
  max_depth = rfGrid$mx_dp[i],
  mtries = rfGrid$ma_tr[i],
  sample_rate = rfGrid$sa_ra[i],
  ntrees = 3000,
  binomial_double_trees = TRUE,
  score_each_iteration = TRUE,
  stopping_rounds = 1,             
  stopping_tolerance = 0,
  seed = 222222222,
  model_id = "baseRf",
  stopping_metric = "logloss"
 )
ex_b <- Sys.time();  
ex_t <- as.numeric(as.character(ex_b - ex_a))

### look at some information about the model
#summary(gbm)
Accrf <- h2o.logloss(rf,valid=T)
print(Accrf)

res_df[i,1] <- Accrf
res_df[i,2] <- rfGrid$mx_dp[i]
res_df[i,3] <- rfGrid$sa_ra[i]
res_df[i,4] <- rfGrid$ma_tr[i]
res_df[i,5] <- ex_t

print(res_df)

#--------------------------------------------------------
#-------------- PREDICTION
#--------------------------------------------------------
### get predictions against the test set and create submission file
p<-as.data.frame(h2o.predict(rf,test))
testIds<-as.data.frame(test$ID)
submission<-data.frame(cbind(testIds,p$p1))
colnames(submission)<-c("ID","PredictedProb")

#--------------------------------------------------------
#-------------- FILE UPLOAD
#--------------------------------------------------------
file_tmp <- paste("Res_xxxxx_H2O_rf_sa_ra_079_", round(Accrf,5),"_.csv", sep="")
write.csv(submission,file_tmp,row.names=F)

} #for( i in 1:nco
write_csv(res_df, "res_df_rf_grid_sa_ra_079_.csv")
b <- Sys.time();b; b-a

names(res_df) <- c('rfAcc', 'max_depth','samp_rate','mtries','ex_time')

library(lattice)
xy_gr <- xyplot(
   rfAcc ~ samp_rate
  ,data = res_df
  ,type = "b"
  ,strip = strip.custom(strip.names = TRUE, strip.levels = TRUE)
)
print(xy_gr)

#--------------------------------------------------------
#-------------- CLOSE H20
#--------------------------------------------------------

### All done, shutdown H2O    
h2o.shutdown(prompt=FALSE)


#--------------------------------------------------------
#-------------- Results

