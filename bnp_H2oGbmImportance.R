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
datIn[is.na(datIn)] <- -1
datIn$target <- as.factor(datIn$target)

datTest <- read_csv("test.csv")
datTest[is.na(datTest)] <- -1

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

load("information_value.RData")
var_god <- information_value$Variable[1:14]
datIn_new <- cbind.data.frame(datIn[,1:2], datIn[, var_god ])
datTest_new <- cbind.data.frame(datTest[,1], datTest[, var_god ])
names(datTest_new)[1] <- names(datTest)[1]

#--------------------------------------------------------
#-------------- READY TO MODEL
#--------------------------------------------------------

library(h2o)
h2o.init(nthreads = -1,max_mem_size = '8G')

### load both files in using H2O's parallel import
# train <- as.h2o(datIn, destination_frame = 'train.hex')
# test <- as.h2o(datTest,destination_frame = "test.hex")
train <- as.h2o(datIn_new, destination_frame = 'train.hex')
test <- as.h2o(datTest_new,destination_frame = "test.hex")

train$target <- as.factor(train$target)

splits <- h2o.splitFrame(train,0.9,destination_frames = c("trainSplit","validSplit"),seed = 111111111)

#---------------------------------
#---------------------- GBM
#---------------------------------
a <- Sys.time();a

gbmGrid <- expand.grid(
  ma_de = seq(7,15, 1),
  le_ra = 0.35,
  sa_ra = 0.9,
  co_ra = 0.9
)

res_df <- data.frame(gbmAcc = 0, ma_de = 0, le_ra = 0,
                     sa_ra = 0, co_ra = 0, ex_t = 0)

for (i in 1:nrow(gbmGrid)) {
print(i)
 
  ex_a <- Sys.time();  
  
gbm <- h2o.gbm(
  x = 3:ncol(datIn_new),
  y = 1,
  training_frame = splits[[1]],
  validation_frame = splits[[2]],
  ntrees = 3000,                    ## let stopping criteria dictate the number of trees
  stopping_rounds = 1,              ## wait until the last round is worse than the previous
                                    ##  this seems low because scoring is not on every tree by default
                                    ##  If that is desired, you can turn on score_each_iteration
                                    ## (and then possibly increase stopping)
  score_each_iteration = TRUE,
  stopping_tolerance = 0,
  max_depth = gbmGrid$ma_de[i],
  learn_rate = gbmGrid$le_ra[i],
  sample_rate = gbmGrid$sa_ra[i],                ## 80% row sampling
  col_sample_rate = gbmGrid$co_ra[i],            ## 70% columns
  seed = 5555555555,
  model_id = "baseGbm",
  stopping_metric = "logloss"
 )

  ex_b <- Sys.time(); 
  ex_t <- as.numeric(as.character(ex_b - ex_a))
  
### look at some information about the model
#summary(gbm)
Accgbm <- h2o.logloss(gbm,valid = T)
Accgbm

res_df[i,1] <- Accgbm
res_df[i,2] <- gbmGrid$ma_de[i]
res_df[i,3] <- gbmGrid$le_ra[i]
res_df[i,4] <- gbmGrid$sa_ra[i]
res_df[i,5] <- gbmGrid$co_ra[i]
res_df[i,6] <- ex_t

print(res_df)


#--------------------------------------------------------
#-------------- PREDICTION
#--------------------------------------------------------
### get predictions against the test set and create submission file
p <- as.data.frame(h2o.predict(gbm,test))
testIds <- as.data.frame(test$ID)
submission <- data.frame(cbind(testIds,p$p1))
colnames(submission) <- c("ID","PredictedProb")

#--------------------------------------------------------
#-------------- FILE UPLOAD
#--------------------------------------------------------
dat_tim <- str_replace_all(Sys.time()," |:","_")
file_tmp <- paste("Res_xxxxx_H2O_gbm_grid_",nrow(gbmGrid),"_", round(Accgbm,5),"_",dat_tim,"_.csv", sep = "")
write.csv(submission,file_tmp,row.names = F)

} #for( i in 1:nco
file_wt <- paste("res_df_gbm_grid_",nrow(gbmGrid),"_",dat_tim,"_.csv", sep = "")
write_csv(res_df, file_wt)
b <- Sys.time();b; b - a

#-------------- END OF PROGRAM ----------------------------- 


#--------------------------------------------------------
#-------------- GRAPHICAL ANALYSIS
#--------------------------------------------------------
names(res_df) <- c('gbmAcc','max_depth', 'learn_rate','samp_rate','col_samp', 'ex_time')
#
library(lattice)
xy_gr <- xyplot(
   gbmAcc ~ samp_rate | col_samp
  ,data = res_df
  ,type = "b"
  ,strip = strip.custom(strip.names = TRUE, strip.levels = TRUE)
)
print(xy_gr)

write_csv(res_df, "res_df_gbm_grid_15.csv")


#--------------------------------------------------------
#-------------- CLOSE H20
#--------------------------------------------------------

### All done, shutdown H2O    
h2o.shutdown(prompt = FALSE)


#--------------------------------------------------------
#-------------- Results
