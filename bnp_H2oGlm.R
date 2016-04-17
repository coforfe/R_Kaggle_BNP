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

library(h2o)
h2o.init(nthreads = -1,max_mem_size = '8G')

### load both files in using H2O's parallel import
#train<-h2o.uploadFile("train.csv",destination_frame = "train.hex")
#test<-h2o.uploadFile("test.csv",destination_frame = "test.hex")
train <- as.h2o(datIn, destination_frame = 'train.hex')
test <- as.h2o(datTest,destination_frame = "test.hex")

train$target <- as.factor(train$target)

splits <- h2o.splitFrame(train,0.9,destination_frames = c("trainSplit","validSplit"),seed = 111111111)

#---------------------------------
#---------------------- GBM
#---------------------------------
a <- Sys.time();a

glmGrid <- expand.grid(
  st_dr = c('FALSE'),
  al_pa = 0,
  la_mb = 0
)
glmGrid$st_dr <- as.logical(glmGrid$st_dr)

res_df <- data.frame(glmAcc = 0, 
                     st_dr = 0, al_pa = 0, la_mb = 0, ex_t = 0)

for (i in 1:nrow(glmGrid)) {
print(i)
 
  ex_a <- Sys.time();  
  
glm <- h2o.glm(
  x = 3:133,
  y = 1,
  training_frame = splits[[1]],
  validation_frame = splits[[2]],
  family = 'binomial',
  model_id = "baseGlm",
  standardize = glmGrid$st_dr[i],
  alpha = glmGrid$al_pa[i],
  lambda = glmGrid$la_mb[i]
 )

  ex_b <- Sys.time(); 
  ex_t <- as.numeric(as.character(ex_b - ex_a))
  
### look at some information about the model
#summary(glm)
Accglm <- h2o.logloss(glm,valid = T)
Accglm

res_df[i,1] <- Accglm
res_df[i,2] <- glmGrid$st_dr[i]
res_df[i,3] <- glmGrid$al_pa[i]
res_df[i,4] <- glmGrid$la_mb[i]
res_df[i,5] <- ex_t

print(res_df)


#--------------------------------------------------------
#-------------- PREDICTION
#--------------------------------------------------------
### get predictions against the test set and create submission file
p <- as.data.frame(h2o.predict(glm,test))
testIds <- as.data.frame(test$ID)
submission <- data.frame(cbind(testIds,p$p1))
colnames(submission) <- c("ID","PredictedProb")

#--------------------------------------------------------
#-------------- FILE UPLOAD
#--------------------------------------------------------
dat_tim <- str_replace_all(Sys.time()," |:","_")
file_tmp <- paste("Res_xxxxx_H2O_glm_grid_",nrow(glmGrid),"_", round(Accglm,5),"_",dat_tim,"_.csv", sep = "")
write.csv(submission,file_tmp,row.names = F)

} #for( i in 1:nco
file_wt <- paste("res_df_glm_grid_",nrow(glmGrid),"_",dat_tim,"_.csv", sep = "")
write_csv(res_df, file_wt)

#--------------------------------------------------------
#-------------- GRAPHICAL ANALYSIS
#--------------------------------------------------------
names(res_df) <- c('glmAcc','standar', 'alpha','lambda', 'ex_time')
#
library(lattice)
xy_gr <- xyplot(
   glmAcc ~ lambda | as.factor(standar) * as.factor(alpha)
  ,data = res_df
  ,type = "b"
  ,strip = strip.custom(strip.names = TRUE, strip.levels = TRUE)
)
print(xy_gr)

write_csv(res_df, "res_df_glm_grid_15.csv")


#--------------------------------------------------------
#-------------- CLOSE H20
#--------------------------------------------------------

### All done, shutdown H2O    
h2o.shutdown(prompt = FALSE)


#--------------------------------------------------------
#-------------- Results
