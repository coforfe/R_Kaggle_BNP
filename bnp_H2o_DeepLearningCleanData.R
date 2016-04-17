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

tra_new <- cbind.data.frame( target = train[, 2], tra_new)

rm(i, lev_tmp)


#--------------------------------------------------------
#-------------- READY TO MODEL
#--------------------------------------------------------

library(h2o)
h2o.init(nthreads = -1, max_mem_size = '8G')

### load both files in using H2O's parallel import
#train<-h2o.uploadFile("train.csv",destination_frame = "train.hex")
#test<-h2o.uploadFile("test.csv",destination_frame = "test.hex")
train <- as.h2o(tra_new, destination_frame = 'train.hex')
test  <- as.h2o(tes_new, destination_frame = 'test.hex')

train$target <- as.factor(train$target)

splits <- h2o.splitFrame(train,0.9,destination_frames = c("trainSplit","validSplit"),seed=111111111)

#---------------------------------
#---------------------- DEEPLEARNING
#---------------------------------
a <- Sys.time();a

dlGrid <- expand.grid(
   hi_dd_A = seq(50,300,50),
   hi_dd_B = seq(50,300,50),
   hi_dd_C = seq(2,300,50),
   ac_ti = c('Rectifier'),
   #ac_ti = c('Tanh'), 
   #mx_w2 = seq(10,50,       length.out = 2),
   mx_w2 = 10,
   #l1_va = seq(1e-5, 1e-3,  length.out = 2),
   l1_va = 1e-5,
   #l2_va = seq(1e-5, 1e-3,  length.out = 2),
   l2_va = 1e-5,
   #ep_si = seq(1e-4, 1e-10, length.out = 2),
   ep_si = 1e-10,
   #rh_oo = seq(0.9, 0.99,   length.out = 2),
   rh_oo = 0.90,
   #ra_te = seq(1e-4, 1e-2,  length.out = 2),
   ra_te = 1e-2,
   #ra_de = seq(0.5, 1,      length.out = 2),
   ra_de = 0.5,
   #ra_an = seq(1e-5, 1e-9,  length.out = 2),
   ra_an = 1e-9,
   #mo_st = seq(0.5, 0.9,    length.out = 2),
   mo_st = 0.7,
   mo_rp = 1/0.7
)

row_sam <- sample(1:nrow(dlGrid), nrow(dlGrid))
dlGrid <- dlGrid[ row_sam , ]

 res_df <- data.frame(
                      dlAcc=0, hi_dd_A=0, hi_dd_B=0, hi_dd_C=0, 
                      ac_ti=0, mx_w2=0, l1_va=0, l2_va=0, 
                      ep_si=0, rh_oo=0, ra_te=0, ra_de=0, 
                      ra_an=0, mo_st=0, mo_rp=0, ex_t=0
                     )

 j <- 0
 
 for( i in 1:nrow(dlGrid)) {
   print(i)
  
  ex_a <- Sys.time();  
  
 res_dl <- h2o.deeplearning(
    x = 3:133,
    y=1,
    training_frame = splits[[1]],
    validation_frame = splits[[2]],
    stopping_rounds = 1,             
    stopping_tolerance = 0,
    seed = 222222222,
    model_id = "baseDL",
    stopping_metric = "logloss",
    nesterov_accelerated_gradient = TRUE,
    epochs = 500,
    momentum_stable = 0.99,
    input_dropout_ratio = 0.2,
    initial_weight_distribution = 'Normal',
    initial_weight_scale = 0.01,
    loss = 'CrossEntropy',
    fast_mode = TRUE,
    diagnostics = TRUE,
    ignore_const_cols = TRUE,
    force_load_balance = FALSE, 
    hidden = c(dlGrid$hi_dd_A[i], dlGrid$hi_dd_B[i], dlGrid$hi_dd_C[i]),
    activation = as.vector(dlGrid$ac_ti[i]),
    max_w2 = dlGrid$mx_w2[i],
    l1 = dlGrid$l1_va[i],
    l2 = dlGrid$l2_va[i],
    epsilon = dlGrid$ep_si[i],
    rho = dlGrid$rh_oo[i],
    rate = dlGrid$ra_te[i],
    rate_decay = dlGrid$ra_de[i],
    rate_annealing = dlGrid$ra_an[i],
    momentum_start = dlGrid$mo_st[i],
    momentum_ramp = dlGrid$mo_rp[i]
  )
  ex_b <- Sys.time();  
  ex_t <- as.numeric(as.character(ex_b - ex_a))
  
  ### look at some information about the model
  Accdl <- h2o.logloss(res_dl,valid=T)
  print(Accdl)
  
#if(Accdl < 0.46) {  
  j <- j+1
  res_df[j,1] <-  Accdl
  res_df[j,2] <-  dlGrid$hi_dd_A[i]
  res_df[j,3] <-  dlGrid$hi_dd_B[i]
  res_df[j,4] <-  dlGrid$hi_dd_C[i]
  res_df[j,5] <-  as.vector(dlGrid$ac_ti[i])
  res_df[j,6] <-  dlGrid$mx_w2[i]
  res_df[j,7] <-  dlGrid$l1_va[i]
  res_df[j,8] <-  dlGrid$l2_va[i]
  res_df[j,9] <-  dlGrid$ep_si[i]
  res_df[j,10] <-  dlGrid$rh_oo[i]
  res_df[j,11] <-  dlGrid$ra_te[i]
  res_df[j,12] <- dlGrid$ra_de[i]
  res_df[j,13] <- dlGrid$ra_an[i]
  res_df[j,14] <- dlGrid$mo_st[i]
  res_df[j,15] <- dlGrid$mo_rp[i]
  res_df[j,16] <- ex_t

  print(res_df)
  print(max(res_df$Accdl))  
  
if(Accdl < 0.49) {  
  #--------------------------------------------------------
  #-------------- PREDICTION
  #--------------------------------------------------------
  ### get predictions against the test set and create submission file
  p<-as.data.frame(h2o.predict(res_dl,test))
  testIds<-as.data.frame(test$ID)
  submission<-data.frame(cbind(testIds,p$p1))
  colnames(submission)<-c("ID","PredictedProb")
  
  #--------------------------------------------------------
  #-------------- FILE UPLOAD
  #--------------------------------------------------------
  file_tmp <- paste("Res_xxxxx_H2O_dl_", round(Accdl,5),"_.csv", sep="")
  write.csv(submission,file_tmp,row.names=F)
  
  #Also write the data.frame with the results..
  write_csv(res_df, "res_df_dl_.csv")
} else next
  
 } #for( i in 1:nco
write_csv(res_df, "res_df_dl_.csv")
b <- Sys.time();b; b-a

# names(res_df) <- c('rfAcc', 'max_depth','samp_rate','mtries','ex_time')
# 
# library(lattice)
# xy_gr <- xyplot(
#   rfAcc ~ samp_rate
#   ,data = res_df
#   ,type = "b"
#   ,strip = strip.custom(strip.names = TRUE, strip.levels = TRUE)
# )
# print(xy_gr)

#--------------------------------------------------------
#-------------- CLOSE H20
#--------------------------------------------------------

### All done, shutdown H2O    
h2o.shutdown(prompt=FALSE)


#--------------------------------------------------------
#-------------- Results



