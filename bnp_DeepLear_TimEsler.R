#------------------------------------------------------------
# BNP - Paribas - Claims Projections - 2016-02-15
#------------------------------------------------------------

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP_Paribas")

#Library loading
library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(xgboost)
library(stringr)


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
comb_high_3 <- as.data.frame(t(combn(high_imp_var[1:3], 2)))

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


#--------------------------------------------------------
#-------------- READY TO MODEL
#--------------------------------------------------------

library(h2o)
h2o.init(nthreads = -1,max_mem_size = '8G')

### load both files in using H2O's parallel import
#train<-h2o.uploadFile("train.csv",destination_frame = "train.hex")
#test<-h2o.uploadFile("test.csv",destination_frame = "test.hex")
train_h <- as.h2o(tra_ex, destination_frame = 'train.hex')
test_h <- as.h2o(tes_ex,destination_frame = "test.hex")

train_h$target <- as.factor(train_h$target)

splits <- h2o.splitFrame(train_h,0.9,destination_frames = c("trainSplit","validSplit"),seed = 111111111)

#-----------------------------------------
#---------------------- DEEP LEARNING
#-----------------------------------------
a <- Sys.time();a

dlGrid <- expand.grid(
  # hi_dd_A = seq(50,300,50),
  # hi_dd_B = seq(50,300,50),
  # hi_dd_C = seq(2,300,50),
  hi_dd_A = 109,
  hi_dd_B = 56,
  hi_dd_C = 2,
  ac_ti = c('Tanh', 'RectifierWithDropout'), 
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

#row_sam <- sample(1:nrow(dlGrid), nrow(dlGrid))
#dlGrid <- dlGrid[ row_sam , ]

res_df <- data.frame(
  dlAcc = 0, hi_dd_A = 0, hi_dd_B = 0, hi_dd_C = 0, 
  ac_ti = 0, mx_w2 = 0, l1_va = 0, l2_va = 0, 
  ep_si = 0, rh_oo = 0, ra_te = 0, ra_de = 0, 
  ra_an = 0, mo_st = 0, mo_rp = 0, ex_t = 0
)

j <- 0

for (i in 1:nrow(dlGrid)) {
  print(i)
  
  ex_a <- Sys.time();  
  
  res_dl <- h2o.deeplearning(
    x = 3:ncol(tra),
    y = 2,
    training_frame = splits[[1]],
    validation_frame = splits[[2]],
    stopping_rounds = 1,             
    stopping_tolerance = 0,
    seed = 1234567890,
    model_id = "baseDL",
    stopping_metric = "logloss",
    nesterov_accelerated_gradient = FALSE,
    epochs = 500,
    momentum_stable = 0.99,
    input_dropout_ratio = 0.2,
    initial_weight_distribution = 'Normal',
    initial_weight_scale = 0.01,
    loss = 'CrossEntropy',
    fast_mode = FALSE,
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
  Accdl <- h2o.logloss(res_dl,valid = T)
  print(Accdl)
  
  #if(Accdl < 0.46) {  
  j <- j + 1
  res_df[j,1]  <-  Accdl
  res_df[j,2]  <-  dlGrid$hi_dd_A[i]
  res_df[j,3]  <-  dlGrid$hi_dd_B[i]
  res_df[j,4]  <-  dlGrid$hi_dd_C[i]
  res_df[j,5]  <-  as.vector(dlGrid$ac_ti[i])
  res_df[j,6]  <-  dlGrid$mx_w2[i]
  res_df[j,7]  <-  dlGrid$l1_va[i]
  res_df[j,8]  <-  dlGrid$l2_va[i]
  res_df[j,9]  <-  dlGrid$ep_si[i]
  res_df[j,10] <-  dlGrid$rh_oo[i]
  res_df[j,11] <-  dlGrid$ra_te[i]
  res_df[j,12] <- dlGrid$ra_de[i]
  res_df[j,13] <- dlGrid$ra_an[i]
  res_df[j,14] <- dlGrid$mo_st[i]
  res_df[j,15] <- dlGrid$mo_rp[i]
  res_df[j,16] <- ex_t
  
  print(res_df)
  print(min(res_df$dlAcc))  
  
  if (Accdl < 0.45) {  
    #--------------------------------------------------------
    #-------------- PREDICTION
    #--------------------------------------------------------
    ### get predictions against the test set and create submission file
    p <- as.data.frame(h2o.predict(res_dl,test_h))
    testIds <- as.data.frame(test$ID)
    submission <- data.frame(cbind(testIds,p$p1))
    colnames(submission) <- c("ID","PredictedProb")
    
    #--------------------------------------------------------
    #-------------- FILE UPLOAD
    #--------------------------------------------------------
    dat_tim  <- str_replace_all(Sys.time()," |:","_")
    file_tmp <- paste("Res_xxxxx_Extended_3_H2O_dl_", round(Accdl,5),"_",dat_tim ,"_.csv", sep = "")
    write.csv(submission,file_tmp,row.names = F)
    
    #Also write the data.frame with the results..
    write_csv(res_df, "res_df_dl_.csv")
  } else next
  
} #for( i in 1:nco
write_csv(res_df, "res_df_dl_.csv")
b <- Sys.time();b; b - a


#--------------------------------------------------------
#---------------------- END OF PROGRAM
#--------------------------------------------------------


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
h2o.shutdown(prompt = FALSE)


#--------------------------------------------------------
#-------------- Results


