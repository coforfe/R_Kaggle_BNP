#------------------------------------------------------------
# BNP - Paribas - Claims Projections - 2016-02-15
#------------------------------------------------------------
setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/02_BNP")

#Library loading
library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(xgboost)
library(stringr)


cat("Read the train and test data\n")
train <- read_csv("train.csv")
test  <- read_csv("test.csv")

cat("Recode NAs to -997\n")
train[is.na(train)] <- -999
test[is.na(test)]   <- -999

cat("Get feature names\n")
feature_names <- names(train)[c(3:ncol(train))]

cat("Remove highly correlated features\n")
highCorrRemovals <- c("v8","v23","v25","v36","v37","v46",
                      "v51","v53","v54","v63","v73","v81",
                      "v82","v89","v92","v95","v105","v107",
                      "v108","v109","v116","v117","v118",
                      "v119","v123","v124","v128")
feature_names <- feature_names[!(feature_names %in% highCorrRemovals)]

cha_col <- 0
cont <- 0
cat("Replace categorical variables with integers\n")
for (f in feature_names) {
  if (class(train[[f]]) == "character") {
    cont <- cont + 1
    cha_col[cont] <- f
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels = levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels = levels))
  }
}

tra <- train[,feature_names]
tes <- test[, feature_names]

#Put together train and test sets for easiness manipulation
all <- rbind.data.frame(tra, tes)

#Feature Engineering....
#High important variables - 10 most improtant - Numeric.
#See end of program about how to calculate them.
high_imp_var <- c(
  "v49", "v65", "v11", "v55", "v39", "v9", "v113", 'v33', 'v30', 'v45'
)
# Create combinations of two and calculate ratios
# They will be new columns in the
#comb_high <- as.data.frame(t(combn(high_imp_var, 2)))
comb_high_3 <- as.data.frame(t(combn(high_imp_var[1:10], 2)))

#----------------------------------------------
# With just the three best one high importance
# Calculate RATIO
ratio_df <- 0
for (j in 1:nrow(comb_high_3)) {
 print(j)
 nam_a  <- as.character(comb_high_3$V1[j])
 nam_b  <- as.character(comb_high_3$V2[j])
 col_a  <- all[,nam_a]
 col_b  <- all[,nam_b]
 col_a[col_a == -999] <- NA
 col_b[col_b == -999] <- NA
 col_ab <- col_a / col_b  
 col_ab[is.na(col_ab)] <- -999 
 ratio_df <- cbind.data.frame( ratio_df, col_ab) 
}
ratio_df <- ratio_df[, 2:ncol(ratio_df)]
names(ratio_df) <- paste(comb_high_3$V1,comb_high_3$V2, sep = "_r_")

all <- cbind.data.frame(all, ratio_df)
all[ all == -999] <- NA

rm(nam_a, nam_b, col_a, col_b, col_ab, ratio_df, j)

#----------------------------------------------
# With just the best one high importance
# Calculate PRODUCT 
prod_df <- 0
for (j in 1:nrow(comb_high_3)) {
  print(j)
  nam_a  <- as.character(comb_high_3$V1[j])
  nam_b  <- as.character(comb_high_3$V2[j])
  col_a  <- all[,nam_a]
  col_b  <- all[,nam_b]
  col_a[col_a == -999] <- NA
  col_b[col_b == -999] <- NA
  col_ab <- col_a * col_b  
  col_ab[is.na(col_ab)] <- -999 
  prod_df <- cbind.data.frame( prod_df, col_ab) 
}
prod_df <- prod_df[, 2:ncol(prod_df)]
names(prod_df) <- paste(comb_high_3$V1,comb_high_3$V2, sep = "_p_")

all <- cbind.data.frame(all, prod_df)
all[ all == -999] <- NA

rm(nam_a, nam_b, col_a, col_b, col_ab, prod_df, j)

#----------------------------------------------
# With just the best one high importance
# Calculate SUM.
log_df <- 0
for (j in 1:nrow(comb_high_3)) {
  print(j)
  nam_a  <- as.character(comb_high_3$V1[j])
  nam_b  <- as.character(comb_high_3$V2[j])
  col_a  <- all[,nam_a]
  col_b  <- all[,nam_b]
  col_a[col_a == -999] <- NA
  col_b[col_b == -999] <- NA
  col_ab <-  col_a + col_b
  col_ab[is.na(col_ab)] <- -999
  log_df <- cbind.data.frame( log_df, col_ab)
}
log_df <- log_df[, 2:ncol(log_df)]
names(log_df) <- paste(comb_high_3$V1,comb_high_3$V2, sep = "_l_")

all <- cbind.data.frame(all, log_df)
all[ all == -999] <- NA

rm(nam_a, nam_b, col_a, col_b, col_ab, log_df, j)


#----------------------------------------------
# With just the best one high importance
# Calculate DIFFERENCE
dif_df <- 0
for (j in 1:nrow(comb_high_3)) {
  print(j)
  nam_a  <- as.character(comb_high_3$V1[j])
  nam_b  <- as.character(comb_high_3$V2[j])
  col_a  <- all[,nam_a]
  col_b  <- all[,nam_b]
  col_a[col_a == -999] <- NA
  col_b[col_b == -999] <- NA
  col_ab <-  col_a - col_b
  col_ab[is.na(col_ab)] <- -999
  dif_df <- cbind.data.frame( dif_df, col_ab)
}
dif_df <- dif_df[, 2:ncol(dif_df)]
names(dif_df) <- paste(comb_high_3$V1,comb_high_3$V2, sep = "_d_")

all <- cbind.data.frame(all, dif_df)
all[ all == -999] <- NA

rm(nam_a, nam_b, col_a, col_b, col_ab, dif_df, j)


#---------- NEW COLUMNS (Justfor's idea) ---------
col_now <- names(all)
N <- length(col_now)
all$fe_NAcountN  <-  rowSums(is.na(all)) / N 
all$fe_NAcountN  <-  rowSums(is.na(all)) / N 
all$fe_NAcount   <-  rowSums(is.na(all))  
all$fe_NAcount   <-  rowSums(is.na(all))  
all$fe_Zero      <-  rowSums(all[, col_now] == 0) / N
all$fe_Zero      <-  rowSums(all[, col_now] == 0) / N
all$fe_below0    <-  rowSums(all[, col_now] < 0) / N
all$fe_below0    <-  rowSums(all[, col_now] < 0) / N

rm(tra, tes, col_now, f, cont, feature_names, 
   high_imp_var, highCorrRemovals, levels, N)

all[ is.na(all)] <- -999

#Separate train and test sets
tra_ex <- all[ 1:nrow(train) ,]
tes_ex <- all[ (nrow(train) + 1):nrow(all) ,]

tra_ex <- cbind.data.frame( target = train[, 2], tra_ex)
# save(tra_ex, tes_ex, file = "extended_2016_08.RData")

#---------------------------------
#---------------------- XGBOOST
#---------------------------------

cat("Sample data for early stopping\n")
h <- sample(nrow(train),2000)
dval      <- xgb.DMatrix(data = data.matrix(tra_ex[h,]), label = train$target[h] , missing = NA)
dtrain    <- xgb.DMatrix(data = data.matrix(tra_ex[-h,]),label = train$target[-h], missing = NA)
watchlist <- list(val = dval,train = dtrain)

# # Run settings - Original settings
# et <- 0.01
# md <- 11
# ss <- 0.96
# cs <- 0.45
# mc <- 1
# np <- 1
# nrounds <- 1501 # CHANGE TO >1500
# early.stop.round <- 300

a <- Sys.time();a

# # et ~ 0.005 overfits.
# xgbGrid <- expand.grid(
#   et = 0.0775,
#   md = 12,
#   ss = 0.93,
#   cs = 0.45,
#   mc = 1,
#   np = 1,
#   nr = 1501,
#   es = 150,
#   rn = 0
# )

xgbGrid <- expand.grid(
  et = 0.01025,
  md = 11,
  ss = 0.96,
  cs = 0.45,
  mc = 1,
  np = 1,
  ms = 1,                   # max_delta_step
  nr = 3501,
  es = 250,
  rn = 0
)


#rn <- c(21219, sample(21219:(21219*2) , 5))
res_df <- data.frame(
                      xgbAcc = 0, xgbIdx = 0, et = 0, md = 0,
                      ss = 0, cs = 0, mc = 0, np = 0, ms = 0,
                      nr = 0, es = 0, rn = 0,
                      ex_t = 0
                     )

cat("Set seed\n")
# rn_val <- round(runif(nrow(xgbGrid) , min = 1e4, max = 1e6) , 0); rn_val
rn_val <- 779667

#Param Iteration
ens_ble <- 0
con_ens <- 0

for (i in 1:nrow(xgbGrid)) {
  print(i)
  
  ex_a <- Sys.time();  

param <- list(  
                objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "logloss",
                eta                 = xgbGrid$et[i],
                max_depth           = xgbGrid$md[i],
                subsample           = xgbGrid$ss[i],
                colsample_bytree    = xgbGrid$cs[i],
                min_child_weight    = xgbGrid$mc[i],
                num_parallel_tree   = xgbGrid$np[i],
                max_delta_step      = xgbGrid$ms[i],
                set.seed            = rn_val[i]
)

cat("Train model\n")
ex_a <- Sys.time();  
clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = xgbGrid$nr[i],
                    early.stop.round    = xgbGrid$es[i],
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    verbose             = 1,  
                    print.every.n       = 50,
                    nthread             = 4 
)

ex_b <- Sys.time();  ex_t <- ex_b - ex_a; (ex_t)

Accxgb <- clf$bestScore
Accidx <- clf$bestInd
(Accxgb)
(Accidx)

#Store results
res_df[i,1]  <- Accxgb
res_df[i,2]  <- Accidx
res_df[i,3]  <- xgbGrid$et[i]
res_df[i,4]  <- xgbGrid$md[i]
res_df[i,5]  <- xgbGrid$ss[i]
res_df[i,6]  <- xgbGrid$cs[i]
res_df[i,7]  <- xgbGrid$mc[i]
res_df[i,8]  <- xgbGrid$np[i]
res_df[i,9]  <- xgbGrid$ms[i]
res_df[i,10] <- xgbGrid$nr[i]
res_df[i,11] <- xgbGrid$es[i]
res_df[i,12] <- rn_val[i]
res_df[i,13] <- ex_t

cat("\n")
print(res_df)
cat("\n")

# if (Accxgb > 0.44) { next }

#--------------------------------------------------------
#-------------- PREDICTION
#--------------------------------------------------------
cat("Calculate predictions\n")
pred1 <- predict(clf,
                 data.matrix(tes_ex), missing = NA
                 #ntreelimit = clf$bestInd #Prefer with the whole set although could overfits
                 )

# if (Accxgb < 0.44) {
#   con_ens <- con_ens + 1
#   ens_ble <- ens_ble + pred1
# }

  con_ens <- con_ens + 1
  ens_ble <- ens_ble + pred1

#--------------------------------------------------------
#-------------- FILE UPLOAD
#--------------------------------------------------------
submission <- data.frame(ID = test$ID, PredictedProb = pred1)

LL <- clf$bestScore
cat(paste("Best AUC: ",LL,"\n",sep = ""))

cat("Create submission file\n")
submission <- submission[order(submission$ID),]

dat_tim  <- str_replace_all(Sys.time()," |:","_")
file_tmp <- paste("Res_xxxxx_Extended_noNA_rat_prod_5_XGB_Acc_", Accxgb ,"_", dat_tim,"_.csv", sep = "")
write.csv(submission,file_tmp,row.names = F)

b <- Sys.time();b; b - a

} #for (i in 1:nrow(
#--- End of loop
Bestxgb <- min(res_df$xgbAcc)
res_csv <- paste("--------- res_df_xgb_noNA_rat_prod_BestAcc_", Bestxgb ,"_",dat_tim,"_.csv", sep = "")
write_csv(res_df, res_csv)


# File Ensemble iterations
cat("Build Ensemble\n")
sub_ensemb <- data.frame(ID = test$ID, PredictedProb = ens_ble/con_ens)
sub_ensemb <- sub_ensemb[order(sub_ensemb$ID),]
dat_tim  <- str_replace_all(Sys.time()," |:","_")
file_ens <- paste("Res_xxxxx_XGB_noNA_Extended_rat_prod_3_GridOri_Ensemble_", con_ens, "_", dat_tim,"_.csv", sep = "")
write.csv(sub_ensemb,file_ens,row.names = F)



#---------------------- AUXILIAR INFO - FUNCTIONS
xgb.plot.tree( model = clf )
# save(clf, dval, dtrain, tra_ex, tes_ex, file = "Execution_clf_2016_04_07_00_01.RData")
load(file = "Execution_2016_05_19_01.RData")

#Importance by using a good model "clf" ~ 0.457...
xgb.dump(clf, 'xgb.model.dump', with.stats = TRUE)
jj <- xgb.importance( filename_dump = "xgb.model.dump")
# head(jj$Feature, 20) #-1st Iteration
# [1] "43" "55" "10" "34" "93" "30" "19" "46" "20" "8"  "12" "27" "40" "98" "92" "44" "91" "21" "89" "67"
# head(jj$Feature, 20) #-2nd Iteration -  Watch out!... are column numbers!!.
# [1] "43"  "55"  "10"  "111" "93"  "19"  "46"  "121" "20"  "34"  "30"  "131" "27"  "12"  "40"  "98"  "8"   "92"  "89"  "21" 
# col_imp <- as.numeric( head(jj$Feature, 20))
# hih_imp <- names(tra_ex[col_imp]); hih_imp
# [1] "v49"       "v65"       "v11"       "v55_r_v93" "v113"      "v20"       "v55"       "v55_p_v93" "v21"       "v39"       "v33"      
# [12] "v55_l_v93" "v30"       "v13"       "v45"       "v122"      "v9"        "v112"      "v106"      "v22"  
# col_imp <- as.numeric( head(jj$Feature, 20)) #-3rd Iteration
# > hih_imp <- names(tra_ex[col_imp]); hih_imp
# [1] "v49"        "v65"        "v11"        "v55"        "v39"        "v9"         "v113"       "v33"        "v30"        "v45"       
# [11] "v21"        "v106"       "v13"        "v113_p_v55" "v127"       "v122"       "v22"        "v20"        "v113_r_v55" "v21_r_v93" 



sub_ensemb_i <- data.frame(ID = test$ID, PredictedProb = ens_ble/i)
sub_ensemb_i <- sub_ensemb_i[order(sub_ensemb_i$ID),]
file_ens_i <- paste("Res_xxxxx_XGB_Extended_3_GridOri_Ensemble_", i, "_", dat_tim,"_.csv", sep = "")
write.csv(sub_ensemb,file_ens_i,row.names = F)

b <- Sys.time();b; b - a

#--------------------------------------------------------
#---------------------- END OF PROGRAM
#--------------------------------------------------------

#Select most important variables out of xgb model *numeric*
#The model used for this was the one that got best scoring.
var_imp <- xgb.importance( model = clf)
top_var <- names(tra)[as.numeric(var_imp$Feature[1:10])]
top_ch <- intersect(top_var, cha_col)
top_num <- setdiff(top_var, top_ch)



#---------------------- SUMMARY INFORMATION
inputs <- c(
  "seed" =  rn,
  "nrounds" = clf$bestInd,
  "eta" = param$eta,
  "max_depth" = param$max_depth,
  "subsample" = param$subsample,
  "colsample_bytree" = param$colsample_bytree,
  "min_child_weight" = param$min_child_weight,
  "num_parallel_tree" = param$num_parallel_tree,
  "bestScore" = LL
)

