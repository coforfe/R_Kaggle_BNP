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
library(Matrix)
library(mi)        # Missing imputations


cat("Read the train and test data\n")
train <- as.data.frame(read_csv("train.csv"))
test  <- as.data.frame(read_csv("test.csv"))
te_id <- test$ID

nams_ini <- names(train)

cat("Remove highly correlated features\n")
#re_move <- c('v8','v23','v25','v31','v36','v37',
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

rm(i, lev_tmp)

# cat("Missing Imputation\n")
# # library(mi)
# options(mc.cores = 2)
# mdf_tranew <- missing_data.frame(tra_new)
# imp_tranew <- mi(mdf_tranew, n.iter = 30, n.chains = 4, max.minutes = 20)
# 
# mdf_tesnew <- missing_data.frame(tes_new)
# imp_tesnew <- mi(mdf_tesnew, n.iter = 30, n.chains = 4, max.minutes = 20)
# 
# # It took around 20 hours to complete...
# # save(imp_tranew, imp_tesnew, file = "Removed_variables_imputedNAs_mi_.RData")
# round(mipply(imp_tranew, mean, to.matrix = TRUE), 3)
# 
# tranew_imp <- complete(imp_tranew, m = 1)
# tesnew_imp <- complete(imp_tesnew, m = 1)
# 
# # Data Imputed
# tranew_nona <- tranew_imp[, 1:ncol(tra_new)]
# tesnew_nona <- tesnew_imp[, 1:ncol(tes_new)]
# save(tranew_nona, tesnew_nona , file = "ImputedSets_ready_.RData")
load("ImputedSets_ready_.RData")

tra_new <- tranew_nona
tes_new <- tesnew_nona

train_y <- train[, 2]

#---------------------------------
#---------------------- XGBOOST
#---------------------------------
rn_v <- sample(1e4:1e5, size = 1); rn_v
set.seed(rn_v)

tra_new$target <- train_y

train <- sparse.model.matrix(target ~ ., data = tra_new)
dtrain <- xgb.DMatrix(data = train, label = train_y)
watchlist <- list(train = dtrain)

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

# # Original
# xgbGrid <- expand.grid(
#   et = 0.01,
#   md = 11,
#   ss = 0.96,
#   cs = 0.45,
#   mc = 1,
#   np = 1,
#   nr = 2501,
#   es = 150,
#   rn = 0
# )

# Exploratory
xgbGrid <- expand.grid(
  et = c(0.0100001, 0.099999),
  md = c(11,12),
  ss = 0.93,
  cs = 0.44999995, 
  mc = 1,
  np = 1,
  nr = 2501,
  es = 150,
  rn = 0
)


#rn <- c(21219, sample(21219:(21219*2) , 5))
res_df <- data.frame(
  xgbAcc = 0, xgbIdx = 0, et = 0, md = 0,
  ss = 0, cs = 0, mc = 0, np = 0,
  nr = 0, es = 0, rn = 0,
  ex_t = 0
)

rn_v <- sample(1e4:1e5, size = nrow(xgbGrid) ); rn_v
#rn_v <- 34992

for (i in 1:nrow(xgbGrid)) {
  print(i)
  
  
  param <- list(  objective           = "binary:logistic", 
                  booster             = "gbtree",
                  eval_metric         = "logloss",
                  eta                 = xgbGrid$et[i],
                  max_depth           = xgbGrid$md[i],
                  subsample           = xgbGrid$ss[i],
                  colsample_bytree    = xgbGrid$cs[i],
                  min_child_weight    = xgbGrid$mc[i],
                  num_paralallel_tree = xgbGrid$np[i],
                  set.seed            = rn_v[i]
  )
  
  ex_a <- Sys.time(); 
  
  n_folds <- 3
  cv_out <- xgb.cv(
    params           = param,   data        = dtrain,  nrounds    = 1500, 
    nfold            = n_folds, prediction  = TRUE,    stratified = TRUE, 
    early.stop.round = 50,      verbose     = TRUE,    maximize   = FALSE,
    print.every.n    = 50,      nthread     = 8 
  )
  
  model_perf <- min(cv_out$dt$test.logloss.mean); model_perf
  best_iter <- which.min(cv_out$dt$test.logloss.mean); best_iter
  
  ex_b <- Sys.time();  ex_t <- ex_b - ex_a; (ex_t)
  cat(paste('\n', ex_t,'\n'))
  
  #Store results
  res_df[i,1]  <- model_perf
  res_df[i,2]  <- best_iter
  res_df[i,3]  <- xgbGrid$et[i]
  res_df[i,4]  <- xgbGrid$md[i]
  res_df[i,5]  <- xgbGrid$ss[i]
  res_df[i,6]  <- xgbGrid$cs[i]
  res_df[i,7]  <- xgbGrid$mc[i]
  res_df[i,8]  <- xgbGrid$np[i]
  res_df[i,9]  <- xgbGrid$nr[i]
  res_df[i,10] <- xgbGrid$es[i]
  res_df[i,11] <- rn_v[i]
  res_df[i,12] <- ex_t
  
  cat("\n")
  print(res_df)
  cat("\n")
  
} #for (i in 1:...)...end of loop


# Now lets get the best result and get a prediction
#res_df <- read.csv("res_df_Best_0.460771_time_2016-03-25_20_17_51_.csv")
res_df <- res_df[ order(res_df$xgbAcc, decreasing = FALSE),]
best_v <- 1
in_err <- res_df$xgbAcc[best_v]; in_err

param_best <- list(
  objective           = "binary:logistic", 
  booster             = "gbtree",
  eval_metric         = "logloss",
  eta                 = res_df$et[best_v],
  max_depth           = res_df$md[best_v],
  subsample           = res_df$ss[best_v],
  colsample_bytree    = res_df$cs[best_v],
  min_child_weight    = res_df$mc[best_v],
  num_paralallel_tree = res_df$np[best_v],
  set.seed            = res_df$rn[best_v]
)

pl_us <- 0.40 #percentage increment in nrounds with respect to cv.
rounds_ext <- (res_df$xgbIdx[best_v] * (1 + pl_us) ); rounds_ext

clf <- xgb.train(   
  params              = param_best, 
  data                = dtrain, 
  nrounds             = rounds_ext, 
  verbose             = 1,
  watchlist           = watchlist,
  maximize            = FALSE,
  early.stop.round    = 50,
  print.every.n       = 100,
  nthread             = 4
)


#--------------------------------------------------------
#-------------- PREDICTION
#--------------------------------------------------------
tes_new$target <- -1
test <- sparse.model.matrix(target ~ ., data = tes_new)

#--------------------------------------------------------
#-------------- FILE UPLOAD
#--------------------------------------------------------
preds <- predict(
  clf, test 
  #ntreelimit = rounds_ext
)
submission <- data.frame(ID = te_id, PredictedProb = preds)
cat("saving the submission file\n")
library(stringr)
dat_tim  <- str_replace_all(Sys.time()," |:","_")
file_out <- paste("Res_XXXX_XGB_Extended_feyfesqr_cv_", n_folds, "_plus_", pl_us, "_Acc_", in_err,"_time_", dat_tim, "_.csv", sep = "" )
write.csv(submission, file = file_out, row.names = F)

file_res <- paste("res_df_Best_",in_err,"_time_", dat_tim, "_.csv", sep = "")
write.csv(res_df, file = file_res, row.names = F)

#*********************************************
#-------------- END OF PROGRAM
#*********************************************



