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


cat("Read the train and test data\n")
train <- as.data.frame(read_csv("train.csv"))
test  <- as.data.frame(read_csv("test.csv"))
te_id <- test$ID

# cat("Recode NAs to -999 or zero\n")
# train[is.na(train)] <- -1500
# test[is.na(test)]   <- -1500

nams_ini <- names(train)

# cat("Get high correlated columns\n")
# all <- rbind.data.frame( train[, 3:ncol(train)] , test[, 2:ncol(test)] )
# nam_all <- names(all)
# num_col <- grep("character" , as.data.frame(mapply(class, all))[,1] , invert = TRUE)
# cor_col <- cor(all[ , num_col])
# 
# fin_cor <- findCorrelation( cor_col, cutoff = 0.999, names = TRUE, exact = TRUE)
# # 97 columns with high correlations... too many...
# #fin_lin <- findLinearCombos( as.matrix(all))
# 
# fin_nzv <- nearZeroVar(all[, num_col], saveMetrics = TRUE, names = TRUE, allowParallel = TRUE, foreach = TRUE)
# nzv_red <- fin_nzv[ fin_nzv$percentUnique < 1 | fin_nzv$zeroVar == TRUE, ]
# # just 4 columns with almost no different values... v38, v62, v72, v129
# 
# all_bad <- c(fin_cor, rownames(nzv_red) )
# 
# cat("Remove just columns with no distint values\n")
# cat("XGB gets info even from correlated columns\n")
# # re_move <- row.names( nzv_red)
# # re_move <- c( 'v38', 'v62', 'v72', 'v129')
# # remove possible duplicates
# re_move <- unique(all_bad)

cat("Remove highly correlated features\n")
#re_move <- c('v8','v23','v25','v31','v36','v37',
# re_move <- c('v8','v23','v25','v31','v36','v37',
#              'v46','v51','v53','v54','v63','v73','v75','v79',
#              'v81','v82','v89','v92','v95','v105','v107','v108',
#              'v109','v110','v116','v117','v118','v119','v123',
#              'v124','v128')


re_move <- c("v8","v23","v25","v36","v37","v46",
             "v51","v53","v54","v63","v73","v81",
             "v82","v89","v92","v95","v105","v107",
             "v108","v109","v116","v117","v118",
             "v119","v123","v124","v128")

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


#---------- NEW COLUMNS ---------
col_now <- names(tra_new)
N <- length(col_now)
tra_new$fe_NAcountN <-  rowSums(is.na(tra_new)) / N 
tes_new$fe_NAcountN <-  rowSums(is.na(tes_new)) / N 
tra_new$fe_NAcount  <-  rowSums(is.na(tra_new))  
tes_new$fe_NAcount  <-  rowSums(is.na(tes_new))  
tra_new$fe_Zero     <-  rowSums(tra_new[, col_now] == 0) / N
tes_new$fe_Zero     <-  rowSums(tes_new[, col_now] == 0) / N
tra_new$fe_below0   <-  rowSums(tra_new[, col_now] < 0) / N
tes_new$fe_below0   <-  rowSums(tes_new[, col_now] < 0) / N


#----------- NEW COLUMNS BAD (TO SKIP) ---------
cat("Create artificial new columns with cut_number , 2 bins for each column\n")
num_bin <- 4
tra_exp <- tra_new
tes_exp <- tes_new
num_col <- ncol(tra_exp)
nam_new <- names(tra_exp)
for (i in 1:num_col ) {
 print(i)
 var_tmp <- try( as.numeric( cut_number( tra_exp[, i], n = num_bin) ), TRUE)
 var_cmp <- grep("Error", var_tmp[1])
  if (length(var_cmp) == 0) {
     var_tmp <- try( as.numeric( cut_number( tra_exp[, i], n = num_bin) ), TRUE)
     tra_exp[ , (num_col + i) ] <- var_tmp
     names(tra_exp)[ (num_col + i)] <- paste( "fe_",names(tra_exp)[i], sep = "")
     
     var_tmp <- try( as.numeric( cut_number( tes_exp[, i], n = num_bin) ), TRUE)
     tes_exp[ , (num_col + i) ] <- var_tmp
     names(tes_exp)[ (num_col + i)] <- paste( "fe_",names(tes_exp)[i], sep = "")
     
  } else {
     var_tmp <- try( as.numeric( cut_number( tra_exp[, i], n = 1) ), TRUE)
     tra_exp[ , (num_col + i) ] <- var_tmp
     names(tra_exp)[ (num_col + i)] <- paste( "Bad_",names(tra_exp)[i], sep = "")
     
     var_tmp <- try( as.numeric( cut_number( tes_exp[, i], n = 1) ), TRUE)
     tes_exp[ , (num_col + i) ] <- var_tmp
     names(tes_exp)[ (num_col + i)] <- paste( "Bad_",names(tes_exp)[i], sep = "")
     
  }
}



cat("Remove columns -Bad- unable to cut 2 bins\n")
nam_god <- grep("Bad", names(tra_exp) , invert = TRUE)
tra_new <- tra_exp[, nam_god]
tes_new <- tes_exp[, nam_god]

#cat("For the transformed columns also generate extra new columns with sqrt(abs) of them\n")
cat("For ALL columns except fe_ also generate extra new columns with sqrt(abs) of them\n")
tra_exp <- tra_new
tes_exp <- tes_new
num_col <- ncol(tra_exp)
nam_tra <- names(tra_exp)[grep("fe", names(tra_exp), invert = TRUE )]

for (i in 1:length(nam_tra)) {
  print(i)
  col_tra <- tra_exp[, nam_tra[i]]
  tra_exp[ , num_col + i] <- sqrt(abs(col_tra)) 
  names(tra_exp)[ num_col + i ] <- paste("fesq_",nam_tra[i], sep = "")
  
  col_tes <- tes_exp[, nam_tra[i]]
  tes_exp[ , num_col + i] <- sqrt(abs(col_tes)) 
  names(tes_exp)[ num_col + i  ] <- paste("fesq_",nam_tra[i], sep = "")
}

tra_new <- tra_exp
tes_new <- tes_exp

#----------- END NEW COLUMNS (TO SKIP) ---------


#---------------------------------
#---------------------- XGBOOST
#---------------------------------
rn_v <- sample(1e4:1e5, size = 1); rn_v
set.seed(rn_v)

train_y <- train[, 2]
tra_new$target <- train_y

# train <- sparse.model.matrix(target ~ ., data = tra_new)
# dtrain <- xgb.DMatrix(data = train, label = train_y, missing = NA)

# dtrain <- xgb.DMatrix(data = as.matrix(tra_new), label = train_y)
dtrain <- xgb.DMatrix(data = as.matrix(tra_new), label = train_y, missing = NA)
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
  et = c(0.0100001),        # eta
  md = 11,                  # max_depth
  ss = 0.93,                # subsample
  cs = 0.44999995,          # colsample_bytree
  mc = 1,                   # min_child_weight
  np = 1,                   # num_parallel_tree
  ms = 1,                   # max_delta_step
  nr = 2501,                # nrounds
  es = 150,                 # early.stop.round
  rn = 0                    # set.seed
)


#rn <- c(21219, sample(21219:(21219*2) , 5))
res_df <- data.frame(
  xgbAcc = 0, xgbIdx = 0, et = 0, md = 0,
  ss = 0, cs = 0, mc = 0, np = 0, ms = 0,
  nr = 0, es = 0, rn = 0,
  ex_t = 0
)

rn_v <- sample(1e4:1e5, size = nrow(xgbGrid) ); rn_v
#rn_v <- 34992

#---------------------- XGBOOST - CROSS VALIDATION
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
                  num_parallel_tree   = xgbGrid$np[i],
                  max_delta_step      = xgbGrid$ms[i],
                  set.seed            = rn_v[i]
  )
  
  ex_a <- Sys.time(); 
  
  n_folds <- 5
  nr_oun  <- 15
  pr_eve  <- round(nr_oun/10, 0) 
  cv_out <- xgb.cv(
    params           = param,   data        = dtrain,  nrounds    = nr_oun, 
    nfold            = n_folds, prediction  = TRUE,    stratified = TRUE, 
    early.stop.round = 3,       verbose     = TRUE,    maximize   = FALSE,
    print.every.n    = pr_eve,  nthread     = 8 
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
  res_df[i,9]  <- xgbGrid$ms[i]
  res_df[i,10] <- xgbGrid$nr[i]
  res_df[i,11] <- xgbGrid$es[i]
  res_df[i,12] <- rn_v[i]
  res_df[i,13] <- ex_t
  
  cat("\n")
  print(res_df)
  cat("\n")
  
} #for (i in 1:...)...end of loop


#---------------------- XGBOOST - ESTIMATION

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
  num_parallel_tree   = res_df$np[best_v],
  max_delta_step      = res_df$ms[best_v],
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
  print.every.n       = 15,
  nthread             = 6
)
sco_end <- clf$bestScore ; sco_end
# Variable Importance
var_imp <- xgb.importance(model = clf) ; head(var_imp)
imp_var <- names(tra_new[, as.numeric(var_imp$Feature[1:6]) ]); imp_var

#--------------------------------------------------------
#-------------- PREDICTION
#--------------------------------------------------------
# tes_new$target <- -1
# test <- sparse.model.matrix(target ~ ., data = tes_new)
xgtest = xgb.DMatrix(as.matrix(tes_new), missing = NA)

#--------------------------------------------------------
#-------------- FILE UPLOAD
#--------------------------------------------------------
preds <- predict(
                 clf, xgtest, missing = NA
                 # clf, test 
                 #ntreelimit = rounds_ext
                )
submission <- data.frame(ID = te_id, PredictedProb = preds)
cat("saving the submission file\n")
library(stringr)
dat_tim  <- str_replace_all(Sys.time()," |:","_")
n_col <- ncol(tra_new)
file_out <- paste("Res_XXXX_XGB_noExtended_ncol_",n_col,"_rounds_", nr_oun, "_cv_", n_folds, "_plus_", pl_us, "_Acc_", sco_end,"_time_", dat_tim, "_.csv", sep = "" )
write.csv(submission, file = file_out, row.names = F)

file_res <- paste("res_df_Best_",in_err,"_time_", dat_tim, "_.csv", sep = "")
write.csv(res_df, file = file_res, row.names = F)

#*********************************************
#-------------- END OF PROGRAM
#*********************************************



# #--------------------------------------------------------
# #-------------- FILE UPLOAD
# #--------------------------------------------------------
# 
# n.folds <- 3
# cv.out <- xgb.cv(
#   params           = param,   data       = dtrain_n, nrounds    = 1500, 
#   nfold            = n.folds, prediction = TRUE,     stratified = TRUE, 
#   early.stop.round = 15,      verbose    = FALSE,    maximize   = TRUE,
#   print.every.n    = 100 
# )
# 
# model.perf <- max(cv.out$dt$test.auc.mean); model.perf
# best.iter <- which.max(cv.out$dt$test.auc.mean); best.iter
# meta.tr <- cv.out$pred; meta.tr
# 
