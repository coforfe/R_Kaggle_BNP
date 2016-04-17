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

cat("Recode NAs to -999\n")
train[is.na(train)] <- -999
test[is.na(test)]   <- -999

nams_ini <- names(train)

cat("Remove highly correlated features\n")
#re_move <- c('v8','v23','v25','v31','v36','v37',
re_move <- c('v8','v25','v31','v36','v37',
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

train_y <- train[, 2]

tra_new$target <- as.factor(train_y)


#---------------------------------
#---------------------- DEEPBOOST
#---------------------------------

best_params <- deepboost.gridSearch( target ~ . , tra_new, k = 3 )

boost <- deepboost(
                   target ~ . , tra_new,
                   num_iter = best_params[2][[1]], 
                   beta = best_params[3][[1]], 
                   lambda = best_params[4][[1]], 
                   loss_type = best_params[5][[1]]
)

print(boost)

labels <- predict( boost, tes_new )
