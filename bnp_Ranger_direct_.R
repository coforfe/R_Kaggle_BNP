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

pro_cas <- as.vector( prop.table(table(tra_ex$target)) )

#---------------------------------
#---------------------- RANGER - directly
#---------------------------------
library(ranger)
library(MLmetrics)

train_idx <- sample(nrow(tra_ex), 2/3 * nrow(tra_ex))
tra_train <- tra_ex[ train_idx, ]
tra_test  <- tra_ex[-train_idx, ]

modFitrang <- ranger( 
                     target ~ .
                    ,data                 = tra_train
                    ,num.trees            = 1000
                    ,mtry                 = 9
                    ,importance           = 'impurity'
                    ,verbose              = TRUE
                    ,classification       = TRUE
                    ,num.threads          = 4
                    ,write.forest         = TRUE
  )

pred_tra <- predict( modFitrang, data = tra_test )
LogLoss( y_true = tra_test$target, y_pred = pred_tra$predictions)
Accuracy( y_true = tra_test$target, y_pred = pred_tra$predictions)

GenerateReport(
               train,
               output_file = "report.html",
               output_dir = getwd(),
               html_document(toc = TRUE, toc_depth = 6, theme = "flatly")
               )

kk <- as.matrix(all)
image(kk)
save(all, file = "allBNP_wNA.RData")
