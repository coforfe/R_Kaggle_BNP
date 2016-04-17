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
h2o.init(nthreads = -1,max_mem_size = '8G')

### load both files in using H2O's parallel import
#train<-h2o.uploadFile("train.csv",destination_frame = "train.hex")
#test<-h2o.uploadFile("test.csv",destination_frame = "test.hex")
train_h <- as.h2o(tra_new, destination_frame = 'train.hex')
test_h  <- as.h2o(tes_new ,destination_frame = "test.hex")

train_h$target <- as.factor(train_h$target)

splits <- h2o.splitFrame(train_h, 0.9, destination_frames = c("trainSplit","validSplit"),seed = 111111111)

#-----------------------------------------
#---------------------- DEEP LEARNING
#----------------------- (scenarios)
#-----------------------------------------
a <- Sys.time();a

#-----------------------------------------
#---------- First Iteration
res_dl <- h2o.deeplearning(
  model_id = "dl_model_faster", 
  training_frame   = splits[[1]],
  validation_frame = splits[[2]],
  x = 2:ncol(train_h),
  y = 1,
  hidden                     = c(25,25,25),     ## small network, runs faster
  activation                 = 'Rectifier',
  epochs                     = 1000000,         ## hopefully converges earlier...
  score_validation_samples   = 10000,           ## sample the validation dataset (faster)
  stopping_rounds            = 50,
  stopping_metric            = "logloss",
  stopping_tolerance         = 0.001,
  adaptive_rate              = TRUE,
  seed                       = 8.541816e+18,
  rate                       = 0.02,
  rate_annealing             = 1e-6,
  input_dropout_ratio        = 0.1,
  max_w2                     = 100,
  l1                         = 1e-5,
  l2                         = 1e-5,
  momentum_start             = 0.1,     ## Tuned with hyper_params
  momentum_stable            = 0.1,     ## Tuned wtih hyper_params     
  momentum_ramp              = 1e9,     ## Tuned with hyper_params
  rho                        = 0.999,   ## Tuned with hyper_params
  epsilon                    = 1e-10    ## Tuned with hyper_params
)

summary(res_dl)
plot(res_dl, las = 1, cex.axis = 0.7, col.axis = "blue", font.axis = 2)

h2o.logloss(res_dl)
h2o.logloss(h2o.performance(res_dl, train = TRUE))
h2o.logloss(h2o.performance(res_dl, valid = TRUE))

h2o.logloss(h2o.performance(res_dl, newdata = train_h))     ## full training data
h2o.logloss(h2o.performance(res_dl, newdata = test_h))         ## full test data
Accdl <- h2o.logloss(h2o.performance(res_dl, valid = TRUE))

b <- Sys.time();b; b - a



#-----------------------------------------
#---------- Second Iteration
res_dl <- h2o.deeplearning(
  model_id = "dl_model_faster",
training_frame = splits[[1]],
validation_frame = splits[[2]],
x = 3:ncol(train_h),
y = 2,
  overwrite_with_best_model = F,    ## Return the final model after 10 epochs, even if not the best
  hidden = c(32,32,32),             ## more hidden layers -> more complex interactions
  epochs = 100000,                  ## to keep it short enough
  score_validation_samples = 10000, ## downsample validation set for faster scoring
  score_duty_cycle = 0.025,         ## don't score more than 2.5% of the wall time
  adaptive_rate = F,                ## manually tuned learning rate
  rate = 0.01,
  rate_annealing = 2e-6,
  momentum_start = 0.2,             ## manually tuned momentum
  momentum_stable = 0.4,
  momentum_ramp = 1e7,
  l1 = 1e-5,                        ## add some L1/L2 regularization
  l2 = 1e-5,
  max_w2 = 10,
  stopping_rounds = 2,
  stopping_tolerance = 0.01,
  stopping_metric = "logloss"
)
summary(res_dl)
plot(res_dl)


#-----------------------------------------
#------- With hyper_params
#-----------------------------------------
hyper_params <- list(
  hidden = list(c(32,32,32),c(64,64)),
  input_dropout_ratio = c(0,0.05),
  rate = c(0.01,0.02),
  rate_annealing = c(1e-8,1e-7,1e-6)
)

hyper_params

grid <- h2o.grid(
  "deeplearning",
  model_id = "dl_grid",
  training_frame = splits[[1]],
  validation_frame = splits[[2]],
  x = 3:ncol(train_h),
  y = 2,
  epochs = 10000,
  stopping_metric = "logloss",
  stopping_tolerance = 1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  stopping_rounds = 2,
  score_validation_samples = 10000, ## downsample validation set for faster scoring
  score_duty_cycle = 0.025,         ## don't score more than 2.5% of the wall time
  adaptive_rate = F,                ## manually tuned learning rate
  momentum_start = 0.5,             ## manually tuned momentum
  momentum_stable = 0.9,
  momentum_ramp = 1e7,
  l1 = 1e-5,
  l2 = 1e-5,
  activation = c("Rectifier"),
  max_w2 = 10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params
)
grid
#
#scores <- cbind(as.data.frame(unlist((lapply(grid@model_ids, function(x) { h2o.confusionMatrix(h2o.performance(h2o.getModel(x),valid=T))$Error[3] })) )), unlist(grid@model_ids))
scores <- cbind(as.data.frame(unlist((lapply(grid@model_ids, function(x) { h2o.logloss(h2o.performance(h2o.getModel(x), valid = T)) })) )), unlist(grid@model_ids))
names(scores) <- c("logloss","model")
sorted_scores <- scores[order(scores$logloss),]
head(sorted_scores)
best_model <- h2o.getModel(as.character(sorted_scores$model[1]))
print(best_model@allparameters)
best_err <- sorted_scores$logloss[1]
print(best_err)


#-----------------------------------------
#---------- Fourth Iterarion
#------- Random Selection of parameters
models <- c()
for (i in 1:10) {
  rand_activation <- c("Rectifier", "RectifierWithDropout")[sample(1:2,1)]
  rand_numlayers <- sample(2:5,1)
  rand_hidden <- c(sample(10:50,rand_numlayers,T))
  rand_l1 <- runif(1, 0, 1e-3)
  rand_l2 <- runif(1, 0, 1e-3)
  rand_dropout <- c(runif(rand_numlayers, 0, 0.6))
  rand_input_dropout <- runif(1, 0, 0.5)
  dlmodel <- h2o.deeplearning(
    model_id = paste0("dl_random_model_", i),
      training_frame = splits[[1]],
      validation_frame = splits[[2]],
      x = 3:ncol(train_h),
      y = 2,
      epochs = 1000,                    ## for real parameters: set high enough to get to convergence
    stopping_metric = "logloss",
    stopping_tolerance = 1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
    stopping_rounds = 2,
    score_validation_samples = 10000, ## downsample validation set for faster scoring
    score_duty_cycle = 0.025,         ## don't score more than 2.5% of the wall time
    max_w2 = 10,                      ## can help improve stability for Rectifier
    
    ### Random parameters
    activation = rand_activation, 
    hidden = rand_hidden, 
    l1 = rand_l1, 
    l2 = rand_l2,
    input_dropout_ratio = rand_input_dropout, 
    hidden_dropout_ratios = rand_dropout
  )                                
  models <- c(models, dlmodel)
}
  
best_err <- 1      ##start with the best reference model from the grid search above, if available
for (i in 1:length(models)) {
  #err <- h2o.confusionMatrix(h2o.performance(models[[i]],valid = T))$Error[3]
  err <- h2o.logloss(models[[i]],valid = T)
  print(c(i,err))
  if (err < best_err) {
    best_err <- err
    best_model <- models[[i]]
  }
}
#h2o.confusionMatrix(best_model,valid = T)
h2o.logloss(best_model, valid = T)
best_params <- best_model@allparameters
best_params$hidden
best_params$l1
best_params$l2
best_params$input_dropout_ratio

# Accdl <- h2o.logloss(res_dl)
# print(Accdl)


#-----------------------------------------
#---------- With hyper_params
#-----------------------------------------
a <- Sys.time();a

hyper_params <- list(
  momentum_start = c(0.1, 0.8),             ## manually tuned momentum
  momentum_stable = c(0.1, 0.9),
  momentum_ramp = c(1e3, 1e9),
  rho =  c(0.9,0.95,0.99,0.999), 
  epsilon = c(1e-10,1e-8,1e-6,1e-4)
)
hyper_params

grid_dl <- h2o.grid(
  'deeplearning',
  model_id = "dl_grid", 
  training_frame = splits[[1]],
  validation_frame = splits[[2]],
  x = 3:ncol(train_h),
  y = 2,
  hidden = c(25,25,25),                ## small network, runs faster
  activation = 'Rectifier',
  epochs = 1000000,                      ## hopefully converges earlier...
  score_validation_samples = 10000,      ## sample the validation dataset (faster)
  stopping_rounds = 50,
  stopping_metric = "logloss",
  stopping_tolerance = 0.001,
  adaptive_rate = TRUE,
  seed = 8.541816e+18,
  rate = 0.02,
  rate_annealing = 1e-6,
  input_dropout_ratio = 0.1,
  max_w2 = 100,
  l1 = 1e-5,
  l2 = 1e-5, 
  nesterov_accelerated_gradient = TRUE,
  hyper_params = hyper_params
)
grid_dl

scores <- cbind(as.data.frame(unlist((lapply(grid_dl@model_ids, function(x) { h2o.logloss(h2o.performance(h2o.getModel(x), valid = TRUE)) })) )), unlist(grid_dl@model_ids))
names(scores) <- c("logloss","model")
sorted_scores <- scores[order(scores$logloss),]
head(sorted_scores)
best_model <- h2o.getModel(as.character(sorted_scores$model[1]))
print(best_model@allparameters)
best_err <- sorted_scores$logloss[1]
print(best_err)

#Params
best_params <- best_model@allparameters
best_params$momentum_start 
best_params$momentum_stable 
best_params$momentum_ramp 
best_params$rho 
best_params$epsilon 
#Others
best_params$hidden
best_params$activation
best_params$max_w2
best_params$input_dropout_ratio
best_params$rate
best_params$rate_annealing
best_params$l1
best_params$l2

b <- Sys.time();b; b - a

library(stringr)
dat_grd  <- str_replace_all(Sys.time()," |:","_")
file_grd <- paste("grid_dl_Extended_5_H2O_dl_", round(best_err,5),"_",dat_grd ,"_.RData", sep = "")
save(grid_dl, file = file_grd)




#--------------------------------------------------------
#-------------- PREDICTION
#--------------------------------------------------------
### get predictions against the test set and create submission file
p <- as.data.frame(h2o.predict(res_dl,test_h))
submission <- data.frame(test$ID,p$p1)
colnames(submission) <- c("ID","PredictedProb")

#--------------------------------------------------------
#-------------- FILE UPLOAD
#--------------------------------------------------------
dat_tim  <- str_replace_all(Sys.time()," |:","_")
file_tmp <- paste("Res_xxxxx_Extended_3_H2O_dl_", round(Accdl,5),"_",dat_tim ,"_.csv", sep = "")
write.csv(submission,file_tmp,row.names = F)

#--------------------------------------------------------
#-------------- CLOSE H20
#--------------------------------------------------------
### All done, shutdown H2O    
h2o.shutdown(prompt = FALSE)

