
# Do cross-validation with xgboost - xgb.cv
docv <- function(param0, iter) {
    model_cv = xgb.cv(
            params = param0
            , nrounds = iter
            , nfold = 3
            , data = dtrain
            , early.stop.round = 50
            , maximize = FALSE
            , nthread = 8,
            , print.every.n = 100
            )
    gc()
    best <- min(model_cv$test.logloss.mean)
    bestIter <- which(model_cv$test.logloss.mean==best)

    cat("\n",best, bestIter,"\n")
    print(model_cv[bestIter])

    bestIter - 1
}

doTest <- function(param0, iter) {
    watchlist <- list('train' = dtrain)
    model = xgb.train(
            nrounds = iter
            , params = param0
            , data = dtrain
            , watchlist = watchlist
            , print.every.n = 50
            , nthread = 8
            )
    p <- predict(model, test)
    rm(model)
    gc()
    p
}

param0 <- list(
        # some generic, non specific params
        "objective"  = "binary:logistic"
        , "eval_metric" = "logloss"
        , "eta" = 0.01
        , "subsample" = 0.93
        , "colsample_bytree" = 0.45
        , "min_child_weight" = 1
        , "max_depth" = 11
        , "set.seed" = 80695
        )

cat("Training a XGBoost classifier with cross-validation\n")
set.seed(2018)
cv <- docv(param0, 1500)

# sample submission total analysis
submission <- read.csv("sample_submission.csv")
ensemble <- rep(0, nrow(test))

cv <- round(cv * 1.4)
cat("Calculated rounds:", cv, " Starting ensemble\n")

# Bagging of single xgboost for ensembling
# change to e.g. 1:10 to get quite good results
for (i in 1:10) {
    print(i)
    set.seed(i + 80695)
    p <- doTest(param0, cv)
    # use 40% to 50% more than the best iter rounds from your cross-fold number.
    # as you have another 50% training data now, which gives longer optimal training time
    ensemble <- ensemble + p
}

# Finalise prediction of the ensemble
cat("Making predictions\n")
submission$PredictedProb <- ensemble/i

# Prepare submission
write.csv(submission, "bnp-xgb-ensemble.csv", row.names = F, quote = F)
summary(submission$PredictedProb)
