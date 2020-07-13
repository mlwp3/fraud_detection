library(tidyverse)
library(caret)
library(dataPreparation)
library(ROCR)
library(xgboost)
library(randomForest)

rm(list = ls())

setwd("/home/marco/Documents/gitrepos/fraud_detection/MDV")

train_data <- read_csv("../data/train_final.csv") %>%
                mutate_if(is.character, as.factor) %>% 
                mutate(FraudFound = as.factor(FraudFound)) %>% 
                rename(y = FraudFound)

test_data <- read_csv("../data/test_final.csv") %>%
                mutate_if(is.character, as.factor) %>%
                mutate(FraudFound = as.factor(FraudFound)) %>% 
                rename(y = FraudFound)


map(train_data, ~sum(is.na(.)))
map(test_data, ~sum(is.na(.)))


# scale
mean_sd_scale <- build_scales(train_data, verbose = FALSE)

train_data <- fastScale(train_data, scales = mean_sd_scale, verbose = FALSE)

test_data <- fastScale(test_data, scales = mean_sd_scale, verbose = FALSE)

# xgboost

table(train_data$y)

table(test_data$y)


train_weights <- ifelse(train_data$y == 0, sum(train_data$y == 0)/nrow(train_data), 
                        sum(train_data$y == 1)/nrow(train_data))

train_data$y <- as.numeric(train_data$y)

test_data$y <- as.numeric(test_data$y)

train_data_enc_rules <- dummyVars(y ~ ., data = train_data)

train_data_encoded <- predict(train_data_enc_rules, newdata = train_data)

test_data_enc_rules <- dummyVars(y ~ ., data = test_data)

test_data_encoded <- predict(test_data_enc_rules, newdata = test_data)

train_xgboost <- xgb.DMatrix(data = train_data_encoded, label = train_data$y - 1)

test_xgboost <- xgb.DMatrix(data = test_data_encoded, label = test_data$y - 1)

xgb_params <-
  list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    tree_method = "hist",
    eta = 0.1,
    subsample = .3,
    scale_pos_weight =  sum(train_data$y == 1) / sum(train_data$y == 2)
  )


xgb_cv <-
  xgb.cv(
    params = xgb_params,
    data = train_xgboost,
    nrounds = 500,
    nfold = 5,
    showsd = T,
    stratified = T,
    print_every_n = 10,
    early_stop_rounds = 20,
    maximize = TRUE
  )

which.max(xgb_cv$evaluation_log$test_auc_mean)


xgb_model <-
  xgb.train (
    params = xgb_params,
    data = train_xgboost,
    nrounds = which.max(xgb_cv$evaluation_log$test_auc_mean),
    print_every_n = 10)


xgb_pred <- predict(xgb_model, test_xgboost)

xgb_pred_labels <- ifelse(xgb_pred > 0.5, 1, 0)

(xgb_acc <- confusionMatrix(as.factor(xgb_pred_labels), as.factor(test_data$y - 1))$overall["Accuracy"])

xgb_roc_data <- prediction(xgb_pred, test_data$y - 1)

(xgb_auc <- performance(xgb_roc_data, measure = "auc")@y.values[[1]])


# randomforest

rf_cv <- rfcv(select(train_data, -y), train_data$y, 
              cv.fold=10, 
              classwt = c("0" = sum(train_data$y == 1)/length(train_data$y), "1" = sum(train_data$y == 0)/length(train_data$y)))

rf_model <- randomForest(y ~ ., 
                   data = train_data, 
                   classwt = c("0" = sum(train_data$y == 1)/length(train_data$y), "1" = sum(train_data$y == 0)/length(train_data$y)),
                   mtry = rf_cv$n.var[which.min(rf_cv$error.cv)])

rf_pred = predict(rf_model, newdata = test_data)

(rf_acc <- confusionMatrix(as.factor(rf_pred), as.factor(test_data$y))$overall["Accuracy"])

rf_roc_data <- prediction(as.numeric(rf_pred), test_data$y)

(rf_auc <- performance(rf_roc_data, measure = "auc")@y.values[[1]])
