library(dummies)
library(ROSE)
library(caret)
library(MASS)
library(tweedie)
library(statmod)
require(xgboost)
library(dplyr)
library(plotly)
library(keras)
library(iml)
library(foreach)
library(doParallel)
library(data.table)
library(ranger)
library(vip)
library(h2o)
library(randomForest)
library(fastDummies)
library(ROCR)
library(MLmetrics)
library(e1071)
library(C50)
library(rpart)
library(purrr)
library(tibble)
library(rpart.plot)

library(readr)
dat <- read_csv("~/Desktop/GIRO 2019/Fraud Detection/Insurance/insurance_fraud_train.csv")
test <- read_csv("~/Desktop/GIRO 2019/Fraud Detection/Insurance/insurance_fraud_test.csv")

dat <- dat[!dat$DaysPolicyClaim == "none", ]
test <- test[!test$DaysPolicyClaim == "none", ]

dat <- dat[!dat$NumberOfCars == "more than 8", ]
test <- test[!test$NumberOfCars == "more than 8", ]

matrix(colnames(dat), ncol = 1)
matrix(colnames(test), ncol = 1)

#Create Vehicle Origin variable
veh_origin_key <- read_csv("~/Desktop/GIRO 2019/Fraud Detection/Insurance/VehOriginKey.csv")
dat <- merge(dat, veh_origin_key, by = "Make")
test <- merge(test, veh_origin_key, by = "Make")
rm(veh_origin_key)

dat <- dat[, c(10, 11, 12, 13, 15, 16, 17, 20, 22, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35)]
test <- test[, c(10, 11, 12, 13, 15, 16, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34)]

dat$Deductible <- as.character(dat$Deductible)
test$Deductible <- as.character(test$Deductible)

dat$IncidentYear <- as.character(dat$IncidentYear)
test$IncidentYear <- as.character(test$IncidentYear)

fraud_count <- as.data.frame(table(dat$FraudFound))
colnames(fraud_count) <- c("FraudFound", "Count")
plot_fraud_count <- plot_ly(fraud_count, x = ~FraudFound, y = ~Count, type = "bar", name = "Bar Chart of Fraud Counts")
plot_fraud_count

grp_by_police_report <- dat %>%
  group_by(PoliceReportFiled, FraudFound) %>%
  summarise(Count = n())
grp_by_police_report <- tidyr::spread(grp_by_police_report, key = "PoliceReportFiled", value = "Count")
plot_police_report <- plot_ly(data = grp_by_police_report, x = ~FraudFound, y = ~No, name = "No", type = "bar") %>%
  add_trace(y = ~Yes, name = "Yes") %>%
  layout(yaxis = list(title = "Count"), barmode = "group")
plot_police_report

grp_by_fault <- dat %>%
  group_by(Fault, FraudFound) %>%
  summarise(Count = n())
grp_by_fault <- tidyr::spread(grp_by_fault, key = "Fault", value = "Count")
plot_fault <- plot_ly(data = grp_by_fault, x = ~FraudFound, y = ~`Policy Holder`, name = "Fault: Policy Holder", type = "bar") %>%
  add_trace(y = ~`Third Party`, name = "Fault: Third Party") %>%
  layout(yaxis = list(title = "Count"), barmode = "group")
plot_fault

grp_by_veh_price <- dat %>%
  group_by(VehiclePrice, FraudFound) %>%
  summarise(Count = n())
grp_by_veh_price <- tidyr::spread(grp_by_veh_price, key = "VehiclePrice", value = "Count")
plot_veh_price <- plot_ly(data = grp_by_veh_price, x = ~FraudFound, y = ~`less than 20000`, name = "Vehicle Price: < 20k", type = "bar") %>%
  add_trace(y = ~`20000 to 29000`, name = "Vehicle Price: 20k-29k") %>%
  add_trace(y = ~`30000 to 39000`, name = "Vehicle Price: 30k-39k") %>%
  add_trace(y = ~`40000 to 59000`, name = "Vehicle Price: 40k-59k") %>%
  add_trace(y = ~`60000 to 69000`, name = "Vehicle Price: 60k-69k") %>%
  add_trace(y = ~`more than 69000`, name = "Vehicle Price: >69k") %>%
  layout(yaxis = list(title = "Count"), barmode = "group")
plot_veh_price

grp_by_veh_cat <- dat %>%
  group_by(VehicleCategory, FraudFound) %>%
  summarise(Count = n())
grp_by_veh_cat <- tidyr::spread(grp_by_veh_cat, key = "VehicleCategory", value = "Count")
plot_veh_cat <- plot_ly(data = grp_by_veh_cat, x = ~FraudFound, y = ~Sedan, name = "Vehicle Category: Sedan", type = "bar") %>%
  add_trace(y = ~Sport, name = "Vehicle Category: Sport") %>%
  add_trace(y = ~Utility, name = "Vehicle Category: Utility") %>%
  layout(yaxis = list(title = "Count"), barmode = "group")
plot_veh_cat

#Convert all Characters to Factors
dat <- dat %>%
  mutate_if(sapply(dat, is.character), as.factor)

test <- test %>%
  mutate_if(sapply(test, is.character), as.factor)

#Define Predictors and Response
targetvar <- c("FraudFound")
predictors <- colnames(dat)[!colnames(dat) %in% c(targetvar)]

#Split Training Data 
set.seed(1248)
ind <- sample(1:nrow(dat), round(0.25 * nrow(dat)))
train <- dat[-ind, ]
val <- dat[ind, ]
rm(ind)

#Data Processing
factors <- c()
for (i in 1:length(predictors)) {
  if (class(unlist(dat[, predictors[i]])) == "factor") {
    factors[i] <- predictors[i]
  } else {
    i <- i + 1
  }
}

factors <- factors[!factors %in% NA]
rm(i)

#Resample Training Data
table(train$FraudFound) #Without Resampling
t <- c("FraudFound")
pp <- append(predictors, targetvar)
pp <- paste(pp, collapse = "+")
f <- as.formula(paste(t,"~",pp,collapse = "+"))
train_bal <- ovun.sample(formula = f, data = train, method = "both", p = 0.4)$data
#train_bal <- train
table(train_bal$FraudFound) #After Resampling - Same as Without
rm(pp)
rm(f)
#train_bal <- train_bal[, -which(colnames(train_bal) == t)]

#One-Hot Encoding - dummy_cols() from fastDummies might be a more efficient way
dat_nn <- dummy_cols(dat, select_columns = factors)
train_nn <- dummy_cols(train_bal, select_columns = factors)
val_nn <- dummy_cols(val, select_columns = factors)
rm(t)

allvars <- colnames(dat_nn)
predictors_nn <- allvars[!allvars %in% c(targetvar, factors)]
rm(allvars)
rm(factors)

dat_nn <- dat_nn[, c(predictors_nn, targetvar)]
train_nn <- train_nn[, c(predictors_nn, targetvar)]
val_nn <- val_nn[, c(predictors_nn, targetvar)]

x_train <- as.matrix(train_nn[predictors_nn]) 
x_train <- x_train %>%
  scale()
y_train <- to_categorical(as.character(train_nn[, targetvar]))

x_val <- as.matrix(val_nn[predictors_nn])
x_val <- x_val %>%
  scale()
y_val <- to_categorical(as.character(val_nn[, targetvar]))

#SET UP H2O ENVIRONMENT
h2o.init(min_mem_size = "5g")
train.h2o <- as.h2o(train_bal)
val.h2o <- as.h2o(val)
y <- targetvar
x <- setdiff(names(train.h2o), y)
train.h2o[, y] <- as.factor(train.h2o[, y])
val.h2o[, y] <- as.factor(val.h2o[, y])

#GBM - HYPERPARAMETERS TUNED USING H2O, MODEL FIT USING XGBOOST
##Hyperparameter Grid Search
gbm_param_grid <- list(learn_rate = c(0.1), ntrees = c(2000, 5000, 8000, 10000), 
                       max_depth = c(10, 25, 50, 60), sample_rate = c(0.6, 0.8, 1), 
                       col_sample_rate = c(0.6, 0.8, 1))
gbm_grid <- h2o.grid("gbm", x = x, y = y, grid_id = "gbm_grid", training_frame = train.h2o, 
                     validation_frame = val.h2o, seed = 13567, hyper_params = gbm_param_grid, categorical_encoding = "AUTO", 
                     distribution = "bernoulli", balance_classes = T)
gbm_gridperf <- h2o.getGrid("gbm_grid", sort_by = "f1", decreasing = TRUE)
print(gbm_gridperf)
best_gbm <- h2o.getModel(gbm_gridperf@model_ids[[1]])
print(best_gbm@model[["model_summary"]])
best_gbm_perf <- h2o.performance(best_gbm, val.h2o)
best_thresh <- h2o.find_threshold_by_max_metric(best_gbm_perf, "f1")
gbm_accuracy <- as.numeric(unlist(h2o.accuracy(best_gbm_perf, best_thresh)))
gbm_accuracy
gbm_auc <- as.numeric(unlist(h2o.auc(best_gbm_perf, best_thresh)))
gbm_auc
gbm_f1 <- as.numeric(unlist(h2o.F1(best_gbm_perf, best_thresh)))
gbm_f1
gbm_precision <- as.numeric(unlist(h2o.precision(best_gbm_perf, best_thresh)))
gbm_precision
gbm_recall <- as.numeric(unlist(h2o.recall(best_gbm_perf, best_thresh)))
gbm_recall
h2o.confusionMatrix(best_gbm_perf, thresholds = best_thresh)
h2o.varimp_plot(best_gbm, num_of_features = 50)

#RANDOM FOREST - TUNING AND TRAINING IN H2O
rf_param_grid <- list(ntrees = c(400, 800, 1000), max_depth = c(10, 25, 40), 
                      sample_rate = c(0.6, 0.8, 1))
rf_grid <- h2o.grid("randomForest", x = x, y = y, grid_id = "rf_grid", training_frame = train.h2o, 
                    validation_frame = val.h2o, seed = 13567, binomial_double_trees = TRUE, 
                    hyper_params = rf_param_grid, categorical_encoding = "AUTO", distribution = "bernoulli", 
                    balance_classes = T)
rf_gridperf <- h2o.getGrid("rf_grid", sort_by = "f1", decreasing = TRUE)
print(rf_gridperf)
best_rf <- h2o.getModel(rf_gridperf@model_ids[[1]])
print(best_rf@model[["model_summary"]])
best_rf_perf <- h2o.performance(best_rf, val.h2o)
best_thresh <- h2o.find_threshold_by_max_metric(best_rf_perf, "f1")
rf_accuracy <- as.numeric(unlist(h2o.accuracy(best_rf_perf, best_thresh)))
rf_accuracy
rf_auc <- as.numeric(unlist(h2o.auc(best_rf_perf, best_thresh)))
rf_auc
rf_f1 <- as.numeric(unlist(h2o.F1(best_rf_perf, best_thresh)))
rf_f1
rf_precision <- as.numeric(h2o.precision(best_rf_perf, best_thresh))
rf_precision
rf_recall <- as.numeric(h2o.recall(best_rf_perf, best_thresh))
rf_recall
h2o.confusionMatrix(best_rf_perf, thresholds = best_thresh)
h2o.varimp_plot(best_rf, num_of_features = 50)

#C5.0
c5 <- C5.0(x = train_bal[, predictors], y = as.factor(train_bal$FraudFound))
summary(c5)
#plot(c5)
c5_preds <- predict(c5, val)
c5_probs <- predict(c5, val, type = "prob")
cm_c5 <- confusionMatrix(c5_preds, as.factor(val$FraudFound), mode = "everything", positive = "1")
c5_accuracy <- as.numeric(cm_c5$overall["Accuracy"])
c5_f1 <- as.numeric(cm_c5$byClass["F1"])
c5_precision <- as.numeric(cm_c5$byClass["Precision"])
c5_recall <- as.numeric(cm_c5$byClass["Recall"])
c5_auc <- AUC(c5_preds, as.factor(val$FraudFound))

#CART
predictors <- paste0(predictors, collapse = "+")
formula <- as.formula(paste(targetvar,"~",predictors,collapse = "+"))
base_tree <- rpart(formula = formula, data = train_bal, method = "class")
plotcp(base_tree)
printcp(base_tree)
base_tree_preds <- predict(base_tree, val, type = "class")
confusionMatrix(base_tree_preds, as.factor(val$FraudFound), mode = "everything", positive = "1")
pruned_tree <- prune(base_tree, cp = base_tree$cptable[which.min(base_tree$cptable[,"xerror"]),"CP"])
pruned_tree_preds <- predict(pruned_tree, val, type = "class")
cm_cart <- confusionMatrix(pruned_tree_preds, as.factor(val$FraudFound), mode = "everything", positive = "1")
cart_accuracy <- as.numeric(cm_cart$overall["Accuracy"])
cart_f1 <- as.numeric(cm_cart$byClass["F1"])
cart_precision <- as.numeric(cm_cart$byClass["Precision"])
cart_recall <- as.numeric(cm_cart$byClass["Recall"])
cart_auc <- AUC(pruned_tree_preds, as.factor(val$FraudFound))

#rpart.plot(pruned_tree, type = 1)

#NEURAL NETWORKS - TRAINED IN H2O
neuralnet_params <- list(hidden = list(c(10, 6), c(50), c(50, 30)), 
                         input_dropout_ratio = c(0.1, 0.2), activation = c("Tanh", "Rectifier"))
neuralnet_grid <- h2o.grid("deeplearning", grid_id = "neuralnet_grid", training_frame = train.h2o, validation_frame = val.h2o, 
                           seed = 13567, x = x, y = y, epochs = 20000, standardize = T, use_all_factor_levels = T, 
                           balance_classes = T, max_after_balance_size = 0.4, distribution = "bernoulli", 
                           stopping_rounds = 1000, stopping_tolerance = 0.0001, variable_importances = T, 
                           adaptive_rate = T, hyper_params = neuralnet_params)
neuralnet_gridperf <- h2o.getGrid("neuralnet_grid", sort_by = "f1", decreasing = TRUE)
print(neuralnet_gridperf)
best_neuralnet <- h2o.getModel(neuralnet_gridperf@model_ids[[1]])
print(best_neuralnet@model[["model_summary"]])
best_neuralnet_perf <- h2o.performance(best_neuralnet, val.h2o)
best_thresh <- h2o.find_threshold_by_max_metric(best_neuralnet_perf, "f1")
nn_accuracy <- as.numeric(unlist(h2o.accuracy(best_neuralnet_perf, best_thresh)))
nn_accuracy
nn_auc <- as.numeric(unlist(h2o.auc(best_neuralnet_perf, best_thresh)))
nn_auc
nn_f1 <- as.numeric(unlist(h2o.F1(best_neuralnet_perf, best_thresh)))
nn_f1
nn_precision <- as.numeric(h2o.precision(best_neuralnet_perf, best_thresh))
nn_precision
nn_recall <- as.numeric(h2o.recall(best_neuralnet_perf, best_thresh))
nn_recall
h2o.confusionMatrix(best_neuralnet_perf, thresholds = best_thresh)

#Logistic Regression
glm <- glm(formula, data = train_bal, family = "binomial")
summary(glm)
preds_glm <- predict(glm, val, type = "response")
preds_glm <- ifelse(preds_glm > 0.5, "1", "0")
cm_glm <- confusionMatrix(as.factor(preds_glm), as.factor(val$FraudFound), mode = "everything", positive = "1")
glm_accuracy <- as.numeric(cm_glm$overall["Accuracy"])
glm_f1 <- as.numeric(cm_glm$byClass["F1"])
glm_precision <- as.numeric(cm_glm$byClass["Precision"])
glm_recall <- as.numeric(cm_glm$byClass["Recall"])
glm_auc <- AUC(preds_glm, as.factor(val$FraudFound))

#Plot ROC Curves for H2O Models
list(best_gbm,best_rf,best_neuralnet) %>% 
  # map a function to each element in the list
  map(function(x) x %>% h2o.performance(valid=T) %>% 
        # from all these 'paths' in the object
        .@metrics %>% .$thresholds_and_metric_scores %>% 
        # extracting true positive rate and false positive rate
        .[c('tpr','fpr')] %>% 
        # add (0,0) and (1,1) for the start and end point of ROC curve
        add_row(tpr=0,fpr=0,.before=T) %>% 
        add_row(tpr=0,fpr=0,.before=F)) %>% 
  # add a column of model name for future grouping in ggplot2
  map2(c('Gradient Boosting Machine','Random Forest','Neural Network'),
       function(x,y) x %>% add_column(Model=y)) %>% 
  # reduce four data.frame to one
  reduce(rbind) %>% 
  # plot fpr and tpr, map model to color as grouping
  ggplot(aes(fpr,tpr,col=Model))+
  geom_line()+
  geom_segment(aes(x=0,y=0,xend = 1, yend = 1),linetype = 2,col='grey')+
  xlab('False Positive Rate')+
  ylab('True Positive Rate')+
  ggtitle('Sample ROC Curve')

#Compile and Visualize Results
models <- c("GBM", "RF", "C5.0", "CART", "Neural Network", "Logistic Regression")
accuracy <- c(gbm_accuracy, rf_accuracy, c5_accuracy, cart_accuracy, nn_accuracy, glm_accuracy)*100
precision <- c(gbm_precision, rf_precision, c5_precision, cart_precision, nn_precision, glm_precision)*100
recall <- c(gbm_recall, rf_recall, c5_recall, cart_recall, nn_recall, glm_recall)*100
f1 <- c(gbm_f1, rf_f1, c5_f1, cart_f1, nn_f1, glm_f1)*100
auc <- c(gbm_auc, rf_auc, c5_auc, cart_auc, nn_auc, glm_auc)
model_kpi <- data.frame(models, accuracy, precision, recall, f1, auc)

model_kpi$models <- factor(model_kpi$models, levels = unique(model_kpi$models)[order(model_kpi$accuracy, decreasing = TRUE)])
plot_model_accuracy <- plot_ly(model_kpi, y = ~accuracy, x = ~models, type = "bar") %>%
  layout(yaxis = list(title = "Accuracy", range = c(0, 100)), xaxis = list(title = "Model"))
plot_model_accuracy

model_kpi$models <- factor(model_kpi$models, levels = unique(model_kpi$models)[order(model_kpi$f1, decreasing = TRUE)])
plot_model_f1 <- plot_ly(model_kpi, y = ~f1, x = ~models, type = "bar") %>%
  layout(yaxis = list(title = "F1", range = c(0, 30)), xaxis = list(title = "Model"))
plot_model_f1

model_kpi$models <- factor(model_kpi$models, levels = unique(model_kpi$models)[order(model_kpi$auc, decreasing = TRUE)])
plot_model_auc <- plot_ly(model_kpi, y = ~auc, x = ~models, type = "bar") %>%
  layout(yaxis = list(title = "AUC", range = c(0, 1)), xaxis = list(title = "Model"))
plot_model_auc

#FEATURE IMPORTANCE
feat_spac <- test[, x]
pred <- function(model, newdata) {
  results <- as.data.frame(h2o.predict(model, as.h2o(newdata)))
  return(results[[3L]])
}

h2o.varimp_plot(best_neuralnet, num_of_features = 50)


