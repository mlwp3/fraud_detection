setwd("../data")

library(caret)
library(pROC)
library(splines)

claims <- read.csv("train_final.csv")
str(claims)
# claims$num_age_of_vehicle <- (claims$AgeOfVehicle=="new")*0.5
# claims$num_age_of_vehicle <- claims$num_age_of_vehicle+(claims$AgeOfVehicle=="2 years")*1.5
# claims$num_age_of_vehicle <- claims$num_age_of_vehicle+(claims$AgeOfVehicle=="3 years")*2.5
# claims$num_age_of_vehicle <- claims$num_age_of_vehicle+(claims$AgeOfVehicle=="4 years")*3.5
# claims$num_age_of_vehicle <- claims$num_age_of_vehicle+(claims$AgeOfVehicle=="5 years")*4.5
# claims$num_age_of_vehicle <- claims$num_age_of_vehicle+(claims$AgeOfVehicle=="6 years")*5.5
# claims$num_age_of_vehicle <- claims$num_age_of_vehicle+(claims$AgeOfVehicle=="7 years")*6.5
# claims$num_age_of_vehicle <- claims$num_age_of_vehicle+(claims$AgeOfVehicle=="more than 7")*7.5
# claims$AgeOfVehicle7Plus <- claims$AgeOfVehicle=="more than 7"
#A model with all the variables
logit_mod <- glm(claims$FraudFound~ ., data=claims,family = binomial(link="logit"))

#Backward regression to eliminate some variables
#YES I REALIZE THIS IS LAZY
#step(logit_mod,direction="backward",trace=FALSE)

#Here is what remains, mostly
#logit_mod2 <- glm(claims$FraudFound~DriverGender+Fault+VehicleCategory+Deductible+ns(num_age_of_vehicle,2)+AgeOfVehicle7Plus+IncidentYear+BasePolicy,data=claims,family=binomial(link="logit"))

#Weirdly this one works slightly better than n-splines which I was sure would help.
logit_mod2 <- glm(claims$FraudFound~DriverGender+Fault+VehicleCategory+Deductible+AgeOfVehicle+IncidentYear+BasePolicy,data=claims,family=binomial(link="logit"))

summary(logit_mod2)

test_data <- read.csv("test_final.csv")
# test_data$num_age_of_vehicle <- (test_data$AgeOfVehicle=="new")*0.5
# test_data$num_age_of_vehicle <- test_data$num_age_of_vehicle+(test_data$AgeOfVehicle=="2 years")*1.5
# test_data$num_age_of_vehicle <- test_data$num_age_of_vehicle+(test_data$AgeOfVehicle=="3 years")*2.5
# test_data$num_age_of_vehicle <- test_data$num_age_of_vehicle+(test_data$AgeOfVehicle=="4 years")*3.5
# test_data$num_age_of_vehicle <- test_data$num_age_of_vehicle+(test_data$AgeOfVehicle=="5 years")*4.5
# test_data$num_age_of_vehicle <- test_data$num_age_of_vehicle+(test_data$AgeOfVehicle=="6 years")*5.5
# test_data$num_age_of_vehicle <- test_data$num_age_of_vehicle+(test_data$AgeOfVehicle=="7 years")*6.5
# test_data$num_age_of_vehicle <- test_data$num_age_of_vehicle+(test_data$AgeOfVehicle=="more than 7")*7.5
# test_data$AgeOfVehicle7Plus <- test_data$AgeOfVehicle=="more than 7"

fraud_prediction <- predict.glm(logit_mod2, newdata=test_data, type="response")

#Cross-entropy
sum(-test_data$FraudFound*log(fraud_prediction)-(1-test_data$FraudFound)*log(1-fraud_prediction))

binary_fraud_prediction <- (fraud_prediction>=quantile(fraud_prediction,.95))*1 #completely arbitrary cutoff; the AUC gets worse the farther away from 1 you get. 0.95 is just a linear decision boundary at the 95th percentile of predicted fraud probability.
roc_obj <- roc(binary_fraud_prediction,test_data$FraudFound)
auc(roc_obj)
confusionMatrix(as.factor(binary_fraud_prediction),as.factor(test_data$FraudFound))
