# ============================================================
# Statistics and Statistical Data Mining
# Coursework 2 – Credit Default Classification
# ============================================================

# This coursework compares multiple classification models
# to predict credit default using a given dataset.

# Models implemented:
# 1. Logistic Regression
# 2. Linear Discriminant Analysis (LDA)
# 3. Decision Tree
# 4. Random Forest
# 5. Gradient Boosting

set.seed(123)  # Ensures reproducibility of results

# ------------------------------------------------------------
# Load required libraries
# ------------------------------------------------------------
library(ISLR)         # Data and statistical learning tools
library(MASS)         # LDA
library(tree)         # Decision trees
library(gbm)          # Gradient Boosting Machines
library(randomForest) # Random Forest

# ------------------------------------------------------------
# Load training and testing datasets
# ------------------------------------------------------------
train = creditdefault_train
test  = creditdefault_test

# ------------------------------------------------------------
# Rename variables for clarity
# ------------------------------------------------------------
custom_names = c(
  "Y", "Credit_Limit", "Gender", "Education", "Marital_Status", "Age",
  "Repay_Sep", "Repay_Aug", "Repay_Jul", "Repay_Jun", "Repay_May", "Repay_Apr",
  "Bill_Sep", "Bill_Aug", "Bill_Jul", "Bill_Jun", "Bill_May", "Bill_Apr",
  "Pay_Sep", "Pay_Aug", "Pay_Jul", "Pay_Jun", "Pay_May", "Pay_Apr"
)

names(train) = custom_names
names(test)  = custom_names

# Preview datasets
head(train)
head(test)

# ------------------------------------------------------------
# Convert target variable to factor (classification task)
# ------------------------------------------------------------
train$Y = as.factor(train$Y)
test$Y  = as.factor(test$Y)

# Store target variables separately
train_target = train$Y
test_target  = test$Y

# ------------------------------------------------------------
# Exploratory Data Analysis
# ------------------------------------------------------------
summary(train)  # Summary statistics
str(train)      # Data structure

# Dataset dimensions
nrow(train); ncol(train)
nrow(test);  ncol(test)

# Check for missing values
miss = sum(is.na(train))
miss

# Pairwise scatter plots for selected feature groups
pairs(train[, c(names(train)[2:7],  "Y")])
pairs(train[, c(names(train)[8:13], "Y")])
pairs(train[, c(names(train)[14:19],"Y")])
pairs(train[, c(names(train)[20:24],"Y")])

# ------------------------------------------------------------
# Feature Scaling (Min-Max Normalisation)
# ------------------------------------------------------------

# Normalisation function
normalize = function(x, min_val, max_val) {
  (x - min_val) / (max_val - min_val)
}

# Remove target variable before scaling
train_set = subset(train, select = -Y)
test_set  = subset(test,  select = -Y)

# Compute min and max values from training data only
mins = sapply(train_set, min)
maxs = sapply(train_set, max)

# Apply scaling
scaled_train_set = as.data.frame(mapply(normalize, train_set, mins, maxs))
scaled_test_set  = as.data.frame(mapply(normalize, test_set,  mins, maxs))

# Add target variable back (target is NOT scaled)
scaled_train_set$Y = train_target
scaled_test_set$Y  = test_target

# ------------------------------------------------------------
# Accuracy Function
# ------------------------------------------------------------
accuracy_fn <- function(pred, actual) {
  sum(pred == actual) / length(actual)
}

# ------------------------------------------------------------
# Evaluation metrics function (without confusion matrix)
# ------------------------------------------------------------
evaluation_metrics = function(pred, actual) {
  
  TP = sum(pred == "1" & actual == "1")
  TN = sum(pred == "0" & actual == "0")
  FP = sum(pred == "1" & actual == "0")
  FN = sum(pred == "0" & actual == "1")
  
  accuracy  = (TP + TN) / length(actual)
  precision = TP / (TP + FP)
  recall    = TP / (TP + FN)
  f1_score  = 2 * (precision * recall) / (precision + recall)
  
  c(
    Accuracy  = accuracy,
    Precision = precision,
    Recall    = recall,
    F1_Score  = f1_score
  )
}

# ------------------------------------------------------------
# Backward Variable Selection (Logistic Regression)
# ------------------------------------------------------------

# Fit full logistic regression model
full_model = glm(Y ~ ., data = scaled_train_set, family = binomial)

# Perform backward stepwise selection using AIC
backward_model = step(full_model, direction = "backward", trace = FALSE)

# Summary of reduced model
summary(backward_model)

# Extract selected variables
selected_vars = names(coef(backward_model))[-1]

# Create reduced training and testing datasets
train_reduced = scaled_train_set[, c(selected_vars, "Y")]
test_reduced  = scaled_test_set[,  c(selected_vars, "Y")]

# ============================================================
# Model Training and Evaluation
# ============================================================

# ------------------------------------------------------------
# 1. Logistic Regression
# ------------------------------------------------------------
log_model = glm(Y ~ ., data = train_reduced, family = binomial)

log_prob = predict(log_model, newdata = test_reduced, type = "response")
log_pred = ifelse(log_prob > 0.5, 1, 0)
log_pred = as.factor(log_pred)

log_metrics = evaluation_metrics(log_pred, test_reduced$Y)

# ------------------------------------------------------------
# 2. Linear Discriminant Analysis (LDA)
# ------------------------------------------------------------
lda_model = lda(Y ~ ., data = train_reduced)

lda_pred = predict(lda_model, newdata = test_reduced)$class
lda_metrics = evaluation_metrics(lda_pred, test_reduced$Y)

# ------------------------------------------------------------
# 3. Decision Tree
# ------------------------------------------------------------
tree_model = tree(Y ~ ., data = train_reduced)

tree_pred = predict(tree_model, newdata = test_reduced, type = "class")
tree_metrics = evaluation_metrics(tree_pred, test_reduced$Y)

# ------------------------------------------------------------
# 4. Random Forest
# ------------------------------------------------------------
rf_model = randomForest(
  Y ~ ., data = train_reduced,
  ntree = 500,
  importance = TRUE
)

rf_pred = predict(rf_model, newdata = test_reduced)
rf_metrics = evaluation_metrics(rf_pred, test_reduced$Y)


# ------------------------------------------------------------
# 5. Gradient Boosting
# ------------------------------------------------------------

# Convert factor response to numeric for GBM (0/1)
train_reduced$Y_gbm = as.numeric(as.character(train_reduced$Y))
test_reduced$Y_gbm  = as.numeric(as.character(test_reduced$Y))

set.seed(123)

gbm_model = gbm(
  Y_gbm ~ . - Y,
  data = train_reduced,
  distribution = "bernoulli",
  n.trees = 3000,
  interaction.depth = 3,
  shrinkage = 0.01,
  cv.folds = 5,
  verbose = FALSE
)

# Select optimal number of trees using cross-validation
best_iter = gbm.perf(gbm_model, method = "cv")

gbm_prob = predict(
  gbm_model, newdata = test_reduced,
  n.trees = best_iter, type = "response"
)

gbm_pred = ifelse(gbm_prob > 0.5, 1, 0)
gbm_pred = as.factor(gbm_pred)

gbm_metrics = evaluation_metrics(gbm_pred, test_reduced$Y)


# ------------------------------------------------------------
# Model Performance Comparison
# ------------------------------------------------------------
performance_table = data.frame(
  Model = c("Logistic Regression", "LDA", "Decision Tree", 
            "Random Forest", "Gradient Boosting"),
  
  Accuracy = c(log_metrics["Accuracy"],
               lda_metrics["Accuracy"],
               tree_metrics["Accuracy"],
               rf_metrics["Accuracy"],
               gbm_metrics["Accuracy"]),
  
  Precision = c(log_metrics["Precision"],
                lda_metrics["Precision"],
                tree_metrics["Precision"],
                rf_metrics["Precision"],
                gbm_metrics["Precision"]),
  
  Recall = c(log_metrics["Recall"],
             lda_metrics["Recall"],
             tree_metrics["Recall"],
             rf_metrics["Recall"],
             gbm_metrics["Recall"]),
  
  F1_Score = c(log_metrics["F1_Score"],
               lda_metrics["F1_Score"],
               tree_metrics["F1_Score"],
               rf_metrics["F1_Score"],
               gbm_metrics["F1_Score"])
)

print(performance_table)

# Identify best-performing model
best_model = performance_table[which.max(performance_table$Accuracy), ]
print(best_model)

# ------------------------------------------------------------
# Variable Importance
# ------------------------------------------------------------

# Random Forest variable importance
varImpPlot(rf_model)

# Gradient Boosting variable importance
summary(gbm_model, n.trees = best_iter)

# ------------------------------------------------------------
# Threshold-based evaluation
# ------------------------------------------------------------
threshold_metrics = function(prob, actual, threshold) {
  
  pred = ifelse(prob >= threshold, 1, 0)
  pred = as.factor(pred)
  
  TP = sum(pred == "1" & actual == "1")
  TN = sum(pred == "0" & actual == "0")
  FP = sum(pred == "1" & actual == "0")
  FN = sum(pred == "0" & actual == "1")
  
  accuracy  = (TP + TN) / length(actual)
  precision = TP / (TP + FP)
  recall    = TP / (TP + FN)
  f1_score  = 2 * (precision * recall) / (precision + recall)
  
  c(Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1_Score = f1_score)
}

# ------------------------------------------------------------
# Fixed classification threshold 
# ------------------------------------------------------------
threshold = 0.15

log_pred_02 = ifelse(log_prob >= threshold, 1, 0)
log_pred_02 = as.factor(log_pred_02)
log_metrics_02 = evaluation_metrics(log_pred_02, test_reduced$Y)


lda_prob = predict(lda_model, newdata = test_reduced)$posterior[, "1"]
lda_pred_02 = ifelse(lda_prob >= threshold, 1, 0)
lda_pred_02 = as.factor(lda_pred_02)
lda_metrics_02 = evaluation_metrics(lda_pred_02, test_reduced$Y)

tree_prob = predict(tree_model, newdata = test_reduced)[, "1"]
tree_pred_02 = ifelse(tree_prob >= threshold, 1, 0)
tree_pred_02 = as.factor(tree_pred_02)
tree_metrics_02 = evaluation_metrics(tree_pred_02, test_reduced$Y)


rf_prob = predict(rf_model, newdata = test_reduced, type = "prob")[, "1"]
rf_pred_02 = ifelse(rf_prob >= threshold, 1, 0)
rf_pred_02 = as.factor(rf_pred_02)
rf_metrics_02 = evaluation_metrics(rf_pred_02, test_reduced$Y)


gbm_pred_02 = ifelse(gbm_prob >= threshold, 1, 0)
gbm_pred_02 = as.factor(gbm_pred_02)
gbm_metrics_02 = evaluation_metrics(gbm_pred_02, test_reduced$Y)


performance_table_02 = data.frame(
  Model = c("Logistic Regression", "LDA",
            "Decision Tree", "Random Forest",
            "Gradient Boosting"),
  
  Accuracy = c(log_metrics_02["Accuracy"],
               lda_metrics_02["Accuracy"],
               tree_metrics_02["Accuracy"],
               rf_metrics_02["Accuracy"],
               gbm_metrics_02["Accuracy"]),
  
  Precision = c(log_metrics_02["Precision"],
                lda_metrics_02["Precision"],
                tree_metrics_02["Precision"],
                rf_metrics_02["Precision"],
                gbm_metrics_02["Precision"]),
  
  Recall = c(log_metrics_02["Recall"],
             lda_metrics_02["Recall"],
             tree_metrics_02["Recall"],
             rf_metrics_02["Recall"],
             gbm_metrics_02["Recall"]),
  
  F1_Score = c(log_metrics_02["F1_Score"],
               lda_metrics_02["F1_Score"],
               tree_metrics_02["F1_Score"],
               rf_metrics_02["F1_Score"],
               gbm_metrics_02["F1_Score"])
)

print(performance_table_02)

